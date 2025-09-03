"""
PPO Trainer for the Trust GNN

This module contains the PPOTrainer class that implements the Proximal Policy Optimization
algorithm for training the trust-based Graph Neural Network.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from typing import Dict, Any, Optional, List

# PPO experience tuple
PPOExperience = namedtuple('PPOExperience', [
    'graph_data', 'action', 'reward', 'log_prob', 'value', 'done', 'next_graph_data', 'global_graph_data'
])


class PPOTrainer:
    """PPO trainer for the trust GNN"""
    
    # Class constants for edge types
    REQUIRED_EDGE_TYPES = [
        ('agent', 'in_fov_and_observed', 'track'),
        ('track', 'observed_and_in_fov_by', 'agent'),
        ('agent', 'in_fov_only', 'track'),
        ('track', 'in_fov_only_by', 'agent'),
        # ('agent', 'isProximal', 'agent')  # TEMPORARILY DISABLED
    ]
    
    def __init__(self, model, learning_rate: float = 3e-3,  # Increased for faster learning
                 device: torch.device = torch.device('cpu')):
        self.model = model.to(device)
        self.device = device
        
        # PPO hyperparameters
        self.lr = learning_rate
        self.eps_clip = 0.2  # PPO clipping parameter
        self.gamma = 0.99    # Discount factor
        self.lam = 0.95      # GAE lambda
        self.value_coef = 1.0  # Value function loss coefficient
        self.entropy_coef = 0.02  # Higher entropy for better exploration during learning 
        
        # Single unified optimizer for better stability
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler for better convergence (much less aggressive)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.95)
        
        # Store all parameters for gradient clipping
        self.all_params = list(self.model.parameters())
        
        # Episode-based experience storage (cleared after each update)
        self.episode_experiences = []  # List of episode sequences
        self.current_episode = []      # Current episode being collected
        
    def select_action_ego(self, ego_graph_data, global_state, deterministic=False):
        """Select action for a single robot using its ego-graph (CTDE: Decentralized Actor)"""
        self.model.eval()
        
        with torch.no_grad():
            # Use ego-graph for policy (decentralized actor)
            x_dict = {k: v.to(self.device) for k, v in ego_graph_data.x_dict.items()}
            edge_index_dict = self._ensure_edge_types(ego_graph_data)
            
            # Forward pass for policy using ego-graph (decentralized actor)
            policy_outputs, _ = self.model(x_dict, edge_index_dict, policy_only=True)
            
            # Use global state for value function (centralized critic)
            global_x_dict = {k: v.to(self.device) for k, v in global_state.x_dict.items()}
            global_edge_index_dict = self._ensure_edge_types(global_state)
            
            # Forward pass for value using global graph (centralized critic)
            _, value_outputs = self.model(global_x_dict, global_edge_index_dict, value_only=True)
            
            actions = {}
            log_probs = {}
            values = {}
            
            # Find ego robot in the ego-graph (should be the first agent)
            ego_robot_id = ego_graph_data._ego_robot_id
            ego_agent_idx = None
            if hasattr(ego_graph_data, 'agent_nodes'):
                for robot_id, agent_idx in ego_graph_data.agent_nodes.items():
                    if robot_id == ego_robot_id:
                        ego_agent_idx = agent_idx
                        break
            
            # Sample action only for the ego robot
            if ('agent' in policy_outputs and ego_agent_idx is not None and 
                ego_agent_idx < policy_outputs['agent']['value_alpha'].shape[0]):
                
                agent_policy = policy_outputs['agent']
                
                # Get parameters for ego robot only
                ego_value_alpha = agent_policy['value_alpha'][ego_agent_idx]
                ego_value_beta = agent_policy['value_beta'][ego_agent_idx]
                ego_confidence = agent_policy['confidence'][ego_agent_idx]
                
                if deterministic:
                    ego_value_action = ego_value_alpha / (ego_value_alpha + ego_value_beta)
                else:
                    ego_value_action = torch.distributions.Beta(ego_value_alpha, ego_value_beta).sample()
                
                actions['agent'] = {
                    'value': ego_value_action.unsqueeze(0),  # Keep batch dimension
                    'confidence': ego_confidence.unsqueeze(0)
                }
                
                # Compute log probabilities
                if deterministic:
                    log_probs['agent'] = {
                        'value': torch.zeros_like(ego_value_action),
                        'confidence': torch.zeros_like(ego_confidence)
                    }
                else:
                    value_dist = torch.distributions.Beta(ego_value_alpha, ego_value_beta)
                    log_probs['agent'] = {
                        'value': value_dist.log_prob(ego_value_action),
                        'confidence': torch.zeros_like(ego_confidence)
                    }
            
            # For tracks, include all tracks in ego-graph
            if 'track' in policy_outputs and policy_outputs['track']['value_alpha'].shape[0] > 0:
                track_policy = policy_outputs['track']
                
                track_value_alpha = track_policy['value_alpha']
                track_value_beta = track_policy['value_beta']
                track_confidence = track_policy['confidence']
                
                if deterministic:
                    track_value_action = track_value_alpha / (track_value_alpha + track_value_beta)
                else:
                    track_value_action = torch.distributions.Beta(track_value_alpha, track_value_beta).sample()
                
                actions['track'] = {
                    'value': track_value_action,
                    'confidence': track_confidence
                }
                
                if deterministic:
                    log_probs['track'] = {
                        'value': torch.zeros_like(track_value_action),
                        'confidence': torch.zeros_like(track_confidence)
                    }
                else:
                    track_value_dist = torch.distributions.Beta(track_value_alpha, track_value_beta)
                    log_probs['track'] = {
                        'value': track_value_dist.log_prob(track_value_action).sum(),
                        'confidence': torch.zeros_like(track_confidence)
                    }
            
            # Use centralized critic value (from global state)
            values = self._extract_global_value(value_outputs, global_state)
        
        self.model.train()
        return actions, log_probs, values
    
    def _extract_global_value(self, value_outputs, global_state):
        """Extract a scalar value from the global state for centralized critic"""
        # With the new architecture, value_outputs is already a scalar tensor from the centralized critic
        if isinstance(value_outputs, torch.Tensor):
            if value_outputs.numel() == 1:
                return value_outputs
            else:
                return value_outputs.mean()
        elif isinstance(value_outputs, (int, float)):
            return torch.tensor(value_outputs, device=self.device)
        else:
            # Fallback for old format
            total_value = 0.0
            total_nodes = 0
            
            if isinstance(value_outputs, dict):
                if 'agent' in value_outputs and hasattr(value_outputs['agent'], 'numel') and value_outputs['agent'].numel() > 0:
                    agent_values = value_outputs['agent'].squeeze()
                    total_value += torch.sum(agent_values).item()
                    total_nodes += agent_values.numel()
                
                if 'track' in value_outputs and hasattr(value_outputs['track'], 'numel') and value_outputs['track'].numel() > 0:
                    track_values = value_outputs['track'].squeeze()
                    total_value += torch.sum(track_values).item()
                    total_nodes += track_values.numel()
                
                global_value = total_value / max(total_nodes, 1)
                return torch.tensor(global_value, device=self.device)
            else:
                return torch.tensor(0.0, device=self.device)
    
    def _ensure_edge_types(self, graph_data_or_dict):
        """Ensure all required edge types exist in edge_index_dict, creating empty ones if missing"""
        # Handle both graph data objects and existing edge index dicts  
        if hasattr(graph_data_or_dict, 'edge_index_dict'):
            source_edges = graph_data_or_dict.edge_index_dict
        elif isinstance(graph_data_or_dict, dict):
            source_edges = graph_data_or_dict
        else:
            source_edges = {}
        
        result = {}
        for edge_type in self.REQUIRED_EDGE_TYPES:
            if edge_type in source_edges:
                result[edge_type] = source_edges[edge_type].to(self.device)
            else:
                result[edge_type] = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        return result

    def _calculate_trust_value(self, alpha, beta):
        """Calculate trust value from alpha and beta parameters"""
        return alpha / (alpha + beta)

    def _get_robot_trust_value(self, robot):
        """Get trust value for a robot"""
        if self._has_trust_attributes(robot):
            return self._calculate_trust_value(robot.trust_alpha, robot.trust_beta)
        return 0.5  # Default trust value

    def _has_trust_attributes(self, obj):
        """Check if object has trust attributes"""
        return hasattr(obj, 'trust_alpha') and hasattr(obj, 'trust_beta')

    def _reset_trust_to_default(self, obj):
        """Reset object trust to default values"""
        obj.trust_alpha = 1.0
        obj.trust_beta = 1.0
    
    def compute_gae_and_returns(self, experiences):
        """Compute both GAE advantages and TD returns for CTDE value function training
        
        CORRECTED: Ensures robots at same timestep share the same advantage
        """
        if not experiences:
            return [], []
            
        # Group experiences by timestep - assume sequential storage: [R0_t0, R1_t0, ..., R0_t1, R1_t1, ...]
        # We need to determine number of robots per timestep
        num_robots = self._estimate_robots_per_timestep(experiences)
        
        # Group experiences by timestep
        timestep_groups = []
        for i in range(0, len(experiences), num_robots):
            timestep_group = experiences[i:i + num_robots]
            if timestep_group:  # Only add non-empty groups
                timestep_groups.append(timestep_group)
        
        # Extract timestep-level data (same for all robots at each timestep)
        timestep_rewards = []
        timestep_values = []
        timestep_dones = []
        
        for timestep_group in timestep_groups:
            # All robots at this timestep should have same reward, value, and done
            first_exp = timestep_group[0]
            
            # Extract reward (should be same for all robots)
            timestep_rewards.append(first_exp.reward)
            timestep_dones.append(first_exp.done)
            
            # Extract value - use centralized critic value (should be same for all robots)
            if isinstance(first_exp.value, torch.Tensor):
                exp_value = first_exp.value.item() if first_exp.value.numel() == 1 else first_exp.value.mean().item()
            elif isinstance(first_exp.value, (int, float)):
                exp_value = first_exp.value
            else:
                # Legacy fallback for old format
                total_nodes = 0
                exp_value = 0.0
                if isinstance(first_exp.value, dict):
                    if 'agent' in first_exp.value and hasattr(first_exp.value['agent'], 'numel') and first_exp.value['agent'].numel() > 0:
                        agent_count = first_exp.value['agent'].numel()
                        exp_value += torch.sum(first_exp.value['agent']).item()
                        total_nodes += agent_count
                    if 'track' in first_exp.value and hasattr(first_exp.value['track'], 'numel') and first_exp.value['track'].numel() > 0:
                        track_count = first_exp.value['track'].numel()
                        exp_value += torch.sum(first_exp.value['track']).item()
                        total_nodes += track_count
                    exp_value = exp_value / max(total_nodes, 1)
                else:
                    exp_value = 0.0
            
            timestep_values.append(exp_value)
        
        # Compute GAE advantages at timestep level
        timestep_advantages = []
        timestep_returns = []
        gae = 0.0
        
        for i in reversed(range(len(timestep_groups))):
            # Proper terminal state handling
            if i == len(timestep_groups) - 1 or timestep_dones[i]:
                # Terminal state: no next value
                next_value = 0.0
            else:
                # Non-terminal state: use next timestep's value
                next_value = timestep_values[i + 1]
            
            # TD error with proper terminal handling
            delta = timestep_rewards[i] + self.gamma * next_value * (1.0 - float(timestep_dones[i])) - timestep_values[i]
            
            # GAE computation - resets when episode ends
            gae = delta + self.gamma * self.lam * gae * (1.0 - float(timestep_dones[i]))
            timestep_advantages.insert(0, gae)
            
            # Returns for value function (advantage + value baseline)
            timestep_returns.insert(0, gae + timestep_values[i])
        
        # Replicate timestep advantages/returns to all robots at each timestep
        advantages = []
        returns = []
        
        for i, timestep_group in enumerate(timestep_groups):
            timestep_advantage = timestep_advantages[i]
            timestep_return = timestep_returns[i]
            
            # All robots at this timestep get the SAME advantage and return
            for exp in timestep_group:
                advantages.append(timestep_advantage)
                returns.append(timestep_return)
        
        # Convert to tensors and normalize for stability
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize advantages for training stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optional: Log advantage statistics occasionally for monitoring
        if not hasattr(self, '_update_count'):
            self._update_count = 0
        self._update_count += 1
        
        # Debug: Print advantage and return statistics every few updates
        if self._update_count % 10 == 0:
            print(f"   [DEBUG] CORRECTED GAE: Timesteps={len(timestep_groups)}, Robots/timestep={num_robots}")
            print(f"   [DEBUG] Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
            print(f"   [DEBUG] Returns: mean={returns.mean():.2f}, std={returns.std():.2f}, range=[{returns.min():.1f}, {returns.max():.1f}]")
            print(f"   [DEBUG] Raw rewards: {[exp.reward for exp in experiences[:3]]}...")
        
        return advantages, returns
    
    def _estimate_robots_per_timestep(self, experiences):
        """Estimate number of robots per timestep from experience pattern"""
        if len(experiences) < 2:
            return 1
        
        # Look for reward changes to detect timestep boundaries
        # Experiences should be stored as: [R0_t0, R1_t0, R2_t0, R0_t1, R1_t1, R2_t1, ...]
        first_reward = experiences[0].reward
        robots_per_timestep = 1
        
        for i in range(1, min(len(experiences), 20)):  # Check first 20 experiences
            if experiences[i].reward != first_reward:
                # Found a timestep boundary
                robots_per_timestep = i
                break
        
        # Validation: ensure this makes sense
        if robots_per_timestep * 2 <= len(experiences):
            # Check if pattern holds for second timestep
            second_timestep_start = robots_per_timestep
            if second_timestep_start < len(experiences):
                second_reward = experiences[second_timestep_start].reward
                if second_reward != first_reward:
                    return robots_per_timestep
        
        # Fallback: assume single robot if pattern detection fails
        return 1
    
    def compute_gae_and_returns_by_episode(self, episode_list):
        """Compute GAE and returns respecting episode boundaries"""
        all_advantages = []
        all_returns = []
        
        # Process each episode separately to maintain temporal structure
        for episode_experiences in episode_list:
            if not episode_experiences:
                continue
                
            # Compute advantages and returns for this episode
            episode_advantages, episode_returns = self.compute_gae_and_returns(episode_experiences)
            all_advantages.extend(episode_advantages)
            all_returns.extend(episode_returns)
        
        return all_advantages, all_returns
    
    def add_experience(self, experience):
        """Add experience to current episode"""
        self.current_episode.append(experience)
    
    def finish_episode(self):
        """Mark current episode as complete and store it"""
        if self.current_episode:
            self.episode_experiences.append(self.current_episode.copy())
            self.current_episode = []
    
    def update_policy(self, min_experiences=32, n_epochs=4):
        """Update policy using PPO with proper experience accumulation"""
        
        # Use all available episode experiences (maintaining temporal order)
        all_experiences = []
        for episode in self.episode_experiences:
            all_experiences.extend(episode)
        
        # Need sufficient experiences for stable learning (not just episodes)
        if len(all_experiences) < min_experiences:
            return {}
        
        # Limit maximum batch size to prevent memory issues
        if len(all_experiences) > 1024:
            # Keep most recent experiences
            excess = len(all_experiences) - 1024
            # Remove oldest episodes first
            while excess > 0 and self.episode_experiences:
                removed_episode = self.episode_experiences.pop(0)
                excess -= len(removed_episode)
                all_experiences = []
                for episode in self.episode_experiences:
                    all_experiences.extend(episode)
        
        # Compute GAE advantages and returns maintaining episode boundaries
        advantages, returns = self.compute_gae_and_returns_by_episode(self.episode_experiences)
        
        # Prepare batch data
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(n_epochs):
            # CRITICAL: Ensure model is in training mode
            self.model.train()
            
            # SMART SHUFFLING: Shuffle episodes but preserve within-episode order
            # This maintains temporal structure while reducing correlation
            import random
            episode_indices = list(range(len(self.episode_experiences)))
            random.shuffle(episode_indices)  # Shuffle episode order only
            
            # Build shuffled experience list maintaining episode boundaries
            shuffled_experiences = []
            shuffled_advantages = []
            shuffled_returns = []
            
            for episode_idx in episode_indices:
                episode = self.episode_experiences[episode_idx]
                episode_start = sum(len(self.episode_experiences[i]) for i in range(episode_idx))
                episode_end = episode_start + len(episode)
                
                shuffled_experiences.extend(episode)
                shuffled_advantages.extend(advantages[episode_start:episode_end])
                shuffled_returns.extend(returns[episode_start:episode_end])
            
            # Accumulate losses for batch update (scalar accumulation to avoid deep graphs)
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy_loss = 0.0
            valid_experience_count = 0
            
            for i, experience in enumerate(shuffled_experiences):
                # FIXED: Use ego-graph for policy, global graph for value (CTDE)
                # Ego-graph for policy (decentralized actor)
                ego_x_dict = {k: v.to(self.device) for k, v in experience.graph_data.x_dict.items()}
                ego_edge_index_dict = self._ensure_edge_types(experience.graph_data)
                
                # Global graph for value (centralized critic) 
                if experience.global_graph_data is not None:
                    global_x_dict = {k: v.to(self.device) for k, v in experience.global_graph_data.x_dict.items()}
                    global_edge_index_dict = self._ensure_edge_types(experience.global_graph_data)
                else:
                    # Fallback to ego-graph if global not available (backward compatibility)
                    global_x_dict = ego_x_dict
                    global_edge_index_dict = ego_edge_index_dict
                
                # Forward pass: CTDE with separate graphs
                self.model.train()  # Force training mode again just before forward pass
                
                # Policy outputs from ego-graph (decentralized actor)
                policy_outputs, _ = self.model(ego_x_dict, ego_edge_index_dict, policy_only=True)
                
                # Value outputs from global graph (centralized critic)  
                _, value_outputs = self.model(global_x_dict, global_edge_index_dict, value_only=True)
                
                # Compute policy loss (simplified PPO loss)
                policy_loss = 0.0
                value_loss = 0.0
                agent_entropy_loss = 0.0
                track_entropy_loss = 0.0
                
                # Agent policy loss
                if 'agent' in policy_outputs and 'agent' in experience.action:
                    agent_policy = policy_outputs['agent']
                    agent_action = experience.action['agent']
                    # Use actual stored log probabilities for PPO ratio calculation
                    agent_old_log_prob = experience.log_prob['agent']
                    old_value_log_prob = agent_old_log_prob['value']
                    old_conf_log_prob = agent_old_log_prob['confidence']
                    # Use the centralized global value (scalar)
                    agent_value = value_outputs
                    
                    # FIXED: Find ego robot in the ego-graph (same logic as in select_action_ego)
                    ego_robot_id = experience.graph_data._ego_robot_id
                    ego_agent_idx = None
                    if hasattr(experience.graph_data, 'agent_nodes'):
                        for robot_id, agent_idx in experience.graph_data.agent_nodes.items():
                            if robot_id == ego_robot_id:
                                ego_agent_idx = agent_idx
                                break
                    
                    # FIXED: Only proceed if we found the ego robot in the ego-graph
                    if (ego_agent_idx is not None and 
                        ego_agent_idx < agent_policy['value_alpha'].shape[0]):
                        
                        # FIXED: Extract ego robot's parameters only
                        ego_value_alpha = agent_policy['value_alpha'][ego_agent_idx]
                        ego_value_beta = agent_policy['value_beta'][ego_agent_idx]
                        ego_confidence = agent_policy['confidence'][ego_agent_idx]
                        
                        # FIXED: Create distribution from ego robot's parameters only
                        ego_value_dist = torch.distributions.Beta(ego_value_alpha, ego_value_beta)
                        
                        # FIXED: Compute log prob for ego robot's action only (no .sum() needed)
                        new_log_prob_value = ego_value_dist.log_prob(agent_action['value'].to(self.device))
                        new_log_prob_conf = torch.zeros_like(new_log_prob_value)  # No sampling for confidence
                        
                        # PPO ratio using actual old probabilities (value only)
                        ratio_value = torch.exp(new_log_prob_value - old_value_log_prob)
                        ratio_conf = torch.tensor(1.0).to(self.device)  # No policy gradient for confidence
                        
                        # Skip experiences with extreme ratios for stability
                        if torch.isnan(ratio_value) or torch.isnan(ratio_conf):
                            continue
                        elif (ratio_value > 5.0 or ratio_conf > 5.0 or 
                              ratio_value < 0.2 or ratio_conf < 0.2):
                            continue
                        
                        # PPO clipped objective - use proper GAE advantage
                        advantage = shuffled_advantages[i]
                        
                        surr1_value = ratio_value * advantage
                        surr2_value = torch.clamp(ratio_value, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                        
                        surr1_conf = ratio_conf * advantage
                        surr2_conf = torch.clamp(ratio_conf, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                        
                        # PPO objective: maximize min(ratio * advantage, clipped_ratio * advantage)  
                        # For loss minimization: negate the objective
                        ppo_loss = -torch.min(surr1_value, surr2_value) - torch.min(surr1_conf, surr2_conf)
                        policy_loss += ppo_loss
                        
                        # FIXED: Compute entropy for ego robot only (accumulate properly)
                        ego_entropy = ego_value_dist.entropy()
                        agent_entropy_loss = -ego_entropy  # Negative because we want to maximize entropy
                
                # Track policy loss - ego robot influences all tracks in its ego-graph
                if 'track' in policy_outputs and 'track' in experience.action:
                    track_policy = policy_outputs['track']
                    track_action = experience.action['track']
                    # Use actual stored log probabilities for PPO ratio calculation
                    track_old_log_prob = experience.log_prob['track']
                    old_value_log_prob = track_old_log_prob['value']
                    old_conf_log_prob = track_old_log_prob['confidence']
                    
                    # FIXED: Ensure dimensions match
                    if (track_policy['value_alpha'].shape[0] > 0 and
                        track_action['value'].shape[0] == track_policy['value_alpha'].shape[0]):
                        
                        # Extract track policy parameters (all tracks in ego-graph)
                        track_value_alpha = track_policy['value_alpha']
                        track_value_beta = track_policy['value_beta']
                        track_confidence = track_policy['confidence']
                        
                        # FIXED: Create distributions for all tracks in ego-graph
                        track_value_dist = torch.distributions.Beta(track_value_alpha, track_value_beta)
                        
                        # FIXED: Compute log prob for all track actions (no incorrect .sum())
                        new_log_prob_value = track_value_dist.log_prob(track_action['value'].to(self.device)).sum()
                        new_log_prob_conf = torch.zeros_like(new_log_prob_value)
                        
                        # PPO ratio using actual old probabilities (value only)
                        ratio_value = torch.exp(new_log_prob_value - old_value_log_prob)
                        ratio_conf = torch.tensor(1.0).to(self.device)
                        
                        # Skip experiences with extreme ratios for stability
                        if torch.isnan(ratio_value) or torch.isnan(ratio_conf):
                            continue
                        elif (ratio_value > 5.0 or ratio_conf > 5.0 or 
                              ratio_value < 0.2 or ratio_conf < 0.2):
                            continue
                        
                        # PPO clipped objective - use proper GAE advantage
                        advantage = shuffled_advantages[i]
                        
                        surr1_value = ratio_value * advantage
                        surr2_value = torch.clamp(ratio_value, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                        
                        surr1_conf = ratio_conf * advantage
                        surr2_conf = torch.clamp(ratio_conf, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                        
                        # PPO objective: maximize min(ratio * advantage, clipped_ratio * advantage)  
                        # For loss minimization: negate the objective
                        track_ppo_loss = -torch.min(surr1_value, surr2_value) - torch.min(surr1_conf, surr2_conf)
                        policy_loss += track_ppo_loss
                        
                        # FIXED: Compute entropy for all tracks in ego-graph (accumulate properly)
                        track_entropy = track_value_dist.entropy().sum()  # Sum across all tracks in ego-graph
                        track_entropy_loss = -track_entropy  # Negative because we want to maximize entropy
                
                # FIXED: Compute value loss only once per experience (centralized critic)
                # Use centralized value for training (scalar)
                return_target = shuffled_returns[i].item() if isinstance(shuffled_returns[i], torch.Tensor) else shuffled_returns[i]
                return_target_tensor = torch.tensor(return_target, device=self.device).float()
                
                # Get current global value (scalar from centralized critic)
                if isinstance(value_outputs, torch.Tensor):
                    if value_outputs.numel() == 1:
                        current_value = value_outputs
                    else:
                        current_value = value_outputs.mean()  # Fallback for old format
                else:
                    current_value = torch.tensor(0.0, device=self.device)
                
                # Value loss using scalar values (computed once per experience)
                value_loss_unclipped = F.mse_loss(current_value, return_target_tensor)
                
                # Add small regularization
                value_reg = 0.01 * current_value ** 2
                value_loss += value_loss_unclipped + value_reg
                
                # FIXED: Combine agent and track entropy losses properly
                total_entropy_loss = agent_entropy_loss + track_entropy_loss
                
                # Accumulate losses for this epoch (scalar accumulation to avoid deep graphs)
                if isinstance(policy_loss, torch.Tensor) and policy_loss.numel() > 0:
                    epoch_policy_loss += policy_loss
                    valid_experience_count += 1
                elif policy_loss != 0.0:
                    epoch_policy_loss += torch.tensor(policy_loss, device=self.device, requires_grad=True)
                    valid_experience_count += 1
                    
                if isinstance(value_loss, torch.Tensor) and value_loss.numel() > 0:
                    epoch_value_loss += value_loss
                elif value_loss != 0.0:
                    epoch_value_loss += torch.tensor(value_loss, device=self.device, requires_grad=True)
                    
                # FIXED: Accumulate entropy loss properly (agent + track)
                if isinstance(total_entropy_loss, torch.Tensor) and total_entropy_loss.numel() > 0:
                    epoch_entropy_loss += total_entropy_loss
                elif total_entropy_loss != 0.0:
                    if isinstance(total_entropy_loss, (int, float)):
                        epoch_entropy_loss += torch.tensor(total_entropy_loss, device=self.device, requires_grad=True)
                    else:
                        epoch_entropy_loss += total_entropy_loss
            
            # BATCH UPDATE: One optimizer step per epoch, not per experience
            if valid_experience_count > 0:
                # Average losses across valid experiences
                epoch_policy_loss = epoch_policy_loss / max(valid_experience_count, 1)
                epoch_value_loss = epoch_value_loss / max(valid_experience_count, 1) if isinstance(epoch_value_loss, torch.Tensor) else torch.tensor(0.0, device=self.device)
                epoch_entropy_loss = epoch_entropy_loss / max(valid_experience_count, 1) if isinstance(epoch_entropy_loss, torch.Tensor) else torch.tensor(0.0, device=self.device)
                
                # Compute total loss for this epoch
                total_epoch_loss = epoch_policy_loss + self.value_coef * epoch_value_loss + self.entropy_coef * epoch_entropy_loss
                
                # Single optimizer update per epoch
                self.optimizer.zero_grad()
                total_epoch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
                self.optimizer.step()
                
                # Store for logging
                policy_losses.append(epoch_policy_loss)
                value_losses.append(epoch_value_loss)
                entropy_losses.append(epoch_entropy_loss)
        
        # Step learning rate scheduler ONCE per policy update (not per epoch)
        self.scheduler.step()
        
        # Final logging
        if policy_losses:
            total_policy_loss = torch.stack(policy_losses).mean()
            total_value_loss = torch.stack(value_losses).mean()
            total_entropy_loss = torch.stack(entropy_losses).mean() if entropy_losses else torch.tensor(0.0)
            total_loss = total_policy_loss + self.value_coef * total_value_loss + self.entropy_coef * total_entropy_loss
            
            
            # Keep some recent episodes for continued learning, but limit buffer size
            max_buffer_episodes = 20  # Keep last 20 episodes for better learning accumulation
            if len(self.episode_experiences) > max_buffer_episodes:
                # Remove oldest episodes to maintain buffer size
                episodes_to_remove = len(self.episode_experiences) - max_buffer_episodes
                self.episode_experiences = self.episode_experiences[episodes_to_remove:]
            
            return {
                'policy_loss': total_policy_loss.item(),
                'value_loss': total_value_loss.item(),
                'entropy_loss': total_entropy_loss.item(),
                'total_loss': total_loss.item(),
                'num_policy_updates': len(policy_losses),
                'num_value_updates': len(value_losses)
            }
        
        # If no policy losses were computed, return empty
        return {}