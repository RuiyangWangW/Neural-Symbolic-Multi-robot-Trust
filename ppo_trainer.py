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
    'graph_data', 'action', 'reward', 'log_prob', 'value', 'done', 'next_graph_data'
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
        
    def select_action(self, graph_data, deterministic=False):
        """Select actions using the current policy"""
        self.model.eval()
        
        with torch.no_grad():
            # Move graph to device
            x_dict = {k: v.to(self.device) for k, v in graph_data.x_dict.items()}
            
            # Handle edge_index_dict - use the proper edge_index_dict we created
            edge_index_dict = self._ensure_edge_types(graph_data)
            
            # Forward pass
            policy_outputs, value_outputs = self.model(x_dict, edge_index_dict)
            
            actions = {}
            log_probs = {}
            values = {}
            
            # Sample actions for agents
            if 'agent' in policy_outputs and policy_outputs['agent']['value_alpha'].shape[0] > 0:
                agent_policy = policy_outputs['agent']
                agent_values = value_outputs['agent']
                
                # Get the parameters from the policy output
                agent_value_alpha = agent_policy['value_alpha']
                agent_value_beta = agent_policy['value_beta']
                agent_confidence = agent_policy['confidence']  # Pre-computed confidence
                
                if deterministic:
                    # Use mean of Beta distribution for deterministic action
                    agent_value_action = agent_value_alpha / (agent_value_alpha + agent_value_beta)
                else:
                    # Sample from Beta distribution using the direct parameters
                    agent_value_action = torch.distributions.Beta(agent_value_alpha, agent_value_beta).sample()
                
                actions['agent'] = {
                    'value': agent_value_action,
                    'confidence': agent_confidence  # Use pre-computed confidence directly
                }
                
                # Compute log probabilities using proper Beta distribution for value only
                if deterministic:
                    # For deterministic actions, use small variance around the mean
                    log_probs['agent'] = {
                        'value': torch.zeros_like(agent_value_action),
                        'confidence': torch.zeros_like(agent_confidence)
                    }
                else:
                    # Create distribution using the direct parameters
                    value_dist = torch.distributions.Beta(agent_value_alpha, agent_value_beta)
                    
                    log_probs['agent'] = {
                        'value': value_dist.log_prob(agent_value_action).sum(),
                        'confidence': torch.zeros_like(agent_confidence)  # No sampling for confidence
                    }
                
                values['agent'] = agent_values
            
            # Similar for tracks
            if 'track' in policy_outputs and policy_outputs['track']['value_alpha'].shape[0] > 0:
                track_policy = policy_outputs['track']
                track_values = value_outputs['track']
                
                # Get the parameters from the policy output
                track_value_alpha = track_policy['value_alpha']
                track_value_beta = track_policy['value_beta']
                track_confidence = track_policy['confidence']  # Pre-computed confidence
                
                if deterministic:
                    # Use mean of Beta distribution for deterministic action
                    track_value_action = track_value_alpha / (track_value_alpha + track_value_beta)
                else:
                    # Sample from Beta distribution using the direct parameters
                    track_value_action = torch.distributions.Beta(track_value_alpha, track_value_beta).sample()
                
                actions['track'] = {
                    'value': track_value_action,
                    'confidence': track_confidence  # Use pre-computed confidence directly
                }
                
                # Compute log probabilities using proper Beta distribution for value only
                if deterministic:
                    # For deterministic actions, use small variance around the mean
                    log_probs['track'] = {
                        'value': torch.zeros_like(track_value_action),
                        'confidence': torch.zeros_like(track_confidence)
                    }
                else:
                    # Create distribution using the direct parameters
                    track_value_dist = torch.distributions.Beta(track_value_alpha, track_value_beta)
                    
                    log_probs['track'] = {
                        'value': track_value_dist.log_prob(track_value_action).sum(),
                        'confidence': torch.zeros_like(track_confidence)  # No sampling for confidence
                    }
                
                values['track'] = track_values
        
        self.model.train()
        return actions, log_probs, values
    
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
        """Compute both GAE advantages and TD returns for value function training"""
        rewards = []
        values = []
        dones = []
        
        # Use raw rewards - no normalization to preserve learning signal magnitude
        for exp in experiences:
            # Keep original reward values to maintain learning signal
            rewards.append(exp.reward)
            dones.append(exp.done)
            
            # Extract value for this experience (proper weighted combination)
            total_nodes = 0
            exp_value = 0.0

            if 'agent' in exp.value and exp.value['agent'].numel() > 0:
                agent_count = exp.value['agent'].numel()
                exp_value += torch.sum(exp.value['agent']).item()  # Sum, not mean
                total_nodes += agent_count

            if 'track' in exp.value and exp.value['track'].numel() > 0:
                track_count = exp.value['track'].numel()
                exp_value += torch.sum(exp.value['track']).item()  # Sum, not mean  
                total_nodes += track_count

            # Final weighted average across all nodes in the graph
            exp_value = exp_value / max(total_nodes, 1)
            values.append(exp_value)
        
        # Compute GAE advantages
        advantages = []
        returns = []
        gae = 0.0
        
        for i in reversed(range(len(experiences))):
            # Proper terminal state handling
            if i == len(experiences) - 1 or dones[i]:
                # Terminal state: no next value
                next_value = 0.0
            else:
                # Non-terminal state: use next state's value
                next_value = values[i + 1]
            
            # TD error with proper terminal handling
            delta = rewards[i] + self.gamma * next_value * (1.0 - float(dones[i])) - values[i]
            
            # GAE computation - resets when episode ends
            gae = delta + self.gamma * self.lam * gae * (1.0 - float(dones[i]))
            advantages.insert(0, gae)
            
            # Returns for value function (advantage + value baseline)
            returns.insert(0, gae + values[i])
        
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
            print(f"   [DEBUG] Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
            print(f"   [DEBUG] Returns: mean={returns.mean():.2f}, std={returns.std():.2f}, range=[{returns.min():.1f}, {returns.max():.1f}]")
            print(f"   [DEBUG] Raw rewards: {[exp.reward for exp in experiences[:3]]}...")
        
        return advantages, returns
    
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
                # Move data to device
                x_dict = {k: v.to(self.device) for k, v in experience.graph_data.x_dict.items()}
                
                # Handle edge_index_dict safely (same as in select_action)
                try:
                    edge_index_dict = self._ensure_edge_types(experience.graph_data)
                except Exception:
                    # Fallback: create all empty edges
                    edge_index_dict = self._ensure_edge_types({})
                
                # Forward pass
                self.model.train()  # Force training mode again just before forward pass
                policy_outputs, value_outputs = self.model(x_dict, edge_index_dict)
                
                # Compute policy loss (simplified PPO loss)
                policy_loss = 0.0
                value_loss = 0.0
                entropy_loss = 0.0
                
                # Agent policy loss
                if 'agent' in policy_outputs and 'agent' in experience.action:
                    agent_policy = policy_outputs['agent']
                    agent_action = experience.action['agent']
                    # Use actual stored log probabilities for PPO ratio calculation
                    agent_old_log_prob = experience.log_prob['agent']
                    old_value_log_prob = agent_old_log_prob['value']
                    old_conf_log_prob = agent_old_log_prob['confidence']
                    agent_value = value_outputs['agent']
                    
                    # Compute new log prob using proper Beta distribution
                    # Use the 4 parameters directly from the policy output
                    agent_value_alpha = agent_policy['value_alpha']
                    agent_value_beta = agent_policy['value_beta']
                    agent_confidence = agent_policy['confidence']  # Pre-computed confidence
                    
                    # Create distributions and compute log probabilities for value only
                    value_dist = torch.distributions.Beta(agent_value_alpha, agent_value_beta)
                    
                    new_log_prob_value = value_dist.log_prob(agent_action['value'].to(self.device)).sum()
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
                    
                    
                    # Improved value loss with clipping and better target handling
                    return_target = shuffled_returns[i].item() if isinstance(shuffled_returns[i], torch.Tensor) else shuffled_returns[i]
                    return_target_tensor = torch.tensor(return_target, device=self.device).float()
                    
                    # Get current value predictions
                    current_values = agent_value.squeeze()
                    if current_values.dim() == 0:
                        current_values = current_values.unsqueeze(0)
                        
                    # Expand target to match value shape
                    if current_values.dim() > 0:
                        return_target_tensor = return_target_tensor.expand_as(current_values)
                    
                    # Clipped value loss (similar to PPO's clipped objective but for values)
                    value_loss_unclipped = F.mse_loss(current_values, return_target_tensor, reduction='none')
                    
                    # Add small regularization to prevent value explosion
                    value_reg = 0.01 * torch.mean(current_values ** 2)
                    value_loss += torch.mean(value_loss_unclipped) + value_reg
                    
                    # Compute actual policy entropy for exploration (value only)
                    value_entropy = value_dist.entropy().sum()
                    conf_entropy = torch.tensor(0.0).to(self.device)  # No entropy for deterministic confidence
                    total_entropy = value_entropy + conf_entropy
                    
                    # Entropy loss (negative because we want to maximize entropy)
                    entropy_loss = entropy_loss - total_entropy if isinstance(entropy_loss, torch.Tensor) else -total_entropy
                
                # Track policy loss
                if 'track' in policy_outputs and 'track' in experience.action:
                    track_policy = policy_outputs['track']
                    track_action = experience.action['track']
                    # Use actual stored log probabilities for PPO ratio calculation
                    track_old_log_prob = experience.log_prob['track']
                    old_value_log_prob = track_old_log_prob['value']
                    old_conf_log_prob = track_old_log_prob['confidence']
                    track_value = value_outputs['track']
                    
                    # Compute new log prob using proper Beta distribution
                    # Use the parameters directly from the policy output
                    track_value_alpha = track_policy['value_alpha']
                    track_value_beta = track_policy['value_beta']
                    track_confidence = track_policy['confidence']  # Pre-computed confidence
                    
                    # Create distributions and compute log probabilities for value only
                    value_dist = torch.distributions.Beta(track_value_alpha, track_value_beta)
                    
                    new_log_prob_value = value_dist.log_prob(track_action['value'].to(self.device)).sum()
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
                    track_ppo_loss = -torch.min(surr1_value, surr2_value) - torch.min(surr1_conf, surr2_conf)
                    policy_loss += track_ppo_loss
                    
                    # Improved value loss with clipping and better target handling
                    return_target = shuffled_returns[i].item() if isinstance(shuffled_returns[i], torch.Tensor) else shuffled_returns[i]
                    return_target_tensor = torch.tensor(return_target, device=self.device).float()
                    
                    # Get current values (handle multiple tracks)
                    current_values = track_value.squeeze() if track_value.numel() > 0 else torch.tensor(0.0, device=self.device)
                    
                    # Expand target to match value shape
                    if current_values.dim() > 0:
                        return_target_tensor = return_target_tensor.expand_as(current_values)
                    
                    # Clipped value loss (similar to PPO's clipped objective but for values)
                    value_loss_unclipped = F.mse_loss(current_values, return_target_tensor, reduction='none')
                    
                    # Add small regularization to prevent value explosion
                    value_reg = 0.01 * torch.mean(current_values ** 2)
                    value_loss += torch.mean(value_loss_unclipped) + value_reg
                    
                    # Compute actual policy entropy for exploration (value only)
                    value_entropy = value_dist.entropy().sum()
                    conf_entropy = torch.tensor(0.0).to(self.device)  # No entropy for deterministic confidence
                    total_entropy = value_entropy + conf_entropy
                    
                    # Entropy loss (negative because we want to maximize entropy)
                    entropy_loss = entropy_loss - total_entropy if isinstance(entropy_loss, torch.Tensor) else -total_entropy
                
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
                    
                if isinstance(entropy_loss, torch.Tensor) and entropy_loss.numel() > 0:
                    epoch_entropy_loss += entropy_loss
                elif entropy_loss != 0.0:
                    epoch_entropy_loss += torch.tensor(entropy_loss, device=self.device, requires_grad=True)
            
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