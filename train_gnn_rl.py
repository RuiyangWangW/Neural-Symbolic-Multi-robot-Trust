#!/usr/bin/env python3
"""
GNN Reinforcement Learning Training with PPO

This script trains the neural symbolic GNN model using Proximal Policy Optimization (PPO)
instead of supervised learning. The GNN learns to make trust update decisions that maximize
overall multi-robot system performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import os
from collections import namedtuple, deque
import random

from neural_symbolic_trust_algorithm import PPOTrustGNN  # Import PPO wrapper
# from neural_symbolic_trust_algorithm import NeuralSymbolicTrustAlgorithm  # COMMENTED OUT: Pure RL training
from simulation_environment import SimulationEnvironment
# from paper_trust_algorithm import PaperTrustAlgorithm  # COMMENTED OUT: Focus on RL algorithm only
from visualize_gnn_input_graph import visualize_gnn_input

# PPO Experience tuple
PPOExperience = namedtuple('PPOExperience', [
    'graph_data', 'action', 'reward', 'log_prob', 'value', 'done', 'next_graph_data'
])



class PPOTrainer:
    """PPO trainer for the trust GNN"""
    
    def __init__(self, model: PPOTrustGNN, learning_rate: float = 3e-3,  # Increased for faster learning
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
            edge_index_dict = {}
            required_edge_types = [
                ('agent', 'observes', 'track'),
                ('track', 'observed_by', 'agent'),
                ('agent', 'in_fov', 'track'),
                ('track', 'in_fov_by', 'agent'),
                # ('agent', 'isProximal', 'agent')  # TEMPORARILY DISABLED
            ]
            
            # Use the edge_index_dict property we added to the graph_data
            if hasattr(graph_data, 'edge_index_dict'):
                for edge_type in required_edge_types:
                    if edge_type in graph_data.edge_index_dict:
                        edge_index_dict[edge_type] = graph_data.edge_index_dict[edge_type].to(self.device)
                    else:
                        edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=self.device)
            else:
                # Fallback: create empty edge tensors
                for edge_type in required_edge_types:
                    edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=self.device)
            
            # Forward pass
            policy_outputs, value_outputs = self.model(x_dict, edge_index_dict)
            
            actions = {}
            log_probs = {}
            values = {}
            
            # Sample actions for agents
            if 'agent' in policy_outputs and policy_outputs['agent']['value'].shape[0] > 0:
                agent_policy = policy_outputs['agent']
                agent_values = value_outputs['agent']
                
                if deterministic:
                    # Use mean of policy for deterministic action
                    agent_value_action = agent_policy['value']
                    agent_conf_action = agent_policy['confidence']
                else:
                    # Sample from policy (treating outputs as logits for Beta distribution)
                    # Convert sigmoid outputs to alpha/beta parameters for Beta distribution
                    # Add minimum values to prevent collapse and extreme probability ratios
                    agent_value_alpha = agent_policy['value'] * agent_policy['confidence'] + 0.1
                    agent_value_beta = (1 - agent_policy['value']) * agent_policy['confidence'] + 0.1
                    agent_conf_alpha = agent_policy['value'] * agent_policy['confidence'] + 0.1
                    agent_conf_beta = (1 - agent_policy['value']) * agent_policy['confidence'] + 0.1

                    # Sample from Beta distributions
                    agent_value_action = torch.distributions.Beta(agent_value_alpha, agent_value_beta).sample()
                    agent_conf_action = torch.distributions.Beta(agent_conf_alpha, agent_conf_beta).sample()
                
                actions['agent'] = {
                    'value': agent_value_action,
                    'confidence': agent_conf_action
                }
                
                # Compute log probabilities using proper Beta distribution
                if deterministic:
                    # For deterministic actions, use small variance around the mean
                    log_probs['agent'] = {
                        'value': torch.zeros_like(agent_value_action),
                        'confidence': torch.zeros_like(agent_conf_action)
                    }
                else:
                    # Compute proper log probabilities from Beta distributions
                    value_dist = torch.distributions.Beta(agent_value_alpha, agent_value_beta)
                    conf_dist = torch.distributions.Beta(agent_conf_alpha, agent_conf_beta)
                    
                    log_probs['agent'] = {
                        'value': value_dist.log_prob(agent_value_action).sum(),
                        'confidence': conf_dist.log_prob(agent_conf_action).sum()
                    }
                
                values['agent'] = agent_values
            
            # Similar for tracks
            if 'track' in policy_outputs and policy_outputs['track']['value'].shape[0] > 0:
                track_policy = policy_outputs['track']
                track_values = value_outputs['track']
                
                if deterministic:
                    track_value_action = track_policy['value']
                    track_conf_action = track_policy['confidence']
                else:
                    track_value_alpha = track_policy['value'] * track_policy['confidence'] + 0.1
                    track_value_beta = (1 - track_policy['value']) * track_policy['confidence'] + 0.1
                    track_conf_alpha = track_policy['value'] * track_policy['confidence'] + 0.1
                    track_conf_beta = (1 - track_policy['value']) * track_policy['confidence'] + 0.1

                    track_value_action = torch.distributions.Beta(track_value_alpha, track_value_beta).sample()
                    track_conf_action = torch.distributions.Beta(track_conf_alpha, track_conf_beta).sample()
                
                actions['track'] = {
                    'value': track_value_action,
                    'confidence': track_conf_action
                }
                
                # Compute log probabilities using proper Beta distribution
                if deterministic:
                    # For deterministic actions, use small variance around the mean
                    log_probs['track'] = {
                        'value': torch.zeros_like(track_value_action),
                        'confidence': torch.zeros_like(track_conf_action)
                    }
                else:
                    # Compute proper log probabilities from Beta distributions
                    track_value_dist = torch.distributions.Beta(track_value_alpha, track_value_beta)
                    track_conf_dist = torch.distributions.Beta(track_conf_alpha, track_conf_beta)
                    
                    log_probs['track'] = {
                        'value': track_value_dist.log_prob(track_value_action).sum(),
                        'confidence': track_conf_dist.log_prob(track_conf_action).sum()
                    }
                
                values['track'] = track_values
        
        self.model.train()
        return actions, log_probs, values
    
    def _ensure_edge_types(self, edge_index_dict):
        """Ensure all required edge types exist"""
        required_edge_types = [
            ('agent', 'observes', 'track'),
            ('track', 'observed_by', 'agent'),
            ('agent', 'in_fov', 'track'),
            ('track', 'in_fov_by', 'agent'),
            # ('agent', 'isProximal', 'agent')  # TEMPORARILY DISABLED
        ]
        
        result = {}
        for edge_type in required_edge_types:
            if edge_type in edge_index_dict:
                result[edge_type] = edge_index_dict[edge_type].to(self.device)
            else:
                result[edge_type] = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        return result
    
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
            
            # Extract value for this experience (handle both agent and track values)
            exp_value = 0.0
            if 'agent' in exp.value:
                exp_value += torch.mean(exp.value['agent']).item()
            if 'track' in exp.value and exp.value['track'].numel() > 0:
                exp_value += torch.mean(exp.value['track']).item()
            values.append(exp_value)
        
        # Compute GAE advantages
        advantages = []
        returns = []
        gae = 0.0
        
        for i in reversed(range(len(experiences))):
            if i == len(experiences) - 1:
                next_value = 0.0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]
            
            # TD error
            delta = rewards[i] + self.gamma * next_value - values[i]
            
            # GAE computation
            gae = delta + self.gamma * self.lam * gae * (1.0 - float(dones[i]))
            advantages.insert(0, gae)
            
            # Returns for value function (advantage + value baseline)
            returns.insert(0, gae + values[i])
        
        # Convert to tensors and normalize for stability
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
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
            
            # Accumulate losses for batch update
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_entropy_losses = []
            
            for i, experience in enumerate(all_experiences):
                # Move data to device
                x_dict = {k: v.to(self.device) for k, v in experience.graph_data.x_dict.items()}
                
                # Handle edge_index_dict safely (same as in select_action)
                try:
                    existing_edges = {}
                    required_edge_types = [
                        ('agent', 'observes', 'track'),
                        ('track', 'observed_by', 'agent'),
                        ('agent', 'in_fov', 'track'),
                        ('track', 'in_fov_by', 'agent'),
                        ('agent', 'isProximal', 'agent')
                    ]
                    
                    for edge_type in required_edge_types:
                        try:
                            if hasattr(experience.graph_data, '_edge_store_dict') and edge_type in experience.graph_data._edge_store_dict:
                                edge_data = experience.graph_data[edge_type]
                                if hasattr(edge_data, 'edge_index'):
                                    existing_edges[edge_type] = edge_data.edge_index
                        except:
                            pass  # Skip if edge type doesn't exist
                    
                    edge_index_dict = self._ensure_edge_types(existing_edges)
                    
                except Exception as e:
                    # Fallback: create all empty edges
                    edge_index_dict = self._ensure_edge_types({})
                
                # Forward pass
                self.model.train()  # Force training mode again just before forward pass
                policy_outputs, value_outputs = self.model(x_dict, edge_index_dict)
                
                # Ensure debug counter exists for other code
                if not hasattr(self, '_debug_count'):
                    self._debug_count = 0
                self._debug_count += 1
                
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
                    # CRITICAL FIX: Use EXACT same parameterization as action selection
                    # Add minimum values to prevent near-zero parameters that cause extreme ratios
                    agent_value_alpha = agent_policy['value'] * agent_policy['confidence'] + 0.1
                    agent_value_beta = (1 - agent_policy['value']) * agent_policy['confidence'] + 0.1
                    agent_conf_alpha = agent_policy['value'] * agent_policy['confidence'] + 0.1
                    agent_conf_beta = (1 - agent_policy['value']) * agent_policy['confidence'] + 0.1
                    
                    # Create distributions and compute log probabilities
                    value_dist = torch.distributions.Beta(agent_value_alpha, agent_value_beta)
                    conf_dist = torch.distributions.Beta(agent_conf_alpha, agent_conf_beta)
                    
                    new_log_prob_value = value_dist.log_prob(agent_action['value'].to(self.device)).sum()
                    new_log_prob_conf = conf_dist.log_prob(agent_action['confidence'].to(self.device)).sum()
                    
                    # PPO ratio using actual old probabilities
                    ratio_value = torch.exp(new_log_prob_value - old_value_log_prob)
                    ratio_conf = torch.exp(new_log_prob_conf - old_conf_log_prob)
                    
                    
                    # Skip experiences with extreme ratios for stability
                    if torch.isnan(ratio_value) or torch.isnan(ratio_conf):
                        continue
                    elif (ratio_value > 5.0 or ratio_conf > 5.0 or 
                          ratio_value < 0.2 or ratio_conf < 0.2):
                        continue
                    
                    # PPO clipped objective - use proper GAE advantage
                    advantage = advantages[i]
                    
                    surr1_value = ratio_value * advantage
                    surr2_value = torch.clamp(ratio_value, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                    
                    surr1_conf = ratio_conf * advantage
                    surr2_conf = torch.clamp(ratio_conf, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                    
                    # PPO objective: maximize min(ratio * advantage, clipped_ratio * advantage)  
                    # For loss minimization: negate the objective
                    ppo_loss = -torch.min(surr1_value, surr2_value) - torch.min(surr1_conf, surr2_conf)
                    policy_loss += ppo_loss
                    
                    
                    # Improved value loss with clipping and better target handling
                    return_target = returns[i].item() if isinstance(returns[i], torch.Tensor) else returns[i]
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
                    
                    # Compute actual policy entropy for exploration
                    value_entropy = value_dist.entropy().sum()
                    conf_entropy = conf_dist.entropy().sum()
                    total_entropy = value_entropy + conf_entropy
                    
                    # Entropy loss (negative because we want to maximize entropy)
                    entropy_loss = entropy_loss - total_entropy if isinstance(entropy_loss, torch.Tensor) else -total_entropy
                
                # Similar for tracks (omitted for brevity)
                
                # Accumulate losses for this epoch
                if isinstance(policy_loss, torch.Tensor) and policy_loss.numel() > 0:
                    epoch_policy_losses.append(policy_loss)
                elif policy_loss != 0.0:
                    epoch_policy_losses.append(torch.tensor(policy_loss, device=self.device, requires_grad=True))
                    
                if isinstance(value_loss, torch.Tensor) and value_loss.numel() > 0:
                    epoch_value_losses.append(value_loss)
                elif value_loss != 0.0:
                    epoch_value_losses.append(torch.tensor(value_loss, device=self.device, requires_grad=True))
                    
                if isinstance(entropy_loss, torch.Tensor) and entropy_loss.numel() > 0:
                    epoch_entropy_losses.append(entropy_loss)
                elif entropy_loss != 0.0:
                    epoch_entropy_losses.append(torch.tensor(entropy_loss, device=self.device, requires_grad=True))
            
            # BATCH UPDATE: One optimizer step per epoch, not per experience
            if epoch_policy_losses:
                epoch_policy_loss = torch.stack(epoch_policy_losses).mean()
                epoch_value_loss = torch.stack(epoch_value_losses).mean() if epoch_value_losses else torch.tensor(0.0, device=self.device)
                epoch_entropy_loss = torch.stack(epoch_entropy_losses).mean() if epoch_entropy_losses else torch.tensor(0.0, device=self.device)
                
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
            max_buffer_episodes = 8  # Keep last 8 episodes for continued learning
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


class RLTrustEnvironment:
    """Environment wrapper for RL training"""
    
    def __init__(self, num_robots=5, num_targets=10, adversarial_ratio=0.3, scenario_config=None, max_steps_per_episode=500, fov_range=50.0, fov_angle=np.pi/3):
        if scenario_config is not None:
            # Use diverse scenario configuration
            self.num_robots = scenario_config['num_robots']
            self.num_targets = scenario_config['num_targets']
            self.adversarial_ratio = scenario_config['adversarial_ratio']
            self.world_size = scenario_config['world_size']
            self.false_positive_rate = scenario_config['false_positive_rate']
            self.false_negative_rate = scenario_config['false_negative_rate']
            self.movement_speed = scenario_config['movement_speed']
            self.proximal_range = scenario_config['proximal_range']
            self.fov_range = scenario_config.get('fov_range', fov_range)  # Use scenario config or fallback to parameter
            self.fov_angle = scenario_config.get('fov_angle', fov_angle)  # Use scenario config or fallback to parameter
        else:
            # Use default values
            self.num_robots = num_robots
            self.num_targets = num_targets
            self.adversarial_ratio = adversarial_ratio
            self.world_size = (50, 50)
            self.false_positive_rate = 0.5
            self.false_negative_rate = 0.3
            self.movement_speed = 1.0
            self.proximal_range = 50.0
            self.fov_range = fov_range
            self.fov_angle = fov_angle
        
        # Store max steps per episode
        self.max_steps_per_episode = max_steps_per_episode
        
        # Initialize neural algorithm for learning (paper algorithm commented out - focus on RL only)
        # self.paper_algo = PaperTrustAlgorithm()  # COMMENTED OUT: Focus on RL algorithm only
        # self.neural_algo = NeuralSymbolicTrustAlgorithm(learning_mode=True)  # COMMENTED OUT: Pure RL training
        
        # Episode-level ego robot tracking
        self.episode_ego_robot_id = None
        
        # Multi-ego training mode
        self.use_multi_ego = True  # Enable multi-ego training by default
        self.accumulated_robot_updates = {}  # robot_id -> list of (delta_alpha, delta_beta)
        self.accumulated_track_updates = {}  # track_id -> list of (delta_alpha, delta_beta)
    
    def reset(self):
        """Reset environment for new episode"""
        # Clear and reset all trust distributions for fresh start each episode
        self._previous_trust_distributions = {}
        
        # Clear multi-ego accumulation structures
        self.accumulated_robot_updates.clear()
        self.accumulated_track_updates.clear()
        
        # FIXED SCENARIO: Set consistent random seed BEFORE creating SimulationEnvironment
        # This ensures identical scenarios across all episodes
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)
        
        # Create new simulation environment (since it doesn't have reset method)
        self.sim_env = SimulationEnvironment(
            num_robots=self.num_robots,
            num_targets=self.num_targets,
            adversarial_ratio=self.adversarial_ratio,
            world_size=self.world_size,
            proximal_range=self.proximal_range,
            fov_range=self.fov_range,
            fov_angle=self.fov_angle
        )
        
        # Select ego robot for this episode and ensure consistency
        self._select_episode_ego_robot()
        
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Verify clean start
        self._verify_clean_trust_reset()
        
        # CRITICAL FIX: Run one simulation step to populate robot_object_tracks
        # This ensures that the simulation environment has generated initial detections
        # before we try to get the current state for visualization/training
        # Reset seed before initial step to ensure deterministic behavior
        np.random.seed(42)
        random.seed(42)
        try:
            self.sim_env.step()
        except Exception as e:
            pass  # print(f"Warning: Initial simulation step failed: {e}")
        
        # Initialize with fresh initial trust values for this episode
        for robot in self.sim_env.robots:
            self._previous_trust_distributions[robot.id] = {
                'alpha': 1.0,
                'beta': 1.0
            }
        
        return self._get_current_state()
    
    def _verify_clean_trust_reset(self):
        """Verify that all trust values start fresh for the episode"""
        if hasattr(self.sim_env, 'robots'):
            for robot in self.sim_env.robots:
                if hasattr(robot, 'trust_alpha') and hasattr(robot, 'trust_beta'):
                    if robot.trust_alpha != 1.0 or robot.trust_beta != 1.0:
                        # print(f"  ⚠️  WARNING: Robot {robot.id} trust not reset! α={robot.trust_alpha:.3f}, β={robot.trust_beta:.3f}")
                        # Force reset to default values
                        robot.trust_alpha = 1.0
                        robot.trust_beta = 1.0
        
        # Ensure no persistent data structures exist
        persistent_attrs = ['_previous_trust_distributions']
        for attr in persistent_attrs:
            if hasattr(self, attr) and getattr(self, attr):
                # print(f"  ⚠️  WARNING: {attr} not properly cleared!")
                setattr(self, attr, {})
    
    def _select_episode_ego_robot(self):
        """Select and fix ego robot for entire episode"""
        # Get available robots
        if hasattr(self.sim_env, 'robots') and self.sim_env.robots:
            if isinstance(self.sim_env.robots, list):
                robots = self.sim_env.robots
            else:
                robots = list(self.sim_env.robots.values()) if self.sim_env.robots else []
            
            if robots:
                # Select first robot as ego robot for this episode
                self.episode_ego_robot_id = robots[0].id
                # print(f"  🎯 Selected Robot {self.episode_ego_robot_id} as ego robot for this episode")
            else:
                self.episode_ego_robot_id = None
        else:
            self.episode_ego_robot_id = None
    
    def _get_current_state(self):
        """Get current state with proper multi-robot track fusion (using episode ego robot)"""
        return self._get_current_state_with_ego(self.episode_ego_robot_id)
    
    def _get_current_state_with_ego(self, ego_robot_id):
        """Get current state with specific robot as ego robot"""
        # Check if simulation has robots
        if not hasattr(self.sim_env, 'robots') or not self.sim_env.robots:
            return self._get_empty_graph()
        
        # Get robots (handle both list and dict formats)
        if isinstance(self.sim_env.robots, list):
            robots = self.sim_env.robots
        else:
            robots = list(self.sim_env.robots.values()) if self.sim_env.robots else []
        
        if not robots:
            return self._get_empty_graph()
        
        # Find the specified ego robot
        ego_robot = None
        proximal_robots = []
        
        if ego_robot_id is not None:
            # Find the ego robot by ID
            for robot in robots:
                if robot.id == ego_robot_id:
                    ego_robot = robot
                else:
                    proximal_robots.append(robot)
        
        # Fallback: if ego robot not found, select first robot
        if ego_robot is None:
            # print(f"  ⚠️  Ego robot {ego_robot_id} not found, falling back to first robot")
            ego_robot = robots[0]
            proximal_robots = robots[1:] if len(robots) > 1 else []
        
        # Step 1: Generate individual track lists for each robot
        individual_robot_tracks = self._generate_individual_robot_tracks(ego_robot, proximal_robots)
        
        # Step 2: Perform track fusion between robots
        fused_tracks, individual_tracks, track_fusion_map = self._perform_track_fusion(
            ego_robot, proximal_robots, individual_robot_tracks)
        
        # Step 3: Build graph with all tracks and proper edges
        graph_data = self._build_multi_robot_graph(
            ego_robot, proximal_robots, fused_tracks, individual_tracks, track_fusion_map)
        
        return graph_data
    
    def _generate_individual_robot_tracks(self, ego_robot, proximal_robots):
        """Generate individual track lists for each robot using CURRENT TIMESTEP tracks only"""
        individual_robot_tracks = {}
        
        # No need for episode track storage - use on-demand fusion
        
        all_robots = [ego_robot] + proximal_robots
        
        # CRITICAL: Use CURRENT timestep tracks only (robot_current_tracks) for graph construction
        # Historical tracks (robot_object_tracks) are maintained for persistence but excluded from graph
        has_current_tracks = (hasattr(self.sim_env, 'robot_current_tracks') and 
                             self.sim_env.robot_current_tracks)
        
        # if has_current_tracks:
            # print("  🎯 Using CURRENT timestep tracks from robot_current_tracks")
        # else:
            # print("  🧪 Using ground truth objects (no current tracks available)")
        
        for robot in all_robots:
            robot_tracks = []
            
            # STEP 1: Get CURRENT timestep tracks from simulation robot_current_tracks
            if has_current_tracks and robot.id in self.sim_env.robot_current_tracks:
                current_tracks = self.sim_env.robot_current_tracks[robot.id]
                
                # print(f"  📡 Robot {robot.id} has {len(current_tracks)} current timestep tracks")
                
                for object_id, current_track in current_tracks.items():
                    track_id = f"current_robot_{robot.id}_obj_{object_id}"
                    
                    # Get trust values from the historical track in robot_object_tracks for persistence
                    # but use CURRENT track data for graph construction
                    historical_track = None
                    if (hasattr(self.sim_env, 'robot_object_tracks') and 
                        robot.id in self.sim_env.robot_object_tracks and
                        object_id in self.sim_env.robot_object_tracks[robot.id]):
                        historical_track = self.sim_env.robot_object_tracks[robot.id][object_id]
                    
                    # Get trust values directly from historical tracks
                    if historical_track:
                        # Use historical track's trust values
                        trust_alpha = getattr(historical_track, 'trust_alpha', 1.0)
                        trust_beta = getattr(historical_track, 'trust_beta', 1.0)
                    else:
                        # Initialize with default values
                        trust_alpha = 1.0
                        trust_beta = 1.0
                    
                    # Create track using CURRENT timestep data
                    from trust_algorithm import Track
                    track = Track(
                        id=track_id,
                        position=current_track.position.copy(),
                        velocity=getattr(current_track, 'velocity', np.array([0.0, 0.0, 0.0])),
                        covariance=getattr(current_track, 'covariance', np.eye(3)),
                        confidence=getattr(current_track, 'confidence', 0.8),
                        timestamp=getattr(current_track, 'timestamp', self.sim_env.time),
                        source_robot=robot.id,
                        trust_alpha=trust_alpha,  # Trust from historical persistence
                        trust_beta=trust_beta,
                        object_id=object_id
                    )
                    robot_tracks.append(track)
            
            # STEP 2: No artificial track generation - robots only have tracks they actually observe
            # if len(robot_tracks) == 0:
                # print(f"  📡 Robot {robot.id} has no current timestep tracks - this is realistic!")
            
            individual_robot_tracks[robot.id] = robot_tracks
            # print(f"  📡 Robot {robot.id} observes {len(robot_tracks)} tracks")
        
        return individual_robot_tracks
    
    
    def _perform_track_fusion(self, ego_robot, proximal_robots, individual_robot_tracks):
        """Perform track fusion between robots with proper trust inheritance"""
        fusion_distance_threshold = 5.0  # Tracks within this distance are considered same object
        
        all_robots = [ego_robot] + proximal_robots
        fused_tracks = []
        individual_tracks = []  # Tracks that remain unfused
        track_fusion_map = {}  # Maps original track IDs to fused/individual track IDs
        
        # Collect all tracks from all robots
        all_tracks = []
        for robot_id, tracks in individual_robot_tracks.items():
            for track in tracks:
                all_tracks.append((robot_id, track))
        
        processed_tracks = set()  # Track IDs that have been processed
        
        for i, (robot_id, track) in enumerate(all_tracks):
            if track.id in processed_tracks:
                continue
            
            # Find all tracks that should be fused with this one
            tracks_to_fuse = [(robot_id, track)]
            
            for j, (other_robot_id, other_track) in enumerate(all_tracks[i+1:], i+1):
                if other_track.id in processed_tracks:
                    continue
                
                # Check if tracks represent same object (same object_id AND within distance threshold)
                same_object = (track.object_id == other_track.object_id and 
                              track.object_id is not None and 
                              other_track.object_id is not None)
                distance = np.linalg.norm(track.position - other_track.position)
                
                if same_object and distance < fusion_distance_threshold:
                    tracks_to_fuse.append((other_robot_id, other_track))
            
            # Process fusion
            if len(tracks_to_fuse) > 1:
                # Multiple robots see the same object - create fused track
                fused_track = self._create_fused_track(tracks_to_fuse, ego_robot.id, all_robots)
                fused_tracks.append(fused_track)
                
                # Map all constituent tracks to the fused track
                for constituent_robot_id, constituent_track in tracks_to_fuse:
                    track_fusion_map[constituent_track.id] = fused_track.id
                    processed_tracks.add(constituent_track.id)
            else:
                # Only one robot sees this object - keep as individual track
                individual_track = tracks_to_fuse[0][1]
                individual_tracks.append(individual_track)
                track_fusion_map[individual_track.id] = individual_track.id
                processed_tracks.add(individual_track.id)
        
        # print(f"  🔀 Track fusion: {len(fused_tracks)} fused, {len(individual_tracks)} individual")
        return fused_tracks, individual_tracks, track_fusion_map
    
    def _create_fused_track(self, tracks_to_fuse, ego_robot_id, all_robots):
        """Create a fused track with proper trust inheritance"""
        # Determine trust inheritance priority
        ego_tracks = [t for r, t in tracks_to_fuse if r == ego_robot_id]
        proximal_tracks = [t for r, t in tracks_to_fuse if r != ego_robot_id]
        
        if ego_tracks:
            # Ego robot is involved - ego track trust takes priority
            primary_track = ego_tracks[0]
            trust_alpha = primary_track.trust_alpha
            trust_beta = primary_track.trust_beta
            source_info = f"fused_ego_{ego_robot_id}"
        else:
            # Only proximal robots involved - highest trust robot takes priority
            robot_trusts = {}
            for robot in all_robots:
                robot_trusts[robot.id] = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
            
            # Find track from highest trust proximal robot
            best_track = None
            best_robot_trust = -1
            for robot_id, track in tracks_to_fuse:
                if robot_trusts[robot_id] > best_robot_trust:
                    best_robot_trust = robot_trusts[robot_id]
                    best_track = track
            
            primary_track = best_track
            trust_alpha = primary_track.trust_alpha
            trust_beta = primary_track.trust_beta  
            source_info = f"fused_proximal_{primary_track.source_robot}"
        
        # Create fused track ID
        contributing_robots = sorted([r for r, t in tracks_to_fuse])
        fused_id = f"fused_{'_'.join(map(str, contributing_robots))}_{primary_track.object_id}"
        
        # Average position and velocity from all contributing tracks
        positions = np.array([t.position for r, t in tracks_to_fuse])
        velocities = np.array([t.velocity for r, t in tracks_to_fuse])
        confidences = [t.confidence for r, t in tracks_to_fuse]
        
        avg_position = np.mean(positions, axis=0)
        avg_velocity = np.mean(velocities, axis=0)
        avg_confidence = np.mean(confidences)
        
        # Create fused track
        from trust_algorithm import Track
        fused_track = Track(
            id=fused_id,
            position=avg_position,
            velocity=avg_velocity,
            covariance=primary_track.covariance,  # Use primary track's covariance
            confidence=avg_confidence,
            timestamp=primary_track.timestamp,
            source_robot=primary_track.source_robot,
            trust_alpha=trust_alpha,
            trust_beta=trust_beta,
            object_id=primary_track.object_id
        )
        
        # No need to store in episode storage - trust values are in the track itself
        
        return fused_track
    
    def _build_multi_robot_graph(self, ego_robot, proximal_robots, fused_tracks, individual_tracks, track_fusion_map):
        """Build graph with all robots and tracks, including proper edge relationships"""
        from torch_geometric.data import HeteroData
        import torch
        
        graph_data = HeteroData()
        all_robots = [ego_robot] + proximal_robots
        all_tracks = fused_tracks + individual_tracks
        
        # Create agent nodes for all robots with rich neural-symbolic features
        agent_nodes = {}
        agent_features = []
        for i, robot in enumerate(all_robots):
            agent_nodes[robot.id] = i
            
            # Compute trust-based predicates
            robot_trust = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
            
            # Feature 1: HighConfidence(robot) - robot confidence > 0.7
            robot_confidence = getattr(robot, 'confidence', robot_trust)  # Use robot.confidence if available, fallback to trust
            high_confidence_pred = 1.0 if robot_confidence > 0.7 else 0.0
            
            # Feature 2: HighlyTrusted(robot) - confident positive classification
            highly_trusted_pred = 1.0 if robot_trust > 0.8 else 0.0
            
            # Feature 3: Suspicious(robot) - likely adversarial
            suspicious_pred = 1.0 if robot_trust < 0.3 else 0.0
            
            # Feature 4: HighConnectivity(robot) - observes many tracks
            robot_track_count = sum(1 for track in all_tracks if self._robot_observes_track(robot, track, fused_tracks, individual_tracks, track_fusion_map))
            high_connectivity_pred = 1.0 if robot_track_count >= 3 else 0.0
            
            # Feature 5: TrustUncertain(robot) - trust close to 0.5 (uncertain)
            trust_uncertain_pred = 1.0 if abs(robot_trust - 0.5) < 0.2 else 0.0
            
            agent_features.append([
                high_confidence_pred,    # Feature 1
                highly_trusted_pred,     # Feature 2  
                suspicious_pred,         # Feature 3
                high_connectivity_pred,  # Feature 4
                trust_uncertain_pred,    # Feature 5
            ])
        
        graph_data['agent'].x = torch.tensor(agent_features, dtype=torch.float)
        graph_data.agent_nodes = agent_nodes
        
        # Create track nodes for all tracks with rich neural-symbolic features  
        track_nodes = {}
        track_features = []
        for i, track in enumerate(all_tracks):
            track_nodes[track.id] = i
            
            # Compute trust and confidence based predicates
            track_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)
            track_confidence = getattr(track, 'confidence', 0.5)
            
            # Feature 1: Trustworthy(track) - basic trust > 0.5
            trustworthy_pred = 1.0 if track_trust > 0.5 else 0.0
            
            # Feature 2: HighConfidence(track) - trust confidence > 0.7
            high_confidence_pred = 1.0 if track_confidence > 0.7 else 0.0
            
            # Feature 3: MultiRobotObserved(track) - observed by multiple robots (fused)
            multi_robot_pred = 1.0 if track in fused_tracks else 0.0
            
            # Feature 4: LikelyFalsePositive(track) - suspicious patterns
            likely_fp_pred = 1.0 if (track_trust < 0.3) else 0.0
            
            # Feature 5: UncertainTrack(track) - trust close to 0.5
            uncertain_track_pred = 1.0 if abs(track_trust - 0.5) < 0.2 else 0.0
            
            track_features.append([
                trustworthy_pred,        # Feature 1
                high_confidence_pred,    # Feature 2
                multi_robot_pred,        # Feature 3  
                likely_fp_pred,          # Feature 4
                uncertain_track_pred,    # Feature 5
            ])
        
        graph_data['track'].x = torch.tensor(track_features, dtype=torch.float)
        graph_data.track_nodes = track_nodes
        
        # Debug: Show neural-symbolic features 
        agent_feature_summaries = []
        for i, af in enumerate(agent_features):
            # Show key predicates: [HighConfidence, HighlyTrusted, Suspicious, IsEgo]
            summary = f"[{af[0]:.0f},{af[1]:.0f},{af[2]:.0f},{af[4]:.0f}]"  # Updated index for IsEgo: now feature 4
            agent_feature_summaries.append(summary)
        trust_values = [f"{robot.trust_alpha/(robot.trust_alpha + robot.trust_beta):.3f}" for robot in all_robots]
        #print(f"  🤖 Agent features (C,H,S,E): {agent_feature_summaries}")
        #print(f"  🎯 Underlying trust: {trust_values}")
        
        track_feature_summaries = []
        for i, tf in enumerate(track_features):
            # Show key predicates: [Trustworthy, HighConf, MultiRobot, LikelyFP, Uncertain]  
            summary = f"[{tf[0]:.0f},{tf[1]:.0f},{tf[2]:.0f},{tf[3]:.0f},{tf[4]:.0f}]"
            track_feature_summaries.append(summary)
        #print(f"  📍 Track features (T,C,M,F,U): {track_feature_summaries[:8]}{'...' if len(track_feature_summaries) > 8 else ''}")
        
        # Build edges: Observes, InFoV, and Proximal relationships
        observes_edges = []  # (agent, track) - robot observes track
        observed_by_edges = []  # (track, agent) - track observed by robot
        in_fov_edges = []  # (agent, track) - track is in robot's field of view
        in_fov_by_edges = []  # (track, agent) - track is in robot's field of view
        # is_proximal_edges = []  # (agent, agent) - robot is proximal to another robot - TEMPORARILY DISABLED
        
        # STEP 1: Create OBSERVES edges - robot observes tracks it owns/contributes to
        for robot in all_robots:
            robot_idx = agent_nodes[robot.id]
            
            for track in all_tracks:
                track_idx = track_nodes[track.id]
                
                # Check if robot observes this track (based on ownership/contribution)
                observes_track = self._robot_observes_track(robot, track, fused_tracks, individual_tracks, track_fusion_map)
                
                if observes_track:
                    observes_edges.append([robot_idx, track_idx])
                    observed_by_edges.append([track_idx, robot_idx])
        
        # STEP 2: Create IN_FOV edges - robot can see tracks from all robots in the group
        for robot in all_robots:
            robot_idx = agent_nodes[robot.id]
            
            for track in all_tracks:
                track_idx = track_nodes[track.id]
                
                # Check if robot observes this track (if so, it MUST be in FoV)
                observes_track = self._robot_observes_track(robot, track, fused_tracks, individual_tracks, track_fusion_map)
                
                # Check if track is in robot's field of view (distance/angle based)
                in_fov_by_distance = self._track_in_robot_fov(robot, track)
                
                # A track is in FoV if: robot observes it OR it's within FoV distance
                in_fov = observes_track or in_fov_by_distance
                
                if in_fov:
                    in_fov_edges.append([robot_idx, track_idx])
                    in_fov_by_edges.append([track_idx, robot_idx])
        
        # STEP 3: Create IS_PROXIMAL edges between robots based on distance
        # TEMPORARILY DISABLED - Comment out proximal edge creation
        # proximal_range = getattr(self.sim_env, 'proximal_range', 50.0)  # Get proximal_range from environment
        # for i, robot1 in enumerate(all_robots):
        #     robot1_idx = agent_nodes[robot1.id]
        #     
        #     for j, robot2 in enumerate(all_robots):
        #         if i >= j:  # Skip self and avoid duplicate edges
        #             continue
        #             
        #         robot2_idx = agent_nodes[robot2.id]
        #         
        #         # Check if robots have positions for distance calculation
        #         if hasattr(robot1, 'position') and hasattr(robot2, 'position'):
        #             distance = np.linalg.norm(robot1.position - robot2.position)
        #             
        #             # Create bidirectional proximal edges if within range
        #             if distance <= proximal_range:
        #                 is_proximal_edges.append([robot1_idx, robot2_idx])
        #                 is_proximal_edges.append([robot2_idx, robot1_idx])
        
        # Debug: Check if any robot has proper orientation
        orientations_set = sum(1 for robot in all_robots if hasattr(robot, 'orientation'))
        # if orientations_set == 0:
            # print(f"  ⚠️  Warning: No robots have orientation set, FoV angle constraints disabled")
        
        # Convert to tensors and create proper edge structure
        edge_types = [
            ('agent', 'observes', 'track'),
            ('track', 'observed_by', 'agent'),
            ('agent', 'in_fov', 'track'),
            ('track', 'in_fov_by', 'agent'),
            # ('agent', 'isProximal', 'agent')  # TEMPORARILY DISABLED
        ]
        
        edge_data = [observes_edges, observed_by_edges, in_fov_edges, in_fov_by_edges]  # Removed is_proximal_edges
        
        for edge_type, edges in zip(edge_types, edge_data):
            if edges:
                # Convert list of [src, dst] pairs to tensor format [2, num_edges]
                edge_tensor = torch.tensor(edges, dtype=torch.long).T
                graph_data[edge_type].edge_index = edge_tensor
            else:
                graph_data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Add x_dict and edge_index_dict properties that the GNN expects
        graph_data.x_dict = {
            'agent': graph_data['agent'].x,
            'track': graph_data['track'].x
        }
        
        graph_data.edge_index_dict = {}
        for edge_type in edge_types:
            if hasattr(graph_data[edge_type], 'edge_index'):
                graph_data.edge_index_dict[edge_type] = graph_data[edge_type].edge_index
        
        # Store additional information for trust updates
        graph_data._current_robots = all_robots
        graph_data._fused_tracks = fused_tracks
        graph_data._individual_tracks = individual_tracks
        graph_data._track_fusion_map = track_fusion_map
        
        return graph_data
    
    def _robot_observes_track(self, robot, track, fused_tracks, individual_tracks, track_fusion_map):
        """Determine if robot observes this track (based on fusion logic)"""
        # Check if this is a fused track and robot contributed to it
        if track in fused_tracks:
            # Check if any of the constituent tracks came from this robot
            for original_id, fused_id in track_fusion_map.items():
                if fused_id == track.id and f"robot_{robot.id}_" in original_id:
                    return True
            return False
        
        # Check if this is an individual track from this robot
        if track in individual_tracks:
            return track.source_robot == robot.id
        
        return False
    
    def _track_in_robot_fov(self, robot, track):
        """Determine if track is in robot's field of view using proper angle and distance constraints"""
        if not (hasattr(robot, 'position') and hasattr(track, 'position')):
            return False
        
        # Use same logic as simulation_environment.py is_in_fov method
        rel_pos = track.position - robot.position
        distance = np.linalg.norm(rel_pos[:2])  # 2D distance
        
        # Check distance constraint - use robot's actual FoV range
        fov_range = getattr(robot, 'fov_range', 100.0)  # Use robot's actual fov_range
        if distance > fov_range:
            return False
        
        # Check angle constraint if robot has orientation
        if hasattr(robot, 'orientation') and hasattr(robot, 'fov_angle'):
            target_angle = np.arctan2(rel_pos[1], rel_pos[0])
            robot_orientation = robot.orientation
            angle_diff = abs(target_angle - robot_orientation)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Wrap around
            
            fov_angle = robot.fov_angle
            return angle_diff <= fov_angle / 2
        else:
            # If no orientation info, only use distance constraint
            return True
    
    def _get_empty_graph(self):
        """Create empty graph when simulation state is invalid"""
        from torch_geometric.data import HeteroData
        import torch
        
        graph_data = HeteroData()
        graph_data['agent'].x = torch.empty((0, 1), dtype=torch.float)
        graph_data['track'].x = torch.empty((0, 1), dtype=torch.float)
        
        required_edge_types = [
            ('agent', 'observes', 'track'),
            ('track', 'observed_by', 'agent'),
            ('agent', 'in_fov', 'track'),
            ('track', 'in_fov_by', 'agent'),
            # ('agent', 'isProximal', 'agent')  # TEMPORARILY DISABLED
        ]
        
        for edge_type in required_edge_types:
            graph_data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        graph_data.agent_nodes = {}
        graph_data.track_nodes = {}
        
        return graph_data
    
    def step(self, actions):
        """Take environment step with given actions (supports multi-ego training)"""
        
        if self.use_multi_ego:
            return self._step_multi_ego(actions)
        else:
            return self._step_single_ego(actions)
    
    def _step_single_ego(self, actions):
        """Single ego robot step (original behavior)"""
        
        # Get current state BEFORE applying updates
        current_state = self._get_current_state()
        
        try:
            # DETERMINISTIC: Reset seed before each simulation step for consistent FP generation
            np.random.seed(42 + self.step_count)  # Different seed per step, but deterministic
            random.seed(42 + self.step_count)
            
            # Advance simulation FIRST
            self.sim_env.step()
            self.step_count += 1
        except Exception as e:
            # print(f"Warning: Simulation step failed: {e}")
            # Continue with current state
            pass
            
        # Get state after simulation step
        next_state = self._get_current_state()
        
        # CRITICAL: Apply trust updates AFTER simulation step to avoid being overwritten
        self._apply_trust_updates(actions, next_state)
        
        # Get next state AFTER applying updates and advancing simulation
        next_state = self._get_current_state()
        
        # Prepare simulation step data for reward computation
        robots = []
        if hasattr(self.sim_env, 'robots') and self.sim_env.robots:
            if isinstance(self.sim_env.robots, list):
                robots = self.sim_env.robots
            else:
                robots = list(self.sim_env.robots.values())
        
        ground_truth_objects = []
        if hasattr(self.sim_env, 'ground_truth_objects'):
            ground_truth_objects = self.sim_env.ground_truth_objects
        
        # Find consistent ego robot for simulation step data
        ego_robot_for_step = None
        proximal_robots_for_step = []
        
        if self.episode_ego_robot_id is not None and robots:
            for robot in robots:
                if robot.id == self.episode_ego_robot_id:
                    ego_robot_for_step = robot
                else:
                    proximal_robots_for_step.append(robot)
        
        # Fallback if ego robot not found
        if ego_robot_for_step is None and robots:
            ego_robot_for_step = robots[0]
            proximal_robots_for_step = robots[1:]
        
        simulation_step_data = {
            'ego_robot': ego_robot_for_step,
            'proximal_robots': proximal_robots_for_step,
            'ground_truth_objects': ground_truth_objects
        }
        
        # Store current trust distributions for comparison in reward computation
        current_trust_distributions = {}
        if hasattr(current_state, '_current_robots'):
            for robot in current_state._current_robots:
                if hasattr(robot, 'trust_alpha'):
                    trust_value = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
                    current_trust_distributions[robot.id] = {
                        'trust': trust_value,
                        'alpha': robot.trust_alpha,
                        'beta': robot.trust_beta
                    }
        
        if hasattr(current_state, '_current_tracks'):
            for track in current_state._current_tracks:
                if hasattr(track, 'trust_alpha'):
                    trust_value = track.trust_alpha / (track.trust_alpha + track.trust_beta)
                    current_trust_distributions[track.id] = {
                        'trust': trust_value,
                        'alpha': track.trust_alpha,
                        'beta': track.trust_beta
                    }
        
        # Add current trust distributions to simulation data for reward computation
        simulation_step_data['current_trust_distributions'] = current_trust_distributions
        
        # Check if episode is done
        num_objects = len(ground_truth_objects)
        num_robots = len(robots)
        
        # Episode ends when max steps reached or system fails
        done = (num_robots == 0 or 
                num_objects == 0 or
                self.step_count >= self.max_steps_per_episode)
        
        # Balanced reward: immediate trust direction + small final episode reward
        reward = self._compute_sparse_reward(next_state, simulation_step_data, done)
        
        
        self.episode_reward += reward
        
        return next_state, reward, done, {
            'step_count': self.step_count,
            'num_robots': num_robots,
            'num_objects': num_objects,
            'ego_robot_id': self.episode_ego_robot_id,
            'trust_updates_applied': True,
            'reward_components': {
                'total': reward
            }
        }
    
    def _step_multi_ego(self, actions):
        """Multi-ego robot step - each robot serves as ego and accumulates updates"""
        
        # This method is called with actions from ONE ego perspective (current state)
        # But we need to get actions from ALL ego perspectives for proper multi-ego training
        
        # Get current state BEFORE applying updates (using original ego for initial state)
        current_state = self._get_current_state()
        
        try:
            # DETERMINISTIC: Reset seed before each simulation step for consistent FP generation
            np.random.seed(42 + self.step_count)  # Different seed per step, but deterministic
            random.seed(42 + self.step_count)
            
            # Advance simulation FIRST
            self.sim_env.step()
            self.step_count += 1
        except Exception as e:
            # print(f"Warning: Simulation step failed: {e}")
            # Continue with current state
            pass
            
        # Get robots for multi-ego processing
        robots = []
        if hasattr(self.sim_env, 'robots') and self.sim_env.robots:
            if isinstance(self.sim_env.robots, list):
                robots = self.sim_env.robots
            else:
                robots = list(self.sim_env.robots.values())
        
        if not robots:
            # Fallback to single ego if no robots
            return self._step_single_ego(actions)
        
        # MULTI-EGO APPROACH: Get actions from each robot's perspective
        self.accumulated_robot_updates.clear()
        self.accumulated_track_updates.clear()
        
        all_ego_states = []
        successful_egos = 0
        
        # Import trainer from training context (this is a bit hacky but works)
        trainer_instance = getattr(self, '_trainer_instance', None)
        
        for robot in robots:
            try:
                # Get state with this robot as ego
                ego_state = self._get_current_state_with_ego(robot.id)
                if ego_state is not None:
                    all_ego_states.append((robot.id, ego_state))
                    
                    # Get actions from this ego robot's perspective
                    if trainer_instance:
                        ego_actions, _, _ = trainer_instance.select_action(ego_state)
                        # Accumulate trust updates from this ego's actions
                        self._accumulate_trust_updates(ego_actions, ego_state, robot.id)
                    else:
                        # Fallback: use provided actions (single-ego mode)
                        self._accumulate_trust_updates(actions, ego_state, robot.id)
                        
                    successful_egos += 1
                    
            except Exception as e:
                # print(f"Warning: Multi-ego processing failed for robot {robot.id}: {e}")
                continue
        
        # Apply all accumulated updates at once
        self._apply_accumulated_trust_updates()
        
        # Get final state after all updates applied
        next_state = self._get_current_state()
        
        # Prepare simulation step data (use first robot as representative ego)
        ego_robot_for_step = robots[0] if robots else None
        proximal_robots_for_step = robots[1:] if len(robots) > 1 else []
        
        ground_truth_objects = []
        if hasattr(self.sim_env, 'ground_truth_objects'):
            ground_truth_objects = self.sim_env.ground_truth_objects
        
        simulation_step_data = {
            'ego_robot': ego_robot_for_step,
            'proximal_robots': proximal_robots_for_step,
            'ground_truth_objects': ground_truth_objects
        }
        
        # Store current trust distributions for reward computation
        current_trust_distributions = {}
        if hasattr(current_state, '_current_robots'):
            for robot in current_state._current_robots:
                if hasattr(robot, 'trust_alpha'):
                    trust_value = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
                    current_trust_distributions[robot.id] = {
                        'trust': trust_value,
                        'alpha': robot.trust_alpha,
                        'beta': robot.trust_beta
                    }
        
        simulation_step_data['current_trust_distributions'] = current_trust_distributions
        
        # Check if episode is done
        num_objects = len(ground_truth_objects)
        num_robots = len(robots)
        
        done = (num_robots == 0 or 
                num_objects == 0 or
                self.step_count >= self.max_steps_per_episode)
        
        # Compute reward using multi-ego enhanced data
        reward = self._compute_sparse_reward(next_state, simulation_step_data, done)
        
        self.episode_reward += reward
        
        return next_state, reward, done, {
            'step_count': self.step_count,
            'num_robots': num_robots,
            'num_objects': num_objects,
            'ego_robot_id': f"multi_ego_{successful_egos}_robots",
            'trust_updates_applied': True,
            'successful_ego_robots': successful_egos,
            'reward_components': {
                'total': reward
            }
        }
    
    def _accumulate_trust_updates(self, actions, ego_state, ego_robot_id):
        """Accumulate trust updates from this ego robot's perspective (don't apply yet)"""
        if not actions or not ego_state:
            return
            
        # Accumulate robot trust updates
        if 'agent' in actions and hasattr(ego_state, 'agent_nodes'):
            agent_actions = actions['agent']
            
            for robot_id, node_idx in ego_state.agent_nodes.items():
                if node_idx < agent_actions['value'].shape[0]:
                    # Convert GNN action to PSM updates
                    psm_value = agent_actions['value'][node_idx].item()
                    psm_confidence = agent_actions['confidence'][node_idx].item()
                    
                    delta_alpha = psm_confidence * psm_value 
                    delta_beta = psm_confidence * (1.0 - psm_value)
                    
                    # Store update from this ego robot's perspective
                    if robot_id not in self.accumulated_robot_updates:
                        self.accumulated_robot_updates[robot_id] = []
                    
                    self.accumulated_robot_updates[robot_id].append((delta_alpha, delta_beta))
        
        # Accumulate track trust updates
        if 'track' in actions and hasattr(ego_state, 'track_nodes'):
            track_actions = actions['track']
            
            for track_id, node_idx in ego_state.track_nodes.items():
                if node_idx < track_actions['value'].shape[0]:
                    # Convert GNN action to PSM updates
                    psm_value = track_actions['value'][node_idx].item()
                    psm_confidence = track_actions['confidence'][node_idx].item()
                    
                    delta_alpha = psm_confidence * psm_value 
                    delta_beta = psm_confidence * (1.0 - psm_value)
                    
                    # Store update from this ego robot's perspective
                    if track_id not in self.accumulated_track_updates:
                        self.accumulated_track_updates[track_id] = []
                    
                    self.accumulated_track_updates[track_id].append((delta_alpha, delta_beta))
    
    def _apply_accumulated_trust_updates(self):
        """Apply all accumulated trust updates from multiple ego perspectives"""
        
        # Apply accumulated robot updates
        if hasattr(self.sim_env, 'robots') and self.sim_env.robots:
            robots = self.sim_env.robots if isinstance(self.sim_env.robots, list) else list(self.sim_env.robots.values())
            
            for robot in robots:
                if robot.id in self.accumulated_robot_updates and hasattr(robot, 'trust_alpha'):
                    updates = self.accumulated_robot_updates[robot.id]
                    
                    # Sum all delta updates from different ego perspectives
                    total_delta_alpha = sum(delta_alpha for delta_alpha, delta_beta in updates)
                    total_delta_beta = sum(delta_beta for delta_alpha, delta_beta in updates)
                    
                    # Apply accumulated updates
                    robot.trust_alpha += total_delta_alpha
                    robot.trust_beta += total_delta_beta
        
        # Apply accumulated track updates to simulation tracks
        if hasattr(self.sim_env, 'robot_object_tracks'):
            for robot_id, track_dict in self.sim_env.robot_object_tracks.items():
                for object_id, track in track_dict.items():
                    # Check various track ID formats that might be accumulated
                    possible_track_ids = [
                        f"current_robot_{robot_id}_obj_{object_id}",
                        f"robot_{robot_id}_obj_{object_id}",
                        f"fused_{object_id}",
                        track.id if hasattr(track, 'id') else None
                    ]
                    
                    for track_id in possible_track_ids:
                        if track_id and track_id in self.accumulated_track_updates:
                            updates = self.accumulated_track_updates[track_id]
                            
                            # Sum all delta updates from different ego perspectives
                            total_delta_alpha = sum(delta_alpha for delta_alpha, delta_beta in updates)
                            total_delta_beta = sum(delta_beta for delta_alpha, delta_beta in updates)
                            
                            # Apply accumulated updates to track
                            if hasattr(track, 'trust_alpha'):
                                track.trust_alpha += total_delta_alpha
                                track.trust_beta += total_delta_beta
                            
                            break  # Only apply once per track
    
    def _apply_trust_updates(self, actions, current_state):
        """Apply GNN action outputs to update trust distributions with proper propagation"""
        if not actions or not current_state:
            return
            
        # Update robot trust distributions directly in simulation environment
        if 'agent' in actions and hasattr(current_state, 'agent_nodes') and hasattr(current_state, '_current_robots'):
            agent_actions = actions['agent']
            current_robots = current_state._current_robots
            
            for robot_id, node_idx in current_state.agent_nodes.items():
                if node_idx < agent_actions['value'].shape[0]:
                    # Find robot in current robots list
                    robot = None
                    for r in current_robots:
                        if r.id == robot_id:
                            robot = r
                            break
                    
                    if robot is not None and hasattr(robot, 'trust_alpha'):
                        
                        # Convert GNN action to trust parameter updates
                        psm_value = agent_actions['value'][node_idx].item()
                        psm_confidence = agent_actions['confidence'][node_idx].item()
                        
                        # Update robot trust using PSM approach
                        delta_alpha = psm_confidence * psm_value 
                        delta_beta = psm_confidence * (1.0 - psm_value) 
                        
                        # Apply updates directly to robot
                        robot.trust_alpha += delta_alpha
                        robot.trust_beta += delta_beta
                        
        # Update track trust distributions with proper propagation
        if 'track' in actions and hasattr(current_state, 'track_nodes'):
            track_actions = actions['track']
            fused_tracks = getattr(current_state, '_fused_tracks', [])
            individual_tracks = getattr(current_state, '_individual_tracks', []) 
            track_fusion_map = getattr(current_state, '_track_fusion_map', {})
            
            all_tracks = fused_tracks + individual_tracks
            
            for track_id, node_idx in current_state.track_nodes.items():
                if node_idx < track_actions['value'].shape[0]:
                    # Find corresponding track
                    track = None
                    for t in all_tracks:
                        if t.id == track_id:
                            track = t
                            break
                    
                    if track is not None and hasattr(track, 'trust_alpha'):
                        
                        # Convert GNN action to trust parameter updates
                        psm_value = track_actions['value'][node_idx].item()
                        psm_confidence = track_actions['confidence'][node_idx].item()
                        
                        # Update track trust using PSM approach
                        delta_alpha = psm_confidence * psm_value 
                        delta_beta = psm_confidence * (1.0 - psm_value)
                        
                        # Apply updates to track trust parameters
                        track.trust_alpha += delta_alpha
                        track.trust_beta += delta_beta
                        
                        # Trust values are already stored in the track itself
                        
                        # CRITICAL: Propagate trust updates to individual robot tracks
                        self._propagate_track_trust_updates(track, track_id, track_fusion_map, delta_alpha, delta_beta)
                        
    
    def _propagate_track_trust_updates(self, updated_track, updated_track_id, track_fusion_map, delta_alpha, delta_beta):
        """Propagate trust updates from fused/individual tracks back to individual robot track lists"""
        
        # Check if this is a fused track - propagate to constituent individual tracks
        if 'fused_' in updated_track_id:
            # Find all individual tracks that contributed to this fused track
            for original_track_id, fused_id in track_fusion_map.items():
                if fused_id == updated_track_id:
                    # This original track contributed to the updated fused track
                    # Trust values are updated in the actual tracks
                    pass
        else:
            # This is an individual track - update is already applied
            # Trust values are stored in the track itself
            pass
    
    def _compute_final_episode_reward(self, final_state, simulation_step_data):
        """
        Compute final episode reward based on correct identification of:
        1. Adversarial robots (trust < 0.5)
        2. False positive tracks (trust < 0.5)
        """
        # Extract simulation data
        ego_robot = simulation_step_data.get('ego_robot')
        proximal_robots = simulation_step_data.get('proximal_robots', [])
        ground_truth_objects = simulation_step_data.get('ground_truth_objects', [])

        all_robots = ([ego_robot] + proximal_robots) if ego_robot else proximal_robots
        
        final_reward = 0.0
        
        # === FINAL ROBOT IDENTIFICATION REWARD ===
        robot_identification_bonus = 0.0
        total_robots = 0
        correct_robot_identifications = 0
        adversarial_robots = 0
        legitimate_robots = 0
        correctly_identified_adversarial = 0
        correctly_identified_legitimate = 0
        
        for robot in all_robots:
            if hasattr(robot, 'trust_alpha') and hasattr(robot, 'trust_beta'):
                total_robots += 1
                final_trust = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
                is_adversarial = getattr(robot, 'is_adversarial', False)
                
                if is_adversarial:
                    adversarial_robots += 1
                    if final_trust < 0.3:
                        correctly_identified_adversarial += 1
                        correct_robot_identifications += 1
                else:
                    legitimate_robots += 1
                    if final_trust >= 0.7:
                        correctly_identified_legitimate += 1
                        correct_robot_identifications += 1
        
        if total_robots > 0:
            robot_accuracy = correct_robot_identifications / total_robots
            robot_identification_bonus = robot_accuracy * 10.0  # Up to 10 points for perfect robot identification
            
            # PENALTIES for wrong robot classifications
            wrong_robot_identifications = total_robots - correct_robot_identifications
            robot_penalty = wrong_robot_identifications * -1.0  # Penalty for each wrong classification

            # Extra penalty for trusting adversarial robots (very dangerous!)
            trusted_adversaries = adversarial_robots - correctly_identified_adversarial
            adversarial_penalty = trusted_adversaries * -5.0  # Higher penalty for security risks

            robot_identification_bonus += robot_penalty + adversarial_penalty
            
            # Additional bonus for balanced classification
            if adversarial_robots > 0 and legitimate_robots > 0:
                adv_accuracy = correctly_identified_adversarial / adversarial_robots
                leg_accuracy = correctly_identified_legitimate / legitimate_robots
                balanced_accuracy = (adv_accuracy + leg_accuracy) / 2.0
                robot_identification_bonus += balanced_accuracy * 5.0  # Up to 5 bonus points for balanced accuracy
        else:
            robot_accuracy = 0.0  # No robots to classify
        
        # === OBJECT-CENTRIC TRACK REWARD WITH PROPER FUSION ===
        track_identification_bonus = 0.0
        
        # Collect all tracks from all robots
        all_tracks = []
        for robot in all_robots:
            if (hasattr(self.sim_env, 'robot_object_tracks') and 
                robot.id in self.sim_env.robot_object_tracks):
                robot_tracks = list(self.sim_env.robot_object_tracks[robot.id].values())
                all_tracks.extend(robot_tracks)
        
        # Separate into ground truth and false positive lists
        ground_truth_tracks = []
        false_positive_tracks = []
        
        for track in all_tracks:
            if (hasattr(track, 'object_id') and 
                hasattr(track, 'trust_alpha') and 
                hasattr(track, 'trust_beta') and
                track.object_id is not None):
                
                # Check if it's a legitimate object by comparing to ground truth
                # Extract numeric part from track object_id (e.g., "gt_obj_0" -> "0")
                track_id_str = str(track.object_id)
                if track_id_str.startswith('gt_obj_'):
                    numeric_id = track_id_str.replace('gt_obj_', '')
                else:
                    numeric_id = track_id_str
                
                is_false_positive = not any(str(getattr(obj, 'id', '')) == numeric_id for obj in ground_truth_objects)
                
                if is_false_positive:
                    false_positive_tracks.append(track)
                else:
                    ground_truth_tracks.append(track)
        
        # Perform object_id fusion for ground truth tracks
        object_trust_groups = {}
        for track in ground_truth_tracks:
            object_id = track.object_id
            track_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)
            if object_id not in object_trust_groups:
                object_trust_groups[object_id] = []
            object_trust_groups[object_id].append(track_trust)
        
        # Perform object_id fusion for false positive tracks
        false_positive_groups = {}
        for track in false_positive_tracks:
            object_id = track.object_id
            track_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)
            if object_id not in false_positive_groups:
                false_positive_groups[object_id] = []
            false_positive_groups[object_id].append(track_trust)
        
        total_tracks_processed = len(all_tracks)
        
        # Evaluate per-object trust (mean of all tracks for each object)
        total_objects = 0
        correct_object_identifications = 0
        
        # Legitimate objects (should be trusted)
        legitimate_objects = len(object_trust_groups)
        correctly_identified_legit_objects = 0
        for object_id, trust_values in object_trust_groups.items():
            total_objects += 1
            avg_object_trust = sum(trust_values) / len(trust_values)
            if avg_object_trust >= 0.7:  # Object correctly trusted
                correctly_identified_legit_objects += 1
                correct_object_identifications += 1
        
        # False positive objects (should be distrusted) 
        false_positive_objects = len(false_positive_groups)
        correctly_identified_fp_objects = 0
        for object_id, trust_values in false_positive_groups.items():
            total_objects += 1
            avg_object_trust = sum(trust_values) / len(trust_values)
            if avg_object_trust < 0.3:  # False positive correctly distrusted
                correctly_identified_fp_objects += 1
                correct_object_identifications += 1
        
        # Calculate object-centric reward
        if total_objects > 0:
            object_accuracy = correct_object_identifications / total_objects
            track_identification_bonus = correct_object_identifications * 8.0 
            
            # PENALTIES for wrong object classifications
            wrong_object_identifications = total_objects - correct_object_identifications
            object_penalty = wrong_object_identifications * -1.0  # Penalty for each wrong object classification

            # Extra penalty for trusting false positive objects (degrades sensor fusion)
            trusted_false_positive_objects = len(false_positive_groups) - correctly_identified_fp_objects
            false_positive_penalty = trusted_false_positive_objects * -1.0 # Higher penalty for trusting bad data

            track_identification_bonus += max(-10.0, object_penalty + false_positive_penalty)
            
            # Additional bonus for balanced object classification
            if len(false_positive_groups) > 0 and len(object_trust_groups) > 0:
                fp_accuracy = correctly_identified_fp_objects / len(false_positive_groups)
                legit_accuracy = correctly_identified_legit_objects / len(object_trust_groups) 
                balanced_object_accuracy = (fp_accuracy + legit_accuracy) / 2.0
                track_identification_bonus += balanced_object_accuracy * 3.0  # Up to 3 bonus points
        else:
            object_accuracy = 0.0  # No objects to classify
        
        final_reward = robot_identification_bonus + track_identification_bonus
        
        # Log detailed classification metrics
        print(f"  🏆 [FINAL REWARD] Episode Complete: {final_reward:.2f} points (assessed {total_tracks_processed} robot tracks)")
        print(f"    🤖 Robots: {correct_robot_identifications}/{total_robots} correct ({robot_accuracy:.2%} accuracy)")
        if adversarial_robots > 0:
            adv_acc = correctly_identified_adversarial / adversarial_robots
            print(f"      - Adversarial: {correctly_identified_adversarial}/{adversarial_robots} ({adv_acc:.2%})")
        if legitimate_robots > 0:
            leg_acc = correctly_identified_legitimate / legitimate_robots
            print(f"      - Legitimate: {correctly_identified_legitimate}/{legitimate_robots} ({leg_acc:.2%})")
        
        # Simple object-centric track logging
        print(f"    📦 Objects: {correct_object_identifications}/{total_objects} correct ({object_accuracy:.2%} accuracy) [from robot tracks]")
        if len(false_positive_groups) > 0:
            fp_acc = correctly_identified_fp_objects / len(false_positive_groups)
            print(f"      - False Positive Objects: {correctly_identified_fp_objects}/{len(false_positive_groups)} ({fp_acc:.2%})")
        if len(object_trust_groups) > 0:
            legit_acc = correctly_identified_legit_objects / len(object_trust_groups)
            print(f"      - Legitimate Objects: {correctly_identified_legit_objects}/{len(object_trust_groups)} ({legit_acc:.2%})")
        
        print(f"      - Total: {len(object_trust_groups)} legitimate objects, {len(false_positive_groups)} false positive objects")
        print(f"      - Processed {total_tracks_processed} robot tracks from {len(all_robots)} robots")
        
        return final_reward
    
    def _compute_sparse_reward(self, next_state, simulation_step_data, done):
        """
        Balanced reward structure:
        - Immediate reward for trust updates in correct direction
        - Small final episode reward (same scale as step rewards)
        """
        
        # === IMMEDIATE TRUST DIRECTION REWARD ===
        immediate_reward = self._compute_trust_direction_reward(next_state, simulation_step_data)
        
        # === SMALL FINAL EPISODE REWARD ===
        final_reward = 0.0
        if done:
            # Small final reward - same scale as step rewards
            episode_classification_score = self._compute_final_episode_reward(next_state, simulation_step_data)
            # Scale to similar magnitude as step rewards (0.1x instead of 3.0x)
            final_reward = episode_classification_score * 0.1  # Keep it small!
        
        total_reward = immediate_reward + final_reward
        
        # SCALE UP REWARD SIGNAL for stronger learning gradients
        scaled_reward = total_reward * 10.0  # 10x amplification
        
        return scaled_reward
    
    def _compute_trust_direction_reward(self, next_state, simulation_step_data):
        """
        Compute immediate reward for trust updates moving in correct direction
        Positive reward for trust moving toward ground truth, negative for moving away
        """
        
        direction_reward = 0.0
        
        # Extract simulation data
        ego_robot = simulation_step_data.get('ego_robot')
        proximal_robots = simulation_step_data.get('proximal_robots', [])
        all_robots = ([ego_robot] + proximal_robots) if ego_robot else proximal_robots
        
        # === ROBOT TRUST DIRECTION REWARD ===
        for robot in all_robots:
            if not hasattr(robot, 'trust_alpha') or not hasattr(robot, 'trust_beta'):
                continue
                
            # Current trust value after update
            current_trust = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
            is_adversarial = getattr(robot, 'is_adversarial', False)
            
            # Get previous trust value from stored distributions
            robot_id = robot.id
            prev_trust = None
            if hasattr(self, '_previous_trust_distributions') and robot_id in self._previous_trust_distributions:
                prev_data = self._previous_trust_distributions[robot_id]
                prev_trust = prev_data['alpha'] / (prev_data['alpha'] + prev_data['beta'])
            
            if prev_trust is not None:
                # Determine correct trust direction
                if is_adversarial:
                    # Adversarial robots should have LOW trust (target: 0.1)
                    target_trust = 0.1
                else:
                    # Legitimate robots should have HIGH trust (target: 0.9)
                    target_trust = 0.9
                
                # Calculate trust movement direction
                prev_distance = abs(prev_trust - target_trust)
                current_distance = abs(current_trust - target_trust)
                
                # Reward for moving closer to target, penalty for moving away
                if current_distance < prev_distance:
                    # Moving in correct direction
                    improvement = prev_distance - current_distance
                    direction_reward += improvement * 5.0  # Increased scale for better learning signal
                elif current_distance > prev_distance:
                    # Moving in wrong direction
                    degradation = current_distance - prev_distance
                    direction_reward -= degradation * 5.0  # Proportional penalty
        
        # === TRACK TRUST DIRECTION REWARD ===
        # Get current tracks from next state
        current_tracks = getattr(next_state, '_current_tracks', [])
        for track in current_tracks:
            if not hasattr(track, 'trust_alpha') or not hasattr(track, 'trust_beta'):
                continue
                
            current_track_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)
            track_id = track.id
            
            # Get previous track trust
            prev_track_trust = None
            if hasattr(self, '_previous_trust_distributions') and f"track_{track_id}" in self._previous_trust_distributions:
                prev_data = self._previous_trust_distributions[f"track_{track_id}"]
                prev_track_trust = prev_data['alpha'] / (prev_data['alpha'] + prev_data['beta'])
            
            if prev_track_trust is not None:
                # For tracks, use ground truth to determine target trust (same as robots)
                track_id_str = str(track.object_id) if hasattr(track, 'object_id') else str(track.id)
                
                if track_id_str.startswith('gt_obj_'):
                    # Ground truth object - should be trusted (high trust)
                    target_track_trust = 0.9
                elif track_id_str.startswith('shared_fp_') or track_id_str.startswith('fp_'):
                    # False positive object - should be distrusted (low trust)
                    target_track_trust = 0.1

                # Calculate track trust movement direction
                prev_distance = abs(prev_track_trust - target_track_trust)
                current_distance = abs(current_track_trust - target_track_trust)
                
                if current_distance < prev_distance:
                    improvement = prev_distance - current_distance
                    direction_reward += improvement * 3.0  # Increased scale for track rewards
                elif current_distance > prev_distance:
                    degradation = current_distance - prev_distance
                    direction_reward -= degradation * 2.0  # Proportional penalty
        
        # Store current trust distributions for next step comparison
        if not hasattr(self, '_previous_trust_distributions'):
            self._previous_trust_distributions = {}
        
        # Store robot trust distributions
        for robot in all_robots:
            if hasattr(robot, 'trust_alpha') and hasattr(robot, 'trust_beta'):
                self._previous_trust_distributions[robot.id] = {
                    'alpha': robot.trust_alpha,
                    'beta': robot.trust_beta
                }
        
        # Store track trust distributions
        for track in current_tracks:
            if hasattr(track, 'trust_alpha') and hasattr(track, 'trust_beta'):
                self._previous_trust_distributions[f"track_{track.id}"] = {
                    'alpha': track.trust_alpha,
                    'beta': track.trust_beta
                }
        
        # Clamp reward to prevent extreme values while preserving learning signal
        direction_reward = max(-10.0, min(10.0, direction_reward))
        
        return direction_reward
    
    def _find_track_by_id(self, track_id, final_state):
        """Find track object by ID in current tracks, fused tracks, or individual tracks"""
        # Check current tracks first
        if hasattr(final_state, '_current_tracks'):
            for track in final_state._current_tracks:
                if hasattr(track, 'id') and track.id == track_id:
                    return track
        
        # Check fused tracks
        if hasattr(final_state, '_fused_tracks'):
            for track in final_state._fused_tracks:
                if hasattr(track, 'id') and track.id == track_id:
                    return track
        
        # Check individual tracks
        if hasattr(final_state, '_individual_tracks'):
            for track in final_state._individual_tracks:
                if hasattr(track, 'id') and track.id == track_id:
                    return track
        
        # Check simulation environment tracks as fallback
        if hasattr(self.sim_env, 'robot_object_tracks'):
            for robot_id, track_dict in self.sim_env.robot_object_tracks.items():
                for obj_id, track in track_dict.items():
                    if hasattr(track, 'id') and track.id == track_id:
                        return track
                    # Also check if the track_id matches the expected format
                    expected_id = f"robot_{robot_id}_obj_{obj_id}"
                    if expected_id == track_id:
                        return track
        
        return None  # Track object not found
    
    def _estimate_track_quality(self, track, ground_truth_objects):
        """
        Estimate track quality by comparing with ground truth objects
        Returns a score between 0 and 1 indicating track reliability
        """
        if not ground_truth_objects or not hasattr(track, 'position'):
            return 0.5  # Default middle score if no ground truth available
        
        track_pos = track.position
        min_distance = float('inf')
        
        # Find closest ground truth object
        for gt_obj in ground_truth_objects:
            if hasattr(gt_obj, 'position'):
                distance = np.linalg.norm(track_pos - gt_obj.position)
                min_distance = min(min_distance, distance)
        
        # Convert distance to quality score (closer = higher quality)
        # Assume good tracks are within 5.0 units, perfect tracks within 1.0 unit
        if min_distance < 1.0:
            return 1.0  # Excellent track
        elif min_distance < 5.0:
            return 1.0 - (min_distance - 1.0) / 4.0  # Linear decay from 1.0 to 0.0
        else:
            return 0.1  # Poor track quality
    
    def get_ego_robot_state(self, ego_robot_id=None):
        """
        Get the current state as a graph representation for a specific ego robot
        Returns the graph with trust distributions as node features
        """
        current_state = self._get_current_state()
        
        # Add ego robot identification to the state
        if ego_robot_id is not None:
            current_state.ego_robot_id = ego_robot_id
        
        return current_state

def train_gnn_with_ppo(episodes=1000, max_steps_per_episode=500, device='cpu', save_path='ppo_trust_gnn.pth', 
                       enable_visualization=True, visualize_frequency=50, visualize_steps=[100, 150, 250, 350],
                       fov_range=50.0, fov_angle=np.pi/3):
    """
    Main PPO training loop
    
    Args:
        episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode (default: 500)
        device: Device to use ('cpu', 'cuda', or 'auto')
        save_path: Path to save the trained model
        enable_visualization: Enable GNN input visualization
        visualize_frequency: Frequency of visualization (every N episodes)
        visualize_steps: List of steps to visualize within episodes
        fov_range: Field of view range for robots (default: 50.0)
        fov_angle: Field of view angle in radians (default: π/3, 60 degrees)
    """
    
    print("🚀 GNN REINFORCEMENT LEARNING TRAINING (PPO)")
    print("=" * 60)
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # print(f"Device: {device}")
    
    # Initialize environment
    # print("🌍 Initializing RL environment...")
    env = RLTrustEnvironment(max_steps_per_episode=max_steps_per_episode, 
                             fov_range=fov_range, fov_angle=fov_angle)
    
    # Initialize PPO model with simplified neural-symbolic features
    # print("🤖 Initializing PPO model...")
    ppo_model = PPOTrustGNN(agent_features=5, track_features=5, hidden_dim=64)  # Updated feature counts
    trainer = PPOTrainer(ppo_model, device=device)
    
    # IMPORTANT: Pass trainer to environment for multi-ego action selection
    env._trainer_instance = trainer
    
    # print(f"Model parameters: {sum(p.numel() for p in ppo_model.parameters()):,}")
    
    # Training loop
    print(f"Starting PPO training for {episodes} episodes on {device}")
    print(f"Using {'multi-ego' if env.use_multi_ego else 'single-ego'} training mode")
    
    episode_rewards = []
    best_reward = float('-inf')
    
    for episode in range(episodes):
        
        # Reset environment
        state = env.reset()
        episode_reward = 0.0
        step_count = 0
        
        # Collect experience for one episode
        while step_count < max_steps_per_episode:  # Max steps per episode
            # Select action with policy outputs for visualization
            actions, log_probs, values = trainer.select_action(state)
            
            # Get policy outputs for visualization (re-run forward pass)
            policy_outputs = None
            if enable_visualization and episode % visualize_frequency == 0 and step_count in visualize_steps:
                try:
                    trainer.model.eval()
                    with torch.no_grad():
                        x_dict = {k: v.to(trainer.device) for k, v in state.x_dict.items()}
                        edge_index_dict = trainer._ensure_edge_types({})
                        policy_outputs, _ = trainer.model(x_dict, edge_index_dict)
                    trainer.model.train()
                except Exception as e:
                    # print(f"Warning: Could not get policy outputs for visualization: {e}")
                    pass
            
            # Visualize GNN input graph if it's time
            if enable_visualization and episode % visualize_frequency == 0 and step_count in visualize_steps:
                try:
                    # print(f"🎯 Visualizing GNN input graph - Episode {episode}, Step {step_count}")
                    visualize_gnn_input(state, episode=episode, timestep=step_count, current_state=state)
                except Exception as e:
                    # print(f"Warning: GNN visualization failed: {e}")
                    pass
            
            # Take environment step
            next_state, reward, done, info = env.step(actions)
            
            # Store experience
            experience = PPOExperience(
                graph_data=state,
                action=actions,
                reward=reward,
                log_prob=log_probs,
                value=values,
                done=done,
                next_graph_data=next_state
            )
            trainer.add_experience(experience)
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            if done:
                break
        
        # Finish the episode
        trainer.finish_episode()
        
        # Update policy when we have enough episodes collected
        losses = trainer.update_policy()  # This handles the min_episodes check internally
        
        episode_rewards.append(episode_reward)
        
        # Show experience buffer status
        total_experiences = sum(len(ep) for ep in trainer.episode_experiences)
        buffer_episodes = len(trainer.episode_experiences)
        
        # Brief episode completion logging with buffer status
        print(f"Episode {episode:4d} completed | Reward: {episode_reward:7.2f} | Steps: {step_count:3d} | Buffer: {total_experiences:3d} exp / {buffer_episodes:2d} eps")
        
        # Detailed logging after policy updates
        if losses:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            current_lr = trainer.scheduler.get_last_lr()[0] if hasattr(trainer, 'scheduler') else trainer.optimizer.param_groups[0]['lr']
            print(f"🔄 POLICY UPDATE after Episode {episode}")
            print(f"   Avg Reward (last 10): {avg_reward:8.2f} | Current: {episode_reward:8.2f}")
            print(f"   Policy Loss: {losses.get('policy_loss', 0):8.4f} | Value Loss: {losses.get('value_loss', 0):8.2f}")
            print(f"   Entropy: {losses.get('entropy_loss', 0):8.3f} | Learning Rate: {current_lr:.6f}")
            print(f"   Updates: Policy={losses.get('num_policy_updates', 0)} Value={losses.get('num_value_updates', 0)}")
            print("-" * 60)
        
        # Periodic learning progress summary
        if episode % 10 == 0 and episode > 0:
            recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
            reward_trend = "↗️" if len(episode_rewards) > 5 and np.mean(episode_rewards[-5:]) > np.mean(episode_rewards[-10:-5]) else "➡️"
            print(f"📊 Episode {episode:3d} Summary: Avg Reward = {np.mean(recent_rewards):6.2f} {reward_trend} | Buffer: {total_experiences} experiences")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'model_state_dict': ppo_model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'episode': episode,
                'best_reward': best_reward,
                'episode_rewards': episode_rewards
            }, save_path)
    
    print(f"\nTraining completed!")
    print(f"Best episode reward: {best_reward:.2f}")
    print(f"Average final 100 episodes: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Model saved to: {save_path}")
    
    return episode_rewards




def main():
    """Main function"""
    import sys
    
    # Parse command line arguments
    device = 'auto'
    episodes = 1000
    enable_visualization = True
    
    # Parse all arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ['--no-viz', '--disable-visualization', '--no-visualization']:
            enable_visualization = False
            # print("📊 Visualization disabled for large-scale training")
        elif arg in ['cpu', 'cuda', 'auto']:
            device = arg
        elif arg.isdigit():
            episodes = int(arg)
        i += 1
    
    try:
        # Run PPO training
        rewards = train_gnn_with_ppo(
            episodes=episodes,
            device=device,
            save_path='ppo_trust_gnn.pth',
            enable_visualization=enable_visualization
        )
        
        # Plot results
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # Moving average
        window_size = 50
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), moving_avg)
            plt.title(f'Moving Average ({window_size} episodes)')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('ppo_training_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training results saved to: ppo_training_results.png")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()