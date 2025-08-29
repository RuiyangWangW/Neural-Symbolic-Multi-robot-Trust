#!/usr/bin/env python3
"""
GNN Reinforcement Learning Training with PPO

This script trains the neural symbolic GNN model using Proximal Policy Optimization (PPO)
instead of supervised learning. The GNN learns to make trust update decisions that maximize
overall multi-robot system performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from collections import namedtuple, deque
import random

from neural_symbolic_trust_algorithm import NeuralSymbolicTrustAlgorithm, TrustGNN
from simulation_environment import SimulationEnvironment
from paper_trust_algorithm import PaperTrustAlgorithm
from visualize_gnn_input_graph import visualize_gnn_input

# PPO Experience tuple
PPOExperience = namedtuple('PPOExperience', [
    'graph_data', 'action', 'reward', 'log_prob', 'value', 'done', 'next_graph_data'
])

class PPOTrustGNN(nn.Module):
    """
    Modified GNN for PPO that outputs action probabilities and values
    instead of direct trust updates
    """
    
    def __init__(self, base_gnn: TrustGNN):
        super(PPOTrustGNN, self).__init__()
        
        # Use the existing GNN architecture
        self.agent_embedding = base_gnn.agent_embedding
        self.track_embedding = base_gnn.track_embedding
        self.conv1 = base_gnn.conv1
        self.conv2 = base_gnn.conv2
        self.symbolic_layer = base_gnn.symbolic_layer
        
        hidden_dim = base_gnn.hidden_dim
        
        # Policy heads for action selection (value and confidence for each node type)
        self.agent_policy_value = nn.Linear(hidden_dim, 1)  # Trust value action
        self.agent_policy_confidence = nn.Linear(hidden_dim, 1)  # Trust confidence action
        self.track_policy_value = nn.Linear(hidden_dim, 1)
        self.track_policy_confidence = nn.Linear(hidden_dim, 1)
        
        # Value function heads for PPO
        self.agent_value_function = nn.Linear(hidden_dim, 1)
        self.track_value_function = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_dict, edge_index_dict, return_features=False):
        """Forward pass that returns both policy and value outputs"""
        
        # Check if we have any tracks
        has_tracks = x_dict['track'].shape[0] > 0
        
        # Initial embeddings
        x_dict_embedded = {
            'agent': F.relu(self.agent_embedding(x_dict['agent']))
        }
        
        # Only process track embeddings if we have tracks
        if has_tracks:
            x_dict_embedded['track'] = F.relu(self.track_embedding(x_dict['track']))
        else:
            # Create empty track tensor with correct dimensions
            device = x_dict['agent'].device
            x_dict_embedded['track'] = torch.empty(0, self.hidden_dim, device=device)
        
        x_dict = x_dict_embedded
        
        # Graph convolution layers
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Apply symbolic reasoning
        x_dict = self.symbolic_layer(x_dict)
        
        if return_features:
            return x_dict
        
        # Generate policy outputs (action probabilities)
        policy_outputs = {}
        value_outputs = {}
        
        if 'agent' in x_dict:
            # Policy: action probabilities for trust updates
            agent_policy_value = torch.sigmoid(self.agent_policy_value(x_dict['agent']))
            agent_policy_confidence = torch.sigmoid(self.agent_policy_confidence(x_dict['agent']))
            
            # Value function: expected return
            agent_values = self.agent_value_function(x_dict['agent'])
            
            policy_outputs['agent'] = {
                'value': agent_policy_value,
                'confidence': agent_policy_confidence
            }
            value_outputs['agent'] = agent_values
        
        if 'track' in x_dict and has_tracks:
            track_policy_value = torch.sigmoid(self.track_policy_value(x_dict['track']))
            track_policy_confidence = torch.sigmoid(self.track_policy_confidence(x_dict['track']))
            
            track_values = self.track_value_function(x_dict['track'])
            
            policy_outputs['track'] = {
                'value': track_policy_value,
                'confidence': track_policy_confidence
            }
            value_outputs['track'] = track_values
        
        return policy_outputs, value_outputs


class PPOTrainer:
    """PPO trainer for the trust GNN"""
    
    def __init__(self, model: PPOTrustGNN, learning_rate: float = 3e-4, 
                 device: torch.device = torch.device('cpu')):
        self.model = model.to(device)
        self.device = device
        
        # PPO hyperparameters
        self.lr = learning_rate
        self.eps_clip = 0.2  # PPO clipping parameter
        self.gamma = 0.99    # Discount factor
        self.lam = 0.95      # GAE lambda
        self.value_coef = 0.5  # Value function loss coefficient
        self.entropy_coef = 0.01  # Entropy bonus coefficient
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)
        
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
                ('track', 'in_fov_by', 'agent')
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
                    agent_value_alpha = agent_policy['value'] * 2 + 0.1
                    agent_value_beta = (1 - agent_policy['value']) * 2 + 0.1
                    agent_conf_alpha = agent_policy['confidence'] * 2 + 0.1
                    agent_conf_beta = (1 - agent_policy['confidence']) * 2 + 0.1
                    
                    # Sample from Beta distributions
                    agent_value_action = torch.distributions.Beta(agent_value_alpha, agent_value_beta).sample()
                    agent_conf_action = torch.distributions.Beta(agent_conf_alpha, agent_conf_beta).sample()
                
                actions['agent'] = {
                    'value': agent_value_action,
                    'confidence': agent_conf_action
                }
                
                # Compute log probabilities (simplified for now)
                log_probs['agent'] = {
                    'value': -((agent_value_action - agent_policy['value']) ** 2).sum(),
                    'confidence': -((agent_conf_action - agent_policy['confidence']) ** 2).sum()
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
                    track_value_alpha = track_policy['value'] * 2 + 0.1
                    track_value_beta = (1 - track_policy['value']) * 2 + 0.1
                    track_conf_alpha = track_policy['confidence'] * 2 + 0.1
                    track_conf_beta = (1 - track_policy['confidence']) * 2 + 0.1
                    
                    track_value_action = torch.distributions.Beta(track_value_alpha, track_value_beta).sample()
                    track_conf_action = torch.distributions.Beta(track_conf_alpha, track_conf_beta).sample()
                
                actions['track'] = {
                    'value': track_value_action,
                    'confidence': track_conf_action
                }
                
                log_probs['track'] = {
                    'value': -((track_value_action - track_policy['value']) ** 2).sum(),
                    'confidence': -((track_conf_action - track_policy['confidence']) ** 2).sum()
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
            ('track', 'in_fov_by', 'agent')
        ]
        
        result = {}
        for edge_type in required_edge_types:
            if edge_type in edge_index_dict:
                result[edge_type] = edge_index_dict[edge_type].to(self.device)
            else:
                result[edge_type] = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        return result
    
    def compute_reward(self, actions, graph_data, next_graph_data, simulation_step_data):
        """
        Compute reward for the taken actions based on ACTUAL trust alignment
        
        Reward components:
        1. Trust alignment with ground truth (based on updated trust distributions)
        2. Detection accuracy improvement
        3. Confidence calibration bonus
        4. System stability reward
        """
        
        # Extract simulation data
        ego_robot = simulation_step_data.get('ego_robot')
        proximal_robots = simulation_step_data.get('proximal_robots', [])
        ground_truth_objects = simulation_step_data.get('ground_truth_objects', [])
        
        all_robots = ([ego_robot] + proximal_robots) if ego_robot else proximal_robots
        
        # Reward 1: Robot Trust Alignment with Ground Truth
        robot_trust_reward = 0.0
        if 'agent' in actions and hasattr(graph_data, 'agent_nodes'):
            for robot_id, node_idx in graph_data.agent_nodes.items():
                if node_idx < actions['agent']['value'].shape[0]:
                    # Find robot object
                    robot = None
                    for r in all_robots:
                        if r.id == robot_id:
                            robot = r
                            break
                    
                    if robot is not None and hasattr(robot, 'trust_alpha'):
                        # Calculate current trust after update
                        current_trust = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
                        is_trustworthy = not getattr(robot, 'is_adversarial', False)
                        
                        if is_trustworthy:
                            # Reward higher trust for legitimate robots (target ~0.7-0.8)
                            optimal_trust_diff = abs(current_trust - 0.75)
                            robot_trust_reward += (1.0 - optimal_trust_diff) * 2.0
                        else:
                            # Reward lower trust for adversarial robots (target ~0.2-0.3)
                            optimal_trust_diff = abs(current_trust - 0.25)
                            robot_trust_reward += (1.0 - optimal_trust_diff) * 2.0
        
        # Reward 2: Track Quality Assessment with Ground Truth Correlation
        track_quality_reward = 0.0
        if 'track' in actions and hasattr(graph_data, 'track_nodes'):
            current_tracks = getattr(graph_data, '_current_tracks', [])
            
            for track_id, track_idx in graph_data.track_nodes.items():
                if track_idx < actions['track']['value'].shape[0]:
                    # Find corresponding track
                    track = None
                    for t in current_tracks:
                        if t.id == track_id:
                            track = t
                            break
                    
                    if track is not None and hasattr(track, 'trust_alpha'):
                        # Calculate current track trust
                        track_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)
                        
                        # Estimate track quality based on confidence and consistency
                        track_confidence = getattr(track, 'confidence', 0.5)
                        
                        # Reward tracks with trust that correlates with their confidence
                        # High confidence tracks should have higher trust, low confidence lower trust
                        target_trust = 0.4 + (track_confidence * 0.4)  # Range: 0.4-0.8 based on confidence
                        trust_alignment = 1.0 - abs(track_trust - target_trust)
                        track_quality_reward += trust_alignment * 0.5
        
        # Reward 3: Gradual Trust Changes (prevent extreme jumps)
        gradual_change_reward = 0.0
        if hasattr(self, '_previous_trust_values'):
            # Check robot trust changes
            for robot in all_robots:
                if (hasattr(robot, 'trust_alpha') and robot.id in self._previous_trust_values):
                    prev_trust = self._previous_trust_values[robot.id]['trust']
                    current_trust = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
                    trust_change = abs(current_trust - prev_trust)
                    
                    # Reward gradual changes (< 0.1 per step), penalize large jumps
                    if trust_change < 0.05:
                        gradual_change_reward += 0.3
                    elif trust_change < 0.1:
                        gradual_change_reward += 0.1
                    else:
                        gradual_change_reward -= 0.2
        
        # Store current trust values for next iteration
        if not hasattr(self, '_previous_trust_values'):
            self._previous_trust_values = {}
        
        for robot in all_robots:
            if hasattr(robot, 'trust_alpha'):
                current_trust = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
                self._previous_trust_values[robot.id] = {'trust': current_trust}
        
        # Reward 4: Confidence Calibration
        confidence_reward = 0.0
        if 'agent' in actions:
            agent_conf = actions['agent']['confidence']
            avg_confidence = torch.mean(agent_conf).item()
            # Reward moderate confidence (avoid overconfidence)
            if 0.4 <= avg_confidence <= 0.8:
                confidence_reward += 0.4
            elif 0.2 <= avg_confidence < 0.4 or 0.8 < avg_confidence <= 0.9:
                confidence_reward += 0.1
            else:
                confidence_reward -= 0.3
        
        # Reward 5: System Stability
        stability_reward = 0.0
        for robot in all_robots:
            if hasattr(robot, 'trust_alpha') and hasattr(robot, 'trust_beta'):
                # Reward stable trust parameters
                if 0.5 <= robot.trust_alpha <= 5.0 and 0.5 <= robot.trust_beta <= 5.0:
                    stability_reward += 0.2
                elif 0.1 <= robot.trust_alpha <= 10.0 and 0.1 <= robot.trust_beta <= 10.0:
                    stability_reward += 0.05
                else:
                    stability_reward -= 0.5
        
        # Reward 6: Base survival reward for maintaining the system
        base_reward = 1.0
        
        # Combine all rewards with carefully tuned weights
        total_reward = (base_reward * 0.3 +                    # Base survival
                       robot_trust_reward * 0.4 +              # Robot trust alignment (most important)
                       track_quality_reward * 0.15 +           # Track quality
                       gradual_change_reward * 0.1 +           # Gradual changes
                       confidence_reward * 0.05 +              # Confidence calibration
                       stability_reward * 0.05)                # System stability
        
        return total_reward
    
    def update_policy(self, batch_size=64, n_epochs=4):
        """Update policy using PPO"""
        
        if len(self.experience_buffer) < batch_size:
            return {}
        
        # Sample batch from experience buffer
        batch = random.sample(self.experience_buffer, min(batch_size, len(self.experience_buffer)))
        
        # Prepare batch data
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(n_epochs):
            for experience in batch:
                # Move data to device
                x_dict = {k: v.to(self.device) for k, v in experience.graph_data.x_dict.items()}
                
                # Handle edge_index_dict safely (same as in select_action)
                try:
                    existing_edges = {}
                    required_edge_types = [
                        ('agent', 'observes', 'track'),
                        ('track', 'observed_by', 'agent'),
                        ('agent', 'in_fov', 'track'),
                        ('track', 'in_fov_by', 'agent')
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
                policy_outputs, value_outputs = self.model(x_dict, edge_index_dict)
                
                # Compute policy loss (simplified PPO loss)
                policy_loss = 0.0
                value_loss = 0.0
                entropy_loss = 0.0
                
                # Agent policy loss
                if 'agent' in policy_outputs and 'agent' in experience.action:
                    agent_policy = policy_outputs['agent']
                    agent_action = experience.action['agent']
                    agent_old_log_prob = experience.log_prob['agent']
                    agent_value = value_outputs['agent']
                    
                    # Compute new log prob
                    new_log_prob_value = -((agent_action['value'].to(self.device) - agent_policy['value']) ** 2).sum()
                    new_log_prob_conf = -((agent_action['confidence'].to(self.device) - agent_policy['confidence']) ** 2).sum()
                    
                    # PPO ratio
                    ratio_value = torch.exp(new_log_prob_value - agent_old_log_prob['value'])
                    ratio_conf = torch.exp(new_log_prob_conf - agent_old_log_prob['confidence'])
                    
                    # PPO clipped objective
                    advantage = experience.reward  # Simplified advantage
                    
                    surr1_value = ratio_value * advantage
                    surr2_value = torch.clamp(ratio_value, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                    
                    surr1_conf = ratio_conf * advantage
                    surr2_conf = torch.clamp(ratio_conf, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                    
                    policy_loss -= (torch.min(surr1_value, surr2_value) + torch.min(surr1_conf, surr2_conf))
                    
                    # Value loss - handle shape mismatch
                    reward_tensor = torch.tensor(experience.reward, device=self.device).float()
                    if agent_value.dim() > 0:
                        # Expand reward to match agent_value shape
                        reward_tensor = reward_tensor.expand_as(agent_value.squeeze())
                    value_loss += F.mse_loss(agent_value.squeeze(), reward_tensor)
                    
                    # Entropy bonus (simplified) - ensure it's a tensor
                    entropy_bonus = torch.tensor(0.01, device=self.device)  # Small constant entropy bonus
                    entropy_loss = entropy_loss - entropy_bonus if isinstance(entropy_loss, torch.Tensor) else -entropy_bonus
                
                # Similar for tracks (omitted for brevity)
                
                # Ensure all losses are tensors before appending
                if isinstance(policy_loss, torch.Tensor):
                    policy_losses.append(policy_loss)
                if isinstance(value_loss, torch.Tensor):
                    value_losses.append(value_loss)
                if isinstance(entropy_loss, torch.Tensor):
                    entropy_losses.append(entropy_loss)
        
        # Compute total loss
        if policy_losses:
            total_policy_loss = torch.stack(policy_losses).mean()
            total_value_loss = torch.stack(value_losses).mean()
            total_entropy_loss = torch.stack(entropy_losses).mean() if entropy_losses else torch.tensor(0.0)
            
            total_loss = total_policy_loss + self.value_coef * total_value_loss + self.entropy_coef * total_entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            return {
                'policy_loss': total_policy_loss.item(),
                'value_loss': total_value_loss.item(),
                'entropy_loss': total_entropy_loss.item(),
                'total_loss': total_loss.item()
            }
        
        return {}


class RLTrustEnvironment:
    """Environment wrapper for RL training"""
    
    def __init__(self, num_robots=5, num_targets=10, adversarial_ratio=0.3, scenario_config=None):
        if scenario_config is not None:
            # Use diverse scenario configuration
            self.num_robots = scenario_config['num_robots']
            self.num_targets = scenario_config['num_targets']
            self.adversarial_ratio = scenario_config['adversarial_ratio']
            self.world_size = scenario_config['world_size']
            self.false_positive_rate = scenario_config['false_positive_rate']
            self.false_negative_rate = scenario_config['false_negative_rate']
            self.sensor_range = scenario_config['sensor_range']
            self.communication_range = scenario_config['communication_range']
            self.movement_speed = scenario_config['movement_speed']
            self.proximal_range = scenario_config['proximal_range']
        else:
            # Use default values
            self.num_robots = num_robots
            self.num_targets = num_targets
            self.adversarial_ratio = adversarial_ratio
            self.world_size = (50, 50)
            self.false_positive_rate = 0.5
            self.false_negative_rate = 0.3
            self.sensor_range = 30.0
            self.communication_range = 50.0
            self.movement_speed = 1.0
            self.proximal_range = 50.0
        
        # Initialize paper algorithm for comparison baseline
        self.paper_algo = PaperTrustAlgorithm()
        self.neural_algo = NeuralSymbolicTrustAlgorithm(learning_mode=True)
        
        # Episode-level ego robot tracking
        self.episode_ego_robot_id = None
        
        self.reset()
    
    def reset(self):
        """Reset environment for new episode"""
        # CRITICAL: Clear all persistent trust data to prevent carryover BETWEEN episodes
        if hasattr(self, '_episode_tracks'):
            self._episode_tracks.clear()
            print("  🧹 Cleared episode track trust data")
        
        if hasattr(self, '_previous_trust_distributions'):
            self._previous_trust_distributions.clear()
            print("  🧹 Cleared previous trust distributions")
        
        # Create new simulation environment (since it doesn't have reset method)
        self.sim_env = SimulationEnvironment(
            num_robots=self.num_robots,
            num_targets=self.num_targets,
            adversarial_ratio=self.adversarial_ratio,
            world_size=self.world_size,
            proximal_range=self.proximal_range
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
        try:
            self.sim_env.step()
        except Exception as e:
            print(f"Warning: Initial simulation step failed: {e}")
        
        return self._get_current_state()
    
    def _verify_clean_trust_reset(self):
        """Verify that all trust values start fresh for the episode"""
        if hasattr(self.sim_env, 'robots'):
            for robot in self.sim_env.robots:
                if hasattr(robot, 'trust_alpha') and hasattr(robot, 'trust_beta'):
                    if robot.trust_alpha != 1.0 or robot.trust_beta != 1.0:
                        print(f"  ⚠️  WARNING: Robot {robot.id} trust not reset! α={robot.trust_alpha:.3f}, β={robot.trust_beta:.3f}")
                        # Force reset to default values
                        robot.trust_alpha = 1.0
                        robot.trust_beta = 1.0
        
        # Ensure no persistent data structures exist
        persistent_attrs = ['_episode_tracks', '_previous_trust_distributions']
        for attr in persistent_attrs:
            if hasattr(self, attr) and getattr(self, attr):
                print(f"  ⚠️  WARNING: {attr} not properly cleared!")
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
                print(f"  🎯 Selected Robot {self.episode_ego_robot_id} as ego robot for this episode")
            else:
                self.episode_ego_robot_id = None
        else:
            self.episode_ego_robot_id = None
    
    def _get_current_state(self):
        """Get current state with proper multi-robot track fusion"""
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
        
        # Find the consistent ego robot for this episode
        ego_robot = None
        proximal_robots = []
        
        if self.episode_ego_robot_id is not None:
            # Find the ego robot by ID to maintain consistency
            for robot in robots:
                if robot.id == self.episode_ego_robot_id:
                    ego_robot = robot
                else:
                    proximal_robots.append(robot)
        
        # Fallback: if ego robot not found, select first robot
        if ego_robot is None:
            print(f"  ⚠️  Ego robot {self.episode_ego_robot_id} not found, falling back to first robot")
            ego_robot = robots[0]
            proximal_robots = robots[1:] if len(robots) > 1 else []
            self.episode_ego_robot_id = ego_robot.id
        
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
        
        # Initialize episode track storage if needed
        if not hasattr(self, '_episode_tracks'):
            self._episode_tracks = {}
        
        all_robots = [ego_robot] + proximal_robots
        
        # CRITICAL: Use CURRENT timestep tracks only (robot_current_tracks) for graph construction
        # Historical tracks (robot_object_tracks) are maintained for persistence but excluded from graph
        has_current_tracks = (hasattr(self.sim_env, 'robot_current_tracks') and 
                             self.sim_env.robot_current_tracks)
        
        if has_current_tracks:
            print("  🎯 Using CURRENT timestep tracks from robot_current_tracks")
        else:
            print("  🧪 Using ground truth objects (no current tracks available)")
        
        for robot in all_robots:
            robot_tracks = []
            
            # STEP 1: Get CURRENT timestep tracks from simulation robot_current_tracks
            if has_current_tracks and robot.id in self.sim_env.robot_current_tracks:
                current_tracks = self.sim_env.robot_current_tracks[robot.id]
                
                print(f"  📡 Robot {robot.id} has {len(current_tracks)} current timestep tracks")
                
                for object_id, current_track in current_tracks.items():
                    track_id = f"current_robot_{robot.id}_obj_{object_id}"
                    
                    # Get trust values from the historical track in robot_object_tracks for persistence
                    # but use CURRENT track data for graph construction
                    historical_track = None
                    if (hasattr(self.sim_env, 'robot_object_tracks') and 
                        robot.id in self.sim_env.robot_object_tracks and
                        object_id in self.sim_env.robot_object_tracks[robot.id]):
                        historical_track = self.sim_env.robot_object_tracks[robot.id][object_id]
                    
                    # Get or initialize trust values with episode persistence
                    if track_id in self._episode_tracks:
                        trust_alpha = self._episode_tracks[track_id]['trust_alpha']
                        trust_beta = self._episode_tracks[track_id]['trust_beta']
                    elif historical_track:
                        # Use historical track's trust values
                        trust_alpha = getattr(historical_track, 'trust_alpha', 1.0)
                        trust_beta = getattr(historical_track, 'trust_beta', 1.0)
                        self._episode_tracks[track_id] = {
                            'trust_alpha': trust_alpha,
                            'trust_beta': trust_beta
                        }
                    else:
                        # Initialize with default values
                        trust_alpha = 1.0
                        trust_beta = 1.0
                        self._episode_tracks[track_id] = {
                            'trust_alpha': trust_alpha,
                            'trust_beta': trust_beta
                        }
                    
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
            if len(robot_tracks) == 0:
                print(f"  📡 Robot {robot.id} has no current timestep tracks - this is realistic!")
            
            individual_robot_tracks[robot.id] = robot_tracks
            print(f"  📡 Robot {robot.id} observes {len(robot_tracks)} tracks")
        
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
        
        print(f"  🔀 Track fusion: {len(fused_tracks)} fused, {len(individual_tracks)} individual")
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
        
        # Store in episode storage
        self._episode_tracks[fused_id] = {
            'trust_alpha': trust_alpha,
            'trust_beta': trust_beta
        }
        
        return fused_track
    
    def _build_multi_robot_graph(self, ego_robot, proximal_robots, fused_tracks, individual_tracks, track_fusion_map):
        """Build graph with all robots and tracks, including proper edge relationships"""
        from torch_geometric.data import HeteroData
        import torch
        
        graph_data = HeteroData()
        all_robots = [ego_robot] + proximal_robots
        all_tracks = fused_tracks + individual_tracks
        
        # Create agent nodes for all robots
        agent_nodes = {}
        agent_features = []
        for i, robot in enumerate(all_robots):
            agent_nodes[robot.id] = i
            # Simple feature: just robot trust value
            robot_trust = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
            agent_features.append([robot_trust])
        
        graph_data['agent'].x = torch.tensor(agent_features, dtype=torch.float)
        graph_data.agent_nodes = agent_nodes
        
        # Create track nodes for all tracks (fused + individual)
        track_nodes = {}
        track_features = []
        for i, track in enumerate(all_tracks):
            track_nodes[track.id] = i
            # Simple feature: track trust value  
            track_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)
            track_features.append([track_trust])
        
        graph_data['track'].x = torch.tensor(track_features, dtype=torch.float)
        graph_data.track_nodes = track_nodes
        
        # Build edges: Observes and InFoV relationships
        observes_edges = []  # (agent, track) - robot observes track
        observed_by_edges = []  # (track, agent) - track observed by robot
        in_fov_edges = []  # (agent, track) - track is in robot's field of view
        in_fov_by_edges = []  # (track, agent) - track is in robot's field of view
        
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
        
        print(f"  🔗 Edge creation: {len(observes_edges)} observes, {len(in_fov_edges)} in_fov")
        
        # Debug: Check if any robot has proper orientation
        orientations_set = sum(1 for robot in all_robots if hasattr(robot, 'orientation'))
        if orientations_set == 0:
            print(f"  ⚠️  Warning: No robots have orientation set, FoV angle constraints disabled")
        
        # Convert to tensors and create proper edge structure
        edge_types = [
            ('agent', 'observes', 'track'),
            ('track', 'observed_by', 'agent'),
            ('agent', 'in_fov', 'track'),
            ('track', 'in_fov_by', 'agent')
        ]
        
        edge_data = [observes_edges, observed_by_edges, in_fov_edges, in_fov_by_edges]
        
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
        
        print(f"  🔗 Graph built: {len(all_robots)} robots, {len(all_tracks)} tracks ({len(fused_tracks)} fused, {len(individual_tracks)} individual)")
        
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
        fov_range = getattr(robot, 'fov_range', 20.0)  # Use robot's actual fov_range
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
            ('track', 'in_fov_by', 'agent')
        ]
        
        for edge_type in required_edge_types:
            graph_data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        graph_data.agent_nodes = {}
        graph_data.track_nodes = {}
        
        return graph_data
    
    def step(self, actions):
        """Take environment step with given actions"""
        
        # Get current state BEFORE applying updates
        current_state = self._get_current_state()
        
        try:
            # Advance simulation FIRST
            self.sim_env.step()
            self.step_count += 1
        except Exception as e:
            print(f"Warning: Simulation step failed: {e}")
            # Continue with current state
            
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
        
        # Compute reward based on ground truth alignment and trust divergence
        reward = self._compute_trust_based_reward(actions, current_state, next_state, simulation_step_data)
        
        # Check if episode is done
        num_objects = len(ground_truth_objects)
        num_robots = len(robots)
        
        done = (num_robots == 0 or 
                num_objects == 0)
        
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
                        # Store original trust for logging
                        original_alpha = robot.trust_alpha
                        original_beta = robot.trust_beta
                        
                        # Convert GNN action to trust parameter updates
                        psm_value = agent_actions['value'][node_idx].item()
                        psm_confidence = agent_actions['confidence'][node_idx].item()
                        
                        # Update robot trust using PSM approach
                        delta_alpha = psm_confidence * psm_value * 0.05  # Gradual changes
                        delta_beta = psm_confidence * (1.0 - psm_value) * 0.05
                        
                        # Apply updates directly to robot
                        robot.trust_alpha += delta_alpha
                        robot.trust_beta += delta_beta
                        
                        # Debug logging
                        print(f"  [TRUST UPDATE] Robot {robot_id}: α {original_alpha:.3f}→{robot.trust_alpha:.3f} (Δ{delta_alpha:+.3f}), β {original_beta:.3f}→{robot.trust_beta:.3f} (Δ{delta_beta:+.3f})")
        
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
                        # Store original trust for logging
                        original_alpha = track.trust_alpha
                        original_beta = track.trust_beta
                        
                        # Convert GNN action to trust parameter updates
                        psm_value = track_actions['value'][node_idx].item()
                        psm_confidence = track_actions['confidence'][node_idx].item()
                        
                        # Update track trust using PSM approach
                        delta_alpha = psm_confidence * psm_value * 0.1  # Slightly larger for tracks
                        delta_beta = psm_confidence * (1.0 - psm_value) * 0.1
                        
                        # Apply updates to track trust parameters
                        track.trust_alpha += delta_alpha
                        track.trust_beta += delta_beta
                        
                        # Store in episode storage
                        if hasattr(self, '_episode_tracks'):
                            self._episode_tracks[track_id] = {
                                'trust_alpha': track.trust_alpha,
                                'trust_beta': track.trust_beta
                            }
                        
                        # CRITICAL: Propagate trust updates to individual robot tracks
                        self._propagate_track_trust_updates(track, track_id, track_fusion_map, delta_alpha, delta_beta)
                        
                        # Debug logging
                        print(f"  [TRUST UPDATE] Track {track_id}: α {original_alpha:.3f}→{track.trust_alpha:.3f} (Δ{delta_alpha:+.3f}), β {original_beta:.3f}→{track.trust_beta:.3f} (Δ{delta_beta:+.3f})")
    
    def _propagate_track_trust_updates(self, updated_track, updated_track_id, track_fusion_map, delta_alpha, delta_beta):
        """Propagate trust updates from fused/individual tracks back to individual robot track lists"""
        
        # Check if this is a fused track - propagate to constituent individual tracks
        if 'fused_' in updated_track_id:
            # Find all individual tracks that contributed to this fused track
            for original_track_id, fused_id in track_fusion_map.items():
                if fused_id == updated_track_id:
                    # This original track contributed to the updated fused track
                    # Update the trust values in episode storage for the original track too
                    if hasattr(self, '_episode_tracks') and original_track_id in self._episode_tracks:
                        # Propagate the same trust update to the individual track
                        self._episode_tracks[original_track_id]['trust_alpha'] += delta_alpha
                        self._episode_tracks[original_track_id]['trust_beta'] += delta_beta
                        
                        print(f"    [PROPAGATE] Original track {original_track_id}: α +{delta_alpha:.3f}, β +{delta_beta:.3f}")
        else:
            # This is an individual track - update is already applied
            # Just ensure it's stored in episode storage
            if hasattr(self, '_episode_tracks'):
                self._episode_tracks[updated_track_id] = {
                    'trust_alpha': updated_track.trust_alpha,
                    'trust_beta': updated_track.trust_beta
                }
    
    def _compute_environment_reward(self, actions):
        """Compute reward based on environment state"""
        
        # Reward components:
        # 1. Detection accuracy
        # 2. False positive/negative rates  
        # 3. Trust calibration
        
        base_reward = 1.0  # Base reward for surviving
        
        # Add performance-based rewards
        num_objects = 0
        num_robots = 0
        
        if hasattr(self.sim_env, 'ground_truth_objects'):
            if isinstance(self.sim_env.ground_truth_objects, list):
                num_objects = len(self.sim_env.ground_truth_objects)
            elif hasattr(self.sim_env.ground_truth_objects, '__len__'):
                num_objects = len(self.sim_env.ground_truth_objects)
        
        if hasattr(self.sim_env, 'robots'):
            if isinstance(self.sim_env.robots, list):
                num_robots = len(self.sim_env.robots)
            elif hasattr(self.sim_env.robots, '__len__'):
                num_robots = len(self.sim_env.robots)
        
        # Reward for maintaining system stability
        stability_reward = 0.1 * (num_robots / max(1, self.step_count * 0.01))
        
        # Reward for object tracking
        tracking_reward = 0.05 * num_objects
        
        return base_reward + stability_reward + tracking_reward
    
    def _compute_trust_based_reward(self, actions, current_state, next_state, simulation_step_data):
        """
        Compute reward based on:
        1. Ground truth alignment for robots (legitimate vs adversarial)
        2. Ground truth alignment for tracks (quality assessment)
        3. Trust distribution divergence penalties (avoid extreme changes)
        4. Action confidence calibration
        """
        # Extract simulation data
        ego_robot = simulation_step_data.get('ego_robot')
        proximal_robots = simulation_step_data.get('proximal_robots', [])
        ground_truth_objects = simulation_step_data.get('ground_truth_objects', [])
        
        all_robots = ([ego_robot] + proximal_robots) if ego_robot else proximal_robots
        
        # === REWARD COMPONENT 1: ROBOT GROUND TRUTH ALIGNMENT ===
        robot_alignment_reward = 0.0
        robot_divergence_penalty = 0.0
        
        if 'agent' in actions and hasattr(next_state, 'agent_nodes'):
            for robot_id, node_idx in next_state.agent_nodes.items():
                if node_idx < actions['agent']['value'].shape[0]:
                    # Find robot object in next state (with updated trust)
                    next_robot = None
                    for r in all_robots:
                        if r.id == robot_id:
                            next_robot = r
                            break
                    
                    if next_robot is not None and hasattr(next_robot, 'trust_alpha'):
                        # Calculate updated trust distribution
                        next_trust = next_robot.trust_alpha / (next_robot.trust_alpha + next_robot.trust_beta)
                        is_legitimate = not getattr(next_robot, 'is_adversarial', False)
                        
                        # Ground truth alignment reward
                        if is_legitimate:
                            # Reward higher trust for legitimate robots (target: 0.8)
                            alignment_error = abs(next_trust - 0.8)
                            robot_alignment_reward += max(0, 1.0 - alignment_error) * 3.0
                        else:
                            # Reward lower trust for adversarial robots (target: 0.2)
                            alignment_error = abs(next_trust - 0.2)
                            robot_alignment_reward += max(0, 1.0 - alignment_error) * 3.0
                        
                        # Trust divergence penalty (compare with previous state if available)
                        if hasattr(self, '_previous_trust_distributions') and robot_id in self._previous_trust_distributions:
                            prev_trust = self._previous_trust_distributions[robot_id]['trust']
                            trust_divergence = abs(next_trust - prev_trust)
                            
                            # Penalty for large changes (> 0.1 per step is too much)
                            if trust_divergence > 0.1:
                                robot_divergence_penalty -= (trust_divergence - 0.1) * 2.0
                            elif trust_divergence > 0.05:
                                robot_divergence_penalty -= (trust_divergence - 0.05) * 1.0
        
        # === REWARD COMPONENT 2: TRACK GROUND TRUTH ALIGNMENT ===
        track_alignment_reward = 0.0
        track_divergence_penalty = 0.0
        
        if 'track' in actions and hasattr(next_state, 'track_nodes'):
            next_tracks = getattr(next_state, '_current_tracks', [])
            
            for track_id, track_idx in next_state.track_nodes.items():
                if track_idx < actions['track']['value'].shape[0]:
                    # Find corresponding track in next state
                    next_track = None
                    for t in next_tracks:
                        if t.id == track_id:
                            next_track = t
                            break
                    
                    if next_track is not None and hasattr(next_track, 'trust_alpha'):
                        # Calculate updated track trust
                        next_track_trust = next_track.trust_alpha / (next_track.trust_alpha + next_track.trust_beta)
                        
                        # Estimate track quality from ground truth correlation
                        track_confidence = getattr(next_track, 'confidence', 0.5)
                        track_quality_score = self._estimate_track_quality(next_track, ground_truth_objects)
                        
                        # Ground truth alignment: trust should correlate with actual track quality
                        target_trust = 0.3 + (track_quality_score * 0.5)  # Range: 0.3-0.8
                        track_alignment_error = abs(next_track_trust - target_trust)
                        track_alignment_reward += max(0, 1.0 - track_alignment_error) * 1.5
                        
                        # Track trust divergence penalty
                        if hasattr(self, '_previous_trust_distributions') and track_id in self._previous_trust_distributions:
                            prev_track_trust = self._previous_trust_distributions[track_id]['trust']
                            track_divergence = abs(next_track_trust - prev_track_trust)
                            
                            if track_divergence > 0.1:
                                track_divergence_penalty -= (track_divergence - 0.1) * 1.5
                            elif track_divergence > 0.05:
                                track_divergence_penalty -= (track_divergence - 0.05) * 0.75
        
        # === REWARD COMPONENT 3: ACTION CONFIDENCE CALIBRATION ===
        confidence_calibration_reward = 0.0
        
        if 'agent' in actions:
            agent_conf = actions['agent']['confidence']
            avg_confidence = torch.mean(agent_conf).item()
            
            # Reward well-calibrated confidence (not too high, not too low)
            if 0.5 <= avg_confidence <= 0.8:
                confidence_calibration_reward += 0.5
            elif 0.3 <= avg_confidence < 0.5 or 0.8 < avg_confidence <= 0.9:
                confidence_calibration_reward += 0.2
            else:
                confidence_calibration_reward -= 0.3
        
        if 'track' in actions:
            track_conf = actions['track']['confidence']
            avg_track_conf = torch.mean(track_conf).item()
            
            # Similar calibration for track confidence
            if 0.5 <= avg_track_conf <= 0.8:
                confidence_calibration_reward += 0.3
            elif 0.3 <= avg_track_conf < 0.5 or 0.8 < avg_track_conf <= 0.9:
                confidence_calibration_reward += 0.1
            else:
                confidence_calibration_reward -= 0.2
        
        # === REWARD COMPONENT 4: SYSTEM HEALTH ===
        system_health_reward = 0.0
        
        # Reward for maintaining reasonable trust parameter ranges
        for robot in all_robots:
            if hasattr(robot, 'trust_alpha') and hasattr(robot, 'trust_beta'):
                if 0.5 <= robot.trust_alpha <= 8.0 and 0.5 <= robot.trust_beta <= 8.0:
                    system_health_reward += 0.1
                elif robot.trust_alpha > 15.0 or robot.trust_beta > 15.0:
                    system_health_reward -= 0.5  # Penalty for extreme values
        
        # Store current trust distributions for next iteration
        if not hasattr(self, '_previous_trust_distributions'):
            self._previous_trust_distributions = {}
        
        # Update stored distributions with current (next) state values
        for robot in all_robots:
            if hasattr(robot, 'trust_alpha'):
                current_trust = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
                self._previous_trust_distributions[robot.id] = {'trust': current_trust}
        
        if hasattr(next_state, '_current_tracks'):
            for track in next_state._current_tracks:
                if hasattr(track, 'trust_alpha'):
                    current_track_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)
                    self._previous_trust_distributions[track.id] = {'trust': current_track_trust}
        
        # === FINAL REWARD COMPUTATION ===
        total_reward = (robot_alignment_reward * 0.4 +          # Robot ground truth alignment (most important)
                       robot_divergence_penalty * 0.15 +       # Robot trust divergence penalty  
                       track_alignment_reward * 0.25 +         # Track ground truth alignment
                       track_divergence_penalty * 0.1 +        # Track trust divergence penalty
                       confidence_calibration_reward * 0.05 +  # Action confidence calibration
                       system_health_reward * 0.05)            # System health maintenance
        
        return total_reward
    
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




def train_gnn_with_ppo(episodes=1000, max_steps=500, device='cpu', save_path='ppo_trust_gnn.pth', 
                       enable_visualization=True, visualize_frequency=50, visualize_steps=[100, 150, 250, 350]):
    """Main PPO training loop"""
    
    print("🚀 GNN REINFORCEMENT LEARNING TRAINING (PPO)")
    print("=" * 60)
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Device: {device}")
    
    # Initialize environment
    print("🌍 Initializing RL environment...")
    env = RLTrustEnvironment()
    
    # Initialize PPO model
    print("🤖 Initializing PPO model...")
    base_gnn = TrustGNN(agent_features=1, track_features=1, hidden_dim=64)
    ppo_model = PPOTrustGNN(base_gnn)
    trainer = PPOTrainer(ppo_model, device=device)
    
    print(f"Model parameters: {sum(p.numel() for p in ppo_model.parameters()):,}")
    
    # Training loop
    print(f"\n🏋️ Starting PPO training for {episodes} episodes...")
    
    episode_rewards = []
    best_reward = float('-inf')
    
    for episode in range(episodes):
        
        # Reset environment
        state = env.reset()
        episode_reward = 0.0
        step_count = 0
        
        # Collect experience for one episode
        while step_count < max_steps:  # Max steps per episode
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
                    print(f"Warning: Could not get policy outputs for visualization: {e}")
            
            # Visualize GNN input graph if it's time
            if enable_visualization and episode % visualize_frequency == 0 and step_count in visualize_steps:
                try:
                    print(f"🎯 Visualizing GNN input graph - Episode {episode}, Step {step_count}")
                    visualize_gnn_input(state, episode=episode, timestep=step_count)
                except Exception as e:
                    print(f"Warning: GNN visualization failed: {e}")
            
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
            trainer.experience_buffer.append(experience)
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            if done:
                break
        
        # Update policy
        if len(trainer.experience_buffer) >= 64:
            losses = trainer.update_policy()
        else:
            losses = {}
        
        episode_rewards.append(episode_reward)
        
        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode:4d} | Avg Reward: {avg_reward:7.2f} | "
                  f"Steps: {step_count:3d} | Buffer: {len(trainer.experience_buffer)}")
            
            if losses:
                print(f"              | Policy Loss: {losses.get('policy_loss', 0):7.4f} | "
                      f"Value Loss: {losses.get('value_loss', 0):7.4f}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'model_state_dict': ppo_model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'episode': episode,
                'best_reward': best_reward,
                'episode_rewards': episode_rewards
            }, save_path)
    
    print(f"\n✅ Training completed!")
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
            print("📊 Visualization disabled for large-scale training")
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
        
        print(f"📊 Training results saved to: ppo_training_results.png")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()