#!/usr/bin/env python3
"""
Neural Symbolic Trust Algorithm using Graph Neural Networks

This module implements a GNN-based trust algorithm for multi-robot trust estimation
using neural symbolic predicates and PPO reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from typing import List, Dict, Optional

from trust_algorithm import TrustAlgorithm, RobotState, Track


class PPOTrustGNN(nn.Module):
    """
    Standalone GNN for PPO that outputs action probabilities and values
    instead of direct trust updates
    """
    
    def __init__(self, agent_features: int, track_features: int, hidden_dim: int = 64):
        super(PPOTrustGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Node embedding layers
        self.agent_embedding = Linear(agent_features, hidden_dim)
        self.track_embedding = Linear(track_features, hidden_dim)
        
        # Heterogeneous graph convolution layers with attention mechanisms
        self.conv1 = HeteroConv({
            ('agent', 'in_fov_and_observed', 'track'): GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1, add_self_loops=False),
            ('track', 'observed_and_in_fov_by', 'agent'): GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1, add_self_loops=False),
            ('agent', 'in_fov_only', 'track'): GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1, add_self_loops=False),
            ('track', 'in_fov_only_by', 'agent'): GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1, add_self_loops=False),
        })
        
        self.conv2 = HeteroConv({
            ('agent', 'in_fov_and_observed', 'track'): GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1, add_self_loops=False),
            ('track', 'observed_and_in_fov_by', 'agent'): GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1, add_self_loops=False),
            ('agent', 'in_fov_only', 'track'): GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1, add_self_loops=False),
            ('track', 'in_fov_only_by', 'agent'): GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1, add_self_loops=False),
        })
        
        # Add third conv layer for deeper representation
        self.conv3 = HeteroConv({
            ('agent', 'in_fov_and_observed', 'track'): GATConv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=0.1, add_self_loops=False),
            ('track', 'observed_and_in_fov_by', 'agent'): GATConv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=0.1, add_self_loops=False),
            ('agent', 'in_fov_only', 'track'): GATConv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=0.1, add_self_loops=False),
            ('track', 'in_fov_only_by', 'agent'): GATConv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=0.1, add_self_loops=False),
        })
        
        # Batch normalization layers for stability
        self.norm1 = nn.ModuleDict({
            'agent': nn.BatchNorm1d(hidden_dim),
            'track': nn.BatchNorm1d(hidden_dim)
        })
        self.norm2 = nn.ModuleDict({
            'agent': nn.BatchNorm1d(hidden_dim),
            'track': nn.BatchNorm1d(hidden_dim)
        })
        self.norm3 = nn.ModuleDict({
            'agent': nn.BatchNorm1d(hidden_dim),
            'track': nn.BatchNorm1d(hidden_dim)
        })
        
        
        # Enhanced value function heads with better architecture for stability
        self.agent_value_function = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        self.track_value_function = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Add value function regularization
        self.value_regularization = nn.Parameter(torch.tensor(0.01))
        
        # Policy heads for Beta distribution parameters - outputs value_alpha, value_beta, conf_alpha, conf_beta
        self.agent_policy_value_alpha = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Output > 0 for Beta distribution alpha parameter
        )
        self.agent_policy_value_beta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Output > 0 for Beta distribution beta parameter
        )
        self.agent_policy_conf_alpha = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Output > 0 for Beta distribution alpha parameter
        )
        self.agent_policy_conf_beta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Output > 0 for Beta distribution beta parameter
        )
        self.track_policy_value_alpha = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Output > 0 for Beta distribution alpha parameter
        )
        self.track_policy_value_beta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Output > 0 for Beta distribution beta parameter
        )
        self.track_policy_conf_alpha = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Output > 0 for Beta distribution alpha parameter
        )
        self.track_policy_conf_beta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Output > 0 for Beta distribution beta parameter
        )
        
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
        
        # Graph convolution layers with skip connections and batch normalization
        # First layer
        x_dict_1 = self.conv1(x_dict, edge_index_dict)
        x_dict_1 = {key: F.relu(x) for key, x in x_dict_1.items()}
        
        # Apply batch normalization (only if batch size > 1)
        for key in x_dict_1:
            if x_dict_1[key].shape[0] > 1:
                x_dict_1[key] = self.norm1[key](x_dict_1[key])
        
        # Second layer with skip connection
        x_dict_2 = self.conv2(x_dict_1, edge_index_dict)
        x_dict_2 = {key: F.relu(x) for key, x in x_dict_2.items()}
        
        # Apply batch normalization
        for key in x_dict_2:
            if x_dict_2[key].shape[0] > 1:
                x_dict_2[key] = self.norm2[key](x_dict_2[key])
        
        # Add skip connection from first layer
        x_dict_2 = {key: x_dict_2[key] + x_dict_1[key] for key in x_dict_2}
        
        # Third layer with skip connection
        x_dict_3 = self.conv3(x_dict_2, edge_index_dict)
        x_dict_3 = {key: F.relu(x) for key, x in x_dict_3.items()}
        
        # Apply batch normalization
        for key in x_dict_3:
            if x_dict_3[key].shape[0] > 1:
                x_dict_3[key] = self.norm3[key](x_dict_3[key])
        
        # Add skip connection from second layer
        x_dict = {key: x_dict_3[key] + x_dict_2[key] for key in x_dict_3}
        
        if return_features:
            return x_dict
        
        # Generate policy outputs (action probabilities)
        policy_outputs = {}
        value_outputs = {}
        
        if 'agent' in x_dict:
            # PSM policy outputs: value_alpha, value_beta, conf_alpha, conf_beta for trust updates
            # Clamp to reasonable ranges to prevent extreme Beta parameters
            agent_policy_value_alpha = self.agent_policy_value_alpha(x_dict['agent']) + 1.0
            agent_policy_value_beta = self.agent_policy_value_beta(x_dict['agent']) + 1.0
            agent_policy_conf_alpha = self.agent_policy_conf_alpha(x_dict['agent']) + 1.0
            agent_policy_conf_beta = self.agent_policy_conf_beta(x_dict['agent']) + 1.0

            # Value function: expected return with regularization
            agent_values = self.agent_value_function(x_dict['agent'])
            # Apply value regularization to reduce oscillations
            agent_values = agent_values * (1.0 - self.value_regularization) + self.value_regularization * torch.tanh(agent_values)
            
            policy_outputs['agent'] = {
                'value_alpha': agent_policy_value_alpha,
                'value_beta': agent_policy_value_beta,
                'conf_alpha': agent_policy_conf_alpha,
                'conf_beta': agent_policy_conf_beta
            }
            value_outputs['agent'] = agent_values
        
        if 'track' in x_dict and has_tracks:
            # PSM policy outputs: value_alpha, value_beta, conf_alpha, conf_beta for trust updates
            # Clamp to reasonable ranges to prevent extreme Beta parameters
            track_policy_value_alpha = self.track_policy_value_alpha(x_dict['track']) + 1.0
            track_policy_value_beta = self.track_policy_value_beta(x_dict['track']) + 1.0
            track_policy_conf_alpha = self.track_policy_conf_alpha(x_dict['track']) + 1.0
            track_policy_conf_beta = self.track_policy_conf_beta(x_dict['track']) + 1.0

            track_values = self.track_value_function(x_dict['track'])
            # Apply value regularization to reduce oscillations
            track_values = track_values * (1.0 - self.value_regularization) + self.value_regularization * torch.tanh(track_values)
            
            policy_outputs['track'] = {
                'value_alpha': track_policy_value_alpha,
                'value_beta': track_policy_value_beta,
                'conf_alpha': track_policy_conf_alpha,
                'conf_beta': track_policy_conf_beta
            }
            value_outputs['track'] = track_values
        
        return policy_outputs, value_outputs

class NeuralSymbolicTrustAlgorithm(TrustAlgorithm):
    """Trust algorithm using neural symbolic GNN for comparison purposes"""
    
    def __init__(self, learning_mode: bool = False, model_path: str = None, device: str = 'cpu'):
        """
        Initialize with optional trained GNN model for inference
        
        Args:
            learning_mode: If True, algorithm is in learning mode (RL training)
            model_path: Path to trained PPO model checkpoint
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.learning_mode = learning_mode
        self.robot_object_trust: Dict[int, Dict[str, tuple]] = {}
        self.model_path = model_path
        self.device = device
        
        # Initialize GNN model if path provided
        self.gnn_model = None
        self.trainer = None
        
        if model_path and not learning_mode:
            self._load_trained_model()
    
    def _load_trained_model(self):
        """Load trained PPO-GNN model for inference"""
        try:
            import torch
            from train_gnn_rl import PPOTrainer
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Create model with correct architecture (6 agent features, 6 track features)
            # Features: 4 original neural-symbolic predicates + alpha + beta
            self.gnn_model = PPOTrustGNN(agent_features=6, track_features=6, hidden_dim=64)
            
            # Load model weights with strict=False to handle architecture differences
            self.gnn_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Create trainer for inference
            device_obj = torch.device(self.device)
            self.trainer = PPOTrainer(self.gnn_model, device=device_obj)
            
            # Set to evaluation mode
            self.gnn_model.eval()
            
            print(f"✅ Loaded trained GNN model from {self.model_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to load GNN model: {e}")
            self.gnn_model = None
            self.trainer = None
    
    def initialize(self, robots: List[RobotState]):
        """Initialize robot trust tracking"""
        for robot in robots:
            self.robot_object_trust[robot.id] = {}
    
    def update_trust(self, robots: List[RobotState], tracks_by_robot: Dict[int, List[Track]], 
                    robot_object_tracks: Dict[int, Dict[str, Track]], current_time: float,
                    robot_current_tracks: Optional[Dict[int, Dict[str, Track]]] = None,
                    simulation_env: Optional = None) -> Dict[str, any]:
        """
        Update trust using trained GNN model if available, otherwise maintain basic trust values
        """
        if self.learning_mode:
            # In learning mode, don't perform updates (handled by training loop)
            return {}
        
        # Initialize robot trust tracking if needed
        for robot in robots:
            if robot.id not in self.robot_object_trust:
                self.robot_object_trust[robot.id] = {}
        
        # If GNN model is available, use it for trust updates
        if self.gnn_model is not None and self.trainer is not None and simulation_env is not None:
            return self._update_trust_with_gnn(robots, tracks_by_robot, robot_object_tracks, 
                                             current_time, robot_current_tracks, simulation_env)
        else:
            # Fallback: maintain basic trust values without updates
            return {}
    
    def _update_trust_with_gnn(self, robots: List[RobotState], tracks_by_robot: Dict[int, List[Track]], 
                              robot_object_tracks: Dict[int, Dict[str, Track]], current_time: float,
                              robot_current_tracks: Optional[Dict[int, Dict[str, Track]]] = None,
                              simulation_env = None) -> Dict[str, any]:
        """
        Use trained GNN model to update trust values
        """
        try:
            # This would require implementing the full graph construction and trust update logic
            # similar to what's in the comparison script. For now, return empty dict.
            # TODO: Implement full GNN-based trust update logic
            return {}
            
        except Exception as e:
            print(f"⚠️ GNN trust update failed: {e}")
            return {}
    
    def get_expected_trust(self, alpha: float, beta: float) -> float:
        """Calculate expected value E[trust] = alpha / (alpha + beta)"""
        return alpha / (alpha + beta)
    
    def get_trust_variance(self, alpha: float, beta: float) -> float:
        """Calculate variance of trust distribution"""
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    
    def predict_trust_updates(self, graph_data):
        """
        Use trained GNN to predict trust updates for given graph
        
        Args:
            graph_data: HeteroData graph with agent and track nodes
            
        Returns:
            Tuple of (robot_actions, track_actions) or None if model not available
        """
        if self.trainer is None or self.gnn_model is None:
            print("⚠️ No trained GNN model available for predictions")
            return None, None
            
        try:
            # Use stochastic inference (same as training)
            actions, policy_outputs, value_outputs = self.trainer.select_action(graph_data, deterministic=False)
            
            # Extract robot and track actions
            robot_actions = actions.get('agent', {})
            track_actions = actions.get('track', {})
            
            return robot_actions, track_actions
            
        except Exception as e:
            print(f"⚠️ GNN prediction failed: {e}")
            return None, None
    
    def is_model_available(self) -> bool:
        """Check if trained GNN model is available"""
        return self.gnn_model is not None and self.trainer is not None