#!/usr/bin/env python3
"""
Neural Symbolic Trust Algorithm using Graph Neural Networks

This module implements a GNN-based trust algorithm for multi-robot trust estimation
using neural symbolic predicates and PPO reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, Linear
from typing import List, Dict, Optional

from trust_algorithm import TrustAlgorithm, RobotState, Track


class HeteroGATConv(nn.Module):
    """
    Simplified heterogeneous GAT layer for multi-robot trust systems.
    Uses GAT for all edge relationships instead of mixed conv types.
    """
    
    def __init__(self, node_types_channels: Dict[str, int], out_channels: int, heads: int = 4):
        super().__init__()
        
        # Define edge types for trust networks
        edge_types = [
            ('agent', 'in_fov_and_observed', 'track'),
            ('track', 'observed_and_in_fov_by', 'agent'),
            ('agent', 'in_fov_only', 'track'),
            ('track', 'in_fov_only_by', 'agent'),
        ]
        
        # Create GAT convolution dictionary for each edge type
        conv_dict = {}
        for src_type, relation, dst_type in edge_types:
            src_dim = node_types_channels[src_type]
            
            # Use GAT for all edge relationships
            conv_dict[(src_type, relation, dst_type)] = GATConv(
                in_channels=src_dim,
                out_channels=out_channels,
                heads=heads,
                concat=False,
                add_self_loops=False,
                dropout=0.1
            )
        
        # Create the heterogeneous convolution
        self.hetero_conv = HeteroConv(conv_dict, aggr='mean')
    
    def forward(self, x_dict, edge_index_dict):
        # Apply heterogeneous GAT convolution
        out_dict = self.hetero_conv(x_dict, edge_index_dict)
        return out_dict

class SharedGNNEncoder(nn.Module):
    """
    Shared GNN encoder for both Actor and Critic in CTDE architecture
    """
    
    def __init__(self, agent_features: int, track_features: int, hidden_dim: int = 64):
        super(SharedGNNEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Node embedding layers
        self.agent_embedding = Linear(agent_features, hidden_dim)
        self.track_embedding = Linear(track_features, hidden_dim)
        
        # Use GAT layers for all convolutions
        # Define node type channels for each layer
        initial_channels = {'agent': hidden_dim, 'track': hidden_dim}
        
        # Layer 1: Initial feature processing with GAT
        self.conv1 = HeteroGATConv(initial_channels, hidden_dim, heads=4)
        
        # Layer 2: Deeper feature refinement with GAT
        self.conv2 = HeteroGATConv(
            {'agent': hidden_dim, 'track': hidden_dim}, 
            hidden_dim, 
            heads=4
        )
        
        # Layer 3: Final feature processing with GAT
        self.conv3 = HeteroGATConv(
            {'agent': hidden_dim, 'track': hidden_dim}, 
            hidden_dim, 
            heads=4
        )
        
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
    
    def forward(self, x_dict, edge_index_dict):
        """Encode graph into node embeddings"""
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
        
        # Add skip connection from first layer (only for non-empty tensors)
        x_dict_2 = {
            key: x_dict_2[key] + x_dict_1[key] if x_dict_2[key].shape[0] > 0 and x_dict_1[key].shape[0] > 0 
            else x_dict_2[key] 
            for key in x_dict_2
        }
        
        # Third layer with skip connection
        x_dict_3 = self.conv3(x_dict_2, edge_index_dict)
        x_dict_3 = {key: F.relu(x) for key, x in x_dict_3.items()}
        
        # Apply batch normalization
        for key in x_dict_3:
            if x_dict_3[key].shape[0] > 1:
                x_dict_3[key] = self.norm3[key](x_dict_3[key])
        
        # Add skip connection from second layer (only for non-empty tensors)
        x_dict = {
            key: x_dict_3[key] + x_dict_2[key] if x_dict_3[key].shape[0] > 0 and x_dict_2[key].shape[0] > 0 
            else x_dict_3[key] 
            for key in x_dict_3
        }
        
        return x_dict


class TrustActor(nn.Module):
    """
    Decentralized Actor for MAPPO-EgoGraph
    Uses ego-graph to produce trust update actions for the ego robot and its observed tracks
    """
    
    def __init__(self, hidden_dim: int = 64):
        super(TrustActor, self).__init__()
        
        # Policy heads for trust value updates (Beta distribution parameters)
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
    
    def forward(self, node_embeddings):
        """
        Generate policy outputs from node embeddings
        
        Args:
            node_embeddings: Dict of node embeddings from SharedGNNEncoder
            
        Returns:
            policy_outputs: Dict containing action distributions
        """
        policy_outputs = {}
        has_tracks = node_embeddings['track'].shape[0] > 0
        
        if 'agent' in node_embeddings:
            # PSM policy outputs: value_alpha, value_beta for trust updates
            # Clamp to reasonable ranges to prevent extreme Beta parameters
            agent_policy_value_alpha = self.agent_policy_value_alpha(node_embeddings['agent']) + 1.0
            agent_policy_value_beta = self.agent_policy_value_beta(node_embeddings['agent']) + 1.0
            
            # Calculate confidence as the mean of the Beta distribution: α/(α+β)
            agent_confidence = agent_policy_value_alpha / (agent_policy_value_alpha + agent_policy_value_beta)
            
            policy_outputs['agent'] = {
                'value_alpha': agent_policy_value_alpha,
                'value_beta': agent_policy_value_beta,
                'confidence': agent_confidence
            }
        
        if 'track' in node_embeddings and has_tracks:
            # PSM policy outputs: value_alpha, value_beta for trust updates
            # Clamp to reasonable ranges to prevent extreme Beta parameters
            track_policy_value_alpha = self.track_policy_value_alpha(node_embeddings['track']) + 1.0
            track_policy_value_beta = self.track_policy_value_beta(node_embeddings['track']) + 1.0
            
            # Calculate confidence as the mean of the Beta distribution: α/(α+β)
            track_confidence = track_policy_value_alpha / (track_policy_value_alpha + track_policy_value_beta)
            
            policy_outputs['track'] = {
                'value_alpha': track_policy_value_alpha,
                'value_beta': track_policy_value_beta,
                'confidence': track_confidence
            }
        
        return policy_outputs


class TrustCritic(nn.Module):
    """
    Centralized Critic for MAPPO-EgoGraph
    Uses global graph to produce a shared value function for all agents
    """
    
    def __init__(self, hidden_dim: int = 64):
        super(TrustCritic, self).__init__()
        
        # Value function heads - more sophisticated architecture for centralized critic
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
        
        # Global value aggregation for permutation-invariant value function
        # FIXED: Updated input size for richer state representation (mean + max + std for agents and tracks)
        self.global_value_head = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),  # Rich features: 6 * hidden_dim input
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Value function regularization
        self.value_regularization = nn.Parameter(torch.tensor(0.01))
    
    def forward(self, node_embeddings):
        """
        Generate value outputs from global node embeddings
        
        Args:
            node_embeddings: Dict of node embeddings from SharedGNNEncoder
            
        Returns:
            value_outputs: Either scalar global value or dict of node values
        """
        has_tracks = node_embeddings['track'].shape[0] > 0
        
        # FIXED: Compute richer global state representation for value function
        # Use multiple aggregation methods to preserve state information
        
        # Agent features: mean, max, std to capture distribution
        agent_mean = torch.mean(node_embeddings['agent'], dim=0, keepdim=True)  # [1, hidden_dim]
        agent_max = torch.max(node_embeddings['agent'], dim=0, keepdim=True)[0]  # [1, hidden_dim]
        agent_std = torch.std(node_embeddings['agent'], dim=0, keepdim=True)   # [1, hidden_dim]
        
        if has_tracks:
            track_mean = torch.mean(node_embeddings['track'], dim=0, keepdim=True)  # [1, hidden_dim]
            track_max = torch.max(node_embeddings['track'], dim=0, keepdim=True)[0]  # [1, hidden_dim]
            track_std = torch.std(node_embeddings['track'], dim=0, keepdim=True)   # [1, hidden_dim]
        else:
            # Create zero track features if no tracks
            device = node_embeddings['agent'].device
            track_mean = torch.zeros(1, node_embeddings['agent'].shape[1], device=device)
            track_max = torch.zeros(1, node_embeddings['agent'].shape[1], device=device)
            track_std = torch.zeros(1, node_embeddings['agent'].shape[1], device=device)
        
        # Concatenate rich global features: mean + max + std for both agents and tracks
        global_features = torch.cat([
            agent_mean, agent_max, agent_std,  # 3 * hidden_dim
            track_mean, track_max, track_std   # 3 * hidden_dim
        ], dim=1)  # [1, hidden_dim * 6]
        
        # Global value function (shared across all agents)
        global_value = self.global_value_head(global_features).squeeze()  # Scalar
        
        # Remove problematic value regularization that was causing values to stick around 0.4
        # global_value = global_value * (1.0 - self.value_regularization) + self.value_regularization * torch.tanh(global_value)
        
        return global_value


class PPOTrustGNN(nn.Module):
    """
    MAPPO-EgoGraph model with separate Actor and Critic for CTDE
    """
    
    def __init__(self, agent_features: int, track_features: int, hidden_dim: int = 64):
        super(PPOTrustGNN, self).__init__()
        
        # Shared GNN encoder for both actor and critic
        self.encoder = SharedGNNEncoder(agent_features, track_features, hidden_dim)
        
        # Decentralized Actor (uses ego-graphs)
        self.actor = TrustActor(hidden_dim)
        
        # Centralized Critic (uses global graph)
        self.critic = TrustCritic(hidden_dim)
        
    def forward(self, x_dict, edge_index_dict, return_features=False, policy_only=False, value_only=False):
        """
        Forward pass using separate Actor and Critic components
        
        Args:
            x_dict: Node features dictionary
            edge_index_dict: Edge indices dictionary  
            return_features: If True, return intermediate features
            policy_only: If True, only compute policy outputs (for decentralized actor)
            value_only: If True, only compute value outputs (for centralized critic)
        """
        
        # Encode the graph using shared GNN encoder
        node_embeddings = self.encoder(x_dict, edge_index_dict)
        
        if return_features:
            return node_embeddings
        
        policy_outputs = {}
        value_outputs = {}
        
        # Generate policy outputs using Actor (for decentralized execution)
        if not value_only:
            policy_outputs = self.actor(node_embeddings)
        
        # Generate value outputs using Critic (for centralized training)
        if not policy_only:
            # Critic returns a scalar global value
            global_value = self.critic(node_embeddings)
            value_outputs = global_value  # Single scalar value for all agents
        
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
            
            # Create model with correct architecture (6 agent features, 7 track features)
            # Features: 5 neural-symbolic predicates + alpha + beta
            self.gnn_model = PPOTrustGNN(agent_features=6, track_features=7, hidden_dim=64)
            
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