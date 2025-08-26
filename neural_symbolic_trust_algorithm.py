#!/usr/bin/env python3
"""
Neural Symbolic Trust Algorithm using Graph Neural Networks

This module implements a GNN-based trust algorithm that learns from
simulation data and uses neural symbolic predicates (InFoV, Observed, Trustworthy)
to estimate trust values for agents and tracks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, Linear
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import pickle
from scipy.optimize import linear_sum_assignment

from trust_algorithm import TrustAlgorithm, RobotState, Track


@dataclass
class GraphNode:
    """Graph node representing either an agent or track"""
    node_id: str
    node_type: str  # 'agent' or 'track'
    features: np.ndarray
    position: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class GraphEdge:
    """Graph edge with relationship type and features"""
    src_node: str
    dst_node: str
    edge_type: str  # 'in_fov', 'observed'
    features: np.ndarray
    confidence: float


@dataclass
class TrainingExample:
    """Training example for the GNN"""
    graph_data: HeteroData
    labels: Dict[str, Dict[str, float]]  # node_id -> {'value': float, 'confidence': float}
    metadata: Dict[str, Any]


class TrustGNN(nn.Module):
    """Graph Neural Network for trust estimation"""
    
    def __init__(self, agent_features: int, track_features: int, hidden_dim: int = 64):
        super(TrustGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Node embedding layers
        self.agent_embedding = Linear(agent_features, hidden_dim)
        self.track_embedding = Linear(track_features, hidden_dim)
        
        # Heterogeneous graph convolution layers using SAGEConv for bipartite support
        self.conv1 = HeteroConv({
            ('agent', 'observes', 'track'): SAGEConv(hidden_dim, hidden_dim),
            ('track', 'observed_by', 'agent'): SAGEConv(hidden_dim, hidden_dim),
            ('agent', 'in_fov', 'track'): SAGEConv(hidden_dim, hidden_dim),
            ('track', 'in_fov_by', 'agent'): SAGEConv(hidden_dim, hidden_dim),
        })
        
        self.conv2 = HeteroConv({
            ('agent', 'observes', 'track'): SAGEConv(hidden_dim, hidden_dim),
            ('track', 'observed_by', 'agent'): SAGEConv(hidden_dim, hidden_dim),
            ('agent', 'in_fov', 'track'): SAGEConv(hidden_dim, hidden_dim),
            ('track', 'in_fov_by', 'agent'): SAGEConv(hidden_dim, hidden_dim),
        })
        
        # Output heads for value and confidence prediction
        self.agent_value_head = Linear(hidden_dim, 1)
        self.agent_confidence_head = Linear(hidden_dim, 1)
        self.track_value_head = Linear(hidden_dim, 1)
        self.track_confidence_head = Linear(hidden_dim, 1)
        
        # Symbolic reasoning layer
        self.symbolic_layer = SymbolicReasoningLayer(hidden_dim)
    
    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        """Forward pass through the GNN"""
        # Initial embeddings
        x_dict = {
            'agent': F.relu(self.agent_embedding(x_dict['agent'])),
            'track': F.relu(self.track_embedding(x_dict['track']))
        }
        
        # Graph convolution layers
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Apply symbolic reasoning
        x_dict = self.symbolic_layer(x_dict)
        
        # Generate predictions - handle missing node types
        predictions = {}
        
        if 'agent' in x_dict:
            agent_values = torch.sigmoid(self.agent_value_head(x_dict['agent']))
            agent_confidences = torch.sigmoid(self.agent_confidence_head(x_dict['agent']))
            predictions['agent'] = {'value': agent_values, 'confidence': agent_confidences}
        
        if 'track' in x_dict:
            track_values = torch.sigmoid(self.track_value_head(x_dict['track']))
            track_confidences = torch.sigmoid(self.track_confidence_head(x_dict['track']))
            predictions['track'] = {'value': track_values, 'confidence': track_confidences}
        
        return predictions


class SymbolicReasoningLayer(nn.Module):
    """Neural symbolic reasoning layer that applies logical constraints"""
    
    def __init__(self, hidden_dim: int):
        super(SymbolicReasoningLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable parameters for symbolic rules
        self.trustworthy_threshold = nn.Parameter(torch.tensor(0.5))
        self.consensus_weight = nn.Parameter(torch.tensor(1.0))
        self.fov_weight = nn.Parameter(torch.tensor(0.8))
        
        # Rule application networks
        self.rule_network = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x_dict):
        """Apply symbolic reasoning rules"""
        # Rule 1: Trustworthy agents should have consistent observations
        if 'agent' in x_dict:
            agent_features = x_dict['agent']
            rule_adjustment = self.rule_network(agent_features)
            x_dict['agent'] = agent_features + 0.1 * rule_adjustment
        
        # Rule 2: Tracks observed by multiple agents are more trustworthy
        if 'track' in x_dict:
            track_features = x_dict['track']
            rule_adjustment = self.rule_network(track_features)
            x_dict['track'] = track_features + 0.1 * rule_adjustment
        
        return x_dict


class GraphBuilder:
    """Builds graph structures from simulation state"""
    
    def __init__(self, fov_threshold: float = 20.0):
        self.fov_threshold = fov_threshold
    
    def build_graph(self, ego_robot: RobotState, ego_fused_tracks: List[Track],
                   proximal_robots: List[RobotState], 
                   proximal_tracks_by_robot: Dict[int, List[Track]]) -> HeteroData:
        """Build heterogeneous graph from simulation state"""
        
        # Create graph data structure
        data = HeteroData()
        
        # Build nodes and features - include ego robot and proximal robots
        all_robots = [ego_robot] + proximal_robots
        agent_nodes, agent_features = self._build_agent_nodes(all_robots)
        track_nodes, track_features = self._build_track_nodes(ego_fused_tracks)
        
        # Build edges with neural symbolic predicates
        edges = self._build_edges(all_robots, ego_fused_tracks, proximal_tracks_by_robot)
        
        # Populate graph data
        data['agent'].x = torch.tensor(agent_features, dtype=torch.float)
        data['track'].x = torch.tensor(track_features, dtype=torch.float)
        
        # Add edges
        for edge_type, edge_indices in edges.items():
            if edge_indices:
                src_nodes, dst_nodes = zip(*edge_indices)
                data[edge_type].edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        
        # Store node mappings for later use
        data.agent_nodes = agent_nodes
        data.track_nodes = track_nodes
        
        return data
    
    def _build_agent_nodes(self, all_robots: List[RobotState]) -> Tuple[Dict, np.ndarray]:
        """Build agent nodes and features"""
        agent_nodes = {}
        agent_features = []
        
        for i, robot in enumerate(all_robots):
            node_id = f"agent_{robot.id}"
            agent_nodes[robot.id] = i
            
            # Agent features: Only trustworthy predicate [trustworthy_predicate(1)]
            trustworthy_pred = self._compute_trustworthy_predicate(robot)
            
            # Store ground truth and current trust for loss computation (not part of GNN features)
            robot._ground_truth_trustworthy = 1.0 - float(robot.is_adversarial)  # Store for loss
            robot._current_trust_alpha = robot.trust_alpha  # Store for loss  
            robot._current_trust_beta = robot.trust_beta   # Store for loss
            
            features = np.array([trustworthy_pred])  # Only trustworthy predicate
            
            agent_features.append(features)
        
        return agent_nodes, np.array(agent_features)
    
    def _build_track_nodes(self, ego_fused_tracks: List[Track]) -> Tuple[Dict, np.ndarray]:
        """Build track nodes and features"""
        track_nodes = {}
        track_features = []
        
        # Add ego fused tracks
        for track_idx, track in enumerate(ego_fused_tracks):
            node_id = f"track_{track.id}"
            track_nodes[track.id] = track_idx
            
            # Track features: only trustworthy predicate
            trustworthy_pred = self._compute_track_trustworthy_predicate(track)
            
            features = np.array([trustworthy_pred])
            
            track_features.append(features)
            track_idx += 1
        
        # Only use ego fused tracks for graph construction
        
        return track_nodes, np.array(track_features) if track_features else np.array([]).reshape(0, 1)
    
    def _build_edges(self, all_robots: List[RobotState], ego_fused_tracks: List[Track],
                    proximal_tracks_by_robot: Dict[int, List[Track]]) -> Dict[str, List[Tuple[int, int]]]:
        """Build edges representing neural symbolic predicates"""
        edges = {
            ('agent', 'observes', 'track'): [],
            ('track', 'observed_by', 'agent'): [],
            ('agent', 'in_fov', 'track'): [],
            ('track', 'in_fov_by', 'agent'): []
        }
        
        # InFoV predicate - agent-to-track visibility
        for robot_idx, robot in enumerate(all_robots):
            for track_idx, track in enumerate(ego_fused_tracks):
                if self._is_track_in_fov_predicate(robot, track.position):
                    edges[('agent', 'in_fov', 'track')].append((robot_idx, track_idx))
                    edges[('track', 'in_fov_by', 'agent')].append((track_idx, robot_idx))
        
        # Observed predicate - agent-to-track observations
        for robot_idx, robot in enumerate(all_robots):
            robot_tracks = proximal_tracks_by_robot.get(robot.id, [])
            for track in robot_tracks:
                track_idx = self._get_track_index(track.id, ego_fused_tracks)
                if track_idx is not None:
                    edges[('agent', 'observes', 'track')].append((robot_idx, track_idx))
                    edges[('track', 'observed_by', 'agent')].append((track_idx, robot_idx))
        
        return edges
    
    def _compute_trustworthy_predicate(self, robot: RobotState, threshold: float = 0.5) -> float:
        """Compute trustworthy predicate for agent using threshold comparison"""
        trust_value = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
        return 1.0 if trust_value > threshold else 0.0
    
    def _compute_track_trustworthy_predicate(self, track: Track, threshold: float = 0.5) -> float:
        """Compute trustworthy predicate for track using threshold comparison"""
        track_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)
        combined_trust = track_trust * track.confidence
        return 1.0 if combined_trust > threshold else 0.0
    
    def _is_track_in_fov_predicate(self, observer: RobotState, track_pos: np.ndarray) -> bool:
        """InFoV binary predicate - whether track is in agent's field of view"""
        rel_pos = track_pos - observer.position
        distance = np.linalg.norm(rel_pos[:2])
        
        if distance > observer.fov_range:
            return False
        
        target_angle = np.arctan2(rel_pos[1], rel_pos[0])
        angle_diff = abs(target_angle - observer.orientation)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        
        return angle_diff <= observer.fov_angle / 2
    
    
    def _get_track_index(self, track_id: str, ego_fused_tracks: List[Track]) -> Optional[int]:
        """Get track index in the node list"""
        # Check ego fused tracks
        for i, track in enumerate(ego_fused_tracks):
            if track.id == track_id:
                return i
        return None


class TrainingDataCollector:
    """Collects and structures training data from simulations"""
    
    def __init__(self):
        self.training_examples = []
        self.graph_builder = GraphBuilder()
    
    def collect_from_simulation_step(self, robots: List[RobotState], 
                                   tracks_by_robot: Dict[int, List[Track]],
                                   robot_object_tracks: Dict[int, Dict[str, Track]],
                                   trust_updates: Dict[int, Dict],
                                   ego_robot: RobotState,
                                   ego_fused_tracks: List[Track]) -> None:
        """Collect training data from a single simulation step"""
        
        # Build graph for this step
        proximal_robots = [r for r in robots if r.id != ego_robot.id]
        proximal_tracks_by_robot = {r_id: tracks for r_id, tracks in tracks_by_robot.items() 
                                  if r_id != ego_robot.id}
        graph_data = self.graph_builder.build_graph(
            ego_robot, ego_fused_tracks, proximal_robots, proximal_tracks_by_robot
        )
        
        # Generate labels from trust updates
        labels = self._generate_labels(robots, ego_fused_tracks, trust_updates)
        
        # Create training example
        example = TrainingExample(
            graph_data=graph_data,
            labels=labels,
            metadata={
                'ego_robot_id': ego_robot.id,
                'num_robots': len(robots),
                'num_tracks': len(ego_fused_tracks),
                'adversarial_robots': [r.id for r in robots if r.is_adversarial]
            }
        )
        
        self.training_examples.append(example)
    
    def _generate_labels(self, robots: List[RobotState], tracks: List[Track], 
                        trust_updates: Dict[int, Dict]) -> Dict[str, Dict[str, float]]:
        """Generate training labels from paper algorithm PSMs"""
        labels = {}
        
        # Agent labels from trust updates
        for robot in robots:
            if robot.id in trust_updates:
                # Use the difference in trust as the label
                trust_before = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
                
                # Simulate PSM value and confidence based on robot behavior
                if robot.is_adversarial:
                    # Adversarial robots should have negative PSM values
                    psm_value = np.random.uniform(-1.0, 0.2)
                    psm_confidence = np.random.uniform(0.3, 0.8)
                else:
                    # Legitimate robots should have positive PSM values
                    psm_value = np.random.uniform(0.2, 1.0)
                    psm_confidence = np.random.uniform(0.6, 0.9)
                
                labels[f"agent_{robot.id}"] = {
                    'value': psm_value,
                    'confidence': psm_confidence,
                    'current_alpha': robot.trust_alpha,
                    'current_beta': robot.trust_beta,
                    'is_trustworthy': 0.0 if robot.is_adversarial else 1.0
                }
        
        # Track labels
        for track in tracks:
            # Generate track PSM based on track characteristics
            track_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)
            psm_value = np.tanh(track_trust * 2 - 1)  # Scale to [-1, 1]
            psm_confidence = track.confidence
            
            labels[f"track_{track.id}"] = {
                'value': psm_value,
                'confidence': psm_confidence,
                'current_alpha': track.trust_alpha,
                'current_beta': track.trust_beta,
                'is_trustworthy': 1.0 if track_trust > 0.5 else 0.0
            }
        
        return labels
    
    def save_training_data(self, filename: str) -> None:
        """Save collected training data"""
        with open(filename, 'wb') as f:
            pickle.dump(self.training_examples, f)
        print(f"Saved {len(self.training_examples)} training examples to {filename}")
    
    def load_training_data(self, filename: str) -> None:
        """Load training data from file"""
        with open(filename, 'rb') as f:
            self.training_examples = pickle.load(f)
        print(f"Loaded {len(self.training_examples)} training examples from {filename}")


class GNNTrainer:
    """Trainer for the GNN model with both supervised and RL capabilities"""
    
    def __init__(self, model: TrustGNN, learning_rate: float = 0.001, device: torch.device = torch.device('cpu')):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')  # Use KL divergence instead of MSE
        
    def train_supervised(self, training_examples: List[TrainingExample], 
                        epochs: int = 100, batch_size: int = 32) -> List[float]:
        """Train using supervised learning"""
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Simple batch training (could be improved with DataLoader)
            for i in range(0, len(training_examples), batch_size):
                batch = training_examples[i:i+batch_size]
                batch_loss = self._train_batch_supervised(batch)
                epoch_loss += batch_loss
                num_batches += 1
            
            avg_loss = epoch_loss / max(1, num_batches)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def _train_batch_supervised(self, batch: List[TrainingExample]) -> float:
        """Train a single batch in supervised mode"""
        total_loss = 0.0
        
        for example in batch:
            self.optimizer.zero_grad()
            
            # Move data to device
            x_dict = {k: v.to(self.device) for k, v in example.graph_data.x_dict.items()}
            
            # Handle edge_index_dict - some graphs may have no edges
            try:
                edge_index_dict = {k: v.to(self.device) for k, v in example.graph_data.edge_index_dict.items()}
            except (KeyError, AttributeError):
                # No edges in this graph - create empty edge_index_dict
                edge_index_dict = {}
            
            # Forward pass
            predictions = self.model(x_dict, edge_index_dict)
            
            # Compute loss
            loss = self._compute_supervised_loss(predictions, example.labels, example.graph_data)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(batch)
    
    def _compute_supervised_loss(self, predictions: Dict, labels: Dict[str, Dict[str, float]], 
                                graph_data: HeteroData) -> torch.Tensor:
        """
        Compute supervised learning loss using KL divergence between Beta distributions
        
        Loss = KL(updated_trust || ground_truth_trust) + λ * KL(current_trust || updated_trust)
        
        Where:
        - First term: Minimize distance from updated trust to ground truth
        - Second term: Regularization to prevent drastic updates
        """
        total_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        regularization_weight = 0.1  # λ parameter for regularization
        
        # Agent losses - only compute if agent predictions exist
        if 'agent' in predictions:
            agent_features = graph_data['agent'].x  # [num_agents, feature_dim]
            
            for node_id, label_dict in labels.items():
                if node_id.startswith('agent_'):
                    agent_id = int(node_id.split('_')[1])
                    if agent_id in graph_data.agent_nodes:
                        idx = graph_data.agent_nodes[agent_id]
                        
                        # With simplified 1D features, get trust parameters from stored metadata
                        # The trust parameters and ground truth should be available in labels
                        current_alpha = label_dict.get('current_alpha', 1.0)
                        current_beta = label_dict.get('current_beta', 1.0)
                        
                        # Ground truth trustworthiness from labels
                        is_trustworthy = label_dict.get('is_trustworthy', 1.0)
                        ground_truth_alpha = 10.0 if is_trustworthy else 1.0  # Strong belief
                        ground_truth_beta = 1.0 if is_trustworthy else 10.0   # Strong belief
                        
                        # Get GNN predictions for PSM updates
                        psm_value = predictions['agent']['value'][idx].squeeze()
                        psm_confidence = predictions['agent']['confidence'][idx].squeeze()
                        
                        # Compute updated trust parameters from PSM
                        delta_alpha = psm_confidence * psm_value
                        delta_beta = psm_confidence * (1 - psm_value)
                        
                        updated_alpha = current_alpha + delta_alpha
                        updated_beta = current_beta + delta_beta
                        
                        # Ensure minimum values
                        updated_alpha = torch.clamp(updated_alpha, min=0.1)
                        updated_beta = torch.clamp(updated_beta, min=0.1)
                        
                        # Convert to tensors for KL divergence computation
                        current_alpha_t = torch.tensor(current_alpha, device=self.device, requires_grad=False)
                        current_beta_t = torch.tensor(current_beta, device=self.device, requires_grad=False)
                        
                        ground_truth_alpha_t = torch.tensor(ground_truth_alpha, device=self.device, requires_grad=False)
                        ground_truth_beta_t = torch.tensor(ground_truth_beta, device=self.device, requires_grad=False)
                        
                        # Compute KL divergence between Beta distributions
                        # KL(Beta(α₁,β₁) || Beta(α₂,β₂)) = ln(B(α₂,β₂)/B(α₁,β₁)) + (α₁-α₂)ψ(α₁) + (β₁-β₂)ψ(β₁) + (α₂-α₁+β₂-β₁)ψ(α₁+β₁)
                        
                        # Loss 1: KL(updated_trust || ground_truth_trust)
                        kl_ground_truth = self._beta_kl_divergence(
                            updated_alpha, updated_beta,
                            ground_truth_alpha_t, ground_truth_beta_t
                        )
                        
                        # Loss 2: KL(current_trust || updated_trust) for regularization  
                        kl_regularization = self._beta_kl_divergence(
                            current_alpha_t, current_beta_t,
                            updated_alpha, updated_beta
                        )
                        
                        # Combined loss
                        agent_loss = kl_ground_truth + regularization_weight * kl_regularization
                        total_loss = total_loss + agent_loss
        
        # Track losses - similar structure
        if 'track' in predictions:
            track_features = graph_data['track'].x  # [num_tracks, feature_dim]
            
            for node_id, label_dict in labels.items():
                if node_id.startswith('track_'):
                    track_id = node_id.split('_', 1)[1]
                    if track_id in graph_data.track_nodes:
                        idx = graph_data.track_nodes[track_id]
                        
                        # With simplified 1D features, get trust parameters from labels
                        current_alpha = label_dict.get('current_alpha', 1.0)
                        current_beta = label_dict.get('current_beta', 1.0)
                        
                        # Ground truth trustworthiness from labels
                        is_trustworthy = label_dict.get('is_trustworthy', 1.0)
                        ground_truth_alpha = 5.0 if is_trustworthy > 0.5 else 1.0
                        ground_truth_beta = 1.0 if is_trustworthy > 0.5 else 5.0
                        
                        # Get GNN predictions
                        psm_value = predictions['track']['value'][idx].squeeze()
                        psm_confidence = predictions['track']['confidence'][idx].squeeze()
                        
                        # Compute updated parameters
                        delta_alpha = psm_confidence * psm_value
                        delta_beta = psm_confidence * (1 - psm_value)
                        
                        updated_alpha = current_alpha + delta_alpha
                        updated_beta = current_beta + delta_beta
                        
                        updated_alpha = torch.clamp(updated_alpha, min=0.1)
                        updated_beta = torch.clamp(updated_beta, min=0.1)
                        
                        # Convert to tensors
                        current_alpha_t = torch.tensor(current_alpha, device=self.device, requires_grad=False)
                        current_beta_t = torch.tensor(current_beta, device=self.device, requires_grad=False)
                        
                        ground_truth_alpha_t = torch.tensor(ground_truth_alpha, device=self.device, requires_grad=False)
                        ground_truth_beta_t = torch.tensor(ground_truth_beta, device=self.device, requires_grad=False)
                        
                        # Compute KL divergences
                        kl_ground_truth = self._beta_kl_divergence(
                            updated_alpha, updated_beta,
                            ground_truth_alpha_t, ground_truth_beta_t
                        )
                        
                        kl_regularization = self._beta_kl_divergence(
                            current_alpha_t, current_beta_t,
                            updated_alpha, updated_beta
                        )
                        
                        # Combined loss
                        track_loss = kl_ground_truth + regularization_weight * kl_regularization
                        total_loss = total_loss + track_loss
        
        return total_loss
    
    def _beta_kl_divergence(self, alpha1: torch.Tensor, beta1: torch.Tensor, 
                           alpha2: torch.Tensor, beta2: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between two Beta distributions
        KL(Beta(α₁,β₁) || Beta(α₂,β₂))
        """
        # Use the digamma function and log-beta function
        from torch.special import digamma, gammaln
        
        # Log-beta function: ln(B(α,β)) = ln(Γ(α)) + ln(Γ(β)) - ln(Γ(α+β))
        def log_beta(alpha, beta):
            return gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
        
        # KL divergence formula
        kl = (log_beta(alpha2, beta2) - log_beta(alpha1, beta1) +
              (alpha1 - alpha2) * digamma(alpha1) +
              (beta1 - beta2) * digamma(beta1) +
              (alpha2 - alpha1 + beta2 - beta1) * digamma(alpha1 + beta1))
        
        # Ensure non-negative (numerical stability)
        return torch.clamp(kl, min=0.0)


class NeuralSymbolicTrustAlgorithm(TrustAlgorithm):
    """Neural Symbolic Trust Algorithm using Graph Neural Networks"""
    
    def __init__(self, model_path: Optional[str] = None, learning_mode: bool = True):
        """
        Initialize the neural symbolic trust algorithm
        
        Args:
            model_path: Path to pre-trained model (if available)
            learning_mode: Whether to collect training data during simulation
        """
        self.learning_mode = learning_mode
        self.graph_builder = GraphBuilder()
        self.training_data_collector = TrainingDataCollector() if learning_mode else None
        
        # Initialize GNN model
        self.model = TrustGNN(agent_features=1, track_features=1)  # Based on simplified trustworthy predicate features
        self.trainer = GNNTrainer(self.model, device=next(self.model.parameters()).device) if learning_mode else None
        
        # Load pre-trained model if available
        if model_path and torch.cuda.is_available():
            try:
                self.model.load_state_dict(torch.load(model_path))
                print(f"Loaded pre-trained model from {model_path}")
            except FileNotFoundError:
                print(f"Model file {model_path} not found, using randomly initialized model")
        
        self.model.eval() if not learning_mode else self.model.train()
        
        # Robot object trust tracking  
        self.robot_object_trust: Dict[int, Dict[str, Tuple[float, float]]] = {}
        
    def initialize(self, robots: List[RobotState]) -> None:
        """Initialize the neural symbolic algorithm with robot states"""
        # Initialize trust tracking
        for robot in robots:
            self.robot_object_trust[robot.id] = {}
        
        print("Neural Symbolic Trust Algorithm initialized with GNN")
    
    def update_trust(self, robots: List[RobotState], tracks_by_robot: Dict[int, List[Track]], 
                    robot_object_tracks: Dict[int, Dict[str, Track]], time: float) -> Dict[int, Dict]:
        """Update trust using GNN-based neural symbolic reasoning"""
        
        trust_updates = {}
        
        # Main loop: iterate over each robot as ego robot
        for ego_robot in robots:
            # Get ego robot's fused tracks from robot_object_tracks
            ego_fused_tracks = list(robot_object_tracks.get(ego_robot.id, {}).values())
            
            if not ego_fused_tracks:
                continue
            
            # Get proximal robots and their tracks
            proximal_robots = [r for r in robots if r.id != ego_robot.id]
            proximal_tracks_by_robot = {r_id: tracks for r_id, tracks in tracks_by_robot.items() 
                                      if r_id != ego_robot.id}
            
            # Build graph with neural symbolic predicates
            graph_data = self.graph_builder.build_graph(
                ego_robot, ego_fused_tracks, proximal_robots, proximal_tracks_by_robot
            )
            
            # Apply GNN to get PSM predictions
            if graph_data['agent'].x.shape[0] > 0:  # Check if we have nodes
                with torch.no_grad() if not self.learning_mode else torch.enable_grad():
                    # Move data to device if needed
                    x_dict = {k: v.to(next(self.model.parameters()).device) for k, v in graph_data.x_dict.items()}
                    edge_index_dict = {}
                    try:
                        # Safely get edge_index_dict if it exists
                        if graph_data.edge_index_dict:
                            edge_index_dict = {k: v.to(next(self.model.parameters()).device) 
                                             for k, v in graph_data.edge_index_dict.items()}
                    except (AttributeError, KeyError):
                        # No edges exist, use empty dict
                        edge_index_dict = {}
                    
                    predictions = self.model(x_dict, edge_index_dict)
                
                # Convert GNN predictions to PSMs and update trust
                # Build the robot list that matches the graph node order
                all_graph_robots = [ego_robot] + proximal_robots
                self._apply_gnn_predictions(predictions, graph_data, all_graph_robots, ego_fused_tracks)
        
        # Generate trust updates for all robots
        for robot in robots:
            trust_updates[robot.id] = {
                'alpha': robot.trust_alpha,
                'beta': robot.trust_beta,
                'mean_trust': robot.trust_alpha / (robot.trust_alpha + robot.trust_beta),
                'algorithm': 'neural_symbolic_gnn'
            }
        
        # Collect training data if in learning mode
        if self.learning_mode and self.training_data_collector:
            # Collect training data for all robots that have fused tracks
            for robot in robots:
                ego_fused_tracks = list(robot_object_tracks.get(robot.id, {}).values())
                if ego_fused_tracks:  # Only collect if we have tracks
                    self.training_data_collector.collect_from_simulation_step(
                        robots, tracks_by_robot, robot_object_tracks, trust_updates, 
                        robot, ego_fused_tracks
                    )
        
        return trust_updates
    
    def _apply_gnn_predictions(self, predictions: Dict, graph_data: HeteroData, 
                              robots: List[RobotState], tracks: List[Track]) -> None:
        """Apply GNN predictions as PSMs to update trust distributions"""
        
        # Update agent trust based on GNN predictions
        if 'agent' in predictions:
            for robot_idx, robot in enumerate(robots):
                if robot_idx < predictions['agent']['value'].shape[0]:
                    psm_value = predictions['agent']['value'][robot_idx].item()
                    psm_confidence = predictions['agent']['confidence'][robot_idx].item()
                    
                    # Apply PSM using Beta-Bernoulli update (similar to paper algorithm)
                    delta_alpha = psm_confidence * psm_value if psm_value > 0 else 0
                    delta_beta = psm_confidence * (1 - psm_value) if psm_value < 0 else 0
                    
                    robot.trust_alpha += delta_alpha
                    robot.trust_beta += delta_beta
                    
                    # Ensure minimum values
                    robot.trust_alpha = max(0.1, robot.trust_alpha)
                    robot.trust_beta = max(0.1, robot.trust_beta)
        
        # Update track trust based on GNN predictions
        if 'track' in predictions:
            for track_idx, track in enumerate(tracks):
                if track_idx < predictions['track']['value'].shape[0]:
                    psm_value = predictions['track']['value'][track_idx].item()
                    psm_confidence = predictions['track']['confidence'][track_idx].item()
                    
                    # Apply PSM to track trust
                    delta_alpha = psm_confidence * psm_value if psm_value > 0 else 0
                    delta_beta = psm_confidence * (1 - psm_value) if psm_value < 0 else 0
                    
                    track.trust_alpha += delta_alpha
                    track.trust_beta += delta_beta
                    
                    # Ensure minimum values
                    track.trust_alpha = max(0.1, track.trust_alpha)
                    track.trust_beta = max(0.1, track.trust_beta)
    
    def train_model(self, training_data_file: str, epochs: int = 100) -> List[float]:
        """Train the GNN model from collected data"""
        if not self.training_data_collector:
            print("Training data collector not available. Enable learning_mode=True")
            return []
        
        # Load training data
        self.training_data_collector.load_training_data(training_data_file)
        
        # Train the model
        losses = self.trainer.train_supervised(
            self.training_data_collector.training_examples,
            epochs=epochs
        )
        
        print(f"Training completed. Final loss: {losses[-1]:.4f}")
        return losses
    
    def save_model(self, model_path: str) -> None:
        """Save the trained model"""
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def save_training_data(self, filename: str) -> None:
        """Save collected training data"""
        if self.training_data_collector:
            self.training_data_collector.save_training_data(filename)
    
    def get_expected_trust(self, alpha: float, beta: float) -> float:
        """Calculate expected value E[trust] = alpha / (alpha + beta)"""
        return alpha / (alpha + beta)
    
    def get_trust_variance(self, alpha: float, beta: float) -> float:
        """Calculate variance of trust distribution"""
        denominator = (alpha + beta) ** 2 * (alpha + beta + 1)
        return (alpha * beta) / denominator
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model state"""
        return {
            'model_type': 'GNN_Neural_Symbolic',
            'learning_mode': self.learning_mode,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'training_examples': len(self.training_data_collector.training_examples) if self.training_data_collector else 0
        }