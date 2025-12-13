#!/usr/bin/env python3
"""
Supervised Learning Trust GNN

This module implements a comprehensive GNN-based trust prediction model with:
- Neural symbolic predicate calculations
- Continuous trust value updates using paper trust algorithm
- Agent-to-agent comparison edge creation
- Direct prediction from Robot/Track objects
- Binary trust classification for supervised learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# Import required modules
from robot_track_classes import Robot


@dataclass
class NodeScores:
    """Container for evidence scores from GNN"""
    agent_scores: Dict[int, float]  # robot_id -> evidence score
    track_scores: Dict[str, float]  # track_id -> evidence score


class TripletEncoder(nn.Module):
    """
    Encodes local edge structure around each node as symbolic triplets using MLP + Attention Pooling.

    For each node, we encode all outgoing edges as triplets:
    τ = (source_type, edge_relation, target_type)

    where:
    - source_type: 0=agent, 1=track (1-bit)
    - edge_relation: one-hot over 6 edge types (6-bit)
    - target_type: 0=agent, 1=track (1-bit)
    Total: 8 dimensions per triplet

    Key Insight: Edges represent INDEPENDENT behavioral observations (co_detection, contradicts, etc.)
    The existence of edge A has no relationship with edge B - they're separate facts.
    Therefore, we don't need Transformer self-attention (which models edge-to-edge dependencies).
    Instead, we use MLP to embed each edge independently, then attention pooling to learn
    which edge TYPES are important (e.g., "contradicts" is more important than "in_fov_only").
    """

    def __init__(self, hidden_dim=128, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Edge embedding MLP: Process each edge independently
        # No need for Transformer because edges are independent observations
        self.edge_embedding = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Attention pooling: Learn which edge types are important
        # This is the KEY component - learns that "contradicts" matters more than "in_fov_only"
        # Permutation-invariant: attention weights sum to 1 regardless of edge order
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, triplets, mask=None):
        """
        Encode symbolic triplets into node embeddings.

        Args:
            triplets: [num_nodes, max_edges, 8] - symbolic triplet representations
            mask: [num_nodes, max_edges] - True for padding positions

        Returns:
            [num_nodes, hidden_dim] - node embeddings
        """
        # Embed each edge independently (no edge-to-edge interaction needed)
        x = self.edge_embedding(triplets)  # [num_nodes, max_edges, hidden_dim]

        # Attention pooling: Learn which edges are important
        # Compute attention scores for each edge
        attn_scores = self.edge_attention(x)  # [num_nodes, max_edges, 1]

        if mask is not None:
            # Mask padded positions with -inf so they get 0 weight after softmax
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), float('-inf'))

        # Softmax over edges (dim=1) to get normalized attention weights
        # This is permutation-invariant: weights sum to 1 regardless of edge order
        attn_weights = torch.softmax(attn_scores, dim=1)  # [num_nodes, max_edges, 1]

        # Weighted sum: aggregate edges weighted by their importance
        # This removes the edge count bias - focuses on patterns, not quantity
        pooled = (x * attn_weights).sum(dim=1)  # [num_nodes, hidden_dim]

        # Final projection
        output = self.output_proj(pooled)  # [num_nodes, hidden_dim]

        return output


class SupervisedTrustGNN(nn.Module):
    """
    Supervised GNN model for binary trust classification with Transformer-based Triplet Encoding

    SYMBOLIC STRUCTURE ENCODING:
    - Triplet encoder: Encodes local edge patterns as symbolic triplets τ = (src_type, relation, dst_type)
    - Transformer: Processes variable-length sequences of triplets per node
    - Rich initial features: Nodes start with meaningful embeddings from local structure
    - Edge types: 6 heterogeneous relations define the graph topology
    """

    def __init__(self, hidden_dim: int = 128):
        super(SupervisedTrustGNN, self).__init__()

        self.hidden_dim = hidden_dim

        # Define edge types for heterogeneous graph
        self.edge_types = [
            ('agent', 'in_fov_and_observed', 'track'),
            ('track', 'observed_and_in_fov_by', 'agent'),
            ('agent', 'in_fov_only', 'track'),
            ('track', 'in_fov_only_by', 'agent'),
            ('agent', 'co_detection', 'agent'),  # Co-detection edges (robots detecting same object)
            ('agent', 'contradicts', 'agent'),   # Contradiction edges (robot A detects track that B should see but doesn't)
        ]

        # Map edge types to indices for one-hot encoding
        self.edge_type_to_idx = {edge_type: i for i, edge_type in enumerate(self.edge_types)}

        # Triplet encoders: Convert local edge structure to node embeddings
        # Separate encoders for agents and tracks (they have different edge patterns)
        self.agent_triplet_encoder = TripletEncoder(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=1,
            dropout=0.1
        )

        self.track_triplet_encoder = TripletEncoder(
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=1,
            dropout=0.1
        )

        # Two layers of heterogeneous convolution (each with independent parameters)
        # FIXED: Create separate conv_dict for each layer to avoid parameter sharing
        self.conv1 = HeteroConv(self._create_conv_dict(), aggr='mean')
        self.conv2 = HeteroConv(self._create_conv_dict(), aggr='mean')

        # Batch normalization layers
        self.norm1 = nn.ModuleDict({
            'agent': nn.BatchNorm1d(hidden_dim),
            'track': nn.BatchNorm1d(hidden_dim)
        })
        self.norm2 = nn.ModuleDict({
            'agent': nn.BatchNorm1d(hidden_dim),
            'track': nn.BatchNorm1d(hidden_dim)
        })

        # Binary classification heads for trust prediction
        self.agent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Standardized to 0.1
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),  # Binary classification
            nn.Sigmoid()  # Output probability of being trustworthy
        )

        self.track_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Standardized to 0.1
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),  # Binary classification
            nn.Sigmoid()  # Output probability of being trustworthy
        )

    def _create_conv_dict(self):
        """
        Create a new GAT convolution dictionary with independent parameters

        This helper method is called for each conv layer to ensure they have
        separate learnable parameters (no parameter sharing).

        Returns:
            Dictionary mapping edge types to GATConv modules
        """
        conv_dict = {}
        for src_type, relation, dst_type in self.edge_types:
            conv_dict[(src_type, relation, dst_type)] = GATConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                heads=4,
                concat=False,
                add_self_loops=False,
                dropout=0.1  # Standardized to 0.1
            )
        return conv_dict

    def forward(self, num_agents, num_tracks, edge_index_dict, device='cpu',
                agent_triplets=None, agent_triplet_mask=None,
                track_triplets=None, track_triplet_mask=None):
        """
        Forward pass for trust classification with triplet encoding

        Args:
            num_agents: Number of agent nodes
            num_tracks: Number of track nodes
            edge_index_dict: Edge indices dictionary
            device: Device for tensors
            agent_triplets: Pre-computed agent triplets [num_agents, max_edges, 8] (optional)
            agent_triplet_mask: Pre-computed agent triplet mask [num_agents, max_edges] (optional)
            track_triplets: Pre-computed track triplets [num_tracks, max_edges, 8] (optional)
            track_triplet_mask: Pre-computed track triplet mask [num_tracks, max_edges] (optional)

        Returns:
            Dict containing trust predictions for agents and tracks
        """
        # Check if we have any tracks
        has_tracks = num_tracks > 0

        # ============================================================
        # STEP 1: Triplet Encoding - Extract and encode local structure
        # ============================================================

        # Use pre-computed triplets if available, otherwise extract them
        if agent_triplets is not None and agent_triplet_mask is not None:
            # Use pre-computed triplets (faster - skip extraction)
            agent_triplets = agent_triplets.to(device)
            agent_triplet_mask = agent_triplet_mask.to(device)
        else:
            # Extract triplets dynamically (slower - for inference)
            agent_triplets, agent_triplet_mask = self._extract_triplets('agent', num_agents, edge_index_dict, device)

        # Encode with transformer
        agent_embeddings = self.agent_triplet_encoder(agent_triplets, agent_triplet_mask)

        x_dict = {'agent': agent_embeddings}

        # Extract and encode triplets for track nodes
        if has_tracks:
            if track_triplets is not None and track_triplet_mask is not None:
                # Use pre-computed triplets (faster - skip extraction)
                track_triplets = track_triplets.to(device)
                track_triplet_mask = track_triplet_mask.to(device)
            else:
                # Extract triplets dynamically (slower - for inference)
                track_triplets, track_triplet_mask = self._extract_triplets('track', num_tracks, edge_index_dict, device)

            track_embeddings = self.track_triplet_encoder(track_triplets, track_triplet_mask)
            x_dict['track'] = track_embeddings
        else:
            # Create empty track tensor with correct dimensions
            x_dict['track'] = torch.empty(0, self.hidden_dim, device=device)

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
        x_dict_final = {
            key: x_dict_2[key] + x_dict_1[key] if x_dict_2[key].shape[0] > 0 and x_dict_1[key].shape[0] > 0
            else x_dict_2[key]
            for key in x_dict_2
        }

        # Apply classification heads
        predictions = {}

        # Agent trust predictions
        if x_dict_final['agent'].shape[0] > 0:
            agent_trust_probs = self.agent_classifier(x_dict_final['agent'])
            predictions['agent'] = agent_trust_probs

        # Track trust predictions (only if we have tracks)
        if has_tracks and x_dict_final['track'].shape[0] > 0:
            track_trust_probs = self.track_classifier(x_dict_final['track'])
            predictions['track'] = track_trust_probs

        return predictions

    def _extract_triplets(self, node_type: str, num_nodes: int, edge_index_dict: Dict, device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract symbolic triplets for all nodes of a given type.

        For each node, we find all outgoing edges and encode them as triplets:
        τ = (source_type, edge_relation, target_type)

        Args:
            node_type: 'agent' or 'track'
            num_nodes: Number of nodes of this type
            edge_index_dict: Dictionary of edge indices
            device: Device for tensors

        Returns:
            Tuple of (triplets, mask)
            - triplets: [num_nodes, max_edges, 8] - symbolic triplet representations
            - mask: [num_nodes, max_edges] - True for padding positions
        """
        # Collect all edges where this node type is the source
        node_edges = []  # List of (node_idx, edge_list) pairs

        for node_idx in range(num_nodes):
            edge_list = []

            # Iterate through all edge types
            for edge_type in self.edge_types:
                src_type, relation, dst_type = edge_type

                # Check if this edge type originates from our node_type
                if src_type == node_type and edge_type in edge_index_dict:
                    edge_index = edge_index_dict[edge_type]

                    if edge_index.numel() > 0:
                        # Find edges where source is node_idx
                        mask = (edge_index[0] == node_idx)
                        if mask.any():
                            # Get target nodes for these edges
                            target_nodes = edge_index[1][mask]

                            # Create triplet for each edge
                            for target_idx in target_nodes:
                                # Encode triplet: (src_type, relation, dst_type)
                                # src_type: 0=agent, 1=track (1 bit)
                                src_bit = 1.0 if src_type == 'track' else 0.0

                                # relation: one-hot over 6 edge types (6 bits)
                                relation_onehot = [0.0] * 6
                                relation_idx = self.edge_type_to_idx[edge_type]
                                relation_onehot[relation_idx] = 1.0

                                # dst_type: 0=agent, 1=track (1 bit)
                                dst_bit = 1.0 if dst_type == 'track' else 0.0

                                # Concatenate: [src_bit, relation_onehot (6), dst_bit] = 8 dims
                                triplet = [src_bit] + relation_onehot + [dst_bit]
                                edge_list.append(triplet)

            node_edges.append(edge_list)

        # Find maximum number of edges across all nodes
        max_edges = max(len(edges) for edges in node_edges) if node_edges else 1
        max_edges = max(max_edges, 1)  # At least 1 to avoid empty tensors

        # Pad all edge lists to max_edges
        triplets_list = []
        mask_list = []

        for edges in node_edges:
            num_edges = len(edges)

            if num_edges > 0:
                # Pad with zeros
                padded_edges = edges + [[0.0] * 8] * (max_edges - num_edges)
                # Mask: False for real edges, True for padding
                mask = [False] * num_edges + [True] * (max_edges - num_edges)
            else:
                # No edges: all padding
                padded_edges = [[0.0] * 8] * max_edges
                mask = [True] * max_edges

            triplets_list.append(padded_edges)
            mask_list.append(mask)

        # Convert to tensors
        triplets = torch.tensor(triplets_list, dtype=torch.float32, device=device)  # [num_nodes, max_edges, 8]
        mask = torch.tensor(mask_list, dtype=torch.bool, device=device)  # [num_nodes, max_edges]

        return triplets, mask


class SupervisedTrustPredictor:
    """
    Wrapper class for supervised trust prediction
    """

    def __init__(self, model_path: str, device: str = 'cpu', proximal_range: float = 50.0):
        """
        Initialize predictor with trained model

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            proximal_range: Proximal range for ego graph building (must match training data)
        """
        self.device = torch.device(device)
        self.model = None
        self.available = False  # Track whether model loaded successfully
        self.ego_graph_builder = EgoGraphBuilder(proximal_range=proximal_range)
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load trained model from checkpoint or initialize fresh weights if unavailable"""
        # Always create the model so we can fall back to fresh weights when needed
        # Structure-only learning: no input features, only learnable type embeddings
        self.model = SupervisedTrustGNN(hidden_dim=128)

        resolved_path = Path(model_path) if model_path else None

        if resolved_path and resolved_path.exists():
            try:
                checkpoint = torch.load(resolved_path, map_location=self.device, weights_only=False)
                state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

                model_state = self.model.state_dict()
                matched_keys = []
                mismatched_keys = []
                unexpected_keys = []

                for key, weight in state_dict.items():
                    if not isinstance(weight, torch.Tensor):
                        continue  # Skip non-tensor entries (metadata, optimizer state, etc.)

                    if key in model_state:
                        target_param = model_state[key]
                        if target_param.shape == weight.shape:
                            model_state[key] = weight.to(device=target_param.device, dtype=target_param.dtype)
                            matched_keys.append(key)
                        else:
                            mismatched_keys.append((key, tuple(weight.shape), tuple(model_state[key].shape)))
                    else:
                        unexpected_keys.append(key)

                if matched_keys:
                    self.model.load_state_dict(model_state)
                    self.available = True  # Model loaded successfully
                    loaded_msg = f"✅ Loaded supervised trust model from {resolved_path} (matched {len(matched_keys)} tensors)"
                    print(loaded_msg)
                else:
                    self.available = False  # No compatible tensors
                    print(f"ℹ️ Checkpoint '{resolved_path}' has no compatible tensors. Using randomly initialized weights.")

                if mismatched_keys:
                    preview = ", ".join([f"{k}:ckpt{src}->model{dst}" for k, src, dst in mismatched_keys[:3]])
                    if len(mismatched_keys) > 3:
                        preview += ", ..."
                    print(f"ℹ️ Skipped {len(mismatched_keys)} tensors with incompatible shapes ({preview})")
                if unexpected_keys:
                    print(f"ℹ️ Ignored {len(unexpected_keys)} unexpected tensors from checkpoint.")

            except Exception as e:
                print(f"⚠️ Failed to load supervised trust model from {resolved_path}: {e}")
                print("ℹ️ Proceeding with randomly initialized GNN weights.")
        else:
            if resolved_path:
                print(f"ℹ️ GNN checkpoint '{resolved_path}' not found. Using randomly initialized weights.")
            else:
                print("ℹ️ No GNN checkpoint provided. Using randomly initialized weights.")

        self.model.to(self.device)
        self.model.eval()

    def _check_ego_cross_validation(self, edge_index_dict: Dict) -> bool:
        """
        Check if ego robot (index 0) has cross-validation with other robots

        Args:
            edge_index_dict: Dictionary of edge indices from ego graph

        Returns:
            True if ego robot has co_detection or contradicts edges with other robots
        """
        # Check for co_detection edges
        if ('agent', 'co_detection', 'agent') in edge_index_dict:
            co_detection_edges = edge_index_dict[('agent', 'co_detection', 'agent')]
            if co_detection_edges.numel() > 0:
                ego_in_co_detection = (co_detection_edges[0] == 0).any() or (co_detection_edges[1] == 0).any()
                if ego_in_co_detection:
                    return True

        # Check for contradicts edges
        if ('agent', 'contradicts', 'agent') in edge_index_dict:
            contradicts_edges = edge_index_dict[('agent', 'contradicts', 'agent')]
            if contradicts_edges.numel() > 0:
                ego_in_contradicts = (contradicts_edges[0] == 0).any() or (contradicts_edges[1] == 0).any()
                if ego_in_contradicts:
                    return True

        return False

    def _identify_meaningful_tracks(self, ego_robot: 'Robot', graph_data, num_tracks: int) -> List[int]:
        """
        Identify meaningful tracks for inference

        A track is meaningful if:
        1. It's currently detected by ego robot (in get_all_current_tracks())
        2. It has edges to >= 2 robots (cross-validation constraint)

        Args:
            ego_robot: The ego robot
            graph_data: Ego graph HeteroData object
            num_tracks: Number of tracks in the ego graph

        Returns:
            List of meaningful track indices
        """
        meaningful_indices = []

        # Get tracks currently detected by ego robot
        ego_current_tracks = ego_robot.get_all_current_tracks()
        # IMPORTANT: Match by object_id, not track_id, because track fusion changes track_id
        ego_object_ids = set(track.object_id for track in ego_current_tracks)

        # Get fused and individual tracks from ego graph
        if hasattr(graph_data, '_fused_tracks') and hasattr(graph_data, '_individual_tracks'):
            all_tracks = graph_data._fused_tracks + graph_data._individual_tracks
        else:
            return []

        edge_index_dict = graph_data.edge_index_dict

        # For each track, check if it's meaningful
        for track_idx, track in enumerate(all_tracks[:num_tracks]):
            # Check 1: Is this track currently detected by ego robot?
            # Match by object_id (not track_id) since fusion changes track_id
            if track.object_id not in ego_object_ids:
                continue

            # Check 2: Does this track have edges to >= 2 robots?
            num_robots_with_edges = self._count_robots_with_edges_to_track(edge_index_dict, track_idx)
            if num_robots_with_edges >= 2:
                meaningful_indices.append(track_idx)

        return meaningful_indices

    def _count_robots_with_edges_to_track(self, edge_index_dict: Dict, track_idx: int) -> int:
        """
        Count how many robots have edges to a specific track

        Args:
            edge_index_dict: Dictionary of edge indices from ego graph
            track_idx: Index of the track to check

        Returns:
            Number of robots with edges to this track
        """
        robots_with_edges = set()

        # Check agent->track edges
        agent_to_track_edge_types = [
            ('agent', 'in_fov_and_observed', 'track'),
            ('agent', 'in_fov_only', 'track')
        ]

        for edge_type in agent_to_track_edge_types:
            if edge_type in edge_index_dict:
                edges = edge_index_dict[edge_type]
                if edges.numel() > 0:
                    agents_to_this_track = edges[0][edges[1] == track_idx]
                    robots_with_edges.update(agents_to_this_track.tolist())

        return len(robots_with_edges)

    def predict_from_robots_tracks(self,
                                 ego_robot: 'Robot',
                                 robots: List['Robot'],
                                 threshold: float = 0.5) -> Dict:
        """
        Predict trust labels directly from ego robot and all robots with cross-validation constraints

        Args:
            ego_robot: The robot for which to build the ego graph
            robots: List of all Robot objects
            threshold: Classification threshold

        Returns:
            Dict containing trust predictions and probabilities, or None if no cross-validation

        Note: Only returns predictions if:
        - Ego robot has cross-validation (co_detection or contradicts edges)
        - There are meaningful tracks (ego-detected AND have edges to >=2 robots)
        """
        if self.model is None:
            raise ValueError("Model not loaded successfully")

        # Create ego graph using proper build_ego_graph method with fusion map
        graph_data = self.ego_graph_builder.build_ego_graph(ego_robot, robots)

        # Check cross-validation: Does ego robot have co_detection or contradicts edges?
        ego_has_cross_validation = self._check_ego_cross_validation(graph_data.edge_index_dict)

        if not ego_has_cross_validation:
            # Ego robot has no cross-validation, skip update
            return None

        # Extract graph size
        num_agents = graph_data.x_dict['agent'].shape[0] if 'agent' in graph_data.x_dict else 0
        num_tracks = graph_data.x_dict['track'].shape[0] if 'track' in graph_data.x_dict else 0

        # Identify meaningful tracks
        meaningful_track_indices = self._identify_meaningful_tracks(ego_robot, graph_data, num_tracks)

        if len(meaningful_track_indices) == 0:
            # No meaningful tracks, skip update
            return None

        # Make predictions
        predictions = self.predict(num_agents, num_tracks, graph_data.edge_index_dict, threshold)

        # Return predictions with meaningful track indices for filtering
        return {
            'predictions': predictions,
            'graph_data': graph_data,
            'meaningful_track_indices': meaningful_track_indices,
            'ego_has_cross_validation': ego_has_cross_validation
        }

    def predict(self, num_agents, num_tracks, edge_index_dict, threshold: float = 0.5):
        """
        Predict trust labels for nodes (structure-only)

        Args:
            num_agents: Number of agent nodes
            num_tracks: Number of track nodes
            edge_index_dict: Edge indices dictionary
            threshold: Threshold for binary classification

        Returns:
            Dict containing binary trust predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded successfully")

        with torch.no_grad():
            # Move edge indices to device
            edge_index_dict_device = {k: v.to(self.device) for k, v in edge_index_dict.items()}

            # Get trust probabilities (no features needed)
            trust_probs = self.model(num_agents, num_tracks, edge_index_dict_device, device=self.device)

            # Convert to binary predictions
            predictions = {}
            for node_type, probs in trust_probs.items():
                binary_preds = (probs > threshold).float()
                predictions[node_type] = {
                    'probabilities': probs.cpu().numpy(),
                    'binary_labels': binary_preds.cpu().numpy(),
                    'trust_scores': probs.cpu().numpy()  # Alias for easier access
                }

            return predictions

    def is_available(self) -> bool:
        """Check if model is available for inference"""
        return self.model is not None

    def get_scores(self, ego_robot, all_robots) -> NodeScores:
        """
        Get evidence scores for ego robot and meaningful tracks only.

        This enforces cross-validation constraints:
        - Only returns ego robot score (index 0)
        - Only returns tracks with >=2 robot observations

        Args:
            ego_robot: Robot for which to build ego graph
            all_robots: List of all robots

        Returns:
            NodeScores containing agent and track evidence scores
        """
        if not self.is_available():
            # Return empty scores if model not available
            return NodeScores(agent_scores={}, track_scores={})

        # Get predictions from supervised GNN
        result = self.predict_from_robots_tracks(ego_robot, all_robots)

        # Check if prediction returned None (no cross-validation or no meaningful tracks)
        if result is None:
            return NodeScores(agent_scores={}, track_scores={})

        predictions = result['predictions']
        graph_data = result['graph_data']
        meaningful_track_indices = result.get('meaningful_track_indices', [])

        # Extract agent scores - ONLY ego robot (index 0)
        agent_scores = {}
        if 'agent' in predictions and hasattr(graph_data, 'agent_nodes'):
            agent_trust_probs = predictions['agent']['trust_scores']
            for robot_id, node_idx in graph_data.agent_nodes.items():
                if node_idx == 0:  # Only ego robot
                    if node_idx < len(agent_trust_probs):
                        agent_scores[robot_id] = float(agent_trust_probs[node_idx])

        # Extract track scores - ONLY meaningful tracks
        track_scores = {}
        if 'track' in predictions and hasattr(graph_data, 'track_nodes'):
            track_trust_probs = predictions['track']['trust_scores']
            for track_id, node_idx in graph_data.track_nodes.items():
                if node_idx in meaningful_track_indices:  # Only meaningful tracks
                    if node_idx < len(track_trust_probs):
                        track_scores[track_id] = float(track_trust_probs[node_idx])

        return NodeScores(agent_scores=agent_scores, track_scores=track_scores)


class EgoGraphBuilder:
    """
    Builds ego graphs for individual robots containing only local observations within comm range
    """

    def __init__(self, proximal_range: float = 50.0):
        self.proximal_range = proximal_range

    def build_ego_graph(self, ego_robot, robots):
        """
        Build ego-graph for a specific robot containing only local observations within comm range

        Args:
            ego_robot: The robot for which to build the ego graph
            robots: List of all robots in the simulation

        Returns:
            HeteroData ego graph or None if build fails
        """
        if ego_robot not in robots or not robots:
            return None

        # Step 1: Find proximal robots within communication range
        proximal_robots = [ego_robot]  # Always include ego robot
        for robot in robots:
            if robot.id != ego_robot.id:
                # Check if robot is within communication range
                distance = np.linalg.norm(np.array(ego_robot.position) - np.array(robot.position))
                if distance <= self.proximal_range:
                    proximal_robots.append(robot)

        # Step 2: Get tracks only from proximal robots (ego + proximal)
        proximal_robot_tracks = {}
        for robot in proximal_robots:
            robot_tracks = robot.get_all_current_tracks()
            proximal_robot_tracks[robot.id] = robot_tracks

        # Step 3: Perform track fusion only among proximal robots
        fused_tracks, individual_tracks, track_fusion_map = self._perform_track_fusion(
            proximal_robots, proximal_robot_tracks)

        # Step 4: Build ego-graph with only proximal robots and their tracks
        # NOTE: This also removes isolated agents and updates robot/track lists
        ego_graph_data = self._build_multi_robot_graph(
            proximal_robots, fused_tracks, individual_tracks, track_fusion_map)

        # Mark this as an ego-graph
        ego_graph_data._is_ego_graph = True
        ego_graph_data._ego_robot_id = ego_robot.id

        # Use filtered robot list (after isolated agent removal) for _proximal_robots
        # This is CRITICAL - ensures labels match the filtered graph structure
        if hasattr(ego_graph_data, '_filtered_robots'):
            ego_graph_data._proximal_robots = ego_graph_data._filtered_robots
            delattr(ego_graph_data, '_filtered_robots')  # Clean up temporary attribute
        else:
            # Fallback if no removal occurred
            ego_graph_data._proximal_robots = proximal_robots

        return ego_graph_data


    def _perform_track_fusion(self, robots, robot_tracks):
        """
        Perform track fusion among proximal robots using same logic as RL environment

        Args:
            robots: List of proximal robots
            robot_tracks: Dictionary mapping robot_id to list of tracks

        Returns:
            Tuple of (fused_tracks, individual_tracks, track_fusion_map)
        """

        fused_tracks = []
        individual_tracks = []
        track_fusion_map = {}

        # Collect all tracks from all robots
        all_tracks = []
        for robot_id, tracks in robot_tracks.items():
            for track in tracks:
                all_tracks.append((robot_id, track))

        # Group tracks by object_id - this is the cleaner approach
        object_to_tracks = {}
        for robot_id, track in all_tracks:
            object_id = track.object_id
            if object_id not in object_to_tracks:
                object_to_tracks[object_id] = []
            object_to_tracks[object_id].append((robot_id, track))

        # Process each object group
        for object_id, tracks_list in object_to_tracks.items():
            if len(tracks_list) > 1:
                # Multiple robots see the same object - create fused track
                fused_track = self._create_fused_track(tracks_list, robots)
                fused_tracks.append(fused_track)

                # Map all constituent tracks to the fused track
                for robot_id, track in tracks_list:
                    track_fusion_map[track.track_id] = fused_track.track_id
            else:
                # Only one robot sees this object - keep as individual track
                robot_id, individual_track = tracks_list[0]
                individual_tracks.append(individual_track)
                track_fusion_map[individual_track.track_id] = individual_track.track_id

        return fused_tracks, individual_tracks, track_fusion_map

    def _create_fused_track(self, tracks_to_fuse, all_robots):
        """
        Create a fused track with proper trust inheritance.

        TRUST-FREE: Simply use the first track as primary instead of selecting
        based on robot trust values. The fusion process itself provides validation.
        """
        # Use first track as primary (TRUST-FREE - arbitrary but consistent choice)
        primary_track = tracks_to_fuse[0][1]
        trust_alpha = primary_track.trust_alpha
        trust_beta = primary_track.trust_beta

        # Create fused track ID
        contributing_robots = sorted([robot_id for robot_id, _ in tracks_to_fuse])
        fused_id = f"fused_{'_'.join(map(str, contributing_robots))}_{primary_track.object_id}"

        # Average position and velocity from all contributing tracks
        positions = np.array([track.position for _, track in tracks_to_fuse])
        velocities = np.array([track.velocity for _, track in tracks_to_fuse])

        avg_position = np.mean(positions, axis=0)
        avg_velocity = np.mean(velocities, axis=0)

        # Create fused track using the Track class from robot_track_classes
        from robot_track_classes import Track
        fused_track = Track(
            track_id=fused_id,
            robot_id=primary_track.robot_id,  # Keep the primary robot's ID
            object_id=primary_track.object_id,
            position=avg_position,
            velocity=avg_velocity,
            trust_alpha=trust_alpha,
            trust_beta=trust_beta,
            timestamp=primary_track.timestamp
        )

        return fused_track

    def _build_multi_robot_graph(self, robots, fused_tracks, individual_tracks, track_fusion_map):
        """
        Build multi-robot graph with neural symbolic features and edges
        Uses the exact same edge calculation logic as RL environment

        Args:
            robots: List of robots
            fused_tracks: List of fused tracks
            individual_tracks: List of individual tracks
            track_fusion_map: Mapping of tracks to fusion groups

        Returns:
            HeteroData graph with features and edges
        """
        from torch_geometric.data import HeteroData
        import torch

        graph_data = HeteroData()
        all_tracks = fused_tracks + individual_tracks

        # No features needed - structure-only learning
        # Create dummy feature tensors just for shape information
        num_agents = len(robots)
        num_tracks = len(all_tracks)

        agent_features = torch.ones(num_agents, 1)  # Dummy features (not used by model)
        track_features = torch.ones(num_tracks, 1)  # Dummy features (not used by model)

        # Set up node features (just for storing graph size)
        graph_data['agent'].x = agent_features
        graph_data['track'].x = track_features

        # Create node mappings
        agent_nodes = {robot.id: i for i, robot in enumerate(robots)}
        track_nodes = {track.track_id: i for i, track in enumerate(all_tracks)}

        graph_data.agent_nodes = agent_nodes
        graph_data.track_nodes = track_nodes

        # Build edges using exact same logic as RL environment
        in_fov_and_observed_edges = []  # (agent, track) - robot observes track AND it's in FoV
        observed_and_in_fov_by_edges = []  # (track, agent) - track observed by robot AND in its FoV
        in_fov_only_edges = []  # (agent, track) - track in robot's FoV but NOT observed by robot
        in_fov_only_by_edges = []  # (track, agent) - track in robot's FoV but NOT observed by robot

        for robot in robots:
            robot_idx = agent_nodes[robot.id]

            for track in all_tracks:
                track_idx = track_nodes[track.track_id]

                # Check if robot observes this track (based on ownership/contribution)
                observes_track = self._robot_observes_track(robot, track, fused_tracks, individual_tracks, track_fusion_map)

                # Check if track is in robot's field of view (distance/angle based)
                in_fov_by_distance = self._track_in_robot_fov(robot, track)

                # Categorize based on precise semantics (same as RL environment)
                if observes_track:
                    # Robot observes the track (and it must be in some form of FoV)
                    in_fov_and_observed_edges.append([robot_idx, track_idx])
                    observed_and_in_fov_by_edges.append([track_idx, robot_idx])
                elif in_fov_by_distance:
                    # Track is in FoV by distance but robot doesn't observe it
                    in_fov_only_edges.append([robot_idx, track_idx])
                    in_fov_only_by_edges.append([track_idx, robot_idx])

        # Create agent-to-agent co-detection edges (TRUST-FREE)
        # Two robots are connected if they detected the same object at current timestep
        agent_codetection_edges = []

        # Build object detection map: object_id -> [robot indices that detected it]
        object_detectors = {}
        for idx, robot in enumerate(robots):
            current_tracks = robot.get_current_timestep_tracks()
            for track in current_tracks:
                if hasattr(track, 'object_id'):
                    obj_id = track.object_id
                    if obj_id not in object_detectors:
                        object_detectors[obj_id] = []
                    object_detectors[obj_id].append(idx)

        # Create bidirectional edges between robots that co-detected the same object
        # DEDUPLICATED: Only one edge per pair, regardless of how many objects they co-detected
        codetection_pairs = set()  # Track (src, dst) pairs to avoid duplicates
        for obj_id, detector_indices in object_detectors.items():
            if len(detector_indices) >= 2:  # At least 2 robots detected this object
                # Create edges between all pairs of co-detectors
                for i in range(len(detector_indices)):
                    for j in range(len(detector_indices)):
                        if i != j:
                            pair = (detector_indices[i], detector_indices[j])
                            if pair not in codetection_pairs:
                                agent_codetection_edges.append([detector_indices[i], detector_indices[j]])
                                codetection_pairs.add(pair)

        # Create agent-to-agent contradiction edges
        # Robot A contradicts Robot B if A detects a track that B should see (in B's FoV) but doesn't
        # DEDUPLICATED: Only one edge per pair, regardless of how many tracks they contradict on
        # Make each contradiction bidirectional for symmetric inconsistency signaling
        agent_contradiction_edges = []
        contradiction_pairs = set()  # Track (robot_a, robot_b) pairs to avoid duplicates

        for robot_a_idx, robot_a in enumerate(robots):
            # Get tracks that robot A detects
            tracks_detected_by_a = robot_a.get_current_timestep_tracks()

            for robot_b_idx, robot_b in enumerate(robots):
                if robot_a_idx == robot_b_idx:
                    continue  # Skip self

                # Skip if we've already added edges for this pair
                if (robot_a_idx, robot_b_idx) in contradiction_pairs:
                    continue

                # Get tracks that robot B detects
                tracks_detected_by_b = robot_b.get_current_timestep_tracks()

                # Check if A detects any track that B should see but doesn't
                found_contradiction = False
                for track_a in tracks_detected_by_a:
                    # Check if this track is in B's FoV
                    if self._track_in_robot_fov(robot_b, track_a):
                        # Track is in B's FoV, check if B detects it
                        # Need to match by object_id since track_ids may differ after fusion
                        track_a_obj_id = getattr(track_a, 'object_id', None)

                        # Check if robot B detects this object
                        b_detects_object = False
                        for track_b in tracks_detected_by_b:
                            track_b_obj_id = getattr(track_b, 'object_id', None)
                            if track_a_obj_id and track_b_obj_id == track_a_obj_id:
                                b_detects_object = True
                                break

                        # If B doesn't detect the object, A contradicts B
                        if not b_detects_object and track_a_obj_id:
                            found_contradiction = True
                            break  # Found one contradiction, that's enough for this pair

                # If contradiction found, add bidirectional edges (only once per pair)
                if found_contradiction:
                    agent_contradiction_edges.append([robot_a_idx, robot_b_idx])
                    agent_contradiction_edges.append([robot_b_idx, robot_a_idx])
                    contradiction_pairs.add((robot_a_idx, robot_b_idx))
                    contradiction_pairs.add((robot_b_idx, robot_a_idx))

        # Convert to tensors and create proper edge structure
        edge_types = [
            ('agent', 'in_fov_and_observed', 'track'),
            ('track', 'observed_and_in_fov_by', 'agent'),
            ('agent', 'in_fov_only', 'track'),
            ('track', 'in_fov_only_by', 'agent'),
            ('agent', 'co_detection', 'agent'),    # Co-detection edges (robots detecting same object)
            ('agent', 'contradicts', 'agent'),     # Contradiction edges (A detects track that B should see but doesn't)
        ]

        edge_data = [in_fov_and_observed_edges, observed_and_in_fov_by_edges,
                     in_fov_only_edges, in_fov_only_by_edges, agent_codetection_edges, agent_contradiction_edges]

        graph_data.edge_index_dict = {}
        for edge_type, edges in zip(edge_types, edge_data):
            if edges:
                # Convert list of [src, dst] pairs to tensor format [2, num_edges]
                edge_tensor = torch.tensor(edges, dtype=torch.long).T
                graph_data.edge_index_dict[edge_type] = edge_tensor
            else:
                graph_data.edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long)

        # Set up x_dict for compatibility
        graph_data.x_dict = {
            'agent': graph_data['agent'].x,
            'track': graph_data['track'].x
        }

        # Store fused_tracks and individual_tracks for sanity checking
        graph_data._fused_tracks = fused_tracks
        graph_data._individual_tracks = individual_tracks

        # Remove isolated agents (agents with no agent-to-agent edges)
        graph_data = self._remove_isolated_agents(graph_data, robots, all_tracks)

        return graph_data

    def _remove_isolated_agents(self, graph_data, robots, all_tracks):
        """
        Remove isolated agents (those with no agent-to-agent edges) from the graph.
        This improves computational efficiency and focuses on robots that provide cross-validation signal.

        Args:
            graph_data: HeteroData graph
            robots: List of robots
            all_tracks: List of all tracks

        Returns:
            Updated HeteroData graph with isolated agents removed
        """
        import torch

        num_agents = len(robots)

        # Identify agents with agent-to-agent edges
        agents_with_edges = set()

        # Check co_detection edges
        co_detection_edges = graph_data.edge_index_dict.get(('agent', 'co_detection', 'agent'), None)
        if co_detection_edges is not None and co_detection_edges.numel() > 0:
            agents_with_edges.update(co_detection_edges[0].tolist())
            agents_with_edges.update(co_detection_edges[1].tolist())

        # Check contradicts edges
        contradicts_edges = graph_data.edge_index_dict.get(('agent', 'contradicts', 'agent'), None)
        if contradicts_edges is not None and contradicts_edges.numel() > 0:
            agents_with_edges.update(contradicts_edges[0].tolist())
            agents_with_edges.update(contradicts_edges[1].tolist())

        # Identify isolated agents (excluding ego robot at index 0)
        isolated_agents = []
        for agent_idx in range(num_agents):
            if agent_idx != 0 and agent_idx not in agents_with_edges:
                isolated_agents.append(agent_idx)

        # If no isolated agents, return unchanged
        if not isolated_agents:
            return graph_data

        # Create mapping from old indices to new indices
        kept_agents = [i for i in range(num_agents) if i not in isolated_agents]
        old_to_new_agent = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_agents)}

        # Identify tracks that should be removed (only connected to isolated agents)
        tracks_to_remove = set()

        # Check which tracks are only connected to isolated agents
        for track_idx in range(len(all_tracks)):
            connected_agents = set()

            # Check in_fov_and_observed edges (agent->track)
            edges = graph_data.edge_index_dict.get(('agent', 'in_fov_and_observed', 'track'), None)
            if edges is not None and edges.numel() > 0:
                mask = edges[1] == track_idx
                connected_agents.update(edges[0][mask].tolist())

            # Check in_fov_only edges (agent->track)
            edges = graph_data.edge_index_dict.get(('agent', 'in_fov_only', 'track'), None)
            if edges is not None and edges.numel() > 0:
                mask = edges[1] == track_idx
                connected_agents.update(edges[0][mask].tolist())

            # If track is only connected to isolated agents, remove it
            if connected_agents and all(agent_idx in isolated_agents for agent_idx in connected_agents):
                tracks_to_remove.add(track_idx)

        # Create mapping from old track indices to new track indices
        kept_tracks = [i for i in range(len(all_tracks)) if i not in tracks_to_remove]
        old_to_new_track = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_tracks)}

        # Update agent features
        if len(kept_agents) > 0:
            graph_data['agent'].x = graph_data['agent'].x[kept_agents]
        else:
            graph_data['agent'].x = torch.empty(0, graph_data['agent'].x.shape[1])

        # Update track features
        if len(kept_tracks) > 0:
            graph_data['track'].x = graph_data['track'].x[kept_tracks]
        else:
            graph_data['track'].x = torch.empty(0, graph_data['track'].x.shape[1])

        # Update all edge indices
        new_edge_index_dict = {}

        for edge_type, edge_index in graph_data.edge_index_dict.items():
            src_type, relation, dst_type = edge_type

            if edge_index.numel() == 0:
                new_edge_index_dict[edge_type] = edge_index
                continue

            # Filter and remap edges
            src_indices = edge_index[0].tolist()
            dst_indices = edge_index[1].tolist()

            new_src = []
            new_dst = []

            for src_idx, dst_idx in zip(src_indices, dst_indices):
                # Check if both src and dst are kept
                if src_type == 'agent':
                    if src_idx in old_to_new_agent:
                        new_src_idx = old_to_new_agent[src_idx]
                    else:
                        continue  # Skip this edge
                else:  # src_type == 'track'
                    if src_idx in old_to_new_track:
                        new_src_idx = old_to_new_track[src_idx]
                    else:
                        continue  # Skip this edge

                if dst_type == 'agent':
                    if dst_idx in old_to_new_agent:
                        new_dst_idx = old_to_new_agent[dst_idx]
                    else:
                        continue  # Skip this edge
                else:  # dst_type == 'track'
                    if dst_idx in old_to_new_track:
                        new_dst_idx = old_to_new_track[dst_idx]
                    else:
                        continue  # Skip this edge

                new_src.append(new_src_idx)
                new_dst.append(new_dst_idx)

            # Create new edge tensor
            if new_src:
                new_edge_index_dict[edge_type] = torch.tensor([new_src, new_dst], dtype=torch.long)
            else:
                new_edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long)

        graph_data.edge_index_dict = new_edge_index_dict

        # Update x_dict
        graph_data.x_dict = {
            'agent': graph_data['agent'].x,
            'track': graph_data['track'].x
        }

        # Update node mappings (if they exist)
        if hasattr(graph_data, 'agent_nodes'):
            new_agent_nodes = {}
            for robot_id, old_idx in graph_data.agent_nodes.items():
                if old_idx in old_to_new_agent:
                    new_agent_nodes[robot_id] = old_to_new_agent[old_idx]
            graph_data.agent_nodes = new_agent_nodes

        if hasattr(graph_data, 'track_nodes'):
            new_track_nodes = {}
            for track_id, old_idx in graph_data.track_nodes.items():
                if old_idx in old_to_new_track:
                    new_track_nodes[track_id] = old_to_new_track[old_idx]
            graph_data.track_nodes = new_track_nodes

        # Update stored robot and track lists to match filtered graph
        # This is CRITICAL for label generation and batch processing
        # Store filtered robot list - will be used to set _proximal_robots
        graph_data._filtered_robots = [robots[i] for i in kept_agents]

        # Update fused and individual track lists (these were set earlier in _build_multi_robot_graph)
        if hasattr(graph_data, '_fused_tracks') and hasattr(graph_data, '_individual_tracks'):
            all_tracks_list = graph_data._fused_tracks + graph_data._individual_tracks
            filtered_all_tracks = [all_tracks_list[i] for i in kept_tracks]

            # Reconstruct fused and individual track lists
            # Check which filtered tracks are fused vs individual
            new_fused_tracks = []
            new_individual_tracks = []
            for track in filtered_all_tracks:
                if track in graph_data._fused_tracks:
                    new_fused_tracks.append(track)
                elif track in graph_data._individual_tracks:
                    new_individual_tracks.append(track)

            graph_data._fused_tracks = new_fused_tracks
            graph_data._individual_tracks = new_individual_tracks

        return graph_data

    def _robot_observes_track(self, robot, track, fused_tracks, individual_tracks, track_fusion_map):
        """Determine if robot observes this track (based on fusion logic)"""
        # Check individual tracks first
        if track in individual_tracks:
            return track.robot_id == robot.id

        # Check if this is a fused track and robot contributed to it
        if track in fused_tracks:
            # Check if this robot contributed to the fused track
            found_contribution = False
            for original_id, fused_id in track_fusion_map.items():
                if fused_id == track.track_id:
                    # Check if this robot contributed to the fused track
                    if original_id.startswith(f"{robot.id}_"):
                        found_contribution = True
                        break

            return found_contribution

        return False

    def _track_in_robot_fov(self, robot, track):
        """Determine if track is in robot's field of view using Robot's built-in method"""
        return robot.is_in_fov(track.position)  # Use Robot class's is_in_fov method
