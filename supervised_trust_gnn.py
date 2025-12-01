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
from typing import Dict, List
from pathlib import Path
import numpy as np

# Import required modules
from robot_track_classes import Robot


# TrustFeatureCalculator class removed - structure-only learning does not require feature calculations


class SupervisedTrustGNN(nn.Module):
    """
    Supervised GNN model for binary trust classification

    SIMPLIFIED DESIGN:
    - No node features - learns purely from graph structure
    - Uses learnable node type embeddings (one for agents, one for tracks)
    - Structure-only learning
    """

    def __init__(self, hidden_dim: int = 128):
        super(SupervisedTrustGNN, self).__init__()

        self.hidden_dim = hidden_dim

        # Learnable node type embeddings (no input features)
        # Each node gets initialized with a shared type-specific embedding
        self.agent_type_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        self.track_type_embedding = nn.Parameter(torch.randn(1, hidden_dim))

        # Graph convolution layers using heterogeneous GAT
        edge_types = [
            ('agent', 'in_fov_and_observed', 'track'),
            ('track', 'observed_and_in_fov_by', 'agent'),
            ('agent', 'in_fov_only', 'track'),
            ('track', 'in_fov_only_by', 'agent'),
            ('agent', 'co_detection', 'agent'),  # Co-detection edges (robots detecting same object)
            ('agent', 'contradicts', 'agent'),   # Contradiction edges (robot A detects track that B should see but doesn't)
        ]

        # Create GAT convolution dictionary for each edge type
        conv_dict = {}
        for src_type, relation, dst_type in edge_types:
            src_dim = hidden_dim  # All use hidden_dim after embedding
            conv_dict[(src_type, relation, dst_type)] = GATConv(
                in_channels=src_dim,
                out_channels=hidden_dim,
                heads=4,
                concat=False,
                add_self_loops=False,
                dropout=0.1
            )

        # Three layers of heterogeneous convolution
        self.conv1 = HeteroConv(conv_dict, aggr='mean')
        self.conv2 = HeteroConv(conv_dict, aggr='mean')
        self.conv3 = HeteroConv(conv_dict, aggr='mean')

        # Batch normalization layers
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

        # Binary classification heads for trust prediction
        self.agent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
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
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),  # Binary classification
            nn.Sigmoid()  # Output probability of being trustworthy
        )


    def forward(self, num_agents, num_tracks, edge_index_dict, device='cpu'):
        """
        Forward pass for trust classification (structure-only)

        Args:
            num_agents: Number of agent nodes
            num_tracks: Number of track nodes
            edge_index_dict: Edge indices dictionary
            device: Device for tensors

        Returns:
            Dict containing trust predictions for agents and tracks
        """
        # Check if we have any tracks
        has_tracks = num_tracks > 0

        # Initialize all nodes with learnable type embeddings
        # Agent nodes: all get the same initial embedding (will be differentiated by graph structure)
        agent_embeddings = self.agent_type_embedding.expand(num_agents, -1)

        x_dict = {'agent': agent_embeddings}

        # Track nodes: all get the same initial embedding
        if has_tracks:
            track_embeddings = self.track_type_embedding.expand(num_tracks, -1)
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

        # Add skip connection from second layer
        x_dict_final = {
            key: x_dict_3[key] + x_dict_2[key] if x_dict_3[key].shape[0] > 0 and x_dict_2[key].shape[0] > 0
            else x_dict_3[key]
            for key in x_dict_3
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
                    loaded_msg = f"✅ Loaded supervised trust model from {resolved_path} (matched {len(matched_keys)} tensors)"
                    print(loaded_msg)
                else:
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

    def predict_from_robots_tracks(self,
                                 ego_robot: 'Robot',
                                 robots: List['Robot'],
                                 threshold: float = 0.5) -> Dict:
        """
        Predict trust labels directly from ego robot and all robots

        Args:
            ego_robot: The robot for which to build the ego graph
            robots: List of all Robot objects
            threshold: Classification threshold

        Returns:
            Dict containing trust predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded successfully")

        # Create ego graph using proper build_ego_graph method with fusion map
        graph_data = self.ego_graph_builder.build_ego_graph(ego_robot, robots)

        # Extract graph size
        num_agents = graph_data.x_dict['agent'].shape[0] if 'agent' in graph_data.x_dict else 0
        num_tracks = graph_data.x_dict['track'].shape[0] if 'track' in graph_data.x_dict else 0

        # Make predictions
        predictions = self.predict(num_agents, num_tracks, graph_data.edge_index_dict, threshold)

        # Return both predictions and graph data for proper node mapping
        return {
            'predictions': predictions,
            'graph_data': graph_data
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
        ego_graph_data = self._build_multi_robot_graph(
            proximal_robots, fused_tracks, individual_tracks, track_fusion_map)

        # Mark this as an ego-graph
        ego_graph_data._is_ego_graph = True
        ego_graph_data._ego_robot_id = ego_robot.id
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
        for obj_id, detector_indices in object_detectors.items():
            if len(detector_indices) >= 2:  # At least 2 robots detected this object
                # Create edges between all pairs of co-detectors
                for i in range(len(detector_indices)):
                    for j in range(len(detector_indices)):
                        if i != j:
                            agent_codetection_edges.append([detector_indices[i], detector_indices[j]])

        # Create agent-to-agent contradiction edges
        # Robot A contradicts Robot B if A detects a track that B should see (in B's FoV) but doesn't
        agent_contradiction_edges = []

        for robot_a_idx, robot_a in enumerate(robots):
            # Get tracks that robot A detects
            tracks_detected_by_a = robot_a.get_current_timestep_tracks()

            for robot_b_idx, robot_b in enumerate(robots):
                if robot_a_idx == robot_b_idx:
                    continue  # Skip self

                # Get tracks that robot B detects
                tracks_detected_by_b = robot_b.get_current_timestep_tracks()

                # Check if A detects any track that B should see but doesn't
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
                        if not b_detects_object:
                            agent_contradiction_edges.append([robot_a_idx, robot_b_idx])
                            break  # Only add one contradiction edge per robot pair

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
