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
import numpy as np
import math

# Import required modules
from paper_trust_algorithm import PaperTrustAlgorithm
from robot_track_classes import Robot, Track


class TrustFeatureCalculator:
    """
    Calculates neural symbolic features and trust values for nodes
    """

    def __init__(self):
        self.paper_trust_algorithm = PaperTrustAlgorithm()

    def _calculate_beta_std(self, alpha: float, beta: float) -> float:
        """Calculate the standard deviation of a Beta(alpha, beta) distribution."""
        alpha_plus_beta = alpha + beta
        variance = (alpha * beta) / ((alpha_plus_beta ** 2) * (alpha_plus_beta + 1))
        return math.sqrt(variance)

    def _robot_observes_track(self, robot: 'Robot', track: 'Track') -> bool:
        """
        Determine if robot observes this track (simplified version)

        Args:
            robot: Robot object
            track: Track object

        Returns:
            True if robot observes the track, False otherwise
        """
        # Check if this track belongs to the robot
        if hasattr(track, 'robot_id') and track.robot_id == robot.id:
            return True

        # Check if track is in robot's field of view
        if hasattr(robot, 'is_in_fov') and hasattr(track, 'position'):
            return robot.is_in_fov(track.position)

        # Fallback: check if robot has this track in its track list
        robot_tracks = robot.get_all_tracks() if hasattr(robot, 'get_all_tracks') else []
        return track in robot_tracks

    def calculate_agent_features(self, robots: List['Robot'], all_tracks: List['Track']) -> torch.Tensor:
        """
        Calculate neural symbolic features for agent nodes

        Args:
            robots: List of Robot objects
            all_tracks: List of all Track objects

        Returns:
            Tensor of agent features [num_agents, 4]
        """
        agent_features = []

        for robot in robots:
            # Compute trust-based predicates using Robot.trust_value property
            robot_trust = robot.trust_value

            # Feature 1: HighConfidence(robot) - robot confidence > 5.0
            robot_std = self._calculate_beta_std(robot.trust_alpha, robot.trust_beta)
            robot_confidence = 1.0 / (robot_std + 1e-6)
            high_confidence_pred = 1.0 if robot_confidence > 5.0 else 0.0

            # Feature 2: HighlyTrusted(robot) - robot trust > 0.7
            highly_trusted_pred = 1.0 if robot_trust > 0.7 else 0.0

            # Feature 3: HighConnectivity(robot) - robot observes many tracks
            robot_track_count = sum(1 for track in all_tracks if self._robot_observes_track(robot, track))
            high_connectivity_pred = 1.0 if robot_track_count >= 3 else 0.0

            # Feature 4: ReliableDetector(robot) - robot's tracks have high average trust
            robot_tracks = robot.get_all_tracks()
            if robot_tracks:
                avg_track_trust = np.mean([track.trust_value for track in robot_tracks])
                reliable_detector_pred = 1.0 if avg_track_trust > 0.6 else 0.0
            else:
                reliable_detector_pred = 0.0

            agent_features.append([
                high_confidence_pred,    # Feature 1
                highly_trusted_pred,     # Feature 2
                high_connectivity_pred,  # Feature 3
                reliable_detector_pred,  # Feature 4
            ])

        return torch.tensor(agent_features, dtype=torch.float)

    def calculate_track_features(self, all_tracks: List['Track'], fused_tracks: List['Track']) -> torch.Tensor:
        """
        Calculate neural symbolic features for track nodes

        Args:
            all_tracks: List of all Track objects
            fused_tracks: List of fused Track objects

        Returns:
            Tensor of track features [num_tracks, 5]
        """
        track_features = []

        for track in all_tracks:
            # Compute trust and confidence based predicates using Track.trust_value property
            track_trust = track.trust_value

            # Feature 1: HighConfidence(track) - track confidence > 5.0
            track_std = self._calculate_beta_std(track.trust_alpha, track.trust_beta)
            track_confidence = 1.0 / (track_std + 1e-6)
            high_confidence_pred = 1.0 if track_confidence > 5.0 else 0.0

            # Feature 2: HighlyTrusted(track) - track trust > 0.7
            highly_trusted_pred = 1.0 if track_trust > 0.7 else 0.0

            # Feature 3: Suspicious(track) - track trust < 0.3
            suspicious_pred = 1.0 if track_trust < 0.3 else 0.0

            # Feature 4: MultiRobotTrack(track) - track appears in fused_tracks
            multi_robot_pred = 1.0 if track in fused_tracks else 0.0

            # Feature 5: WellObserved(track) - track has been observed/updated many times
            well_observed_pred = 1.0 if track.observation_count > 50 else 0.0

            track_features.append([
                high_confidence_pred,    # Feature 1
                highly_trusted_pred,     # Feature 2
                suspicious_pred,         # Feature 3
                multi_robot_pred,        # Feature 4
                well_observed_pred,      # Feature 5
            ])

        return torch.tensor(track_features, dtype=torch.float)

    def create_agent_comparison_edges(self, robots: List['Robot']) -> torch.Tensor:
        """
        Create agent-to-agent comparison edges based on trust values

        Args:
            robots: List of Robot objects

        Returns:
            Edge tensor [2, num_edges] for agent comparison edges
        """
        source_indices = []
        target_indices = []

        # Compare each pair of agents
        for i, robot_i in enumerate(robots):
            for j, robot_j in enumerate(robots):
                if i != j:
                    # Only create edge if robot_i is more trustworthy than robot_j
                    if robot_i.trust_value > robot_j.trust_value:
                        source_indices.append(i)
                        target_indices.append(j)

        if source_indices:
            return torch.tensor([source_indices, target_indices], dtype=torch.long)
        else:
            return torch.empty((2, 0), dtype=torch.long)

    def update_trust_continuously(self, robots: List['Robot'], environment) -> Dict:
        """
        Update trust values using paper trust algorithm

        Args:
            robots: List of Robot objects
            environment: Simulation environment

        Returns:
            Dictionary of trust updates
        """
        if self.paper_trust_algorithm is None:
            return {}

        return self.paper_trust_algorithm.update_trust(robots, environment)


class SupervisedTrustGNN(nn.Module):
    """
    Supervised GNN model for binary trust classification
    """

    def __init__(self, agent_features: int = 5, track_features: int = 5, hidden_dim: int = 64):
        super(SupervisedTrustGNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.feature_calculator = TrustFeatureCalculator()

        # Node embedding layers
        self.agent_embedding = Linear(agent_features, hidden_dim)
        self.track_embedding = Linear(track_features, hidden_dim)

        # Graph convolution layers using heterogeneous GAT
        edge_types = [
            ('agent', 'in_fov_and_observed', 'track'),
            ('track', 'observed_and_in_fov_by', 'agent'),
            ('agent', 'in_fov_only', 'track'),
            ('track', 'in_fov_only_by', 'agent'),
            ('agent', 'more_trustworthy_than', 'agent'),  # Agent comparison edges
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


    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass for trust classification

        Args:
            x_dict: Node features dictionary
            edge_index_dict: Edge indices dictionary

        Returns:
            Dict containing trust predictions for agents and tracks
        """
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
        self.feature_calculator = TrustFeatureCalculator()
        self.ego_graph_builder = EgoGraphBuilder(proximal_range=proximal_range)
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Create model with correct architecture
            self.model = SupervisedTrustGNN(
                agent_features=4,  # 4 neural-symbolic predicates
                track_features=5,  # 5 neural-symbolic predicates
                hidden_dim=64
            )

            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            print(f"✅ Loaded supervised trust model from {model_path}")

        except Exception as e:
            print(f"⚠️ Failed to load supervised trust model: {e}")
            self.model = None

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

        # Make predictions
        predictions = self.predict(graph_data.x_dict, graph_data.edge_index_dict, threshold)

        # Return both predictions and graph data for proper node mapping
        return {
            'predictions': predictions,
            'graph_data': graph_data
        }

    def predict(self, x_dict, edge_index_dict, threshold: float = 0.5):
        """
        Predict trust labels for nodes

        Args:
            x_dict: Node features dictionary
            edge_index_dict: Edge indices dictionary
            threshold: Threshold for binary classification

        Returns:
            Dict containing binary trust predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded successfully")

        with torch.no_grad():
            # Move data to device
            x_dict_device = {k: v.to(self.device) for k, v in x_dict.items()}
            edge_index_dict_device = {k: v.to(self.device) for k, v in edge_index_dict.items()}

            # Get trust probabilities
            trust_probs = self.model(x_dict_device, edge_index_dict_device)

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

    def update_trust_and_predict(self,
                                ego_robot: 'Robot',
                                robots: List['Robot'],
                                environment,
                                threshold: float = 0.5) -> Tuple[Dict, Dict]:
        """
        Update trust values using paper algorithm and then make predictions

        Args:
            ego_robot: The robot for which to build the ego graph
            robots: List of all Robot objects
            environment: Simulation environment
            threshold: Classification threshold

        Returns:
            Tuple of (trust_updates, predictions)
        """
        # Update trust values using paper algorithm
        trust_updates = self.feature_calculator.update_trust_continuously(robots, environment)

        # Make predictions with updated trust values
        predictions = self.predict_from_robots_tracks(ego_robot, robots, threshold)

        return trust_updates, predictions

    def is_available(self) -> bool:
        """Check if model is available for inference"""
        return self.model is not None

    def has_features(self) -> bool:
        """Check if features are available"""
        return self.feature_calculator is not None


class EgoGraphBuilder:
    """
    Builds ego graphs for individual robots containing only local observations within comm range
    """

    def __init__(self, proximal_range: float = 50.0):
        self.proximal_range = proximal_range
        self.feature_calculator = TrustFeatureCalculator()

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
        """Create a fused track with proper trust inheritance"""
        # Use highest trust robot's track as primary track
        robot_trusts = {}
        for robot in all_robots:
            robot_trusts[robot.id] = robot.trust_value  # Use Robot.trust_value property

        # Find track from highest trust robot
        best_track = None
        best_robot_trust = -1
        for robot_id, track in tracks_to_fuse:
            if robot_trusts[robot_id] > best_robot_trust:
                best_robot_trust = robot_trusts[robot_id]
                best_track = track

        primary_track = best_track
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

        # Use TrustFeatureCalculator to compute features properly
        agent_features = self.feature_calculator.calculate_agent_features(robots, all_tracks)
        track_features = self.feature_calculator.calculate_track_features(all_tracks, fused_tracks)

        # Set up node features
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

        # Create agent-to-agent comparison edges based on trust values
        agent_comparison_edges = []
        for i, robot_i in enumerate(robots):
            for j, robot_j in enumerate(robots):
                if i != j:
                    # Only create edge if robot_i is more trustworthy than robot_j
                    if robot_i.trust_value > robot_j.trust_value:
                        agent_comparison_edges.append([i, j])

        # Convert to tensors and create proper edge structure
        edge_types = [
            ('agent', 'in_fov_and_observed', 'track'),
            ('track', 'observed_and_in_fov_by', 'agent'),
            ('agent', 'in_fov_only', 'track'),
            ('track', 'in_fov_only_by', 'agent'),
            ('agent', 'more_trustworthy_than', 'agent'),  # Agent comparison edges
        ]

        edge_data = [in_fov_and_observed_edges, observed_and_in_fov_by_edges,
                     in_fov_only_edges, in_fov_only_by_edges, agent_comparison_edges]

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


