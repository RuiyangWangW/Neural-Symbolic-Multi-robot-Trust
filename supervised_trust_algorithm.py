#!/usr/bin/env python3
"""
Supervised GNN Trust Algorithm Implementation

This module provides a trust update algorithm based on a trained Graph Neural Network (GNN)
that predicts trust values from the structure of ego-centric observation graphs.

Key Properties:
- Ego-centric: Each robot updates only its own trust and detected tracks
- Ego robot cross-validation: Only updates when ego robot has co-detection or contradicts edges
- Track cross-validation: Only updates tracks detected by ego AND with edges to >=2 robots
- Neural-symbolic: GNN predictions guide Beta distribution trust updates
"""

from typing import List, Dict, Optional
from robot_track_classes import Robot, Track
from supervised_trust_gnn import SupervisedTrustPredictor
from simulation_environment import SimulationEnvironment


class SupervisedTrustAlgorithm:
    """
    Supervised GNN-based trust update algorithm

    This algorithm uses a trained Graph Neural Network to predict trust values
    based on the structure of ego-centric observation graphs. The GNN learns
    patterns of legitimate vs adversarial behavior from graph features like:
    - Co-detection edges (robots observing same objects)
    - Contradicts edges (robots reporting conflicting observations)
    - In-FOV edges (robots within each other's field of view)

    The trust update follows an ego-centric design:
    1. Each robot constructs its own ego-graph from local observations
    2. The GNN predicts trust for the ego robot and its meaningful tracks
    3. Only the ego robot and cross-validated tracks get updated
    4. Other robots update themselves when building their own ego-graphs
    """

    def __init__(self, model_path: Optional[str] = None, proximal_range: float = 50.0):
        """
        Initialize supervised trust algorithm

        Args:
            model_path: Path to trained GNN model (.pth file). If None, uses fresh weights.
            proximal_range: Maximum distance for considering robots as proximal neighbors
        """
        self.proximal_range = proximal_range
        self.predictor = SupervisedTrustPredictor(
            model_path=model_path,
            proximal_range=proximal_range
        )

    def update_trust(self, robots: List[Robot], environment: Optional[SimulationEnvironment] = None) -> Dict[int, Dict]:
        """
        Update trust values using supervised GNN predictions

        This method implements the ego-centric trust update:
        1. For each robot (as ego):
           a. Build ego-centric graph from local observations
           b. Get GNN predictions for ego robot and meaningful tracks
           c. Update ONLY ego robot and meaningful tracks (not other robots in graph)
        2. Collect and return trust update statistics

        Args:
            robots: List of all robots in the simulation
            environment: Optional simulation environment (for compatibility with paper algorithm)

        Returns:
            Dictionary mapping robot_id to trust update information:
            {
                robot_id: {
                    'alpha': updated alpha value,
                    'beta': updated beta value,
                    'mean_trust': updated trust value,
                    'ego_updated': True if this robot was updated as ego,
                    'track_trust_info': {
                        track_id: {
                            'alpha': track alpha,
                            'beta': track beta,
                            'mean_trust': track trust,
                            'object_id': associated object ID,
                            'meaningful': True if track was updated
                        }
                    }
                }
            }
        """
        trust_updates = {}

        # Track which robots were updated as ego
        updated_as_ego = set()

        # Track which tracks were updated (for statistics)
        updated_tracks = {}  # robot_id -> set of track_ids

        # For each robot, build its ego-graph and update its trust
        for ego_robot in robots:
            try:
                # Get predictions from supervised model for this ego-graph
                result = self.predictor.predict_from_robots_tracks(
                    ego_robot=ego_robot,
                    robots=robots,
                    threshold=0.5
                )

                # Check if result is None (no cross-validation or meaningful tracks)
                if result is None:
                    continue

                # Extract prediction results
                predictions = result['predictions']
                graph_data = result['graph_data']
                meaningful_track_indices = result['meaningful_track_indices']

                # Update trust from predictions
                # IMPORTANT: This only updates ego_robot and its meaningful tracks
                updated_track_ids = self._update_trust_from_predictions(
                    predictions=predictions,
                    graph_data=graph_data,
                    meaningful_track_indices=meaningful_track_indices
                )

                # Record that this robot was updated as ego
                updated_as_ego.add(ego_robot.id)
                updated_tracks[ego_robot.id] = updated_track_ids

            except Exception as e:
                # Continue with other robots if one fails
                print(f"⚠️ Supervised model error for robot {ego_robot.id}: {e}")
                continue

        # Collect trust information for all robots (similar to paper algorithm format)
        for robot in robots:
            track_trust_info = {}

            # Collect track information
            for track in robot.get_all_current_tracks():
                was_meaningful = (
                    robot.id in updated_tracks and
                    track.track_id in updated_tracks[robot.id]
                )

                track_trust_info[track.track_id] = {
                    'alpha': track.trust_alpha,
                    'beta': track.trust_beta,
                    'mean_trust': track.trust_value,
                    'object_id': track.object_id,
                    'meaningful': was_meaningful
                }

            trust_updates[robot.id] = {
                'alpha': robot.trust_alpha,
                'beta': robot.trust_beta,
                'mean_trust': robot.trust_value,
                'ego_updated': robot.id in updated_as_ego,
                'track_trust_info': track_trust_info
            }

        return trust_updates

    def _update_trust_from_predictions(
        self,
        predictions: Dict,
        graph_data,
        meaningful_track_indices: List[int]
    ) -> set:
        """
        Update alpha/beta values based on supervised model predictions

        IMPORTANT: Only updates:
        1. Ego robot (index 0 in the graph)
        2. Meaningful tracks (those in meaningful_track_indices)

        This design ensures:
        - Ego-centric updates (each robot only updates itself)
        - Cross-validated tracks only (ego-detected + edges to >=2 robots)
        - Privacy-preserving (no robot updates another robot's trust)

        Args:
            predictions: Model predictions containing trust probabilities
            graph_data: Graph data with node mappings (agent_nodes, track_nodes)
            meaningful_track_indices: List of track indices (ego-detected + edges to >=2 robots)

        Returns:
            Set of track_ids that were updated
        """
        updated_track_ids = set()

        # Update ONLY ego robot (index 0 in the graph)
        if 'agent' in predictions and hasattr(graph_data, 'agent_nodes'):
            agent_probs = predictions['agent']['probabilities']

            # Ego robot is always at index 0
            ego_idx = 0
            if ego_idx < len(agent_probs):
                p = agent_probs[ego_idx][0]  # Extract probability from array

                # Convert probability to alpha/beta update
                # High probability (p ≥ 0.5) → increase alpha (positive evidence)
                # Low probability (p < 0.5) → increase beta (negative evidence)
                if p >= 0.5:
                    delta_alpha = p
                    delta_beta = 0.0
                else:
                    delta_alpha = 0.0
                    delta_beta = (1 - p)

                # Find ego robot object (first robot in proximal_robots list)
                ego_robot = graph_data._proximal_robots[0]
                ego_robot.update_trust(delta_alpha, delta_beta)

        # Update ONLY meaningful tracks (those that pass cross-validation)
        if 'track' in predictions and hasattr(graph_data, 'track_nodes') and meaningful_track_indices:
            track_probs = predictions['track']['probabilities']

            # Get reverse mapping: graph_index -> track_id
            index_to_track_id = {
                node_idx: track_id
                for track_id, node_idx in graph_data.track_nodes.items()
            }

            # Update only meaningful tracks
            for track_idx in meaningful_track_indices:
                if track_idx < len(track_probs):
                    p = track_probs[track_idx][0]  # Extract probability from array

                    # Convert probability to alpha/beta update
                    if p >= 0.5:
                        # High trust prediction - increase alpha
                        delta_alpha = p
                        delta_beta = 0.0
                    else:
                        # Low trust prediction - increase beta
                        delta_alpha = 0.0
                        delta_beta = (1 - p)

                    # Find the track object using the mapping
                    track_id = index_to_track_id.get(track_idx)
                    if track_id:
                        # IMPORTANT: Only update ego robot's track (index 0)
                        # Each robot manages its own tracks independently
                        ego_robot = graph_data._proximal_robots[0]
                        for track in ego_robot.get_all_current_tracks():
                            if track.track_id == track_id:
                                track.update_trust(delta_alpha, delta_beta)
                                updated_track_ids.add(track_id)
                                break

        return updated_track_ids

