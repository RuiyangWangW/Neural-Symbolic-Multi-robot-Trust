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

    def __init__(self, model_path: Optional[str] = None, proximal_range: float = 50.0,
                 ablation: Optional[str] = None, temporal_mode: str = 'beta'):
        """
        Initialize supervised trust algorithm

        Args:
            model_path: Path to trained GNN model (.pth file). If None, uses fresh weights.
            proximal_range: Maximum distance for considering robots as proximal neighbors
            ablation: Architecture ablation the checkpoint was trained with (forwarded to
                SupervisedTrustPredictor; None auto-detects from the checkpoint tag).
            temporal_mode: How per-timestep GNN scores are aggregated into a final trust:
                - 'beta'        : (default) accumulate delta_alpha/delta_beta into a Beta
                                  distribution every timestep (the paper's contribution #2).
                - 'last_step'   : ignore accumulation; final trust = last timestep's raw GNN
                                  probability.
                - 'mean_scores' : final trust = mean of per-timestep raw GNN probabilities.
                For 'last_step'/'mean_scores', per-step scores are recorded and the benchmark
                must call finalize_temporal(robots) once at the end of the episode to write
                the aggregated trust back onto each robot/track.
        """
        if temporal_mode not in ('beta', 'last_step', 'mean_scores'):
            raise ValueError(f"Unknown temporal_mode '{temporal_mode}'")
        self.proximal_range = proximal_range
        self.temporal_mode = temporal_mode
        self.predictor = SupervisedTrustPredictor(
            model_path=model_path,
            proximal_range=proximal_range,
            ablation=ablation,
        )
        # Per-step raw GNN probability history for the non-beta temporal ablations.
        # robot_id -> list[float];  (robot_id, object_id) -> list[float]
        self._agent_score_history: Dict = {}
        self._track_score_history: Dict = {}

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
        updated_tracks = {}  # robot_id -> set of object_ids

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

            # NEW ARCHITECTURE: Collect track information from reported_tracks
            # (these are the tracks that were actually shared with neighbors)
            for track in robot.get_reported_tracks_list():
                was_meaningful = (
                    robot.id in updated_tracks and
                    track.object_id in updated_tracks[robot.id]
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

    def finalize_temporal(self, robots: List[Robot]) -> None:
        """
        For the temporal ablation ('last_step' / 'mean_scores'), aggregate the per-timestep
        raw GNN probabilities recorded during the episode into a single final trust value
        and write it onto each robot (and its meaningful tracks) by setting the Beta params
        so that trust_value == aggregated_score (alpha=s, beta=1-s).

        No-op for temporal_mode='beta' (trust was already accumulated online), so it is safe
        to call unconditionally at the end of an episode.

        Args:
            robots: The robots evaluated this episode (same list passed to update_trust).
        """
        if self.temporal_mode == 'beta':
            return

        def _aggregate(scores: List[float]) -> Optional[float]:
            if not scores:
                return None
            if self.temporal_mode == 'last_step':
                return scores[-1]
            return sum(scores) / len(scores)  # 'mean_scores'

        eps = 1e-3

        def _set_trust(obj, score: float):
            s = min(max(score, eps), 1.0 - eps)  # keep alpha,beta strictly positive
            obj.trust_alpha = s
            obj.trust_beta = 1.0 - s

        robots_by_id = {r.id: r for r in robots}

        # Agent (robot) trust
        for robot_id, scores in self._agent_score_history.items():
            agg = _aggregate(scores)
            if agg is None or robot_id not in robots_by_id:
                continue
            _set_trust(robots_by_id[robot_id], agg)

        # Track trust (only tracks that were meaningful at some step)
        for (robot_id, object_id), scores in self._track_score_history.items():
            agg = _aggregate(scores)
            if agg is None or robot_id not in robots_by_id:
                continue
            robot = robots_by_id[robot_id]
            track = robot.all_tracks.get(object_id)
            if track is not None:
                _set_trust(track, agg)

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
                raw_p = agent_probs[ego_idx][0]  # Extract raw probability from array

                # Use raw probability from model (no calibration)
                p = float(raw_p)

                # Find ego robot object (first robot in proximal_robots list)
                ego_robot = graph_data._proximal_robots[0]

                if self.temporal_mode == 'beta':
                    # Convert probability to alpha/beta update (contribution #2): standard
                    # soft-evidence Beta update - every step adds p to alpha and (1-p) to beta.
                    delta_alpha, delta_beta = p, (1 - p)
                    ego_robot.update_trust(delta_alpha, delta_beta)
                else:
                    # Temporal ablation: record the raw per-step score instead of
                    # accumulating a Beta distribution (finalized in finalize_temporal()).
                    self._agent_score_history.setdefault(ego_robot.id, []).append(p)

        # Update ONLY meaningful tracks (those that pass cross-validation)
        if 'track' in predictions and hasattr(graph_data, '_fused_tracks') and hasattr(graph_data, '_individual_tracks') and meaningful_track_indices:
            track_probs = predictions['track']['probabilities']

            # Index directly into the graph's own track list (same order as track_nodes/
            # edge_index_dict). IMPORTANT: Do NOT look up by track_id via ego_robot's
            # reported_tracks - for fused tracks (>=2 robots observing the same object),
            # graph_data.track_nodes keys are the synthesized "fused_..." id, which never
            # matches any individual robot's own track_id. Since cross-validated
            # ("meaningful") tracks are fused tracks by construction, matching on track_id
            # silently dropped trust updates for exactly the tracks this algorithm targets.
            # object_id is the stable identifier preserved through fusion, so use that.
            all_tracks = graph_data._fused_tracks + graph_data._individual_tracks

            # Update only meaningful tracks
            for track_idx in meaningful_track_indices:
                if track_idx < len(track_probs) and track_idx < len(all_tracks):
                    raw_p = track_probs[track_idx][0]  # Extract raw probability from array
                    p = float(raw_p)  # Use raw probability (no calibration)

                    # Find the track object using the graph's own track list
                    object_id = all_tracks[track_idx].object_id

                    # IMPORTANT: Only update ego robot's track (index 0)
                    # Each robot manages its own tracks independently
                    ego_robot = graph_data._proximal_robots[0]

                    # Forward the update if ego robot is either currently reporting this
                    # object, OR has detected it before (missed-detection case: ego saw
                    # this object at some point but isn't reporting it this timestep even
                    # though a proximal robot currently corroborates it - see
                    # _identify_meaningful_tracks criterion 1b). Both cases have a real
                    # all_tracks entry to update; anything else was never a track ego owns.
                    if not (object_id in ego_robot.get_reported_object_ids()
                            or object_id in ego_robot.all_tracks):
                        continue

                    if self.temporal_mode == 'beta':
                        # Convert probability to alpha/beta update (contribution #2): standard
                        # soft-evidence Beta update - add p to alpha and (1-p) to beta.
                        delta_alpha, delta_beta = p, (1 - p)
                        # Forward trust update to all_tracks (persistent storage)
                        ego_robot.forward_trust_update_to_all_tracks(
                            object_id=object_id,
                            delta_alpha=delta_alpha,
                            delta_beta=delta_beta
                        )
                    else:
                        # Temporal ablation: record the raw per-step score, keyed by the
                        # ego robot + object, finalized in finalize_temporal().
                        self._track_score_history.setdefault((ego_robot.id, object_id), []).append(p)

                    updated_track_ids.add(object_id)

        return updated_track_ids

