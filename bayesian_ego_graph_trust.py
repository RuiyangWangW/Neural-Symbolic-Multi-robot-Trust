#!/usr/bin/env python3
"""
Bayesian Ego Graph Trust Algorithm - Simple Baseline

This baseline uses edge counts from ego graphs to update trust values:
- For robot nodes: co_detection edges (positive) vs contradicts edges (negative)
- For track nodes: in_fov_and_observed edges (positive) vs in_fov_only edges (negative)

Updates use simple Bayesian (Beta distribution) hyperparameter increments:
  α_{t+1} = α_t + count(positive_evidence)
  β_{t+1} = β_t + count(negative_evidence)
"""

import numpy as np
from typing import List, Dict, Optional
from robot_track_classes import Robot, Track
from supervised_trust_gnn import EgoGraphBuilder


class BayesianEgoGraphTrust:
    """
    Simple Bayesian trust update using ego graph edge counts as evidence.

    For each ego robot:
    1. Build ego graph (includes proximal robots and their tracks)
    2. Count positive and negative evidence edges from ALL agents in the graph
    3. Update trust hyperparameters (alpha, beta) for ego robot and its tracks only

    Key principle: Uses neighborhood evidence but only updates ego's own entities.
    """

    def __init__(self, proximity_radius: float = 50.0):
        """
        Args:
            proximity_radius: Distance threshold for considering robots as neighbors
        """
        self.ego_graph_builder = EgoGraphBuilder(proximal_range=proximity_radius)
        self.proximity_radius = proximity_radius

    def update_trust(self, robots: List[Robot], environment=None) -> Dict[int, Dict]:
        """
        Update trust values using ego graph edge counts.

        For each ego robot, builds an ego graph and updates:
        - The ego robot's trust based on its co_detection and contradicts edges
        - Trust for ONLY tracks detected by the ego robot (not all tracks in graph)
        - Uses ALL edges from ANY agent in the ego graph for counting evidence

        Args:
            robots: List of all robots in the environment
            environment: Optional environment reference for proximity calculation

        Returns:
            Dictionary of trust updates for each robot
        """
        trust_updates = {}

        # Build ego graph for each robot and count evidence edges
        for ego_robot in robots:
            # Get proximal robots (neighbors within proximity radius)
            if environment:
                proximal_robots = environment.get_proximal_robots(ego_robot)
            else:
                # Fallback: use distance-based proximity
                proximal_robots = self._get_proximal_robots(ego_robot, robots)

            if not proximal_robots:
                # No neighbors - no update possible, but still record current state
                trust_updates[ego_robot.id] = {
                    'alpha': ego_robot.trust_alpha,
                    'beta': ego_robot.trust_beta,
                    'mean_trust': ego_robot.trust_value,
                    'num_robot_positive': 0,
                    'num_robot_negative': 0,
                    'track_trust_info': {}
                }
                continue

            # Build ego graph for this robot
            ego_graph_data = self.ego_graph_builder.build_ego_graph(ego_robot, robots)

            if ego_graph_data is None:
                # Failed to build ego graph - no update
                trust_updates[ego_robot.id] = {
                    'alpha': ego_robot.trust_alpha,
                    'beta': ego_robot.trust_beta,
                    'mean_trust': ego_robot.trust_value,
                    'num_robot_positive': 0,
                    'num_robot_negative': 0,
                    'track_trust_info': {}
                }
                continue

            # Count evidence edges and update trust for ego robot and its tracks
            robot_updates, track_updates = self._count_and_update_trust(
                ego_robot, ego_graph_data, robots
            )

            trust_updates[ego_robot.id] = {
                'alpha': ego_robot.trust_alpha,
                'beta': ego_robot.trust_beta,
                'mean_trust': ego_robot.trust_value,
                'num_robot_positive': robot_updates['positive'],
                'num_robot_negative': robot_updates['negative'],
                'track_trust_info': track_updates
            }

        return trust_updates

    def _get_proximal_robots(self, ego_robot: Robot, all_robots: List[Robot]) -> List[Robot]:
        """Get robots within proximity radius of ego robot."""
        proximal = []
        for robot in all_robots:
            if robot.id == ego_robot.id:
                continue
            distance = np.linalg.norm(ego_robot.position[:2] - robot.position[:2])
            if distance <= self.proximity_radius:
                proximal.append(robot)
        return proximal

    def _count_and_update_trust(
        self,
        ego_robot: Robot,
        ego_graph_data,
        all_robots: List[Robot]
    ) -> tuple:
        """
        Count edges in ego graph and update trust values for ego robot and its detected tracks.

        Returns:
            Tuple of (robot_updates, track_updates) where:
            - robot_updates: dict with 'positive' and 'negative' counts for ego robot
            - track_updates: dict mapping track_id to trust info
        """
        edge_index_dict = ego_graph_data.edge_index_dict

        # ===== EGO ROBOT NODE UPDATE =====
        # Positive evidence: co_detection edges
        # Negative evidence: contradicts edges

        robot_positive = 0
        robot_negative = 0

        # Count co_detection edges (positive evidence for ego robot)
        # Ego robot is always at index 0 in the ego graph
        # We count ALL co_detection edges in the ego graph where ego robot is involved
        if ('agent', 'co_detection', 'agent') in edge_index_dict:
            co_detection_edges = edge_index_dict[('agent', 'co_detection', 'agent')]
            if co_detection_edges.numel() > 0:
                # Count ALL edges involving ego robot (index 0) in this ego graph
                # This includes edges: (ego, other) and (other, ego)
                ego_edges = (co_detection_edges[0] == 0).sum() + (co_detection_edges[1] == 0).sum()
                robot_positive = int(ego_edges.item())

        # Count contradicts edges (negative evidence for ego robot)
        # We count ALL contradicts edges in the ego graph where ego robot is involved
        if ('agent', 'contradicts', 'agent') in edge_index_dict:
            contradicts_edges = edge_index_dict[('agent', 'contradicts', 'agent')]
            if contradicts_edges.numel() > 0:
                # Count ALL edges involving ego robot (index 0) in this ego graph
                # This includes edges: (ego, other) and (other, ego)
                ego_edges = (contradicts_edges[0] == 0).sum() + (contradicts_edges[1] == 0).sum()
                robot_negative = int(ego_edges.item())

        # Update ego robot's trust based on its edges
        if robot_positive > 0 or robot_negative > 0:
            ego_robot.update_trust(
                delta_alpha=float(robot_positive),
                delta_beta=float(robot_negative)
            )

        robot_updates = {
            'positive': robot_positive,
            'negative': robot_negative
        }

        # ===== TRACK NODE UPDATES =====
        # IMPORTANT: Only update tracks that the EGO ROBOT detected
        # But use ALL edges from ANY agent in the ego graph for counting evidence
        # Positive evidence: in_fov_and_observed edges
        # Negative evidence: in_fov_only edges

        track_updates = {}

        # Use the track_nodes mapping directly from the ego graph
        # This mapping is created by the EgoGraphBuilder: {track_id: node_index}
        if not hasattr(ego_graph_data, 'track_nodes'):
            return robot_updates, track_updates

        track_node_map = ego_graph_data.track_nodes  # track_id -> node_index

        # Get all tracks from the graph (fused + individual) for matching
        all_graph_tracks = []
        if hasattr(ego_graph_data, '_fused_tracks'):
            all_graph_tracks.extend(ego_graph_data._fused_tracks)
        if hasattr(ego_graph_data, '_individual_tracks'):
            all_graph_tracks.extend(ego_graph_data._individual_tracks)

        # Create object_id -> graph track mapping for easy lookup
        # NOTE: object_id is globally unique (e.g., "gt_obj_5"), while track_id is robot-local (e.g., "3_gt_obj_5")
        object_id_to_graph_track = {}
        for graph_track in all_graph_tracks:
            object_id_to_graph_track[graph_track.object_id] = graph_track

        # ONLY update tracks detected by the ego robot in the CURRENT timestep
        ego_tracks = ego_robot.get_all_current_tracks()

        for ego_track in ego_tracks:
            # Find the corresponding track in the ego graph by object_id
            # (The graph may have fused this track, so track_id might differ)
            # Match by object_id (global) not track_id (robot-local)
            graph_track = object_id_to_graph_track.get(ego_track.object_id)

            if graph_track is None:
                # Ego track not in the graph (shouldn't happen, but handle gracefully)
                continue

            # Get the node index for this track in the graph
            if graph_track.track_id not in track_node_map:
                # Track not in graph node mapping, skip
                continue

            track_node_idx = track_node_map[graph_track.track_id]
            track_positive = 0
            track_negative = 0

            # Count in_fov_and_observed edges (positive evidence)
            # We count ALL edges from ANY agent in the ego graph to this track
            if ('agent', 'in_fov_and_observed', 'track') in edge_index_dict:
                observed_edges = edge_index_dict[('agent', 'in_fov_and_observed', 'track')]
                if observed_edges.numel() > 0:
                    # Count ALL edges pointing to this track from ANY agent in the ego graph
                    # This uses all observation evidence in the local neighborhood
                    track_edges = (observed_edges[1] == track_node_idx).sum()
                    track_positive = int(track_edges.item())

            # Count in_fov_only edges (negative evidence - missed detections)
            # We count ALL edges from ANY agent in the ego graph to this track
            if ('agent', 'in_fov_only', 'track') in edge_index_dict:
                fov_only_edges = edge_index_dict[('agent', 'in_fov_only', 'track')]
                if fov_only_edges.numel() > 0:
                    # Count ALL edges pointing to this track from ANY agent in the ego graph
                    # This uses all "missed detection" evidence in the local neighborhood
                    track_edges = (fov_only_edges[1] == track_node_idx).sum()
                    track_negative = int(track_edges.item())

            # Update the EGO ROBOT'S ORIGINAL track (not the graph copy)
            if track_positive > 0 or track_negative > 0:
                ego_track.update_trust(
                    delta_alpha=float(track_positive),
                    delta_beta=float(track_negative)
                )

            track_updates[ego_track.track_id] = {
                'alpha': ego_track.trust_alpha,
                'beta': ego_track.trust_beta,
                'mean_trust': ego_track.trust_value,
                'num_positive': track_positive,
                'num_negative': track_negative,
                'object_id': ego_track.object_id
            }

        return robot_updates, track_updates

