#!/usr/bin/env python3
"""
Evidence Extraction for Trust Updates

Wraps the supervised GNN to extract evidence scores for robots and tracks.
"""

from typing import Dict, List
from dataclasses import dataclass
from supervised_trust_gnn import SupervisedTrustPredictor


@dataclass
class NodeScores:
    """Container for evidence scores from GNN"""
    agent_scores: Dict[int, float]  # robot_id -> evidence score
    track_scores: Dict[str, float]  # track_id -> evidence score


class EvidenceExtractor:
    """
    Extracts evidence scores from supervised GNN for trust updates
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize evidence extractor

        Args:
            model_path: Path to trained supervised GNN model
            device: Device for inference ('cpu' or 'cuda')
        """
        self.predictor = SupervisedTrustPredictor(model_path, device=device)
        self.available = self.predictor.model is not None

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
        if not self.available:
            # Return empty scores if model not available
            return NodeScores(agent_scores={}, track_scores={})

        # Get predictions from supervised GNN
        result = self.predictor.predict_from_robots_tracks(ego_robot, all_robots)

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
