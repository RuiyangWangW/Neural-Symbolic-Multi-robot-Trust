#!/usr/bin/env python3
"""
Simple Evidence Extractor

Gets agent_score and track_score from the GNN for nodes in the ego graph.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from supervised_trust_gnn import SupervisedTrustPredictor
from robot_track_classes import Robot, Track


@dataclass
class NodeScores:
    """GNN scores for nodes in ego graph"""
    agent_scores: Dict[int, float]  # robot_id -> score [0,1]
    track_scores: Dict[str, float] # track_id -> score [0,1]


class EvidenceExtractor:
    """Gets GNN scores for ego graph nodes"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        try:
            self.predictor = SupervisedTrustPredictor(model_path=model_path, device=device)
            self.available = True
        except Exception as e:
            print(f"Failed to load GNN: {e}")
            self.predictor = None
            self.available = False

    def get_scores(self, ego_robot: Robot, all_robots: List[Robot]) -> NodeScores:
        """Get GNN scores for ego graph nodes"""
        if not self.available:
            return NodeScores({}, {})

        try:
            result = self.predictor.predict_from_robots_tracks(ego_robot, all_robots)
            predictions = result['predictions']
            graph_data = result['graph_data']

            # Get trust_scores arrays
            agent_scores_array = predictions.get('agent', {}).get('trust_scores', np.array([]))
            track_scores_array = predictions.get('track', {}).get('trust_scores', np.array([]))

            # Get node mappings from graph_data
            agent_nodes = getattr(graph_data, 'agent_nodes', {})
            track_nodes = getattr(graph_data, 'track_nodes', {})

            # Convert to dict with clipping to [0,1]
            agent_scores = {}
            for robot_id, idx in agent_nodes.items():
                if idx < len(agent_scores_array):
                    agent_scores[robot_id] = float(np.clip(agent_scores_array[idx], 0.0, 1.0))
                else:
                    agent_scores[robot_id] = 0.5

            track_scores = {}
            for track_id, idx in track_nodes.items():
                if idx < len(track_scores_array):
                    track_scores[track_id] = float(np.clip(track_scores_array[idx], 0.0, 1.0))
                else:
                    track_scores[track_id] = 0.5

            return NodeScores(agent_scores, track_scores)

        except Exception as e:
            print(f"Error getting scores: {e}")
            return NodeScores({}, {})