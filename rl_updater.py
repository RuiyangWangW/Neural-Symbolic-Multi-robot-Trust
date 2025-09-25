#!/usr/bin/env python3
"""
Learnable Updater Policy

Follows the exact framework:
- Inputs: permutation-invariant summaries of current trust + simple aggregates of today's GNN scores
- Outputs: step scale [0,1] per participating robot and track
- No raw graph, only summaries
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from robot_track_classes import Robot, Track


@dataclass
class EgoGraphSummary:
    """Permutation-invariant summary for one ego graph"""
    # Current trust summaries (for nodes in this ego graph only)
    robot_trust_mean: float      # Mean of robot trust means in ego graph
    robot_trust_std: float       # Std of robot trust means in ego graph
    robot_trust_count: int       # Number of robots in ego graph

    track_trust_mean: float      # Mean of track trust means in ego graph
    track_trust_std: float       # Std of track trust means in ego graph
    track_trust_count: int       # Number of tracks in ego graph

    # Simple aggregates of today's GNN scores
    robot_score_mean: float      # Mean of today's robot scores
    robot_score_std: float       # Std of today's robot scores
    track_score_mean: float      # Mean of today's track scores
    track_score_std: float       # Std of today's track scores


@dataclass
class UpdateDecision:
    """Step scales for participating nodes in this ego graph"""
    robot_steps: Dict[int, float]   # robot_id -> step_scale [0,1]
    track_steps: Dict[str, float]   # track_id -> step_scale [0,1]


class UpdaterPolicy(nn.Module):
    """
    Learnable updater policy

    Input: EgoGraphSummary (permutation-invariant, no raw graph)
    Output: Step scales [0,1] for each participating robot/track
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Input: 10 features from EgoGraphSummary
        input_dim = 10

        self.summary_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Per-node features for step scale prediction
        # Robot: [current_trust_mean, current_trust_confidence, agent_score]
        # Track: [current_trust_mean, current_trust_confidence, track_score, maturity]

        self.robot_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 3, hidden_dim // 4),  # summary + robot features
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output [0,1]
        )

        self.track_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 4, hidden_dim // 4),  # summary + track features
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output [0,1]
        )

    def forward(self,
                summary: torch.Tensor,           # [1, 10] ego graph summary
                robot_features: torch.Tensor,   # [num_robots, 3]
                track_features: torch.Tensor    # [num_tracks, 4]
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Encode summary
        summary_encoded = self.summary_encoder(summary)  # [1, hidden_dim//2]

        # Robot step scales
        if robot_features.size(0) > 0:
            robot_input = torch.cat([
                summary_encoded.expand(robot_features.size(0), -1),
                robot_features
            ], dim=1)
            robot_steps = self.robot_head(robot_input).squeeze(-1)  # [num_robots]
        else:
            robot_steps = torch.zeros(0, device=summary.device)

        # Track step scales
        if track_features.size(0) > 0:
            track_input = torch.cat([
                summary_encoded.expand(track_features.size(0), -1),
                track_features
            ], dim=1)
            track_steps = self.track_head(track_input).squeeze(-1)  # [num_tracks]
        else:
            track_steps = torch.zeros(0, device=summary.device)

        return robot_steps, track_steps


class LearnableUpdater:
    """Wrapper for the learnable updater policy"""

    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.policy = UpdaterPolicy().to(self.device)

        if model_path:
            try:
                self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded updater policy from {model_path}")
            except Exception as e:
                print(f"Failed to load updater policy: {e}")

        self.policy.eval()

    def create_ego_graph_summary(self,
                                ego_robots: List[Robot],      # robots in ego graph
                                ego_tracks: List[Track],      # tracks in ego graph
                                robot_scores: Dict[int, float],
                                track_scores: Dict[str, float]) -> EgoGraphSummary:
        """Create permutation-invariant summary of ego graph"""

        # Current trust summaries
        if ego_robots:
            robot_means = [robot.trust_value for robot in ego_robots]
            robot_trust_mean = np.mean(robot_means)
            robot_trust_std = np.std(robot_means) if len(robot_means) > 1 else 0.0
        else:
            robot_trust_mean = 0.5
            robot_trust_std = 0.0

        if ego_tracks:
            track_means = [track.trust_value for track in ego_tracks]
            track_trust_mean = np.mean(track_means)
            track_trust_std = np.std(track_means) if len(track_means) > 1 else 0.0
        else:
            track_trust_mean = 0.5
            track_trust_std = 0.0

        # Today's GNN score aggregates
        ego_robot_scores = [robot_scores.get(robot.id, 0.5) for robot in ego_robots]
        ego_track_scores = [track_scores.get(track.track_id, 0.5) for track in ego_tracks]

        if ego_robot_scores:
            robot_score_mean = np.mean(ego_robot_scores)
            robot_score_std = np.std(ego_robot_scores) if len(ego_robot_scores) > 1 else 0.0
        else:
            robot_score_mean = 0.5
            robot_score_std = 0.0

        if ego_track_scores:
            track_score_mean = np.mean(ego_track_scores)
            track_score_std = np.std(ego_track_scores) if len(ego_track_scores) > 1 else 0.0
        else:
            track_score_mean = 0.5
            track_score_std = 0.0

        return EgoGraphSummary(
            robot_trust_mean=robot_trust_mean,
            robot_trust_std=robot_trust_std,
            robot_trust_count=len(ego_robots),
            track_trust_mean=track_trust_mean,
            track_trust_std=track_trust_std,
            track_trust_count=len(ego_tracks),
            robot_score_mean=robot_score_mean,
            robot_score_std=robot_score_std,
            track_score_mean=track_score_mean,
            track_score_std=track_score_std
        )

    def get_step_scales(self,
                       ego_robots: List[Robot],        # robots in ego graph
                       ego_tracks: List[Track],        # tracks in ego graph
                       participating_robots: List[Robot],  # robots that detected something
                       participating_tracks: List[Track],  # tracks that were detected
                       robot_scores: Dict[int, float],
                       track_scores: Dict[str, float]) -> UpdateDecision:
        """
        Get step scales for participating robots and tracks in this ego graph
        """

        if not participating_robots and not participating_tracks:
            return UpdateDecision({}, {})

        # Create permutation-invariant summary
        summary = self.create_ego_graph_summary(ego_robots, ego_tracks, robot_scores, track_scores)

        with torch.no_grad():
            # Convert summary to tensor
            summary_tensor = torch.tensor([
                summary.robot_trust_mean, summary.robot_trust_std, summary.robot_trust_count,
                summary.track_trust_mean, summary.track_trust_std, summary.track_trust_count,
                summary.robot_score_mean, summary.robot_score_std,
                summary.track_score_mean, summary.track_score_std
            ], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 10]

            # Build per-node features for participating nodes only
            robot_features = []
            robot_ids = []
            for robot in participating_robots:
                trust_mean = robot.trust_value
                trust_conf = min(1.0, (robot.trust_alpha + robot.trust_beta) / 20.0)
                robot_score = robot_scores.get(robot.id, 0.5)
                robot_features.append([trust_mean, trust_conf, robot_score])
                robot_ids.append(robot.id)

            track_features = []
            track_ids = []
            for track in participating_tracks:
                trust_mean = track.trust_value
                trust_conf = min(1.0, (track.trust_alpha + track.trust_beta) / 20.0)
                track_score = track_scores.get(track.track_id, 0.5)
                maturity = min(1.0, track.observation_count / 10.0)
                track_features.append([trust_mean, trust_conf, track_score, maturity])
                track_ids.append(track.track_id)

            robot_features_tensor = torch.tensor(robot_features, dtype=torch.float32, device=self.device) if robot_features else torch.zeros(0, 3, device=self.device)
            track_features_tensor = torch.tensor(track_features, dtype=torch.float32, device=self.device) if track_features else torch.zeros(0, 4, device=self.device)

            # Get step scales
            robot_step_scales, track_step_scales = self.policy(summary_tensor, robot_features_tensor, track_features_tensor)

            # Convert to dict
            robot_steps = {}
            for i, robot_id in enumerate(robot_ids):
                robot_steps[robot_id] = float(robot_step_scales[i])

            track_steps = {}
            for i, track_id in enumerate(track_ids):
                track_steps[track_id] = float(track_step_scales[i])

            return UpdateDecision(robot_steps, track_steps)