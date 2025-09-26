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
import torch.distributions as dist
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


class SetEncoder(nn.Module):
    """
    Simple set encoder using MLP + mean/max pooling
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim] set of N entities (may be padded)
            mask: [N] binary mask (1 for real entities, 0 for padding)
        Returns:
            [output_dim] pooled representation
        """
        if x.size(0) == 0:
            return torch.zeros(self.mlp[-1].out_features, device=x.device)

        # Apply MLP to each entity
        encoded = self.mlp(x)  # [N, output_dim]

        if mask is not None:
            # Apply mask to ignore padded entities
            mask = mask.unsqueeze(-1)  # [N, 1]
            encoded = encoded * mask  # Zero out padded entities

            # Masked pooling
            valid_count = mask.sum(dim=0, keepdim=True).clamp(min=1)  # [1, 1]
            mean_pool = encoded.sum(dim=0) / valid_count.squeeze()  # [output_dim]

            # For max pooling, set padded positions to very negative values
            masked_encoded = encoded + (1 - mask) * (-1e9)
            max_pool = masked_encoded.max(dim=0)[0]  # [output_dim]
        else:
            # No masking - use all entities
            mean_pool = encoded.mean(dim=0)  # [output_dim]
            max_pool = encoded.max(dim=0)[0]  # [output_dim]

        # Combine mean and max (could also just use mean)
        return (mean_pool + max_pool) / 2  # [output_dim]


class CentralizedCritic(nn.Module):
    """
    Enhanced centralized critic with Tier 0 + Tier 2 features
    """

    def __init__(self, tier0_dim: int = 44, hidden: int = 128):
        super().__init__()

        # Tier 2 set encoders
        self.robot_encoder = SetEncoder(input_dim=6, hidden_dim=64, output_dim=32)  # 6 -> 32
        self.track_encoder = SetEncoder(input_dim=7, hidden_dim=64, output_dim=32)  # 7 -> 32

        # Combined input: Tier 0 (34) + robot_emb (32) + track_emb (32) = 98 dims
        total_dim = tier0_dim + 32 + 32  # 98 with tier0_dim=34

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, tier0_features: torch.Tensor,
                robot_features: torch.Tensor,
                track_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            tier0_features: [B, 34] Tier 0 global summary
            robot_features: [B, N_R, 6] per-robot features
            track_features: [B, N_T, 7] per-track features
        Returns:
            scalar value: [B]
        """
        batch_size = tier0_features.size(0)

        # Encode sets (handle batch dimension)
        robot_embeddings = []
        track_embeddings = []

        for i in range(batch_size):
            robot_emb = self.robot_encoder(robot_features[i])  # [32]
            track_emb = self.track_encoder(track_features[i])  # [32]
            robot_embeddings.append(robot_emb)
            track_embeddings.append(track_emb)

        robot_embeddings = torch.stack(robot_embeddings)  # [B, 32]
        track_embeddings = torch.stack(track_embeddings)  # [B, 32]

        # Concatenate all features
        combined = torch.cat([tier0_features, robot_embeddings, track_embeddings], dim=1)  # [B, 108]

        # Final MLP
        return self.mlp(combined).squeeze(-1)


class UpdaterPolicy(nn.Module):
    """
    Learnable updater policy with shared architecture for MAPPO

    Input: EgoGraphSummary (permutation-invariant, no raw graph)
    Output: Step scales [0,1] for each participating robot/track
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Input: 10 features from EgoGraphSummary
        input_dim = 10

        # Shared encoder - used by both actor and critic
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

        # Robot head outputs Beta parameters (alpha, beta)
        self.robot_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 3, hidden_dim // 4),  # summary + robot features
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2)  # Output [alpha, beta] for Beta distribution
        )

        # Track head outputs Beta parameters (alpha, beta)
        self.track_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 4, hidden_dim // 4),  # summary + track features
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2)  # Output [alpha, beta] for Beta distribution
        )

        # Value head for PPO - shares the encoder with actor
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # Takes summary encoding
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)  # State value estimation
        )

    def forward(self,
                summary: torch.Tensor,           # [1, D_local] ego graph summary
                robot_features: torch.Tensor,   # [N_R, F_R] participating robots
                track_features: torch.Tensor    # [N_T, F_T] participating tracks
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get Beta distribution parameters

        Returns:
            robot_params: [N_R, 2] - (alpha, beta) for each robot
            track_params: [N_T, 2] - (alpha, beta) for each track
        """

        # Encode summary
        summary_encoded = self.summary_encoder(summary)  # [1, hidden_dim//2]

        # Robot Beta parameters
        if robot_features.size(0) > 0:
            robot_input = torch.cat([
                summary_encoded.expand(robot_features.size(0), -1),
                robot_features
            ], dim=1)
            robot_params = self.robot_head(robot_input)  # [N_R, 2]
            # Apply softplus + epsilon to ensure alpha, beta > 0
            robot_params = torch.nn.functional.softplus(robot_params) + 1e-3
        else:
            robot_params = torch.zeros(0, 2, device=summary.device)

        # Track Beta parameters
        if track_features.size(0) > 0:
            track_input = torch.cat([
                summary_encoded.expand(track_features.size(0), -1),
                track_features
            ], dim=1)
            track_params = self.track_head(track_input)  # [N_T, 2]
            # Apply softplus + epsilon to ensure alpha, beta > 0
            track_params = torch.nn.functional.softplus(track_params) + 1e-3
        else:
            track_params = torch.zeros(0, 2, device=summary.device)

        return robot_params, track_params

    def get_value(self, summary: torch.Tensor) -> torch.Tensor:
        """Get state value estimate for PPO"""
        summary_encoded = self.summary_encoder(summary)
        value = self.value_head(summary_encoded)
        return value.squeeze(-1)  # Remove last dimension

    def sample_actions(self,
                      summary: torch.Tensor,           # [1, D_local]
                      robot_features: torch.Tensor,   # [N_R, F_R]
                      track_features: torch.Tensor    # [N_T, F_T]
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from Beta distributions

        Returns:
            a_R: [N_R] step scales for robots ∈ (0,1)
            a_T: [N_T] step scales for tracks ∈ (0,1)
            logp_R: [N_R] per-entity log probabilities
            logp_T: [N_T] per-entity log probabilities
            entropy_scalar: scalar total entropy
        """
        robot_params, track_params = self.forward(summary, robot_features, track_features)

        # Sample from Beta distributions
        robot_actions = torch.zeros(robot_params.size(0), device=summary.device)
        robot_log_probs = torch.zeros(robot_params.size(0), device=summary.device)
        robot_entropy = 0.0

        if robot_params.size(0) > 0:
            robot_alphas = robot_params[:, 0]  # [N_R]
            robot_betas = robot_params[:, 1]   # [N_R]
            robot_dists = dist.Beta(robot_alphas, robot_betas)
            robot_actions = robot_dists.sample()  # [N_R]
            # Clamp actions to [0, 1] for numerical stability
            robot_actions = torch.clamp(robot_actions, 0.0, 1.0)
            robot_log_probs = robot_dists.log_prob(robot_actions)  # [N_R]
            robot_entropy = robot_dists.entropy().sum()  # scalar

        track_actions = torch.zeros(track_params.size(0), device=summary.device)
        track_log_probs = torch.zeros(track_params.size(0), device=summary.device)
        track_entropy = 0.0

        if track_params.size(0) > 0:
            track_alphas = track_params[:, 0]  # [N_T]
            track_betas = track_params[:, 1]   # [N_T]
            track_dists = dist.Beta(track_alphas, track_betas)
            track_actions = track_dists.sample()  # [N_T]
            # Clamp actions to [0, 1] for numerical stability
            track_actions = torch.clamp(track_actions, 0.0, 1.0)
            track_log_probs = track_dists.log_prob(track_actions)  # [N_T]
            track_entropy = track_dists.entropy().sum()  # scalar

        entropy_scalar = robot_entropy + track_entropy

        return robot_actions, track_actions, robot_log_probs, track_log_probs, entropy_scalar

    def evaluate_actions(self,
                        summary: torch.Tensor,           # [1, D_local]
                        robot_features: torch.Tensor,   # [N_R, F_R]
                        track_features: torch.Tensor,   # [N_T, F_T]
                        robot_actions: torch.Tensor,    # [N_R] - a_R
                        track_actions: torch.Tensor     # [N_T] - a_T
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate specific actions for PPO update

        Returns:
            logp_R_new: [N_R] per-entity log probabilities
            logp_T_new: [N_T] per-entity log probabilities
            entropy_new: scalar total entropy
            value: scalar state value
            total_log_prob: scalar sum of all log probs (for compatibility)
        """

        # Forward pass to get current Beta distribution parameters
        robot_params, track_params = self.forward(summary, robot_features, track_features)

        # Evaluate robot actions
        robot_log_probs = torch.zeros(robot_actions.size(0), device=summary.device)
        robot_entropy = 0.0

        if robot_params.size(0) > 0 and robot_actions.size(0) > 0:
            robot_alphas = robot_params[:, 0]  # [N_R]
            robot_betas = robot_params[:, 1]   # [N_R]
            robot_dists = dist.Beta(robot_alphas, robot_betas)
            # Clamp actions to valid Beta range (0, 1) exclusive
            robot_actions_clamped = torch.clamp(robot_actions, 1e-6, 1 - 1e-6)
            robot_log_probs = robot_dists.log_prob(robot_actions_clamped)  # [N_R]
            robot_entropy = robot_dists.entropy().sum()  # scalar

        # Evaluate track actions
        track_log_probs = torch.zeros(track_actions.size(0), device=summary.device)
        track_entropy = 0.0

        if track_params.size(0) > 0 and track_actions.size(0) > 0:
            track_alphas = track_params[:, 0]  # [N_T]
            track_betas = track_params[:, 1]   # [N_T]
            track_dists = dist.Beta(track_alphas, track_betas)
            # Clamp actions to valid Beta range (0, 1) exclusive
            track_actions_clamped = torch.clamp(track_actions, 1e-6, 1 - 1e-6)
            track_log_probs = track_dists.log_prob(track_actions_clamped)  # [N_T]
            track_entropy = track_dists.entropy().sum()  # scalar

        # Total quantities
        total_log_prob = robot_log_probs.sum() + track_log_probs.sum()
        total_entropy = robot_entropy + track_entropy

        # Get value estimate
        value = self.get_value(summary)

        return robot_log_probs, track_log_probs, total_entropy, value, total_log_prob


class LearnableUpdater:
    """Wrapper for the learnable updater policy with centralized critic for MAPPO"""

    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.policy = UpdaterPolicy().to(self.device)
        self.critic = CentralizedCritic(tier0_dim=34, hidden=128).to(self.device)

        if model_path:
            try:
                self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded updater policy from {model_path}")
            except Exception as e:
                print(f"Failed to load updater policy: {e}")

        # Always initialize a fresh critic for training
        print("Initialized fresh centralized critic for training")

        self.policy.eval()
        self.critic.eval()

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

            # Get Beta distribution parameters from policy
            robot_params, track_params = self.policy(summary_tensor, robot_features_tensor, track_features_tensor)

            # Sample step scales from Beta distributions (deterministic inference)
            robot_steps = {}
            if robot_params.size(0) > 0:
                robot_alphas = robot_params[:, 0]  # [N_R]
                robot_betas = robot_params[:, 1]   # [N_R]
                # Use the mean of Beta distribution for deterministic inference
                robot_means = robot_alphas / (robot_alphas + robot_betas)  # [N_R]
                for i, robot_id in enumerate(robot_ids):
                    robot_steps[robot_id] = float(robot_means[i])

            track_steps = {}
            if track_params.size(0) > 0:
                track_alphas = track_params[:, 0]  # [N_T]
                track_betas = track_params[:, 1]   # [N_T]
                # Use the mean of Beta distribution for deterministic inference
                track_means = track_alphas / (track_alphas + track_betas)  # [N_T]
                for i, track_id in enumerate(track_ids):
                    track_steps[track_id] = float(track_means[i])

            return UpdateDecision(robot_steps, track_steps)