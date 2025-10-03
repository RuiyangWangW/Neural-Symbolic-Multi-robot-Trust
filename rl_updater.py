#!/usr/bin/env python3
"""
Learnable Updater Policy

Simplified MAPPO actor/critic stack:
- Actor encodes robot/track features with small MLPs, pools to a global context, then emits Beta params per entity
- Critic pools ground-truth-aware features to score the global state
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from robot_track_classes import Robot, Track


@dataclass
class UpdateDecision:
    """Step scales for participating nodes in this ego graph"""
    robot_steps: Dict[int, float]   # robot_id -> step_scale [0,1]
    track_steps: Dict[str, float]   # track_id -> step_scale [0,1]


class SetTransformerCritic(nn.Module):
    """Simplified centralized critic that pools ground-truth-aware features"""

    def __init__(self,
                 robot_input_dim: int = 3,
                 track_input_dim: int = 3,
                 tier0_dim: int = 8,
                 hidden_dim: int = 64):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.robot_projector = nn.Sequential(
            nn.Linear(robot_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.track_projector = nn.Sequential(
            nn.Linear(track_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(tier0_dim + 2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    @staticmethod
    def _masked_mean(encoded: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute masked mean over set dimension"""
        if mask is None:
            return encoded.mean(dim=1) if encoded.size(1) > 0 else torch.zeros(encoded.size(0), encoded.size(-1), device=encoded.device)

        if mask.dim() == 2:
            mask_expanded = mask.unsqueeze(-1)
        else:
            mask_expanded = mask

        mask_expanded = mask_expanded.to(encoded.dtype)
        denom = mask_expanded.sum(dim=1).clamp(min=1e-6)
        summed = (encoded * mask_expanded).sum(dim=1)
        return summed / denom

    def forward(self,
                tier0_features: torch.Tensor,
                robot_features: torch.Tensor,
                track_features: torch.Tensor,
                robot_mask: Optional[torch.Tensor] = None,
                track_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass that averages projected robot/track features and fuses with tier-0 summary"""
        if robot_features.dim() == 2:
            robot_features = robot_features.unsqueeze(0)
        if track_features.dim() == 2:
            track_features = track_features.unsqueeze(0)
        if robot_mask is not None and robot_mask.dim() == 1:
            robot_mask = robot_mask.unsqueeze(0)
        if track_mask is not None and track_mask.dim() == 1:
            track_mask = track_mask.unsqueeze(0)

        batch_size = tier0_features.size(0)

        robot_summary = torch.zeros(batch_size, self.hidden_dim, device=tier0_features.device)
        if robot_features.size(1) > 0:
            projected_robot = self.robot_projector(robot_features)
            robot_summary = self._masked_mean(projected_robot, robot_mask)

        track_summary = torch.zeros(batch_size, self.hidden_dim, device=tier0_features.device)
        if track_features.size(1) > 0:
            projected_track = self.track_projector(track_features)
            track_summary = self._masked_mean(projected_track, track_mask)

        critic_input = torch.cat([tier0_features, robot_summary, track_summary], dim=-1)
        values = self.value_mlp(critic_input).squeeze(-1)
        return values


class SetTransformerActor(nn.Module):
    """Simplified actor that pools robot/track features and emits Beta parameters"""

    def __init__(self,
                 robot_input_dim: int = 6,
                 track_input_dim: int = 7,
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Lightweight per-entity encoders
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.track_encoder = nn.Sequential(
            nn.Linear(track_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # Global context from pooled robot/track embeddings
        self.context_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # Per-entity action heads conditioned on global context
        self.robot_action_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

        self.track_action_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

    @staticmethod
    def _masked_mean(embeddings: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if embeddings.shape[1] == 0:
            return torch.zeros(embeddings.shape[0], embeddings.shape[-1], device=embeddings.device)
        if mask is None or mask.numel() == 0:
            return embeddings.mean(dim=1)
        mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        summed = (embeddings * mask).sum(dim=1)
        return summed / denom

    @staticmethod
    def _masked_pair_mean(source_embeddings: torch.Tensor,
                          pair_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Masked mean over source embeddings for each target using adjacency mask."""
        if pair_mask is None or pair_mask.numel() == 0 or source_embeddings.size(1) == 0:
            return None

        if source_embeddings.dim() == 2:
            source_embeddings = source_embeddings.unsqueeze(0)
        if pair_mask.dim() == 2:
            pair_mask = pair_mask.unsqueeze(0)

        pair_mask = pair_mask.to(source_embeddings.dtype)

        weighted = source_embeddings.unsqueeze(1) * pair_mask.unsqueeze(-1)
        denom = pair_mask.sum(dim=2, keepdim=True).clamp(min=1e-6)
        return weighted.sum(dim=2) / denom

    def _beta_forward(self,
                      mu: torch.Tensor,
                      kappa: torch.Tensor,
                      mask: Optional[torch.Tensor],
                      actions: Optional[torch.Tensor] = None):
        """Helper to sample Beta actions or evaluate log-probs with empty-set guards"""
        if mu.numel() == 0:
            shape = mu.shape
            zero = torch.zeros(shape, dtype=mu.dtype, device=mu.device)
            entropy = torch.zeros(mu.shape[0], dtype=mu.dtype, device=mu.device)
            if actions is None:
                return zero, zero.clone(), entropy, zero.clone(), zero.clone()
            return zero.clone(), entropy

        eps = 1e-4
        mu = mu.clamp(eps, 1.0 - eps)
        alpha = mu * kappa
        beta = (1 - mu) * kappa
        dist = torch.distributions.Beta(alpha, beta)

        samples = dist.rsample() if actions is None else actions
        log_prob = dist.log_prob(samples)
        if mask is not None:
            log_prob = log_prob * mask.float()

        entropy_matrix = dist.entropy() - torch.log(kappa + 1e-8) * 0.01
        if mask is not None:
            entropy_matrix = entropy_matrix * mask.float()
        entropy = entropy_matrix.sum(dim=1)

        if actions is None:
            return samples, log_prob, entropy, alpha, beta
        return log_prob, entropy

    def forward(self,
                robot_features: torch.Tensor,
                track_features: torch.Tensor,
                robot_mask: Optional[torch.Tensor] = None,
                track_mask: Optional[torch.Tensor] = None,
                robot_track_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for action generation with defensive guards for empty sets"""

        if robot_features.dim() == 2:
            robot_features = robot_features.unsqueeze(0)
        if track_features.dim() == 2:
            track_features = track_features.unsqueeze(0)

        B = robot_features.shape[0] if robot_features.numel() > 0 else track_features.shape[0]
        if B == 0:
            B = 1
        device = robot_features.device if robot_features.numel() > 0 else track_features.device
        dtype = robot_features.dtype if robot_features.numel() > 0 else track_features.dtype

        # If no nodes at all, return empty tensors immediately
        if robot_features.shape[1] == 0 and track_features.shape[1] == 0:
            zero = torch.zeros(B, 0, dtype=dtype, device=device)
            return (zero, zero.clone()), (zero.clone(), zero.clone())

        try:
            # Encode robot features when available
            if robot_features.shape[1] > 0:
                robot_embeddings = self.robot_encoder(robot_features)
                if robot_mask is None:
                    robot_mask = torch.ones(B, robot_features.shape[1], device=device)
            else:
                robot_embeddings = torch.zeros(B, 0, self.hidden_dim, dtype=dtype, device=device)
                if robot_mask is None:
                    robot_mask = torch.zeros(B, 0, dtype=dtype, device=device)

            # Encode track features when available
            if track_features.shape[1] > 0:
                track_embeddings = self.track_encoder(track_features)
                if track_mask is None:
                    track_mask = torch.ones(B, track_features.shape[1], device=device)
            else:
                track_embeddings = torch.zeros(B, 0, self.hidden_dim, dtype=dtype, device=device)
                if track_mask is None:
                    track_mask = torch.zeros(B, 0, dtype=dtype, device=device)

            robot_mask_float = robot_mask.float() if robot_mask is not None and robot_mask.numel() > 0 else None
            track_mask_float = track_mask.float() if track_mask is not None and track_mask.numel() > 0 else None

            if robot_track_mask is not None and robot_track_mask.numel() > 0:
                if robot_track_mask.dim() == 2:
                    robot_track_mask = robot_track_mask.unsqueeze(0)
                robot_track_mask = robot_track_mask.to(dtype)
            else:
                robot_track_mask = None

            robot_summary = self._masked_mean(robot_embeddings, robot_mask_float)
            track_summary = self._masked_mean(track_embeddings, track_mask_float)

            context = torch.cat([robot_summary, track_summary], dim=-1)
            context_embed = self.context_mlp(context)

            if robot_embeddings.shape[1] > 0:
                context_expanded = context_embed.unsqueeze(1).repeat(1, robot_embeddings.shape[1], 1)
                if track_embeddings.shape[1] > 0:
                    track_context = self._masked_pair_mean(track_embeddings, robot_track_mask)
                    track_context_fallback = track_summary.unsqueeze(1).repeat(1, robot_embeddings.shape[1], 1)
                    if track_context is None:
                        track_context = track_context_fallback
                    else:
                        if robot_track_mask is not None:
                            mask_sums = robot_track_mask.sum(dim=2, keepdim=True)
                            zero_mask = mask_sums <= 0
                            track_context = torch.where(zero_mask, track_context_fallback, track_context)
                else:
                    track_context = track_summary.unsqueeze(1).repeat(1, robot_embeddings.shape[1], 1)

                robot_inputs = torch.cat([robot_embeddings, context_expanded, track_context], dim=-1)
                robot_params = self.robot_action_head(robot_inputs)
            else:
                robot_params = torch.zeros(B, 0, 2, dtype=dtype, device=device)

            if track_embeddings.shape[1] > 0:
                context_expanded = context_embed.unsqueeze(1).repeat(1, track_embeddings.shape[1], 1)
                if robot_embeddings.shape[1] > 0:
                    track_robot_mask = robot_track_mask.transpose(1, 2) if robot_track_mask is not None else None
                    robot_context = self._masked_pair_mean(robot_embeddings, track_robot_mask)
                    robot_context_fallback = robot_summary.unsqueeze(1).repeat(1, track_embeddings.shape[1], 1)
                    if robot_context is None:
                        robot_context = robot_context_fallback
                    else:
                        if track_robot_mask is not None:
                            mask_sums = track_robot_mask.sum(dim=2, keepdim=True)
                            zero_mask = mask_sums <= 0
                            robot_context = torch.where(zero_mask, robot_context_fallback, robot_context)
                else:
                    robot_context = robot_summary.unsqueeze(1).repeat(1, track_embeddings.shape[1], 1)

                track_inputs = torch.cat([track_embeddings, context_expanded, robot_context], dim=-1)
                track_params = self.track_action_head(track_inputs)
            else:
                track_params = torch.zeros(B, 0, 2, dtype=dtype, device=device)

            kappa_min = 1e-3
            kappa_max = 50.0

            robot_mu = torch.sigmoid(robot_params[..., 0])
            robot_kappa = kappa_max * torch.sigmoid(robot_params[..., 1]) + kappa_min

            track_mu = torch.sigmoid(track_params[..., 0])
            track_kappa = kappa_max * torch.sigmoid(track_params[..., 1]) + kappa_min

            return (robot_mu, robot_kappa), (track_mu, track_kappa)

        except Exception as exc:
            print("Actor forward failure")
            print("  robot_features:", robot_features.shape)
            print("  track_features:", track_features.shape)
            print("  robot_mask:", None if robot_mask is None else robot_mask.shape)
            print("  track_mask:", None if track_mask is None else track_mask.shape)
            raise

    def sample_actions(self,
                      robot_features: torch.Tensor,
                      track_features: torch.Tensor,
                      robot_mask: Optional[torch.Tensor] = None,
                      track_mask: Optional[torch.Tensor] = None,
                      robot_track_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                                               Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Sample Beta actions while safely handling empty robot/track sets."""

        (robot_mu, robot_kappa), (track_mu, track_kappa) = self.forward(
            robot_features, track_features, robot_mask, track_mask, robot_track_mask)

        robot_actions, robot_log_probs, robot_entropy, robot_alpha, robot_beta = self._beta_forward(
            robot_mu, robot_kappa, robot_mask)
        track_actions, track_log_probs, track_entropy, track_alpha, track_beta = self._beta_forward(
            track_mu, track_kappa, track_mask)

        total_entropy = robot_entropy + track_entropy

        return robot_actions, track_actions, robot_log_probs, track_log_probs, total_entropy, \
               (robot_alpha, robot_beta), (track_alpha, track_beta)

    def evaluate_actions(self,
                        robot_features: torch.Tensor,
                        track_features: torch.Tensor,
                        robot_actions: torch.Tensor,
                        track_actions: torch.Tensor,
                        robot_mask: Optional[torch.Tensor] = None,
                        track_mask: Optional[torch.Tensor] = None,
                        robot_track_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probs/entropy for Beta actions with empty-set guards."""

        (robot_mu, robot_kappa), (track_mu, track_kappa) = self.forward(
            robot_features, track_features, robot_mask, track_mask, robot_track_mask)

        robot_log_probs, robot_entropy = self._beta_forward(
            robot_mu, robot_kappa, robot_mask, actions=robot_actions)
        track_log_probs, track_entropy = self._beta_forward(
            track_mu, track_kappa, track_mask, actions=track_actions)

        total_entropy = robot_entropy + track_entropy

        return robot_log_probs, track_log_probs, total_entropy

class LearnableUpdater:
    """Wrapper for the learnable updater policy with centralized critic for MAPPO"""

    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.device = torch.device(device)

        # Initialize simplified actor and critic
        self.policy = SetTransformerActor(
            robot_input_dim=6,  # trust_mean, strength, Δtrust, agent_score, score_conf, degree
            track_input_dim=7,  # trust_mean, strength, Δtrust, maturity, track_score, score_conf, degree
            hidden_dim=128,
            dropout=0.1
        ).to(self.device)

        self.critic = SetTransformerCritic(
            robot_input_dim=3,
            track_input_dim=3,
            tier0_dim=8,
            hidden_dim=64
        ).to(self.device)

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

    def get_step_scales(self,
                       ego_robots: List[Robot],        # robots in ego graph
                       ego_tracks: List[Track],        # tracks in ego graph
                       participating_robots: List[Robot],  # robots that detected something
                       participating_tracks: List[Track],  # tracks that were detected or in FoV
                       robot_scores: Dict[int, float],
                       track_scores: Dict[str, float],
                       track_observers: Optional[Dict[str, List[int]]] = None) -> UpdateDecision:
        """
        Get step scales for participating robots and tracks in this ego graph
        """

        if not participating_robots and not participating_tracks:
            return UpdateDecision({}, {})

        with torch.no_grad():
            # Build per-node features for participating nodes only (match actor input dimensions)
            robot_features = []
            robot_ids = []
            robot_lookup = {}
            for robot in participating_robots:
                trust_mean = robot.trust_value
                strength = robot.trust_alpha + robot.trust_beta
                delta_trust = 0.0  # No previous step info in inference mode
                agent_score = robot_scores.get(robot.id, 0.5)
                score_conf = 2 * abs(agent_score - 0.5)  # Confidence as distance from 0.5
                if track_observers:
                    degree = sum(1 for observers in track_observers.values() if robot.id in observers)
                else:
                    degree = len(robot.get_current_timestep_tracks())
                robot_features.append([trust_mean, strength, delta_trust, agent_score, score_conf, degree])
                robot_ids.append(robot.id)
                robot_lookup[robot.id] = robot

            track_features = []
            track_ids = []
            track_lookup = {}
            for track in participating_tracks:
                trust_mean = track.trust_value
                strength = track.trust_alpha + track.trust_beta
                delta_trust = 0.0  # No previous step info in inference mode
                maturity = min(1.0, track.observation_count / 10.0)
                track_score = track_scores.get(track.track_id, 0.5)
                score_conf = 2 * abs(track_score - 0.5)  # Confidence as distance from 0.5
                observers = track_observers.get(track.track_id, []) if track_observers else []
                if not observers:
                    observers = [robot.id for robot in ego_robots if robot.is_in_fov(track.position)]
                degree = len(observers)
                track_features.append([trust_mean, strength, delta_trust, maturity, track_score, score_conf, degree])
                track_ids.append(track.track_id)
                track_lookup[track.track_id] = track

            robot_features_tensor = torch.tensor(robot_features, dtype=torch.float32, device=self.device) if robot_features else torch.zeros(0, 6, device=self.device)
            track_features_tensor = torch.tensor(track_features, dtype=torch.float32, device=self.device) if track_features else torch.zeros(0, 7, device=self.device)

            # Build robot-to-track adjacency mask based on detections and field-of-view
            robot_track_mask = None
            if robot_features and track_features:
                mask = torch.zeros((len(robot_ids), len(track_ids)), dtype=torch.float32, device=self.device)
                for r_idx, robot_id in enumerate(robot_ids):
                    robot = robot_lookup[robot_id]
                    detected_ids = {t.track_id for t in robot.get_current_timestep_tracks()}
                    for t_idx, track_id in enumerate(track_ids):
                        track = track_lookup[track_id]
                        in_detection = track_id in detected_ids
                        in_fov = robot.is_in_fov(track.position)
                        if in_detection or in_fov:
                            mask[r_idx, t_idx] = 1.0
                robot_track_mask = mask

            # Get (μ, κ) parameters from policy (ego-centric)
            # Ensure proper batching for SetTransformer (needs 3D tensors)
            robot_features_batched = robot_features_tensor.unsqueeze(0) if robot_features_tensor.dim() == 2 else robot_features_tensor
            track_features_batched = track_features_tensor.unsqueeze(0) if track_features_tensor.dim() == 2 else track_features_tensor

            (robot_mu, robot_kappa), (track_mu, track_kappa) = self.policy(
                robot_features_batched,
                track_features_batched,
                robot_track_mask=robot_track_mask.unsqueeze(0) if robot_track_mask is not None else None
            )

            # Use μ directly as action (deterministic inference)
            robot_steps = {}
            if robot_mu.size(1) > 0:  # Check actual number of robots (batch=0, robots=1)
                robot_mus = robot_mu[0, :]  # [N_R] - remove batch dimension
                # μ is already the mean of the Beta distribution, use directly
                for i, robot_id in enumerate(robot_ids):
                    robot_steps[robot_id] = float(robot_mus[i])

            track_steps = {}
            if track_mu.size(1) > 0:  # Check actual number of tracks (batch=0, tracks=1)
                track_mus = track_mu[0, :]  # [N_T] - remove batch dimension
                # μ is already the mean of the Beta distribution, use directly
                for i, track_id in enumerate(track_ids):
                    track_steps[track_id] = float(track_mus[i])

            return UpdateDecision(robot_steps, track_steps)
