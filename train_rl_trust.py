#!/usr/bin/env python3
"""
PPO Training for RL Trust System

Trains the updater policy using PPO on scenarios from rl_scenario_generator.py
Follows the exact framework with proper reward computation.
"""

import torch
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass

from robot_track_classes import Robot
from rl_trust_system import RLTrustSystem
# rl_updater imports are accessed through RLTrustSystem
from rl_scenario_generator import RLScenarioGenerator
from rl_updater import UpdateDecision


@dataclass
class TrainingConfig:
    """Training configuration following the framework defaults"""
    # PPO hyperparameters (safe starting point from blueprint)
    lr: float = 3e-4  # Adam lr for both actor & critic
    gamma: float = 0.99  # Standard discount factor
    gae_lambda: float = 0.95  # Standard GAE lambda
    clip_epsilon: float = 0.2  # PPO clipping epsilon
    value_loss_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.005  # Entropy coefficient
    max_grad_norm: float = 0.5  # Gradient clipping (0.5-1.0 range)

    # Fixed padding limits (to prevent tensor size mismatches)
    max_robots: int = 50  # Maximum number of robots to pad to
    max_tracks: int = 100  # Maximum number of tracks to pad to

    # Training schedule (safe starting point)
    num_episodes: int = 5000  # Total episodes for training
    steps_per_episode: int = 100
    update_every_episodes: int = 10  # How often to run PPO updates
    ppo_epochs: int = 4  # PPO epochs per update
    batch_size: int = 128  # Ego batch size (64-256 range, using middle value)
    
    # Trust system defaults from framework
    step_size: float = 0.25
    strength_cap: float = 50.0

    # Thresholds for classification
    robot_negative_threshold: float = 0.30  # Below this = adversarial
    robot_positive_threshold: float = 0.70  # Above this = legitimate
    track_negative_threshold: float = 0.30  # Below this = false track
    track_positive_threshold: float = 0.70  # Above this = legitimate track

    # Confidence and cross-weight floors
    c_min: float = 0.2
    rho_min: float = 0.2

    # Training options
    reward_scale: float = 1.0  # Let advantage normalization handle scaling
    clip_rewards: bool = True  # Clip rewards for stability
    reward_clip_range: float = 10.0  # Reward clipping range

    # Advantage processing (blueprint options)
    divide_advantage_by_ego_count: bool = True  # Divide by ego count for balanced gradient mass
    standardize_advantages: bool = True  # Advantage standardization over ego samples

    # Reward wiring (blueprint specification)
    classification_bonus_multiplier: float = 2.0  # Small multiplier (1-3×) for final step classification bonus


class PPOTrainer:
    """PPO trainer for the updater policy"""

    def __init__(self, config: TrainingConfig, supervised_model_path: str, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)

        # Initialize components
        self.scenario_generator = RLScenarioGenerator(curriculum_learning=True)

        # Create trust system once (reuse across episodes)
        self.trust_system = RLTrustSystem(
            evidence_model_path=supervised_model_path,
            updater_model_path=None,
            device=str(self.device),
            rho_min=config.rho_min,
            c_min=config.c_min,
            step_size=config.step_size,
            strength_cap=config.strength_cap
        )

        # Use the trust system's updater directly
        self.updater = self.trust_system.updater

        # Use the centralized critic from the updater
        self.critic = self.updater.critic

        # Optimizer for both actor and critic
        actor_params = list(self.updater.policy.parameters())
        critic_params = list(self.critic.parameters())
        self.optimizer = optim.Adam(actor_params + critic_params, lr=config.lr)

        # Add learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=50
        )

        # Training state
        self.episode_count = 0
        self.best_reward = float('-inf')

        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.ppo_losses = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_values = []
        self.total_updates = 0
        self.trajectory_counts = []

        # Episode state tracking (reset at episode start to prevent leakage)
        self.prev_robot_trusts = {}   # Dict[int, float] - previous step robot trust values
        self.prev_track_trusts = {}   # Dict[str, float] - previous step track trust values
        self.prev_step_reward = 0.0   # float - previous step reward

        # Step-level storage (for critic/GAE)
        self.step_S = []           # List[Tensor[D_global]] - Global state tensors S_t
        self.step_V = []           # List[Tensor[1]] - Value estimates V_t
        self.step_R = []           # List[float] - Rewards R_t
        self.step_done = []        # List[bool] - Done flags done_t
        self.step_to_item_indices = {}  # Dict[int, List[int]] - map step t -> ego item indices

        # Ego-level storage (for actors)
        self.ego_summary = []      # List[Tensor[D_local]] - Ego graph summaries
        self.ego_R_feats = []      # List[Tensor[N_R, F_R]] - Robot features
        self.ego_T_feats = []      # List[Tensor[N_T, F_T]] - Track features
        self.ego_a_R = []          # List[Tensor[N_R]] - Robot actions
        self.ego_a_T = []          # List[Tensor[N_T]] - Track actions
        self.ego_logp_R = []       # List[Tensor[N_R]] - Robot log probs (old)
        self.ego_logp_T = []       # List[Tensor[N_T]] - Track log probs (old)
        self.ego_entropy = []      # List[float] - Entropy values
        self.ego_step_idx = []     # List[int] - Step indices for broadcasting A_t
        self.ego_robot_mask = []   # List[Tensor[N_R]] - Robot masks
        self.ego_track_mask = []   # List[Tensor[N_T]] - Track masks


    def compute_step_reward(self, all_robots: List[Robot], ground_truth: Dict) -> float:
        """
        Compute step reward based on how close trust values are to ground truth targets
        - Adversarial robots should have trust → 0
        - Legitimate robots should have trust → 1
        - False tracks should have trust → 0
        - True tracks should have trust → 1
        """
        total_reward = 0.0
        entity_count = 0

        # Ground truth
        true_adversarial = set(ground_truth.get('adversarial_agents', []))
        true_false_tracks = set(ground_truth.get('false_tracks', []))

        # Robot trust alignment rewards (simpler reward computation)
        for robot in all_robots:
            entity_count += 1
            if robot.id in true_adversarial:
                # Adversarial robot: reward for low trust (closer to 0 is better)
                total_reward += (1.0 - robot.trust_value)  # Max reward 1.0 when trust = 0
            else:
                # Legitimate robot: reward for high trust (closer to 1 is better)
                total_reward += robot.trust_value  # Max reward 1.0 when trust = 1

        # Track trust alignment rewards
        all_track_ids = set()
        for robot in all_robots:
            for track in robot.get_current_timestep_tracks():
                if track.track_id not in all_track_ids:  # Avoid double counting
                    all_track_ids.add(track.track_id)
                    entity_count += 1

                    if track.track_id in true_false_tracks:
                        # False track: reward for low trust
                        total_reward += (1.0 - track.trust_value)
                    else:
                        # True track: reward for high trust
                        total_reward += track.trust_value

        # Normalize by entity count to keep rewards manageable
        return total_reward / max(entity_count, 1)

    def compute_episode_end_reward(self, all_robots: List[Robot], ground_truth: Dict) -> float:
        """
        Compute episode-end classification reward using dual thresholds
        - trust > positive_threshold = legitimate (positive class)
        - trust < negative_threshold = adversarial/false (negative class)
        - negative_threshold <= trust <= positive_threshold = uncertain (penalized)
        """
        # Ground truth sets
        true_adversarial = set(ground_truth.get('adversarial_agents', []))
        true_false_tracks = set(ground_truth.get('false_tracks', []))

        all_robot_ids = set(robot.id for robot in all_robots)
        true_legitimate = all_robot_ids - true_adversarial

        all_track_ids = set()
        for robot in all_robots:
            for track in robot.get_current_timestep_tracks():
                all_track_ids.add(track.track_id)
        true_legitimate_tracks = all_track_ids - true_false_tracks

        # Classify robots using dual thresholds
        predicted_legitimate_robots = set()
        predicted_adversarial_robots = set()
        uncertain_robots = set()

        for robot in all_robots:
            if robot.trust_value >= self.config.robot_positive_threshold:
                predicted_legitimate_robots.add(robot.id)
            elif robot.trust_value <= self.config.robot_negative_threshold:
                predicted_adversarial_robots.add(robot.id)
            else:
                uncertain_robots.add(robot.id)

        # Classify tracks using dual thresholds
        predicted_legitimate_tracks = set()
        predicted_false_tracks = set()
        uncertain_tracks = set()

        for robot in all_robots:
            for track in robot.get_current_timestep_tracks():
                if track.trust_value >= self.config.track_positive_threshold:
                    predicted_legitimate_tracks.add(track.track_id)
                elif track.trust_value <= self.config.track_negative_threshold:
                    predicted_false_tracks.add(track.track_id)
                else:
                    uncertain_tracks.add(track.track_id)

        # Calculate classification metrics for robots
        robot_tp = len(true_legitimate & predicted_legitimate_robots)
        robot_fp = len(predicted_legitimate_robots & true_adversarial)  # Adversarial predicted as legitimate
        robot_fn = len(true_legitimate & predicted_adversarial_robots)  # Legitimate predicted as adversarial

        # Calculate classification metrics for tracks
        track_tp = len(true_legitimate_tracks & predicted_legitimate_tracks)
        track_fp = len(predicted_legitimate_tracks & true_false_tracks)  # False predicted as legitimate
        track_fn = len(true_legitimate_tracks & predicted_false_tracks)  # Legitimate predicted as false

        # Penalty for uncertain classifications
        robot_uncertainty_penalty = len(uncertain_robots) * 0.5  # Each uncertain robot costs 0.5
        track_uncertainty_penalty = len(uncertain_tracks) * 0.5  # Each uncertain track costs 0.5

        # Calculate F1 scores for legitimate classification
        if len(true_legitimate) == 0:
            robot_f1 = 1.0 if len(predicted_legitimate_robots) == 0 else 0.0
        else:
            robot_precision = robot_tp / (robot_tp + robot_fp) if (robot_tp + robot_fp) > 0 else 1.0
            robot_recall = robot_tp / (robot_tp + robot_fn) if (robot_tp + robot_fn) > 0 else 0.0
            robot_f1 = 2 * robot_precision * robot_recall / (robot_precision + robot_recall) if (robot_precision + robot_recall) > 0 else 0.0

        if len(true_legitimate_tracks) == 0:
            track_f1 = 1.0 if len(predicted_legitimate_tracks) == 0 else 0.0
        else:
            track_precision = track_tp / (track_tp + track_fp) if (track_tp + track_fp) > 0 else 1.0
            track_recall = track_tp / (track_tp + track_fn) if (track_tp + track_fn) > 0 else 0.0
            track_f1 = 2 * track_precision * track_recall / (track_precision + track_recall) if (track_precision + track_recall) > 0 else 0.0

        # Combined reward with uncertainty penalty
        base_reward = (robot_f1 + track_f1) / 2.0
        total_entities = len(all_robot_ids) + len(all_track_ids)
        uncertainty_penalty = (robot_uncertainty_penalty + track_uncertainty_penalty) / total_entities if total_entities > 0 else 0.0

        classification_reward = base_reward - uncertainty_penalty

        return max(0.0, classification_reward)  # Ensure non-negative reward

    def build_global_state_tensor(self, robots: List[Robot], gnn_scores_all: Dict,
                                 step_idx: int, total_steps: int,
                                 participating_robots: List[Robot] = None,
                                 participating_tracks: List = None) -> torch.Tensor:
        """
        Build Tier 0 global summary with fixed-length vector

        Args:
            robots: All robots in the environment
            gnn_scores_all: Dictionary of all GNN scores from this step
            step_idx: Current step index
            total_steps: Total steps in episode
            participating_robots: Robots participating in this step
            participating_tracks: Tracks participating in this step

        Returns:
            Tier 0 global state summary tensor [34 dims]
        """
        if not robots:
            # Return default features if no robots
            return torch.zeros(34, device=self.device)

        # Get all tracks (avoid duplicates)
        all_tracks = []
        track_lookup = {}
        for robot in robots:
            for track in robot.get_current_timestep_tracks():
                if track.track_id not in track_lookup:
                    all_tracks.append(track)
                    track_lookup[track.track_id] = track

        # Participating counts
        n_participating_robots = len(participating_robots) if participating_robots else 0
        n_participating_tracks = len(participating_tracks) if participating_tracks else 0

        # Robot and track data
        robot_trusts = [robot.trust_value for robot in robots]
        robot_strengths = [robot.trust_alpha + robot.trust_beta for robot in robots]
        track_trusts = [track.trust_value for track in all_tracks]
        track_strengths = [track.trust_alpha + track.trust_beta for track in all_tracks]
        track_maturities = [min(1.0, track.observation_count / 10.0) for track in all_tracks]

        # GNN scores - ALIGNED by ID to prevent mismatched pairing
        rid2score = gnn_scores_all.get('agent_scores', {})
        robot_scores = [rid2score.get(robot.id, 0.5) for robot in robots]

        tid2score = gnn_scores_all.get('track_scores', {})
        track_scores = [tid2score.get(track.track_id, 0.5) for track in all_tracks]

        features = []

        # === Counts / Context (6) ===
        features.append(np.log1p(len(robots)))  # log1p(n_robots)
        features.append(np.log1p(len(all_tracks)))  # log1p(n_tracks)
        features.append(np.log1p(n_participating_robots))  # log1p(n_participating_robots)
        features.append(np.log1p(n_participating_tracks))  # log1p(n_participating_tracks)
        features.append(step_idx / max(total_steps, 1))  # t_over_T
        # Previous step reward (using explicit member to avoid episode leakage)
        features.append(self.prev_step_reward)  # mean_prev_step_reward

        # === Robots - Trust & Strength (8) ===
        features.append(np.mean(robot_trusts) if robot_trusts else 0.5)  # robot_trust_mean
        features.append(np.std(robot_trusts) if len(robot_trusts) > 1 else 0.0)  # robot_trust_std
        features.append(np.mean(robot_strengths) if robot_strengths else 2.0)  # robot_strength_mean
        features.append(np.std(robot_strengths) if len(robot_strengths) > 1 else 0.0)  # robot_strength_std

        # Dual threshold fractions for robots
        neg_thresh = self.config.robot_negative_threshold  # 0.30
        pos_thresh = self.config.robot_positive_threshold  # 0.70
        if robot_trusts:
            frac_robot_low = sum(1 for t in robot_trusts if t <= neg_thresh) / len(robot_trusts)
            frac_robot_high = sum(1 for t in robot_trusts if t >= pos_thresh) / len(robot_trusts)
            frac_robot_gray = 1.0 - frac_robot_low - frac_robot_high
        else:
            frac_robot_low = frac_robot_high = frac_robot_gray = 0.0

        features.append(frac_robot_low)
        features.append(frac_robot_high)
        features.append(frac_robot_gray)

        # Mean absolute delta robot trust - track trust changes from previous step
        mean_abs_delta_robot_trust = 0.0
        if hasattr(self, 'prev_robot_trusts') and self.prev_robot_trusts and robot_trusts:
            # Calculate absolute trust changes for robots that exist in both steps
            trust_deltas = []
            robot_id_to_trust = {robot.id: robot.trust_value for robot in robots}
            for robot_id, prev_trust in self.prev_robot_trusts.items():
                if robot_id in robot_id_to_trust:
                    current_trust = robot_id_to_trust[robot_id]
                    trust_deltas.append(abs(current_trust - prev_trust))
            mean_abs_delta_robot_trust = np.mean(trust_deltas) if trust_deltas else 0.0

        # Store current robot trusts for next step
        self.prev_robot_trusts = {robot.id: robot.trust_value for robot in robots}
        features.append(mean_abs_delta_robot_trust)  # mean_abs_delta_robot_trust

        # === Tracks - Trust, Strength, Maturity (10) ===
        features.append(np.mean(track_trusts) if track_trusts else 0.5)  # track_trust_mean
        features.append(np.std(track_trusts) if len(track_trusts) > 1 else 0.0)  # track_trust_std
        features.append(np.mean(track_strengths) if track_strengths else 2.0)  # track_strength_mean
        features.append(np.std(track_strengths) if len(track_strengths) > 1 else 0.0)  # track_strength_std
        features.append(np.mean(track_maturities) if track_maturities else 0.0)  # track_maturity_mean
        features.append(np.std(track_maturities) if len(track_maturities) > 1 else 0.0)  # track_maturity_std

        # Dual threshold fractions for tracks
        neg_thresh_t = self.config.track_negative_threshold  # 0.30
        pos_thresh_t = self.config.track_positive_threshold  # 0.70
        if track_trusts:
            frac_track_low = sum(1 for t in track_trusts if t <= neg_thresh_t) / len(track_trusts)
            frac_track_high = sum(1 for t in track_trusts if t >= pos_thresh_t) / len(track_trusts)
            frac_track_gray = 1.0 - frac_track_low - frac_track_high
        else:
            frac_track_low = frac_track_high = frac_track_gray = 0.0

        features.append(frac_track_low)
        features.append(frac_track_high)
        features.append(frac_track_gray)

        # Mean absolute delta track trust - track trust changes from previous step
        mean_abs_delta_track_trust = 0.0
        if hasattr(self, 'prev_track_trusts') and self.prev_track_trusts and track_trusts:
            # Calculate absolute trust changes for tracks that exist in both steps
            trust_deltas = []
            track_id_to_trust = {track.track_id: track.trust_value for track in all_tracks}
            for track_id, prev_trust in self.prev_track_trusts.items():
                if track_id in track_id_to_trust:
                    current_trust = track_id_to_trust[track_id]
                    trust_deltas.append(abs(current_trust - prev_trust))
            mean_abs_delta_track_trust = np.mean(trust_deltas) if trust_deltas else 0.0

        # Store current track trusts for next step
        self.prev_track_trusts = {track.track_id: track.trust_value for track in all_tracks}
        features.append(mean_abs_delta_track_trust)  # mean_abs_delta_track_trust

        # === Current GNN Evidence (6) ===
        features.append(np.mean(robot_scores) if robot_scores else 0.5)  # robot_score_mean
        features.append(np.std(robot_scores) if len(robot_scores) > 1 else 0.0)  # robot_score_std
        features.append(np.mean(track_scores) if track_scores else 0.5)  # track_score_mean
        features.append(np.std(track_scores) if len(track_scores) > 1 else 0.0)  # track_score_std

        # Score confidence (distance from 0.5)
        robot_score_conf = np.mean([2 * abs(score - 0.5) for score in robot_scores]) if robot_scores else 0.0
        track_score_conf = np.mean([2 * abs(score - 0.5) for score in track_scores]) if track_scores else 0.0
        features.append(robot_score_conf)  # robot_score_conf_mean
        features.append(track_score_conf)  # track_score_conf_mean

        # === Trust-Evidence Alignment (4) ===
        # Compute correlations between trust and GNN scores
        robot_trust_score_corr = 0.0
        track_trust_score_corr = 0.0

        # Safe correlation computation with variance checks
        if (len(robot_trusts) > 1 and len(robot_scores) > 1 and
            len(robot_trusts) == len(robot_scores)):
            # Check if both arrays have variance (avoid division by zero)
            if np.std(robot_trusts) > 1e-8 and np.std(robot_scores) > 1e-8:
                robot_trust_score_corr = np.corrcoef(robot_trusts, robot_scores)[0, 1]
                if np.isnan(robot_trust_score_corr):
                    robot_trust_score_corr = 0.0
            # If no variance, correlation is undefined (set to 0)

        if (len(track_trusts) > 1 and len(track_scores) > 1 and
            len(track_trusts) == len(track_scores)):
            # Check if both arrays have variance (avoid division by zero)
            if np.std(track_trusts) > 1e-8 and np.std(track_scores) > 1e-8:
                track_trust_score_corr = np.corrcoef(track_trusts, track_scores)[0, 1]
                if np.isnan(track_trust_score_corr):
                    track_trust_score_corr = 0.0
            # If no variance, correlation is undefined (set to 0)

        features.append(robot_trust_score_corr)  # cov(robot_trust, robot_score)
        features.append(track_trust_score_corr)  # cov(track_trust, track_score)

        # Disagreement rates - measure trust vs GNN score disagreement
        robot_disagreement_rate = 0.0
        track_disagreement_rate = 0.0

        # Robot disagreement: trust says adversarial but GNN says legitimate (or vice versa)
        if robot_trusts and robot_scores and len(robot_trusts) == len(robot_scores):
            disagreements = 0
            for trust, score in zip(robot_trusts, robot_scores):
                trust_says_bad = trust < 0.5  # Low trust = adversarial
                gnn_says_bad = score < 0.5    # Low score = adversarial
                if trust_says_bad != gnn_says_bad:  # Disagreement
                    disagreements += 1
            robot_disagreement_rate = disagreements / len(robot_trusts)

        # Track disagreement: trust says false but GNN says legitimate (or vice versa)
        if track_trusts and track_scores and len(track_trusts) == len(track_scores):
            disagreements = 0
            for trust, score in zip(track_trusts, track_scores):
                trust_says_bad = trust < 0.5  # Low trust = false track
                gnn_says_bad = score < 0.5    # Low score = false track
                if trust_says_bad != gnn_says_bad:  # Disagreement
                    disagreements += 1
            track_disagreement_rate = disagreements / len(track_trusts)

        features.append(track_disagreement_rate)  # disagreement_rate_tracks
        features.append(robot_disagreement_rate)  # disagreement_rate_robots

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def build_tier2_robot_features(self, robots: List[Robot], gnn_scores_all: Dict) -> torch.Tensor:
        """
        Build per-robot features for Tier 2 set encoder

        Returns:
            [N_R, 6] tensor with features: [trust_mean, strength, Δtrust, agent_score, score_conf, degree]
        """
        if not robots:
            return torch.zeros(0, 6, device=self.device)

        robot_features = []
        for robot in robots:
            # Basic features
            trust_mean = robot.trust_value
            strength = robot.trust_alpha + robot.trust_beta
            # Delta trust: change from previous step
            prev_trust = self.prev_robot_trusts.get(robot.id, trust_mean)
            delta_trust = abs(trust_mean - prev_trust)

            # GNN score
            agent_score = gnn_scores_all.get('agent_scores', {}).get(robot.id, 0.5)
            score_conf = 2 * abs(agent_score - 0.5)  # Confidence as distance from 0.5

            # Degree (number of tracks this robot has)
            degree = len(robot.get_current_timestep_tracks())

            robot_features.append([trust_mean, strength, delta_trust, agent_score, score_conf, degree])

        return torch.tensor(robot_features, dtype=torch.float32, device=self.device)

    def build_tier2_track_features(self, robots: List[Robot], gnn_scores_all: Dict) -> torch.Tensor:
        """
        Build per-track features for Tier 2 set encoder

        Returns:
            [N_T, 7] tensor with features: [trust_mean, strength, Δtrust, maturity, track_score, score_conf, degree]
        """
        # Get all unique tracks
        all_tracks = []
        track_lookup = {}
        track_detectors = {}  # track_id -> list of robot_ids that detected it

        for robot in robots:
            for track in robot.get_current_timestep_tracks():
                if track.track_id not in track_lookup:
                    all_tracks.append(track)
                    track_lookup[track.track_id] = track
                    track_detectors[track.track_id] = []
                track_detectors[track.track_id].append(robot.id)

        if not all_tracks:
            return torch.zeros(0, 7, device=self.device)

        track_features = []
        for track in all_tracks:
            # Basic features
            trust_mean = track.trust_value
            strength = track.trust_alpha + track.trust_beta
            # Delta trust: change from previous step
            prev_trust = self.prev_track_trusts.get(track.track_id, trust_mean)
            delta_trust = abs(trust_mean - prev_trust)
            maturity = min(1.0, track.observation_count / 10.0)

            # GNN score
            track_score = gnn_scores_all.get('track_scores', {}).get(track.track_id, 0.5)
            score_conf = 2 * abs(track_score - 0.5)  # Confidence as distance from 0.5

            # Degree (number of robots that detected this track)
            degree = len(track_detectors.get(track.track_id, []))

            track_features.append([trust_mean, strength, delta_trust, maturity, track_score, score_conf, degree])

        return torch.tensor(track_features, dtype=torch.float32, device=self.device)

    def build_complete_global_state(self, robots: List[Robot], gnn_scores_all: Dict,
                                   step_idx: int, total_steps: int,
                                   participating_robots: List[Robot] = None,
                                   participating_tracks: List = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build complete global state representation for the critic

        Returns:
            Tuple of (tier0_summary, tier2_robot_features, tier2_track_features)
        """
        tier0_summary = self.build_global_state_tensor(
            robots, gnn_scores_all, step_idx, total_steps,
            participating_robots, participating_tracks
        )
        tier2_robot_features = self.build_tier2_robot_features(robots, gnn_scores_all)
        tier2_track_features = self.build_tier2_track_features(robots, gnn_scores_all)

        return tier0_summary, tier2_robot_features, tier2_track_features

    def pad_and_mask_tensors(self, robot_features: torch.Tensor, track_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pad robot and track features to fixed sizes and create masks

        Args:
            robot_features: [N_R, 6] current robot features
            track_features: [N_T, 7] current track features

        Returns:
            Tuple of (padded_robot_features, padded_track_features, robot_mask, track_mask)
        """
        device = self.device
        max_robots = self.config.max_robots
        max_tracks = self.config.max_tracks

        # Get current sizes
        n_robots = robot_features.size(0)
        n_tracks = track_features.size(0)

        # Validate that we don't exceed the fixed limits
        if n_robots > max_robots:
            raise ValueError(f"Number of robots ({n_robots}) exceeds maximum limit ({max_robots})")
        if n_tracks > max_tracks:
            raise ValueError(f"Number of tracks ({n_tracks}) exceeds maximum limit ({max_tracks})")

        # Create padded robot features
        padded_robot_features = torch.zeros(max_robots, 6, device=device)
        robot_mask = torch.zeros(max_robots, device=device)
        if n_robots > 0:
            padded_robot_features[:n_robots] = robot_features
            robot_mask[:n_robots] = 1.0

        # Create padded track features
        padded_track_features = torch.zeros(max_tracks, 7, device=device)
        track_mask = torch.zeros(max_tracks, device=device)
        if n_tracks > 0:
            padded_track_features[:n_tracks] = track_features
            track_mask[:n_tracks] = 1.0

        return padded_robot_features, padded_track_features, robot_mask, track_mask

    def run_episode(self) -> float:
        """Run one training episode"""

        # Reset episode state to prevent leakage across episodes
        self.prev_robot_trusts = {}
        self.prev_track_trusts = {}
        self.prev_step_reward = 0.0

        # Note: Using fixed padding limits from config (no dynamic episode-level padding)

        # Generate scenario parameters
        scenario_params = self.scenario_generator.sample_scenario_parameters(self.episode_count)

        # Create scenario environment (not mock!)
        sim_env, _ = self.scenario_generator.create_scenario_environment(scenario_params)

        episode_reward = 0.0

        # Run episode using proper simulation
        for step in range(scenario_params.episode_length):
            try:
                # Step the simulation environment
                sim_env.step()

                # Get current robots from simulation
                robots = sim_env.robots
                if not robots:
                    continue

                # Update robot tracks from simulation
                for robot in robots:
                    robot.update_current_timestep_tracks()

                # Get detection info for this step
                robot_detections, track_detectors = self.trust_system.get_detections_this_step(robots)

                # Build track lookup
                track_lookup = {}
                for robot in robots:
                    for track in robot.get_current_timestep_tracks():
                        track_lookup[track.track_id] = track

                # Collect all GNN scores for global summary
                all_gnn_scores = {'agent_scores': {}, 'track_scores': {}}

                # Pre-compute step decisions for ALL ego robots to maintain on-policy behavior
                precomputed_decisions = {}
                all_trajectory_data = []  # Will store data for ALL participating egos

                # First pass: collect all GNN scores for global state (detached for critic)
                for ego_robot in robots:
                    scores = self.trust_system.evidence_extractor.get_scores(ego_robot, robots)
                    if scores.agent_scores or scores.track_scores:
                        # Detach GNN scores to prevent backprop into evidence GNN from RL training
                        detached_agent_scores = {k: float(v) for k, v in scores.agent_scores.items()}
                        detached_track_scores = {k: float(v) for k, v in scores.track_scores.items()}
                        all_gnn_scores['agent_scores'].update(detached_agent_scores)
                        all_gnn_scores['track_scores'].update(detached_track_scores)

                # Build tier0 features early (needed for SetTransformer)
                tier0_features = self.build_global_state_tensor(
                    robots, all_gnn_scores, step, scenario_params.episode_length,
                    participating_robots=None, participating_tracks=None  # Will be filled per ego
                )

                # Loop over all robots as ego robots (like in ego_sweep_step)
                for ego_robot in robots:
                    # Get GNN scores for this ego graph
                    scores = self.trust_system.evidence_extractor.get_scores(ego_robot, robots)

                    if not scores.agent_scores and not scores.track_scores:
                        continue  # No scores from GNN

                    # Accumulate GNN scores for global summary (detached for critic)
                    # Detach GNN scores to prevent backprop into evidence GNN from RL training
                    detached_agent_scores = {k: float(v) for k, v in scores.agent_scores.items()}
                    detached_track_scores = {k: float(v) for k, v in scores.track_scores.items()}
                    all_gnn_scores['agent_scores'].update(detached_agent_scores)
                    all_gnn_scores['track_scores'].update(detached_track_scores)

                    # Get ego graph nodes (same logic as in ego_sweep_step)
                    ego_robot_ids = list(scores.agent_scores.keys())
                    ego_track_ids = list(scores.track_scores.keys())

                    ego_robots = [robot for robot in robots if robot.id in ego_robot_ids]
                    ego_tracks = [track_lookup[track_id] for track_id in ego_track_ids if track_id in track_lookup]

                    # Get participating nodes (those that detected/were detected this step)
                    participating_robots = [robot for robot in ego_robots if robot.id in robot_detections]
                    participating_tracks = [track for track in ego_tracks if track.track_id in track_detectors]

                    if not participating_robots and not participating_tracks:
                        continue  # No participating nodes in this ego graph

                    # STABLE ORDERING: Sort IDs for consistent ordering across calls
                    robot_ids = sorted([r.id for r in participating_robots])
                    track_ids = sorted([t.track_id for t in participating_tracks])

                    # Build participating entities in sorted order
                    robot_lookup_local = {r.id: r for r in participating_robots}
                    track_lookup_local = {t.track_id: t for t in participating_tracks}

                    participating_robots_sorted = [robot_lookup_local[rid] for rid in robot_ids]
                    participating_tracks_sorted = [track_lookup_local[tid] for tid in track_ids]

                    # State encoding is no longer needed since we use tier0_features directly

                    # Build features for participating nodes in sorted order (SetTransformer format)
                    robot_features = []
                    for robot in participating_robots_sorted:
                        trust_mean = robot.trust_value
                        strength = robot.trust_alpha + robot.trust_beta
                        # Delta trust: change from previous step
                        prev_trust = self.prev_robot_trusts.get(robot.id, trust_mean)
                        delta_trust = abs(trust_mean - prev_trust)
                        robot_score = scores.agent_scores.get(robot.id, 0.5)
                        score_conf = 2 * abs(robot_score - 0.5)  # Confidence as distance from 0.5
                        # Degree (number of tracks this robot has)
                        degree = len(robot.get_current_timestep_tracks())
                        robot_features.append([trust_mean, strength, delta_trust, robot_score, score_conf, degree])

                    track_features = []
                    for track in participating_tracks_sorted:
                        trust_mean = track.trust_value
                        strength = track.trust_alpha + track.trust_beta
                        # Delta trust: change from previous step
                        prev_trust = self.prev_track_trusts.get(track.track_id, trust_mean)
                        delta_trust = abs(trust_mean - prev_trust)
                        maturity = min(1.0, track.observation_count / 10.0)
                        track_score = scores.track_scores.get(track.track_id, 0.5)
                        score_conf = 2 * abs(track_score - 0.5)  # Confidence as distance from 0.5
                        # Degree (number of participating robots that detected this track)
                        degree = sum(1 for r in participating_robots_sorted if any(t.track_id == track.track_id for t in r.get_current_timestep_tracks()))
                        track_features.append([trust_mean, strength, delta_trust, maturity, track_score, score_conf, degree])

                    robot_features_tensor = torch.tensor(robot_features, dtype=torch.float32, device=self.device) if robot_features else torch.zeros(0, 6, device=self.device)
                    track_features_tensor = torch.tensor(track_features, dtype=torch.float32, device=self.device) if track_features else torch.zeros(0, 7, device=self.device)

                    # Create ego graph summary for this ego robot (local stats only)
                    ego_summary = self.updater.create_ego_graph_summary(
                        ego_robots, ego_tracks, scores.agent_scores, scores.track_scores
                    )
                    ego_summary_tensor = torch.tensor([
                        ego_summary.robot_trust_mean, ego_summary.robot_trust_std, ego_summary.robot_trust_count,
                        ego_summary.track_trust_mean, ego_summary.track_trust_std, ego_summary.track_trust_count,
                        ego_summary.robot_score_mean, ego_summary.robot_score_std,
                        ego_summary.track_score_mean, ego_summary.track_score_std
                    ], dtype=torch.float32, device=self.device)

                    # Sample actions using SetTransformer actor policy
                    robot_features_tensor = robot_features_tensor.to(self.device)
                    track_features_tensor = track_features_tensor.to(self.device)

                    # Create masks (no padding needed here since these are actual participating entities)
                    robot_mask = torch.ones(robot_features_tensor.shape[0], device=self.device) if robot_features_tensor.shape[0] > 0 else None
                    track_mask = torch.ones(track_features_tensor.shape[0], device=self.device) if track_features_tensor.shape[0] > 0 else None

                    robot_actions, track_actions, robot_log_probs, track_log_probs, entropy, _, _ = self.updater.policy.sample_actions(
                        ego_summary_tensor.unsqueeze(0), robot_features_tensor.unsqueeze(0), track_features_tensor.unsqueeze(0),
                        robot_mask.unsqueeze(0) if robot_mask is not None else None,
                        track_mask.unsqueeze(0) if track_mask is not None else None
                    )
                    # Remove batch dimension from outputs
                    robot_actions = robot_actions.squeeze(0)
                    track_actions = track_actions.squeeze(0)
                    robot_log_probs = robot_log_probs.squeeze(0)
                    track_log_probs = track_log_probs.squeeze(0)
                    entropy = entropy.squeeze(0)

                    # Convert sampled actions to step decision format for trust updater
                    step_decision = UpdateDecision({}, {})
                    for i, robot_id in enumerate(robot_ids):
                        if i < len(robot_actions):
                            step_decision.robot_steps[robot_id] = float(robot_actions[i].detach())
                    for i, track_id in enumerate(track_ids):
                        if i < len(track_actions):
                            step_decision.track_steps[track_id] = float(track_actions[i].detach())

                    precomputed_decisions[ego_robot.id] = step_decision

                    # Store trajectory data for this ego (detached to avoid gradient conflicts)
                    # Include IDs to maintain exact ordering consistency and ego summary for SetTransformer
                    ego_trajectory_data = [ego_summary_tensor.detach(), robot_features_tensor.detach(), track_features_tensor.detach(),
                                         robot_actions.detach(), track_actions.detach(), robot_log_probs.detach(), track_log_probs.detach(),
                                         robot_ids.copy(), track_ids.copy(), robot_mask, track_mask]  # Store everything needed for SetTransformer
                    all_trajectory_data.append(ego_trajectory_data)

                # Fix scope bug: compute environment-level participating entities from detection maps
                env_participating_robot_ids = set(robot_detections.keys())
                env_participating_track_ids = set(track_detectors.keys())

                env_participating_robots = [r for r in robots if r.id in env_participating_robot_ids]
                env_participating_tracks = [track_lookup[tid] for tid in env_participating_track_ids if tid in track_lookup]

                # Build complete global state (combines Tier 0 + Tier 2) with detached GNN evidence
                tier0_summary, tier2_robot_features, tier2_track_features = self.build_complete_global_state(
                    robots, all_gnn_scores, step, scenario_params.episode_length,
                    participating_robots=env_participating_robots,
                    participating_tracks=env_participating_tracks
                )

                # Apply fixed padding and masking for consistent tensor sizes
                padded_robot_features, padded_track_features, robot_mask, track_mask = self.pad_and_mask_tensors(
                    tier2_robot_features, tier2_track_features
                )

                # Create global state tensor for storage (combines all components with fixed padding)
                global_state_tensor = {
                    'tier0': tier0_summary,
                    'tier2_robots': padded_robot_features,
                    'tier2_tracks': padded_track_features,
                    'robot_mask': robot_mask,
                    'track_mask': track_mask,
                    'max_robots': self.config.max_robots,
                    'max_tracks': self.config.max_tracks,
                    'actual_robots': tier2_robot_features.size(0),
                    'actual_tracks': tier2_track_features.size(0)
                }

                # Get centralized critic value estimate for this step with proper masking
                with torch.no_grad():
                    step_value = self.critic(
                        tier0_summary.unsqueeze(0),  # [1, 34]
                        padded_robot_features.unsqueeze(0),  # [1, max_robots, 6]
                        padded_track_features.unsqueeze(0),  # [1, max_tracks, 7]
                        robot_mask.unsqueeze(0),  # [1, max_robots]
                        track_mask.unsqueeze(0)   # [1, max_tracks]
                    ).item()

                # Apply trust updates using precomputed decisions
                self.trust_system.update_trust(robots, precomputed_decisions)

                # Update ground truth for dynamic false tracks (changes as robots move)
                current_ground_truth = self.scenario_generator._extract_ground_truth(sim_env)

                # Compute step reward based on trust alignment with current ground truth
                raw_step_reward = self.compute_step_reward(robots, current_ground_truth)
                step_reward = raw_step_reward * self.config.reward_scale

                # REWARD WIRING: Add classification bonus to R_t at the last step (blueprint requirement)
                if step == scenario_params.episode_length - 1:
                    try:
                        episode_end_reward = self.compute_episode_end_reward(robots, current_ground_truth)
                        # Small multiplier (1-3×) as specified in blueprint
                        step_reward += episode_end_reward * self.config.classification_bonus_multiplier * self.config.reward_scale
                    except Exception as e:
                        print(f"Error computing episode-end reward in step: {e}")

                # Apply reward clipping at step level if enabled
                if self.config.clip_rewards:
                    step_reward = max(-self.config.reward_clip_range, min(self.config.reward_clip_range, step_reward))

                episode_reward += step_reward

                # Update previous step reward for next step's feature computation
                self.prev_step_reward = step_reward

                # Store step-level data using new buffer format with proper padding
                current_step_idx = len(self.step_R)

                self.step_S.append(global_state_tensor)  # Store complete global state with padding
                self.step_V.append(torch.tensor([step_value], device=self.device))  # Tensor[1]
                self.step_R.append(step_reward)  # float
                self.step_done.append(step == scenario_params.episode_length - 1)  # bool

                # Initialize step mapping if not exists
                if current_step_idx not in self.step_to_item_indices:
                    self.step_to_item_indices[current_step_idx] = []

                # Collect ego-level data for ALL participating egos using new buffer format
                for ego_trajectory_data in all_trajectory_data:
                    ego_summary_ego, robot_features_tensor, track_features_tensor, robot_actions, track_actions, robot_log_probs, track_log_probs, robot_ids, track_ids, robot_mask, track_mask = ego_trajectory_data

                    # Store ego-level data using new buffer format
                    ego_item_idx = len(self.ego_summary)

                    self.ego_summary.append(ego_summary_ego.detach())  # Tensor[ego_summary_dim] - ego graph summary
                    self.ego_R_feats.append(robot_features_tensor.detach())  # Tensor[N_R, F_R] - robot features
                    self.ego_T_feats.append(track_features_tensor.detach())  # Tensor[N_T, F_T] - track features
                    self.ego_a_R.append(robot_actions.detach())  # Tensor[N_R] - robot actions
                    self.ego_a_T.append(track_actions.detach())  # Tensor[N_T] - track actions
                    self.ego_logp_R.append(robot_log_probs.detach())  # Tensor[N_R] - robot log probs (old)
                    self.ego_logp_T.append(track_log_probs.detach())  # Tensor[N_T] - track log probs (old)
                    self.ego_entropy.append(float(entropy.item() if hasattr(entropy, 'item') else entropy))  # float - entropy
                    self.ego_step_idx.append(current_step_idx)  # int - step index for broadcasting A_t
                    self.ego_robot_mask.append(robot_mask.detach() if robot_mask is not None else torch.empty(0, device=self.device))
                    self.ego_track_mask.append(track_mask.detach() if track_mask is not None else torch.empty(0, device=self.device))

                    # Map step to ego item index
                    self.step_to_item_indices[current_step_idx].append(ego_item_idx)

            except Exception as e:
                print(f"Error in episode step {step}: {e}")
                break

        # Episode-end reward is now included in the final step above

        # Normalize episode reward by episode length for fair comparison across different scenario sizes
        normalized_episode_reward = episode_reward / max(scenario_params.episode_length, 1)
        return normalized_episode_reward

    def clear_trajectories(self):
        """Clear trajectory storage after PPO update"""
        # Clear step-level storage
        self.step_S.clear()
        self.step_V.clear()
        self.step_R.clear()
        self.step_done.clear()
        self.step_to_item_indices.clear()

        # Clear ego-level storage
        self.ego_summary.clear()
        self.ego_R_feats.clear()
        self.ego_T_feats.clear()
        self.ego_a_R.clear()
        self.ego_a_T.clear()
        self.ego_logp_R.clear()
        self.ego_logp_T.clear()
        self.ego_entropy.clear()
        self.ego_step_idx.clear()
        self.ego_robot_mask.clear()
        self.ego_track_mask.clear()

    def ppo_update(self):
        """Perform PPO policy update using collected trajectories"""
        if len(self.step_R) == 0:
            return

        # Compute advantages using centralized critic at step level, then broadcast to ego samples
        advantages, returns = self.compute_step_level_advantages()

        # Advantage standardization (blueprint requirement)
        if self.config.standardize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Note: single advantage values are left unchanged as normalization would set them to 0

        # PPO update loop with KL watchdog
        for epoch in range(self.config.ppo_epochs):
            # Create random mini-batches over ego items
            num_ego_items = len(self.ego_summary)
            batch_size = min(self.config.batch_size, num_ego_items)
            indices = torch.randperm(num_ego_items)

            for start in range(0, num_ego_items, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Process each item in the batch
                total_policy_loss = 0.0
                total_value_loss = 0.0
                total_entropy = 0.0
                batch_count = 0

                # RAGGED SAMPLES APPROACH: Process each sample individually
                for idx, i in enumerate(batch_indices):
                    try:
                        # Move sample's tensors to device (ragged - no padding needed)
                        ego_summary = self.ego_summary[i].to(self.device)  # [ego_summary_dim] - ego graph summary
                        robot_features = self.ego_R_feats[i].to(self.device)  # [N_R, F_R] - variable N_R
                        track_features = self.ego_T_feats[i].to(self.device)  # [N_T, F_T] - variable N_T
                        robot_mask = self.ego_robot_mask[i].to(self.device) if self.ego_robot_mask[i].numel() > 0 else None
                        track_mask = self.ego_track_mask[i].to(self.device) if self.ego_track_mask[i].numel() > 0 else None

                        robot_actions = self.ego_a_R[i].to(self.device)  # [N_R] - taken robot actions
                        track_actions = self.ego_a_T[i].to(self.device)  # [N_T] - taken track actions

                        logp_R_old = self.ego_logp_R[i].to(self.device)  # [N_R] - old log-probs
                        logp_T_old = self.ego_logp_T[i].to(self.device)  # [N_T] - old log-probs

                        # Get new log-probs from current policy using SetTransformer interface
                        logp_R_new, logp_T_new, entropy = self.updater.policy.evaluate_actions(
                            ego_summary.unsqueeze(0), robot_features.unsqueeze(0), track_features.unsqueeze(0),
                            robot_actions.unsqueeze(0), track_actions.unsqueeze(0),
                            robot_mask.unsqueeze(0) if robot_mask is not None else None,
                            track_mask.unsqueeze(0) if track_mask is not None else None
                        )
                        # Remove batch dimension from outputs
                        logp_R_new = logp_R_new.squeeze(0)
                        logp_T_new = logp_T_new.squeeze(0)
                        entropy = entropy.squeeze(0)

                        # Compute element-wise log probability ratios (much more stable)
                        robot_log_ratios = logp_R_new - logp_R_old  # [N_R]
                        track_log_ratios = logp_T_new - logp_T_old  # [N_T]

                        # Combine ratios (weighted by mask if available)
                        if robot_mask is not None and robot_mask.numel() > 0:
                            robot_log_ratios = robot_log_ratios * robot_mask.float()
                        if track_mask is not None and track_mask.numel() > 0:
                            track_log_ratios = track_log_ratios * track_mask.float()

                        # Sum log ratios for total policy ratio (more numerically stable than summing log probs)
                        total_log_ratio = robot_log_ratios.sum() + track_log_ratios.sum()
                        ratio = torch.exp(torch.clamp(total_log_ratio, -10, 10))  # Clamp for numerical stability

                        step_idx = self.ego_step_idx[i]  # Get step index for broadcasting advantage
                        advantage = batch_advantages[idx]  # Broadcast advantage for this step

                        # PPO clipped surrogate objective
                        surr1 = ratio * advantage
                        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantage
                        policy_loss = -torch.min(surr1, surr2)

                        # Use centralized critic for value loss
                        if step_idx >= 0 and step_idx < len(self.step_S):
                            global_state = self.step_S[step_idx]
                            tier0_summary = global_state['tier0'].unsqueeze(0).to(self.device)
                            tier2_robot_features = global_state['tier2_robots'].unsqueeze(0).to(self.device)
                            tier2_track_features = global_state['tier2_tracks'].unsqueeze(0).to(self.device)
                            robot_mask = global_state['robot_mask'].unsqueeze(0).to(self.device)
                            track_mask = global_state['track_mask'].unsqueeze(0).to(self.device)
                            centralized_value = self.critic(tier0_summary, tier2_robot_features, tier2_track_features, robot_mask, track_mask)

                            # Target comes from step-level returns (already computed)
                            step_target = batch_returns[idx]  # Use position in batch
                            value_loss = torch.nn.SmoothL1Loss()(centralized_value.squeeze(), step_target)
                        else:
                            # Fallback to zero loss if invalid step index
                            value_loss = torch.tensor(0.0, device=self.device)

                        # Use computed entropy
                        entropy = entropy

                        total_policy_loss += policy_loss
                        total_value_loss += value_loss
                        total_entropy += entropy
                        batch_count += 1

                    except Exception as e:
                        print(f"Warning: Failed to process sample {i}: {e}")
                        continue

                if batch_count > 0:
                    # Average losses over the batch
                    avg_policy_loss = total_policy_loss / batch_count
                    avg_value_loss = total_value_loss / batch_count
                    avg_entropy = total_entropy / batch_count

                    # Total loss 
                    loss = avg_policy_loss + self.config.value_loss_coef * avg_value_loss - self.config.entropy_coef * avg_entropy

                    # Store losses for monitoring
                    self.ppo_losses.append(loss.item())
                    self.policy_losses.append(avg_policy_loss.item())
                    self.value_losses.append(avg_value_loss.item())
                    self.entropy_values.append(avg_entropy.item())

                    # Update policy
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Clip gradients for both actor and critic (since they're optimized together)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.updater.policy.parameters()) + list(self.critic.parameters()),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

        self.total_updates += 1
        print(f"PPO update completed. Total loss: {np.mean(self.ppo_losses[-10:]):.4f}")

    def compute_step_level_advantages(self):
        """
        Compute advantages using centralized critic values at step level,
        then broadcast to all ego samples from that step
        """
        if not self.step_R:
            return torch.tensor([]), torch.tensor([])

        # Convert step-level data to tensors
        step_rewards = torch.tensor(self.step_R, dtype=torch.float32, device=self.device)
        step_dones = torch.tensor(self.step_done, dtype=torch.float32, device=self.device)

        # Re-evaluate centralized critic on stored features (ragged approach)
        fresh_step_values = []

        with torch.no_grad():
            for global_state in self.step_S:
                tier0_summary = global_state['tier0'].unsqueeze(0).to(self.device)  # [1, 34]
                tier2_robot_features = global_state['tier2_robots'].unsqueeze(0).to(self.device)  # [1, max_robots, 6]
                tier2_track_features = global_state['tier2_tracks'].unsqueeze(0).to(self.device)  # [1, max_tracks, 7]
                robot_mask = global_state['robot_mask'].unsqueeze(0).to(self.device)  # [1, max_robots]
                track_mask = global_state['track_mask'].unsqueeze(0).to(self.device)  # [1, max_tracks]

                step_value = self.critic(tier0_summary, tier2_robot_features, tier2_track_features, robot_mask, track_mask)
                fresh_step_values.append(step_value.squeeze())

        fresh_step_values = torch.stack(fresh_step_values).to(self.device)  # [T]

        # Compute step-level advantages and returns using GAE
        advantages = []
        returns = []
        gae = 0

        # Process in reverse order for GAE computation
        for t in reversed(range(len(step_rewards))):
            if step_dones[t]:
                # Terminal step - no bootstrapping
                delta = step_rewards[t] - fresh_step_values[t]
                gae = delta
            else:
                # Non-terminal step
                delta = step_rewards[t] + self.config.gamma * fresh_step_values[t+1] - fresh_step_values[t]
                gae = delta + self.config.gamma * self.config.gae_lambda * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + fresh_step_values[t])

        step_advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        step_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Now broadcast these step-level advantages to per-ego samples
        # Optionally divide by number of egos per step (for multi-ego training)
        ego_advantages = []
        ego_returns = []

        for step_idx in self.ego_step_idx:
            if step_idx >= 0 and step_idx < len(step_advantages):
                step_advantage = step_advantages[step_idx]
                step_return = step_returns[step_idx]

                # Optionally divide by number of egos for this step (to normalize per step)
                if hasattr(self.config, 'divide_advantage_by_ego_count') and self.config.divide_advantage_by_ego_count:
                    num_egos_for_step = len(self.step_to_item_indices.get(step_idx, []))
                    if num_egos_for_step > 0:
                        step_advantage = step_advantage / num_egos_for_step
                        step_return = step_return / num_egos_for_step

                ego_advantages.append(step_advantage)
                ego_returns.append(step_return)
            else:
                # Fallback for invalid indices
                ego_advantages.append(torch.tensor(0.0, device=self.device))
                ego_returns.append(torch.tensor(0.0, device=self.device))

        return torch.stack(ego_advantages), torch.stack(ego_returns)

    def encode_state(self, ego_robots, ego_tracks, robot_scores, track_scores) -> torch.Tensor:
        """Encode ego graph state for PPO"""
        # Create ego graph summary (same as in rl_updater.py)
        summary = self.trust_system.updater.create_ego_graph_summary(
            ego_robots, ego_tracks, robot_scores, track_scores
        )

        # Convert to tensor
        summary_tensor = torch.tensor([
            summary.robot_trust_mean, summary.robot_trust_std, summary.robot_trust_count,
            summary.track_trust_mean, summary.track_trust_std, summary.track_trust_count,
            summary.robot_score_mean, summary.robot_score_std,
            summary.track_score_mean, summary.track_score_std
        ], dtype=torch.float32)

        return summary_tensor

    def collect_trajectory_step(self, summary, robot_features, track_features,
                               robot_actions, track_actions, robot_log_probs, track_log_probs,
                               reward, value, done, robot_ids=None, track_ids=None, step_idx=None):
        """Collect one step of trajectory for PPO with complete state information"""

        # Store complete state information
        full_state = {
            'summary': summary,
            'robot_features': robot_features,
            'track_features': track_features,
            'robot_ids': robot_ids if robot_ids is not None else [],
            'track_ids': track_ids if track_ids is not None else []
        }

        # Store complete action information
        full_actions = {
            'robot_actions': robot_actions,
            'track_actions': track_actions
        }

        # Store complete log probabilities
        full_log_probs = {
            'robot_log_probs': robot_log_probs,
            'track_log_probs': track_log_probs
        }

        self.trajectories['states'].append(full_state)
        self.trajectories['actions'].append(full_actions)
        self.trajectories['log_probs'].append(full_log_probs)
        self.trajectories['rewards'].append(reward)
        self.trajectories['values'].append(value)
        self.trajectories['dones'].append(done)
        self.trajectories['step_indices'].append(step_idx if step_idx is not None else -1)

    def train(self):
        """Main training loop"""
        print(f"Starting PPO training for RL Trust System")
        print(f"Episodes: {self.config.num_episodes}")

        for episode in range(self.config.num_episodes):
            self.episode_count = episode

            # Run episode
            start_time = time.time()
            episode_reward = self.run_episode()
            episode_time = time.time() - start_time

            # Collect metrics
            self.episode_rewards.append(episode_reward)

            # Perform PPO update every few episodes (separate from ppo_epochs)
            if episode > 0 and episode % self.config.update_every_episodes == 0 and len(self.ego_summary) > 0:
                traj_count = len(self.ego_summary)
                self.trajectory_counts.append(traj_count)
                print(f"Running PPO update at episode {episode} with {traj_count} ego samples")
                self.ppo_update()
                self.clear_trajectories()

                # Update learning rate based on recent performance
                recent_avg = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
                self.scheduler.step(recent_avg)

            # More frequent logging for monitoring
            if True:
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                recent_loss = np.mean(self.ppo_losses[-10:]) if self.ppo_losses else 0
                current_lr = self.optimizer.param_groups[0]['lr']
                traj_count = self.trajectory_counts[-1] if self.trajectory_counts else 0
                print(f"Episode {episode}: Reward = {episode_reward:.4f}, Avg = {avg_reward:.4f}, Loss = {recent_loss:.4f}")
                print(f"           Updates = {self.total_updates}, LR = {current_lr:.6f}, Trajectories = {traj_count}, Time = {episode_time:.2f}s")

            # Save best model
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                torch.save(self.updater.policy.state_dict(), 'rl_trust_model.pth')
                print(f"New best model saved: {episode_reward:.3f}")

        # Save final model
        torch.save(self.updater.policy.state_dict(), 'rl_trust_model_final.pth')

        # Plot training rewards
        self.plot_training_rewards()

        print(f"Training completed. Best reward: {self.best_reward:.4f}")
        print(f"Final learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"Total PPO updates: {self.total_updates}")
        print(f"Average trajectory count: {np.mean(self.trajectory_counts) if self.trajectory_counts else 0:.1f}")

    def plot_training_rewards(self):
        """Plot training rewards over episodes"""
        if not self.episode_rewards:
            print("⚠️ No episode rewards to plot")
            return

        plt.figure(figsize=(12, 6))

        episodes = list(range(len(self.episode_rewards)))

        # Plot raw episode rewards
        plt.subplot(1, 2, 1)
        plt.plot(episodes, self.episode_rewards, alpha=0.7, label='Episode Rewards')

        # Calculate and plot moving average
        window_size = min(10, len(self.episode_rewards))
        if len(self.episode_rewards) >= window_size:
            moving_avg = []
            for i in range(len(self.episode_rewards)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(np.mean(self.episode_rewards[start_idx:i+1]))
            plt.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size})')

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards Over Episodes')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot reward distribution
        plt.subplot(1, 2, 2)
        plt.hist(self.episode_rewards, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.grid(True, alpha=0.3)

        # Add statistics
        mean_reward = np.mean(self.episode_rewards)
        std_reward = np.std(self.episode_rewards)
        plt.axvline(mean_reward, color='red', linestyle='--',
                   label=f'Mean: {mean_reward:.4f}')
        plt.axvline(mean_reward + std_reward, color='orange', linestyle=':',
                   label=f'Mean + Std: {mean_reward + std_reward:.4f}')
        plt.axvline(mean_reward - std_reward, color='orange', linestyle=':',
                   label=f'Mean - Std: {mean_reward - std_reward:.4f}')
        plt.legend()

        plt.tight_layout()
        plt.savefig('rl_training_rewards.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Training rewards plot saved to: rl_training_rewards.png")
        print(f"📈 Training Statistics:")
        print(f"   • Total Episodes: {len(self.episode_rewards)}")
        print(f"   • Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
        print(f"   • Best Reward: {max(self.episode_rewards):.4f}")
        print(f"   • Worst Reward: {min(self.episode_rewards):.4f}")
        print(f"   • Final Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"   • Total Updates: {self.total_updates}")


def main():
    """Main training function"""
    import argparse
    parser = argparse.ArgumentParser(description='Train RL Trust System')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cpu, cuda, cuda:0, cuda:1, etc.')
    args = parser.parse_args()

    config = TrainingConfig()

    # Device selection logic
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"🖥️  Using device: {device}")

    # Start training (need to provide path to supervised model)
    supervised_model_path = "supervised_trust_model.pth"

    trainer = PPOTrainer(config, supervised_model_path, device)

    # Check if supervised model exists
    import os
    if not os.path.exists(supervised_model_path):
        print(f"⚠️  Supervised model not found: {supervised_model_path}")
        print("   Training will continue but GNN evidence will not be available")

    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()