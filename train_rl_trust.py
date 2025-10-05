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
import copy
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from robot_track_classes import Robot
from rl_trust_system import RLTrustSystem
# rl_updater imports are accessed through RLTrustSystem
from rl_scenario_generator import RLScenarioGenerator
from rl_updater import UpdateDecision
from robot_track_classes import Track


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
    max_tracks: int = 200  # Maximum number of tracks to pad to

    # Training schedule (safe starting point)
    num_episodes: int = 5000  # Total episodes for training
    steps_per_episode: int = 100
    update_every_episodes: int = 5  # How often to run PPO updates
    ppo_epochs: int = 4  # PPO epochs per update
    batch_size: int = 128  # Ego batch size (64-256 range, using middle value)
    
    # Trust system defaults from framework
    step_size: float = 1.0
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
    reward_clip_range: float = 200.0  # Reward clipping range
    baseline_step_scale: float = 0.5  # Reference step scale used for baseline comparison
    step_reward_scale: float = 1000.0  # Scale factor applied to per-step alignment improvements
    final_classification_weight: float = 100.0  # Scale factor for final classification bonus

    # Advantage processing (blueprint options)
    divide_advantage_by_ego_count: bool = True  # Divide by ego count for balanced gradient mass
    standardize_advantages: bool = True  # Advantage standardization over ego samples

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
        self.actor_optimizer = optim.Adam(actor_params, lr=config.lr)
        self.critic_optimizer = optim.Adam(critic_params, lr=config.lr)
        self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='min', factor=0.8, patience=10
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
        self.final_classification_scores = []
        self.final_accuracy_stats = []
        self.best_model_score = float('-inf')
        self.best_accuracy_stats = None

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
        self.ego_robot_track_mask = []  # List[Tensor[N_R, N_T]] - Robot-to-track adjacency masks

    def _build_actor_robot_inputs(self,
                                  ego_robots: List[Robot],
                                  agent_scores: Dict[int, float],
                                  robot_detections: Dict[int, List[str]],
                                  track_observers: Optional[Dict[str, List[int]]] = None) -> Tuple[List[int], torch.Tensor, Optional[torch.Tensor]]:
        """Create actor feature tensor and mask for ego robots"""
        if not ego_robots:
            empty = torch.zeros(0, 6, device=self.device)
            return [], empty, None

        robot_lookup = {robot.id: robot for robot in ego_robots}
        robot_ids = sorted(robot_lookup)
        participating_ids = set(robot_detections.keys())
        if track_observers:
            for observers in track_observers.values():
                participating_ids.update(observers)

        rows = []
        mask_vals = []
        for robot_id in robot_ids:
            robot = robot_lookup[robot_id]
            trust_mean = robot.trust_value
            strength = robot.trust_alpha + robot.trust_beta
            prev_trust = self.prev_robot_trusts.get(robot_id, 0.5)
            delta_trust = abs(trust_mean - prev_trust)
            agent_score = agent_scores.get(robot_id, 0.5)
            score_conf = 2 * abs(agent_score - 0.5)
            if track_observers:
                degree = sum(1 for observers in track_observers.values() if robot_id in observers)
            else:
                degree = len(robot.get_current_timestep_tracks())
            rows.append([trust_mean, strength, delta_trust, agent_score, score_conf, degree])
            mask_vals.append(1.0 if robot_id in participating_ids else 0.0)

        features = torch.tensor(rows, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(mask_vals, dtype=torch.float32, device=self.device)
        return robot_ids, features, mask_tensor

    def _build_actor_track_inputs(self,
                                  participating_tracks: List[Track],
                                  track_scores: Dict[str, float],
                                  track_detectors: Dict[str, List[int]]) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor]]:
        """Create actor feature tensor for detected tracks"""
        if not participating_tracks:
            empty = torch.zeros(0, 7, device=self.device)
            return [], empty, None

        track_lookup = {track.track_id: track for track in participating_tracks}
        track_ids = sorted(track_lookup)

        rows = []
        for track_id in track_ids:
            track = track_lookup[track_id]
            trust_mean = track.trust_value
            strength = track.trust_alpha + track.trust_beta
            prev_trust = self.prev_track_trusts.get(track_id, 0.5)
            delta_trust = abs(trust_mean - prev_trust)
            maturity = min(1.0, track.observation_count / 10.0)
            track_score = track_scores.get(track_id, 0.5)
            score_conf = 2 * abs(track_score - 0.5)
            degree = len(track_detectors.get(track_id, []))
            rows.append([trust_mean, strength, delta_trust, maturity, track_score, score_conf, degree])

        features = torch.tensor(rows, dtype=torch.float32, device=self.device)
        mask_tensor = torch.ones(features.shape[0], dtype=torch.float32, device=self.device)
        return track_ids, features, mask_tensor

    def _build_robot_track_relation_mask(self,
                                         robot_ids: List[int],
                                         track_ids: List[str],
                                         robot_lookup: Dict[int, Robot],
                                         track_lookup: Dict[str, Track],
                                         robot_detections: Dict[int, List[str]]) -> torch.Tensor:
        """Build adjacency mask marking which robots should attend to which tracks"""
        if not robot_ids or not track_ids:
            return torch.zeros((len(robot_ids), len(track_ids)), dtype=torch.float32, device=self.device)

        mask = torch.zeros((len(robot_ids), len(track_ids)), dtype=torch.float32, device=self.device)

        for r_idx, robot_id in enumerate(robot_ids):
            robot = robot_lookup[robot_id]
            detected_tracks = set(robot_detections.get(robot_id, []))
            for t_idx, track_id in enumerate(track_ids):
                track = track_lookup[track_id]
                in_detection = track_id in detected_tracks
                in_fov = robot.is_in_fov(track.position)
                if in_detection or in_fov:
                    mask[r_idx, t_idx] = 1.0

        return mask


    def compute_alignment_score(self, all_robots: List[Robot], ground_truth: Dict) -> float:
        """Average trust alignment with ground truth for robots and tracks."""
        total_score = 0.0
        entity_count = 0

        true_adversarial = set(ground_truth.get('adversarial_agents', []))
        true_false_tracks = set(ground_truth.get('false_tracks', []))

        for robot in all_robots:
            entity_count += 1
            if robot.id in true_adversarial:
                total_score += (1.0 - robot.trust_value)
            else:
                total_score += robot.trust_value

        for robot in all_robots:
            for track in robot.get_current_timestep_tracks():
                entity_count += 1

                if track.track_id in true_false_tracks:
                    total_score += (1.0 - track.trust_value)
                else:
                    total_score += track.trust_value

        return total_score / max(entity_count, 1)

    def compute_classification_score(self, all_robots: List[Robot], ground_truth: Dict) -> float:
        """Signed classification quality (F1 - uncertainty penalty)."""
        true_adversarial = set(ground_truth.get('adversarial_agents', []))
        all_robot_ids = {robot.id for robot in all_robots}
        true_legitimate = all_robot_ids - true_adversarial

        object_min_trust, false_object_ids = self._aggregate_track_objects(all_robots, ground_truth)
        object_ids = set(object_min_trust.keys())
        true_legitimate_objects = object_ids - false_object_ids

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

        predicted_legitimate_objects = set()
        predicted_false_objects = set()
        uncertain_objects = set()

        for obj_id, trust in object_min_trust.items():
            if trust >= self.config.track_positive_threshold:
                predicted_legitimate_objects.add(obj_id)
            elif trust <= self.config.track_negative_threshold:
                predicted_false_objects.add(obj_id)
            else:
                uncertain_objects.add(obj_id)

        robot_tp = len(true_legitimate & predicted_legitimate_robots)
        robot_fp = len(predicted_legitimate_robots & true_adversarial)
        robot_fn = len(true_legitimate & predicted_adversarial_robots)

        track_tp = len(true_legitimate_objects & predicted_legitimate_objects)
        track_fp = len(predicted_legitimate_objects & false_object_ids)
        track_fn = len(true_legitimate_objects & predicted_false_objects)

        robot_precision = robot_tp / (robot_tp + robot_fp) if (robot_tp + robot_fp) > 0 else 1.0
        robot_recall = robot_tp / (robot_tp + robot_fn) if (robot_tp + robot_fn) > 0 else 0.0
        robot_f1 = (
            2 * robot_precision * robot_recall / (robot_precision + robot_recall)
            if (robot_precision + robot_recall) > 0 else 0.0
        ) if true_legitimate else (1.0 if not predicted_legitimate_robots else 0.0)

        track_precision = track_tp / (track_tp + track_fp) if (track_tp + track_fp) > 0 else 1.0
        track_recall = track_tp / (track_tp + track_fn) if (track_tp + track_fn) > 0 else 0.0
        track_f1 = (
            2 * track_precision * track_recall / (track_precision + track_recall)
            if (track_precision + track_recall) > 0 else 0.0
        ) if true_legitimate_objects else (1.0 if not predicted_legitimate_objects else 0.0)

        total_entities = len(all_robot_ids) + len(object_ids)
        uncertainty_penalty = (
            (len(uncertain_robots) * 0.5 + len(uncertain_objects) * 0.5) / total_entities
            if total_entities > 0 else 0.0
        )

        base_reward = (robot_f1 + track_f1) / 2.0
        return base_reward - uncertainty_penalty

    def compute_episode_end_reward(self, all_robots: List[Robot], ground_truth: Dict) -> float:
        """
        Compute episode-end classification reward using dual thresholds
        - trust > positive_threshold = legitimate (positive class)
        - trust < negative_threshold = adversarial/false (negative class)
        - negative_threshold <= trust <= positive_threshold = uncertain (penalized)
        """
        classification_reward = self.compute_classification_score(all_robots, ground_truth)

        return max(0.0, classification_reward)  # Ensure non-negative reward

    def compute_classification_breakdown(self, all_robots: List[Robot], ground_truth: Dict) -> Dict:
        """Compute per-category classification accuracy for robots and tracks."""
        true_adversarial = set(ground_truth.get('adversarial_agents', []))
        object_min_trust, false_object_ids = self._aggregate_track_objects(all_robots, ground_truth)

        robot_legit_total = 0
        robot_legit_correct = 0
        robot_adv_total = 0
        robot_adv_correct = 0

        track_true_total = 0
        track_true_correct = 0
        track_false_total = 0
        track_false_correct = 0

        for robot in all_robots:
            trust_val = robot.trust_value
            if robot.id in true_adversarial:
                robot_adv_total += 1
                if trust_val <= self.config.robot_negative_threshold:
                    robot_adv_correct += 1
            else:
                robot_legit_total += 1
                if trust_val >= self.config.robot_positive_threshold:
                    robot_legit_correct += 1

        for obj_id, trust in object_min_trust.items():
            if obj_id in false_object_ids:
                track_false_total += 1
                if trust <= self.config.track_negative_threshold:
                    track_false_correct += 1
            else:
                track_true_total += 1
                if trust >= self.config.track_positive_threshold:
                    track_true_correct += 1

        def build_stats(correct: int, total: int) -> Dict[str, float]:
            return {
                'correct': correct,
                'total': total,
                'accuracy': (correct / total) if total > 0 else None
            }

        return {
            'robot_legitimate': build_stats(robot_legit_correct, robot_legit_total),
            'robot_adversarial': build_stats(robot_adv_correct, robot_adv_total),
            'track_true': build_stats(track_true_correct, track_true_total),
            'track_false': build_stats(track_false_correct, track_false_total)
        }

    def _aggregate_track_objects(self, all_robots: List[Robot], ground_truth: Dict) -> Tuple[Dict[str, float], Set[str]]:
        """Aggregate min trust per underlying object and identify false objects."""
        object_min_trust: Dict[str, float] = {}
        for robot in all_robots:
            for track in robot.get_current_timestep_tracks():
                obj_id = getattr(track, 'object_id', track.track_id)
                trust = track.trust_value
                if obj_id not in object_min_trust or trust < object_min_trust[obj_id]:
                    object_min_trust[obj_id] = trust

        false_object_ids: Set[str] = set()
        for track_key in ground_truth.get('false_tracks', []):
            parts = track_key.split('_', 1)
            if len(parts) == 2:
                false_object_ids.add(parts[1])

        if not false_object_ids:
            for obj_id in object_min_trust:
                if obj_id.startswith('fp_obj_'):
                    false_object_ids.add(obj_id)

        return object_min_trust, false_object_ids

    def compute_model_selection_score(self, final_classification: float, accuracy_stats: Dict[str, Dict[str, float]]) -> float:
        """Composite metric for deciding when to save best model."""
        total_correct = sum(stats['correct'] for stats in accuracy_stats.values())
        total_entities = sum(stats['total'] for stats in accuracy_stats.values())
        return final_classification * (1 + total_entities) + total_correct

    def build_global_state_tensor(self, robots: List[Robot], ground_truth: Dict,
                                 step_idx: int, total_steps: int) -> torch.Tensor:
        """Build compact Tier-0 summary driven by ground-truth alignment"""
        if not robots:
            return torch.zeros(8, device=self.device)

        true_adversarial = set(ground_truth.get('adversarial_agents', []))
        false_tracks = set(ground_truth.get('false_tracks', []))

        track_lookup = {}
        for robot in robots:
            for track in robot.get_current_timestep_tracks():
                track_lookup[track.track_id] = track

        adv_trusts = [robot.trust_value for robot in robots if robot.id in true_adversarial]
        legit_trusts = [robot.trust_value for robot in robots if robot.id not in true_adversarial]

        false_trusts = [track_lookup[tid].trust_value for tid in track_lookup if tid in false_tracks]
        true_trusts = [track_lookup[tid].trust_value for tid in track_lookup if tid not in false_tracks]

        def mean_with_default(values: List[float], default: float = 0.5) -> float:
            return float(np.mean(values)) if values else default

        total_robots = len(robots)
        total_tracks = len(track_lookup)
        adv_present = len(adv_trusts)
        false_present = len(false_trusts)

        step_progress = step_idx / max(total_steps, 1)
        prev_reward = self.prev_step_reward
        frac_adversarial = adv_present / max(total_robots, 1)
        frac_false_tracks = false_present / max(total_tracks, 1) if total_tracks > 0 else 0.0

        features = [
            step_progress,
            prev_reward,
            frac_adversarial,
            frac_false_tracks,
            mean_with_default(adv_trusts, 0.5),
            mean_with_default(legit_trusts, 0.5),
            mean_with_default(false_trusts, 0.5),
            mean_with_default(true_trusts, 0.5),
        ]

        self.prev_robot_trusts = {robot.id: robot.trust_value for robot in robots}
        self.prev_track_trusts = {track_id: track.trust_value for track_id, track in track_lookup.items()}

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def build_tier2_robot_features(self, robots: List[Robot], ground_truth: Dict) -> torch.Tensor:
        """Per-robot critic features using ground-truth labels"""
        if not robots:
            return torch.zeros(0, 3, device=self.device)

        true_adversarial = set(ground_truth.get('adversarial_agents', []))

        features = []
        for robot in robots:
            trust = robot.trust_value
            label = 0.0 if robot.id in true_adversarial else 1.0
            error = abs(trust - label)
            features.append([trust, label, error])

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def build_tier2_track_features(self, robots: List[Robot], ground_truth: Dict) -> torch.Tensor:
        """Per-track critic features using ground-truth labels"""
        track_lookup = {}
        for robot in robots:
            for track in robot.get_current_timestep_tracks():
                track_lookup[track.track_id] = track

        if not track_lookup:
            return torch.zeros(0, 3, device=self.device)

        false_tracks = set(ground_truth.get('false_tracks', []))

        features = []
        for track_id, track in track_lookup.items():
            trust = track.trust_value
            label = 0.0 if track_id in false_tracks else 1.0
            error = abs(trust - label)
            features.append([trust, label, error])

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def build_complete_global_state(self, robots: List[Robot], ground_truth: Dict,
                                   step_idx: int, total_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build global critic inputs from simplified ground-truth-driven features"""
        tier0_summary = self.build_global_state_tensor(robots, ground_truth, step_idx, total_steps)
        tier2_robot_features = self.build_tier2_robot_features(robots, ground_truth)
        tier2_track_features = self.build_tier2_track_features(robots, ground_truth)

        return tier0_summary, tier2_robot_features, tier2_track_features

    def pad_and_mask_tensors(self, robot_features: torch.Tensor, track_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad simplified robot/track features to the configured max counts"""
        device = self.device
        max_robots = self.config.max_robots
        max_tracks = self.config.max_tracks

        n_robots = robot_features.size(0)
        n_tracks = track_features.size(0)

        if n_robots > max_robots:
            raise ValueError(f"Number of robots ({n_robots}) exceeds maximum limit ({max_robots})")
        if n_tracks > max_tracks:
            raise ValueError(f"Number of tracks ({n_tracks}) exceeds maximum limit ({max_tracks})")

        padded_robot_features = torch.zeros(max_robots, 3, device=device)
        robot_mask = torch.zeros(max_robots, device=device)
        if n_robots > 0:
            padded_robot_features[:n_robots] = robot_features
            robot_mask[:n_robots] = 1.0

        padded_track_features = torch.zeros(max_tracks, 3, device=device)
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
        final_classification_score = 0.0
        final_accuracy_stats = {
            'robot_legitimate': {'correct': 0, 'total': 0, 'accuracy': None},
            'robot_adversarial': {'correct': 0, 'total': 0, 'accuracy': None},
            'track_true': {'correct': 0, 'total': 0, 'accuracy': None},
            'track_false': {'correct': 0, 'total': 0, 'accuracy': None}
        }

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

                # Pre-compute step decisions for ALL ego robots to maintain on-policy behavior
                precomputed_decisions = {}
                all_trajectory_data = []  # Will store data for ALL participating egos

                # Pull ground truth before applying updates for critic features
                ground_truth_pre_update = self.scenario_generator._extract_ground_truth(sim_env)

                alignment_before = self.compute_alignment_score(robots, ground_truth_pre_update)

                # Loop over all robots as ego robots (like in ego_sweep_step)
                for ego_robot in robots:
                    # Get GNN scores for this ego graph
                    scores = self.trust_system.evidence_extractor.get_scores(ego_robot, robots)

                    if not scores.agent_scores and not scores.track_scores:
                        continue  # No scores from GNN

                    # Get ego graph nodes (same logic as in ego_sweep_step)
                    ego_robot_ids = list(scores.agent_scores.keys())
                    ego_track_ids = list(scores.track_scores.keys())

                    ego_robots = [robot for robot in robots if robot.id in ego_robot_ids]
                    ego_tracks = [track_lookup[track_id] for track_id in ego_track_ids if track_id in track_lookup]

                    # Identify observers (detections or FoV sightings)
                    track_observers = {}
                    for track in ego_tracks:
                        detectors = set(track_detectors.get(track.track_id, []))
                        fov_watchers = {robot.id for robot in ego_robots if robot.is_in_fov(track.position)}
                        observers = detectors | fov_watchers
                        if observers:
                            track_observers[track.track_id] = sorted(observers)

                    observer_robot_ids = set(robot_detections.keys())
                    for observers in track_observers.values():
                        observer_robot_ids.update(observers)

                    participating_robots = [robot for robot in ego_robots if robot.id in observer_robot_ids]
                    participating_tracks = [track for track in ego_tracks if track.track_id in track_observers]

                    if not participating_robots and not participating_tracks:
                        continue  # No participating nodes in this ego graph

                    # Build actor inputs (robots include ego + proximal, tracks only detections)
                    robot_ids, robot_features_tensor, robot_mask = self._build_actor_robot_inputs(
                        ego_robots, scores.agent_scores, robot_detections, track_observers)
                    track_ids, track_features_tensor, track_mask = self._build_actor_track_inputs(
                        participating_tracks, scores.track_scores, track_observers)

                    robot_lookup_local = {robot.id: robot for robot in ego_robots}
                    track_lookup_local = {track.track_id: track for track in participating_tracks}
                    robot_track_mask = self._build_robot_track_relation_mask(
                        robot_ids, track_ids, robot_lookup_local, track_lookup_local, robot_detections
                    )

                    try:
                        robot_actions, track_actions, robot_log_probs, track_log_probs, entropy, _, _ = self.updater.policy.sample_actions(
                            robot_features_tensor.unsqueeze(0), track_features_tensor.unsqueeze(0),
                            robot_mask.unsqueeze(0) if robot_mask is not None else None,
                            track_mask.unsqueeze(0) if track_mask is not None else None,
                            robot_track_mask.unsqueeze(0) if robot_track_mask.numel() > 0 else None
                        )
                    except Exception as exc:
                        print("Actor sample failure:")
                        print("  robot_features shape:", robot_features_tensor.unsqueeze(0).shape)
                        print("  track_features shape:", track_features_tensor.unsqueeze(0).shape)
                        if robot_mask is not None:
                            print("  robot_mask shape:", robot_mask.unsqueeze(0).shape)
                        if track_mask is not None:
                            print("  track_mask shape:", track_mask.unsqueeze(0).shape)
                        raise
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
                    # Include IDs to maintain exact ordering consistency for the actor
                    ego_trajectory_data = [
                        robot_features_tensor.detach(),
                        track_features_tensor.detach(),
                        robot_actions.detach(),
                        track_actions.detach(),
                        robot_log_probs.detach(),
                        track_log_probs.detach(),
                        robot_ids.copy(),
                        track_ids.copy(),
                        robot_mask,
                        track_mask,
                        robot_track_mask.detach(),
                        float(entropy.detach())
                    ]
                    all_trajectory_data.append(ego_trajectory_data)

                # Build complete global state for the critic using simplified features
                tier0_summary, tier2_robot_features, tier2_track_features = self.build_complete_global_state(
                    robots, ground_truth_pre_update, step, scenario_params.episode_length
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
                        tier0_summary.unsqueeze(0),  # [1, 8]
                        padded_robot_features.unsqueeze(0),  # [1, max_robots, 3]
                        padded_track_features.unsqueeze(0),  # [1, max_tracks, 3]
                        robot_mask.unsqueeze(0),
                        track_mask.unsqueeze(0)
                    ).item()

                # Baseline comparison using fixed step scales
                baseline_alignment_delta = 0.0
                baseline_decisions = {}
                baseline_scale = max(0.0, float(self.config.baseline_step_scale))
                if baseline_scale > 0 and precomputed_decisions:
                    for ego_id, decision in precomputed_decisions.items():
                        baseline_robot_steps = {rid: baseline_scale for rid in decision.robot_steps}
                        baseline_track_steps = {tid: baseline_scale for tid in decision.track_steps}
                        if baseline_robot_steps or baseline_track_steps:
                            baseline_decisions[ego_id] = UpdateDecision(baseline_robot_steps, baseline_track_steps)

                if baseline_decisions:
                    robot_trust_snapshot = {
                        robot.id: (robot.trust_alpha, robot.trust_beta)
                        for robot in robots
                    }
                    track_trust_snapshot = {
                        track_id: (track.trust_alpha, track.trust_beta)
                        for track_id, track in track_lookup.items()
                    }

                    self.trust_system.update_trust(robots, baseline_decisions)
                    baseline_alignment_after = self.compute_alignment_score(robots, ground_truth_pre_update)
                    baseline_alignment_delta = baseline_alignment_after - alignment_before

                    # Restore original trust state before applying actual policy decisions
                    for robot in robots:
                        if robot.id in robot_trust_snapshot:
                            alpha, beta = robot_trust_snapshot[robot.id]
                            robot.trust_alpha = alpha
                            robot.trust_beta = beta
                    for robot in robots:
                        for track in robot.get_current_timestep_tracks():
                            if track.track_id in track_trust_snapshot:
                                alpha, beta = track_trust_snapshot[track.track_id]
                                track.trust_alpha = alpha
                                track.trust_beta = beta

                # Apply trust updates using policy decisions
                self.trust_system.update_trust(robots, precomputed_decisions)

                # Update ground truth for dynamic false tracks (changes as robots move)
                current_ground_truth = self.scenario_generator._extract_ground_truth(sim_env)

                alignment_after = self.compute_alignment_score(robots, current_ground_truth)

                alignment_delta = alignment_after - alignment_before
                improvement_alignment = alignment_delta - baseline_alignment_delta

                step_reward = improvement_alignment * self.config.step_reward_scale * self.config.reward_scale

                # REWARD WIRING: Add classification bonus to R_t at the last step (blueprint requirement)
                if step == scenario_params.episode_length - 1:
                    try:
                        episode_end_reward = self.compute_episode_end_reward(robots, current_ground_truth)
                        final_classification_score = episode_end_reward
                        final_accuracy_stats = self.compute_classification_breakdown(robots, current_ground_truth)
                        step_reward += episode_end_reward * self.config.final_classification_weight
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
                    (robot_features_tensor, track_features_tensor, robot_actions, track_actions,
                     robot_log_probs, track_log_probs, robot_ids, track_ids, robot_mask,
                     track_mask, robot_track_mask_tensor, entropy_value) = ego_trajectory_data

                    # Store ego-level data using new buffer format
                    ego_item_idx = len(self.ego_R_feats)
                    self.ego_R_feats.append(robot_features_tensor.detach())  # Tensor[N_R, F_R] - robot features
                    self.ego_T_feats.append(track_features_tensor.detach())  # Tensor[N_T, F_T] - track features
                    self.ego_a_R.append(robot_actions.detach())  # Tensor[N_R] - robot actions
                    self.ego_a_T.append(track_actions.detach())  # Tensor[N_T] - track actions
                    self.ego_logp_R.append(robot_log_probs.detach())  # Tensor[N_R] - robot log probs (old)
                    self.ego_logp_T.append(track_log_probs.detach())  # Tensor[N_T] - track log probs (old)
                    self.ego_entropy.append(float(entropy_value))  # float - entropy
                    self.ego_step_idx.append(current_step_idx)  # int - step index for broadcasting A_t
                    self.ego_robot_mask.append(robot_mask.detach() if robot_mask is not None else torch.empty(0, device=self.device))
                    self.ego_track_mask.append(track_mask.detach() if track_mask is not None else torch.empty(0, device=self.device))
                    self.ego_robot_track_mask.append(robot_track_mask_tensor.detach() if robot_track_mask_tensor.numel() > 0 else torch.empty(0, 0, device=self.device))

                    # Map step to ego item index
                    self.step_to_item_indices[current_step_idx].append(ego_item_idx)

            except Exception as e:
                print(f"Error in episode step {step}: {e}")
                break

        # Episode-end reward is now included in the final step above

        return episode_reward, final_classification_score, final_accuracy_stats

    def clear_trajectories(self):
        """Clear trajectory storage after PPO update"""
        # Clear step-level storage
        self.step_S.clear()
        self.step_V.clear()
        self.step_R.clear()
        self.step_done.clear()
        self.step_to_item_indices.clear()

        # Clear ego-level storage
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
        self.ego_robot_track_mask.clear()

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
            num_ego_items = len(self.ego_R_feats)
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
                        robot_features = self.ego_R_feats[i].to(self.device)  # [N_R, F_R] - variable N_R
                        track_features = self.ego_T_feats[i].to(self.device)  # [N_T, F_T] - variable N_T
                        robot_mask = self.ego_robot_mask[i].to(self.device) if self.ego_robot_mask[i].numel() > 0 else None
                        track_mask = self.ego_track_mask[i].to(self.device) if self.ego_track_mask[i].numel() > 0 else None
                        robot_track_mask = self.ego_robot_track_mask[i].to(self.device) if self.ego_robot_track_mask[i].numel() > 0 else None

                        robot_actions = self.ego_a_R[i].to(self.device)  # [N_R] - taken robot actions
                        track_actions = self.ego_a_T[i].to(self.device)  # [N_T] - taken track actions

                        logp_R_old = self.ego_logp_R[i].to(self.device)  # [N_R] - old log-probs
                        logp_T_old = self.ego_logp_T[i].to(self.device)  # [N_T] - old log-probs

                        # Get new log-probs from current policy using the simplified actor interface
                        logp_R_new, logp_T_new, entropy = self.updater.policy.evaluate_actions(
                            robot_features.unsqueeze(0), track_features.unsqueeze(0),
                            robot_actions.unsqueeze(0), track_actions.unsqueeze(0),
                            robot_mask.unsqueeze(0) if robot_mask is not None else None,
                            track_mask.unsqueeze(0) if track_mask is not None else None,
                            robot_track_mask.unsqueeze(0) if robot_track_mask is not None else None
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

                    actor_loss = avg_policy_loss - self.config.entropy_coef * avg_entropy
                    critic_loss = self.config.value_loss_coef * avg_value_loss

                    # Actor update
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.updater.policy.parameters(),
                        self.config.max_grad_norm
                    )
                    self.actor_optimizer.step()

                    # Critic update
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        self.config.max_grad_norm
                    )
                    self.critic_optimizer.step()

                    actor_loss_val = actor_loss.item()
                    critic_loss_val = critic_loss.item()
                    combined_loss = actor_loss_val + critic_loss_val

                    # Store losses for monitoring
                    self.ppo_losses.append(combined_loss)
                    self.policy_losses.append(actor_loss_val)
                    self.value_losses.append(critic_loss_val)
                    self.entropy_values.append(avg_entropy.item())

        self.total_updates += 1
        recent_actor = np.mean(self.policy_losses[-10:]) if self.policy_losses else 0.0
        recent_critic = np.mean(self.value_losses[-10:]) if self.value_losses else 0.0
        recent_total = np.mean(self.ppo_losses[-10:]) if self.ppo_losses else 0.0
        print(f"PPO update completed. Loss(actor/critic/total): {recent_actor:.4f} / {recent_critic:.4f} / {recent_total:.4f}")
        self.critic_scheduler.step(recent_critic)

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
                tier0_summary = global_state['tier0'].unsqueeze(0).to(self.device)  # [1, 8]
                tier2_robot_features = global_state['tier2_robots'].unsqueeze(0).to(self.device)  # [1, max_robots, 3]
                tier2_track_features = global_state['tier2_tracks'].unsqueeze(0).to(self.device)  # [1, max_tracks, 3]
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

    def train(self):
        """Main training loop"""
        print(f"Starting PPO training for RL Trust System")
        print(f"Episodes: {self.config.num_episodes}")

        for episode in range(self.config.num_episodes):
            self.episode_count = episode

            # Run episode
            start_time = time.time()
            episode_reward, final_classification, accuracy_stats = self.run_episode()
            episode_time = time.time() - start_time

            # Collect metrics
            self.episode_rewards.append(episode_reward)
            self.final_classification_scores.append(final_classification)
            self.final_accuracy_stats.append(accuracy_stats)

            # Update curriculum progression based on current performance
            self.scenario_generator.update_curriculum(final_classification)

            # Perform PPO update every few episodes (separate from ppo_epochs)
            if episode > 0 and episode % self.config.update_every_episodes == 0 and len(self.ego_R_feats) > 0:
                traj_count = len(self.ego_R_feats)
                self.trajectory_counts.append(traj_count)
                print(f"Running PPO update at episode {episode} with {traj_count} ego samples")
                self.ppo_update()
                self.clear_trajectories()

            # More frequent logging for monitoring
            if True:
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                avg_final_cls = np.mean(self.final_classification_scores[-10:]) if self.final_classification_scores else 0
                recent_actor_loss = np.mean(self.policy_losses[-10:]) if self.policy_losses else 0
                recent_critic_loss = np.mean(self.value_losses[-10:]) if self.value_losses else 0
                recent_total_loss = np.mean(self.ppo_losses[-10:]) if self.ppo_losses else 0
                actor_lr = self.actor_optimizer.param_groups[0]['lr']
                critic_lr = self.critic_optimizer.param_groups[0]['lr']
                traj_count = self.trajectory_counts[-1] if self.trajectory_counts else 0
                def fmt_acc(stats_key):
                    stats = accuracy_stats[stats_key]
                    if stats['total'] == 0 or stats['accuracy'] is None:
                        return 'N/A'
                    return f"{stats['accuracy']*100:.1f}%"

                print(f"Episode {episode}: Reward = {episode_reward:.4f}, Avg = {avg_reward:.4f}, FinalCls = {final_classification:.3f}, AvgCls = {avg_final_cls:.3f}")
                print(f"           Robot Acc (legit/adv) = {fmt_acc('robot_legitimate')}/{fmt_acc('robot_adversarial')} | "
                      f"Track Acc (true/false) = {fmt_acc('track_true')}/{fmt_acc('track_false')}")
                print(f"           Loss(actor/critic/total) = {recent_actor_loss:.4f}/{recent_critic_loss:.4f}/{recent_total_loss:.4f}")
                print(f"           Updates = {self.total_updates}, LR(actor/critic) = {actor_lr:.6f}/{critic_lr:.6f}, Trajectories = {traj_count}, Time = {episode_time:.2f}s")

            total_correct = sum(stats['correct'] for stats in accuracy_stats.values())
            total_entities = sum(stats['total'] for stats in accuracy_stats.values())
            selection_score = self.compute_model_selection_score(final_classification, accuracy_stats)

            # Save best model based on composite selection score
            if selection_score > self.best_model_score:
                self.best_model_score = selection_score
                self.best_reward = episode_reward
                self.best_accuracy_stats = copy.deepcopy(accuracy_stats)
                torch.save(self.updater.policy.state_dict(), 'rl_trust_model.pth')
                print(f"New best model saved: reward={episode_reward:.3f}, finalCls={final_classification:.3f}, "
                      f"correct={total_correct}, total={total_entities}, score={selection_score:.2f}")

        # Save final model
        torch.save(self.updater.policy.state_dict(), 'rl_trust_model_final.pth')

        # Plot training rewards
        self.plot_training_rewards()

        print(f"Training completed. Best reward: {self.best_reward:.4f}")
        print(f"Final actor LR: {self.actor_optimizer.param_groups[0]['lr']:.6f}")
        print(f"Final critic LR: {self.critic_optimizer.param_groups[0]['lr']:.6f}")
        print(f"Total PPO updates: {self.total_updates}")
        print(f"Average trajectory count: {np.mean(self.trajectory_counts) if self.trajectory_counts else 0:.1f}")
        if self.final_classification_scores:
            print(f"Average final classification score: {np.mean(self.final_classification_scores):.4f}")
        if self.best_accuracy_stats:
            print("Best model classification stats:")
            for key, stats in self.best_accuracy_stats.items():
                acc = stats['accuracy']
                acc_str = f"{acc*100:.1f}%" if acc is not None else 'N/A'
                print(f"  {key}: {stats['correct']}/{stats['total']} ({acc_str})")
        if self.final_accuracy_stats:
            def average_accuracy(key: str) -> float:
                values = [stats[key]['accuracy'] for stats in self.final_accuracy_stats if stats[key]['accuracy'] is not None]
                return float(np.mean(values)) if values else None

            avg_robot_legit = average_accuracy('robot_legitimate')
            avg_robot_adv = average_accuracy('robot_adversarial')
            avg_track_true = average_accuracy('track_true')
            avg_track_false = average_accuracy('track_false')

            def fmt_avg(value):
                return f"{value*100:.1f}%" if value is not None else 'N/A'

            print("Average classification accuracy:")
            print(f"  Robot legitimate: {fmt_avg(avg_robot_legit)}")
            print(f"  Robot adversarial: {fmt_avg(avg_robot_adv)}")
            print(f"  Track true: {fmt_avg(avg_track_true)}")
            print(f"  Track false: {fmt_avg(avg_track_false)}")

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
        print(f"   • Final Actor LR: {self.actor_optimizer.param_groups[0]['lr']:.6f}")
        print(f"   • Final Critic LR: {self.critic_optimizer.param_groups[0]['lr']:.6f}")
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
