#!/usr/bin/env python3
"""
PPO Training for RL Trust System

Trains the updater policy using PPO on scenarios from rl_scenario_generator.py
Follows the exact framework with proper reward computation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List
from dataclasses import dataclass

from robot_track_classes import Robot
from rl_trust_system import RLTrustSystem
# rl_updater imports are accessed through RLTrustSystem
from rl_scenario_generator import RLScenarioGenerator

@dataclass
class TrainingConfig:
    """Training configuration following the framework defaults"""
    # PPO hyperparameters (conservative/stable settings)
    lr: float = 3e-4  # Standard PPO learning rate
    gamma: float = 0.99  # Standard discount factor
    gae_lambda: float = 0.95  # Standard GAE lambda
    clip_epsilon: float = 0.2  # Standard PPO clipping
    value_loss_coef: float = 0.5  # Standard value loss coefficient
    entropy_coef: float = 0.005  # Lower entropy for more focused learning
    max_grad_norm: float = 0.5  # Conservative gradient clipping

    # Training schedule
    num_episodes: int = 1000  # Start with just 100 episodes for testing
    steps_per_episode: int = 100
    update_every_episodes: int = 10  # How often to run PPO updates (separate from ppo_epochs)
    ppo_epochs: int = 4  # Inner gradient passes per update
    batch_size: int = 64
    
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

    # KL divergence control
    target_kl: float = 0.02  # Early stop epoch if KL exceeds this
    kl_penalty_coef: float = 0.1  # Optional KL penalty coefficient


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
        self.optimizer = optim.Adam(self.updater.policy.parameters(), lr=config.lr)

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
        self.kl_divergences = []
        self.total_updates = 0
        self.trajectory_counts = []
        self.early_stops = 0

        # PPO trajectory storage
        self.trajectories = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }


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

    def run_episode(self) -> float:
        """Run one training episode"""

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

                # Pre-compute step decisions for ALL ego robots to maintain on-policy behavior
                precomputed_decisions = {}
                all_trajectory_data = []  # Will store data for ALL participating egos

                # Get detection info for this step
                robot_detections, track_detectors = self.trust_system.get_detections_this_step(robots)

                # Build track lookup
                track_lookup = {}
                for robot in robots:
                    for track in robot.get_current_timestep_tracks():
                        track_lookup[track.track_id] = track

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

                    # Get participating nodes (those that detected/were detected this step)
                    participating_robots = [robot for robot in ego_robots if robot.id in robot_detections]
                    participating_tracks = [track for track in ego_tracks if track.track_id in track_detectors]

                    if not participating_robots and not participating_tracks:
                        continue  # No participating nodes in this ego graph

                    # Use the updater's method to get step decision with our policy
                    step_decision = self.updater.get_step_scales(
                        ego_robots, ego_tracks, participating_robots, participating_tracks,
                        scores.agent_scores, scores.track_scores
                    )

                    precomputed_decisions[ego_robot.id] = step_decision

                    # Collect trajectory data for EVERY participating ego robot
                    # Encode state for this ego
                    state = self.encode_state(ego_robots, ego_tracks, scores.agent_scores, scores.track_scores)

                    # Get the actual actions used (convert step decision back to tensors for consistency)
                    robot_ids = [r.id for r in participating_robots]
                    track_ids = [t.track_id for t in participating_tracks]

                    robot_actions = torch.tensor([step_decision.robot_steps.get(rid, 0.0) for rid in robot_ids], dtype=torch.float32)
                    track_actions = torch.tensor([step_decision.track_steps.get(tid, 0.0) for tid in track_ids], dtype=torch.float32)

                    # Get features for the used actions
                    robot_features = []
                    for robot in participating_robots:
                        trust_mean = robot.trust_value
                        trust_conf = min(1.0, (robot.trust_alpha + robot.trust_beta) / 20.0)
                        robot_score = scores.agent_scores.get(robot.id, 0.5)
                        robot_features.append([trust_mean, trust_conf, robot_score])

                    track_features = []
                    for track in participating_tracks:
                        trust_mean = track.trust_value
                        trust_conf = min(1.0, (track.trust_alpha + track.trust_beta) / 20.0)
                        track_score = scores.track_scores.get(track.track_id, 0.5)
                        maturity = min(1.0, track.observation_count / 10.0)
                        track_features.append([trust_mean, trust_conf, track_score, maturity])

                    robot_features_tensor = torch.tensor(robot_features, dtype=torch.float32, device=self.device) if robot_features else torch.zeros(0, 3, device=self.device)
                    track_features_tensor = torch.tensor(track_features, dtype=torch.float32, device=self.device) if track_features else torch.zeros(0, 4, device=self.device)

                    # Evaluate the log probabilities for the actions that were actually used
                    # Ensure all tensors are on the same device before policy call
                    summary_tensor = state.unsqueeze(0).to(self.device)
                    robot_features_tensor = robot_features_tensor.to(self.device)
                    track_features_tensor = track_features_tensor.to(self.device)
                    robot_actions = robot_actions.to(self.device)
                    track_actions = track_actions.to(self.device)

                    _, _, _, robot_log_probs, track_log_probs = self.updater.policy.evaluate_actions(
                        summary_tensor, robot_features_tensor, track_features_tensor,
                        robot_actions, track_actions
                    )

                    # Store trajectory data for this ego (detached to avoid gradient conflicts)
                    # Include IDs to maintain exact ordering consistency
                    ego_trajectory_data = [state.detach(), robot_features_tensor.detach(), track_features_tensor.detach(),
                                         robot_actions.detach(), track_actions.detach(), robot_log_probs.detach(), track_log_probs.detach(),
                                         robot_ids.copy(), track_ids.copy()]  # Store IDs for ordering verification
                    all_trajectory_data.append(ego_trajectory_data)

                # Apply trust updates using precomputed decisions
                self.trust_system.update_trust(robots, precomputed_decisions)

                # Update ground truth for dynamic false tracks (changes as robots move)
                current_ground_truth = self.scenario_generator._extract_ground_truth(sim_env)

                # Compute step reward based on trust alignment with current ground truth
                raw_step_reward = self.compute_step_reward(robots, current_ground_truth)
                step_reward = raw_step_reward * self.config.reward_scale

                # Add episode-end classification reward to the final step
                if step == scenario_params.episode_length - 1:
                    try:
                        episode_end_reward = self.compute_episode_end_reward(robots, current_ground_truth)
                        # Add with moderate weight (3x) to avoid swamping step learning
                        step_reward += episode_end_reward * 3.0 * self.config.reward_scale
                    except Exception as e:
                        print(f"Error computing episode-end reward in step: {e}")

                # Apply reward clipping at step level if enabled
                if self.config.clip_rewards:
                    step_reward = max(-self.config.reward_clip_range, min(self.config.reward_clip_range, step_reward))

                episode_reward += step_reward

                # Collect trajectory data for ALL participating egos
                for ego_trajectory_data in all_trajectory_data:
                    state, robot_features_tensor, track_features_tensor, robot_actions, track_actions, robot_log_probs, track_log_probs, robot_ids, track_ids = ego_trajectory_data

                    # Get value estimate for this specific ego (detach to avoid gradient conflicts)
                    with torch.no_grad():
                        value = self.updater.policy.get_value(state.unsqueeze(0).to(self.device))

                    # Collect trajectory data with complete state information
                    # All egos share the same step reward (global environment reward)
                    # Treat as contextual bandit: each ego decision is independent/terminal
                    self.collect_trajectory_step(
                        state, robot_features_tensor, track_features_tensor,
                        robot_actions, track_actions, robot_log_probs, track_log_probs,
                        step_reward, value, True,  # Always done=True for contextual bandit
                        robot_ids, track_ids
                    )

            except Exception as e:
                print(f"Error in episode step {step}: {e}")
                break

        # Episode-end reward is now included in the final step above


        return episode_reward

    def clear_trajectories(self):
        """Clear trajectory storage after PPO update"""
        for key in self.trajectories:
            self.trajectories[key].clear()

    def ppo_update(self):
        """Perform PPO policy update using collected trajectories"""
        if len(self.trajectories['rewards']) == 0:
            return

        # Extract trajectory components
        states = self.trajectories['states']  # List of state dicts
        actions = self.trajectories['actions']  # List of action dicts
        old_log_probs = self.trajectories['log_probs']  # List of log_prob dicts
        rewards = torch.tensor(self.trajectories['rewards'], dtype=torch.float32).to(self.device)
        values = torch.stack(self.trajectories['values']).to(self.device)
        dones = torch.tensor(self.trajectories['dones'], dtype=torch.float32).to(self.device)

        # Apply reward clipping if enabled
        if self.config.clip_rewards:
            rewards = torch.clamp(rewards, -self.config.reward_clip_range, self.config.reward_clip_range)

        # Compute advantages using GAE (detach values to avoid gradient issues)
        advantages, returns = self.compute_gae(rewards, values.detach(), dones)

        # More robust advantage normalization
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages  # Don't normalize single advantage

        # PPO update loop with KL watchdog
        for epoch in range(self.config.ppo_epochs):
            # Create random mini-batches
            batch_size = min(self.config.batch_size, len(states))
            indices = torch.randperm(len(states))

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Process each item in the batch
                total_policy_loss = 0.0
                total_value_loss = 0.0
                total_entropy = 0.0
                total_kl_div = 0.0
                batch_count = 0

                for idx, i in enumerate(batch_indices):
                    try:
                        # Extract state components
                        state_dict = states[i]
                        action_dict = actions[i]
                        old_log_prob_dict = old_log_probs[i]

                        summary = state_dict['summary'].unsqueeze(0).to(self.device)
                        robot_features = state_dict['robot_features'].to(self.device)
                        track_features = state_dict['track_features'].to(self.device)

                        robot_actions = action_dict['robot_actions'].to(self.device)
                        track_actions = action_dict['track_actions'].to(self.device)

                        old_robot_log_probs = old_log_prob_dict['robot_log_probs'].to(self.device)
                        old_track_log_probs = old_log_prob_dict['track_log_probs'].to(self.device)

                        # Evaluate the stored actions using current policy
                        _, value_pred, entropy, new_robot_log_probs, new_track_log_probs = self.updater.policy.evaluate_actions(
                            summary, robot_features, track_features, robot_actions, track_actions
                        )

                        # Compute total log-probs using identical method for old vs new
                        new_log_prob_total = new_robot_log_probs.sum() + new_track_log_probs.sum()
                        old_log_prob_total = old_robot_log_probs.sum() + old_track_log_probs.sum()

                        # Compute KL divergence (approximate)
                        kl_div = (old_log_prob_total - new_log_prob_total)

                        # Compute policy loss for this sample
                        ratio = torch.exp(new_log_prob_total - old_log_prob_total)
                        advantage = batch_advantages[idx]  # Use position in batch, not batch_count
                        surr1 = ratio * advantage
                        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantage
                        policy_loss = -torch.min(surr1, surr2)

                        # Compute value loss for this sample
                        target_return = batch_returns[idx]  # Use position in batch, not batch_count

                        # Ensure both are scalars, preserving gradients
                        # value_pred should be scalar from evaluate_actions
                        value_pred_flat = value_pred.view(-1)
                        target_return_flat = target_return.view(-1) if target_return.numel() > 1 else target_return.unsqueeze(0)

                        # Take first element if needed (should be only 1 element)
                        value_pred_scalar = value_pred_flat[0]
                        target_return_scalar = target_return_flat[0]

                        # Use Huber loss (SmoothL1) for more robust critic training
                        value_loss = torch.nn.SmoothL1Loss()(value_pred_scalar.unsqueeze(0), target_return_scalar.unsqueeze(0))

                        # Use computed entropy
                        entropy = entropy

                        total_policy_loss += policy_loss
                        total_value_loss += value_loss
                        total_entropy += entropy
                        total_kl_div += kl_div
                        batch_count += 1

                    except Exception as e:
                        print(f"Warning: Failed to process sample {i}: {e}")
                        continue

                if batch_count > 0:
                    # Average losses over the batch
                    avg_policy_loss = total_policy_loss / batch_count
                    avg_value_loss = total_value_loss / batch_count
                    avg_entropy = total_entropy / batch_count
                    avg_kl_div = total_kl_div / batch_count

                    # KL divergence watchdog - early stop if KL too high
                    approx_kl = avg_kl_div.item()
                    if approx_kl > self.config.target_kl:
                        print(f"Early stopping epoch {epoch} due to high KL divergence: {approx_kl:.4f}")
                        self.early_stops += 1
                        return  # Early stop this entire PPO update

                    # Total loss with optional KL penalty
                    loss = avg_policy_loss + self.config.value_loss_coef * avg_value_loss - self.config.entropy_coef * avg_entropy
                    if self.config.kl_penalty_coef > 0:
                        loss += self.config.kl_penalty_coef * avg_kl_div

                    # Store losses for monitoring
                    self.ppo_losses.append(loss.item())
                    self.policy_losses.append(avg_policy_loss.item())
                    self.value_losses.append(avg_value_loss.item())
                    self.entropy_values.append(avg_entropy.item())
                    self.kl_divergences.append(approx_kl)

                    # Update policy
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.updater.policy.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

        self.total_updates += 1
        recent_kl = np.mean(self.kl_divergences[-10:]) if self.kl_divergences else 0
        print(f"PPO update completed. Total loss: {np.mean(self.ppo_losses[-10:]):.4f}, KL: {recent_kl:.4f}, Early stops: {self.early_stops}")

    def compute_gae(self, rewards, values, _):
        """Compute advantages for contextual bandit (no bootstrapping)"""
        # Ensure values is 1D for easier computation
        if len(values.shape) > 1:
            values = values.squeeze()

        # For contextual bandit: advantages = rewards - baseline (no GAE)
        # Each decision is independent, so no temporal credit assignment needed
        advantages = rewards - values
        returns = rewards  # No discounting for independent decisions

        return advantages, returns

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
                               reward, value, done, robot_ids=None, track_ids=None):
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
            if episode > 0 and episode % self.config.update_every_episodes == 0 and len(self.trajectories['rewards']) > 0:
                traj_count = len(self.trajectories['rewards'])
                self.trajectory_counts.append(traj_count)
                print(f"Running PPO update at episode {episode} with {traj_count} trajectory steps")
                self.ppo_update()
                self.clear_trajectories()

                # Update learning rate based on recent performance
                recent_avg = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
                self.scheduler.step(recent_avg)

            # More frequent logging for monitoring
            if episode % 10 == 0 or episode < 50:
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                recent_loss = np.mean(self.ppo_losses[-10:]) if self.ppo_losses else 0
                recent_kl = np.mean(self.kl_divergences[-10:]) if self.kl_divergences else 0
                current_lr = self.optimizer.param_groups[0]['lr']
                traj_count = self.trajectory_counts[-1] if self.trajectory_counts else 0
                print(f"Episode {episode}: Reward = {episode_reward:.4f}, Avg = {avg_reward:.4f}, Loss = {recent_loss:.4f}, KL = {recent_kl:.4f}")
                print(f"           Updates = {self.total_updates}, LR = {current_lr:.6f}, Trajectories = {traj_count}, Early stops = {self.early_stops}, Time = {episode_time:.2f}s")

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
        print(f"Early stops due to high KL: {self.early_stops}")
        print(f"Average trajectory count: {np.mean(self.trajectory_counts) if self.trajectory_counts else 0:.1f}")
        print(f"Average KL divergence: {np.mean(self.kl_divergences) if self.kl_divergences else 0:.4f}")

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
    config = TrainingConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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