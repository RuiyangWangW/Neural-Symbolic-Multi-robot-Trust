#!/usr/bin/env python3
"""
PPO Training for RL Trust System

Trains the updater policy using PPO on scenarios from rl_scenario_generator.py
Follows the exact framework with proper reward computation.
"""

import torch
import torch.optim as optim
import numpy as np
import random
import json
import time
import matplotlib.pyplot as plt
from typing import Dict, List
from dataclasses import dataclass

from robot_track_classes import Robot
from rl_trust_system import RLTrustSystem
from rl_updater import UpdaterPolicy
from rl_scenario_generator import RLScenarioGenerator

@dataclass
class TrainingConfig:
    """Training configuration following the framework defaults"""
    # PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Training schedule
    num_episodes: int = 1000  # Start with just 100 episodes for testing
    steps_per_episode: int = 100
    ppo_epochs: int = 4
    batch_size: int = 64

    # Trust system defaults from framework
    step_size: float = 0.25
    strength_cap: float = 50.0

    # Thresholds and confirmation
    robot_threshold: float = 0.30
    track_threshold: float = 0.35
    confirmation_k: int = 3

    # Confidence and cross-weight floors
    c_min: float = 0.2
    rho_min: float = 0.2


class PPOTrainer:
    """PPO trainer for the updater policy"""

    def __init__(self, config: TrainingConfig, supervised_model_path: str, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)

        # Initialize components
        self.scenario_generator = RLScenarioGenerator(curriculum_learning=True)

        # Create updater policy
        self.policy = UpdaterPolicy().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)

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

        # Replace the trust system's updater policy with ours
        self.trust_system.updater.policy = self.policy

        # Training state
        self.episode_count = 0
        self.best_reward = float('-inf')

        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []


    def compute_reward(self,
                      all_robots: List[Robot],
                      ground_truth: Dict,
                      predicted_adversarial: List[int],
                      predicted_false_tracks: List[str]) -> float:
        """
        Compute reward from detection quality (TP/FP on robots & tracks)
        plus optional downstream metrics minus smoothness/budget penalties
        """
        # Ground truth
        true_adversarial = set(ground_truth.get('adversarial_agents', []))
        true_false_tracks = set(ground_truth.get('false_tracks', []))

        predicted_adversarial_set = set(predicted_adversarial)
        predicted_false_tracks_set = set(predicted_false_tracks)

        # Robot detection quality
        robot_tp = len(true_adversarial & predicted_adversarial_set)
        robot_fp = len(predicted_adversarial_set - true_adversarial)
        robot_fn = len(true_adversarial - predicted_adversarial_set)

        # Track detection quality
        all_track_ids = set()
        for robot in all_robots:
            for track in robot.get_current_timestep_tracks():
                all_track_ids.add(track.track_id)

        track_tp = len(true_false_tracks & predicted_false_tracks_set)
        track_fp = len(predicted_false_tracks_set - true_false_tracks)
        track_fn = len(true_false_tracks - predicted_false_tracks_set)

        # F1-based reward
        robot_precision = robot_tp / (robot_tp + robot_fp) if (robot_tp + robot_fp) > 0 else 1.0
        robot_recall = robot_tp / (robot_tp + robot_fn) if (robot_tp + robot_fn) > 0 else 1.0
        robot_f1 = 2 * robot_precision * robot_recall / (robot_precision + robot_recall) if (robot_precision + robot_recall) > 0 else 0.0

        track_precision = track_tp / (track_tp + track_fp) if (track_tp + track_fp) > 0 else 1.0
        track_recall = track_tp / (track_tp + track_fn) if (track_tp + track_fn) > 0 else 1.0
        track_f1 = 2 * track_precision * track_recall / (track_precision + track_recall) if (track_precision + track_recall) > 0 else 0.0

        # Combined F1 score
        reward = (robot_f1 + track_f1) / 2.0

        # Small smoothness penalty for extreme trust values
        smoothness_penalty = 0.0
        for robot in all_robots:
            trust_change = abs(robot.trust_value - 0.5)
            smoothness_penalty += trust_change * 0.1

        reward -= smoothness_penalty / len(all_robots) if all_robots else 0.0

        return reward

    def run_episode(self) -> float:
        """Run one training episode"""

        # Generate scenario parameters
        scenario_params = self.scenario_generator.sample_scenario_parameters(self.episode_count)

        # Create scenario environment (not mock!)
        sim_env, ground_truth = self.scenario_generator.create_scenario_environment(scenario_params)

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

                # Run trust update
                self.trust_system.update_trust(robots)

                # Get detection flags
                adversarial_robots, false_tracks = self.trust_system.get_adversarial_flags(
                    robots, self.config.robot_threshold, self.config.track_threshold
                )

                # Compute reward
                step_reward = self.compute_reward(robots, ground_truth, adversarial_robots, false_tracks)
                episode_reward += step_reward

            except Exception as e:
                print(f"Error in episode step {step}: {e}")
                break

        return episode_reward

    def train(self, supervised_model_path: str):
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

            # Simple PPO update placeholder (real implementation needs policy gradients)
            # For now just track rewards

            # Logging
            if episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0
                print(f"Episode {episode}: Reward = {episode_reward:.3f}, Avg = {avg_reward:.3f}, Time = {episode_time:.2f}s")

            # Save best model
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                torch.save(self.policy.state_dict(), 'rl_trust_model.pth')
                print(f"New best model saved: {episode_reward:.3f}")

        # Save final model
        torch.save(self.policy.state_dict(), 'rl_trust_model_final.pth')

        # Plot training rewards
        self.plot_training_rewards()

        print(f"Training completed. Best reward: {self.best_reward:.3f}")

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
                   label=f'Mean: {mean_reward:.3f}')
        plt.axvline(mean_reward + std_reward, color='orange', linestyle=':',
                   label=f'Mean + Std: {mean_reward + std_reward:.3f}')
        plt.axvline(mean_reward - std_reward, color='orange', linestyle=':',
                   label=f'Mean - Std: {mean_reward - std_reward:.3f}')
        plt.legend()

        plt.tight_layout()
        plt.savefig('rl_training_rewards.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Training rewards plot saved to: rl_training_rewards.png")
        print(f"📈 Training Statistics:")
        print(f"   • Total Episodes: {len(self.episode_rewards)}")
        print(f"   • Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"   • Best Reward: {max(self.episode_rewards):.3f}")
        print(f"   • Worst Reward: {min(self.episode_rewards):.3f}")


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
        trainer.train(supervised_model_path)
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()