#!/usr/bin/env python3
"""
RL Scenario Generator

This module generates diverse scenarios for RL training episodes, similar to the approach
used in generate_supervised_data.py but adapted for RL training with dynamic environments.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
from collections import deque
from simulation_environment import SimulationEnvironment


@dataclass
class ScenarioParameters:
    """Parameters for a single RL training scenario"""
    num_robots: int
    num_targets: int
    adversarial_ratio: float
    false_positive_rate: float
    false_negative_rate: float
    world_size: Tuple[float, float]
    proximal_range: float
    episode_length: int
    difficulty_level: float  # 0.0 = easy, 1.0 = hard


class RLScenarioGenerator:
    """
    Generates diverse scenarios for RL training episodes with curriculum learning support
    """

    def __init__(self,
                 num_robots_range: Tuple[int, int] = (3, 10),
                 num_targets_range: Tuple[int, int] = (10, 30),
                 adversarial_ratio_range: Tuple[float, float] = (0.2, 0.5),
                 false_positive_rate_range: Tuple[float, float] = (0.1, 0.7),
                 false_negative_rate_range: Tuple[float, float] = (0.0, 0.3),
                 world_size_range: Tuple[Tuple[float, float], Tuple[float, float]] = ((40, 40), (100, 100)),
                 proximal_range: float = 50.0,
                 fov_range: float = 50.0,
                 fov_angle: float = np.pi/3,
                 episode_length: int = 100,  # Fixed episode length (matches steps_per_episode in train_rl_trust.py)
                 curriculum_learning: bool = True,
                 curriculum_levels: int = 6,
                 curriculum_window_size: int = 20,
                 curriculum_min_episodes: int = 10,
                 curriculum_progress_threshold: float = 0.7):
        """
        Initialize RL scenario generator

        Args:
            num_robots_range: Range of number of robots (min, max)
            num_targets_range: Range of number of targets (min, max)
            adversarial_ratio_range: Range of adversarial robot ratios (min, max)
            false_positive_rate_range: Range of false positive rates (min, max)
            false_negative_rate_range: Range of false negative rates (min, max)
            world_size_range: Range of world sizes ((min_x, min_y), (max_x, max_y))
            proximal_range: Communication/sensing range
            fov_range: Field of view range
            fov_angle: Field of view angle
            episode_length: Fixed episode length (should match steps_per_episode in train_rl_trust.py)
            curriculum_learning: Whether to use curriculum learning
            curriculum_levels: Number of discrete curriculum stages from easy to hard
            curriculum_window_size: How many recent episodes to consider when judging performance
            curriculum_min_episodes: Minimum episodes to run at a level before considering promotion
            curriculum_progress_threshold: Average performance required to advance to the next level
        """
        self.num_robots_range = num_robots_range
        self.num_targets_range = num_targets_range
        self.adversarial_ratio_range = adversarial_ratio_range
        self.false_positive_rate_range = false_positive_rate_range
        self.false_negative_rate_range = false_negative_rate_range
        self.world_size_range = world_size_range
        self.proximal_range = proximal_range
        self.fov_range = fov_range
        self.fov_angle = fov_angle
        self.episode_length = episode_length
        self.curriculum_learning = curriculum_learning

        # Curriculum progression settings
        curriculum_levels = max(2, curriculum_levels)
        self.level_complexities = np.linspace(0.0, 1.0, curriculum_levels)
        self.curriculum_level = 0
        self.curriculum_window = deque(maxlen=max(1, curriculum_window_size))
        self.curriculum_min_episodes = max(1, curriculum_min_episodes)
        self.curriculum_progress_threshold = curriculum_progress_threshold
        self.episodes_in_level = 0

    def sample_scenario_parameters(self, episode_num: int) -> ScenarioParameters:
        """
        Sample parameters for a new scenario

        Args:
            episode_num: Current episode number for curriculum learning

        Returns:
            ScenarioParameters for the episode
        """
        if self.curriculum_learning:
            complexity = float(self.level_complexities[self.curriculum_level])
        else:
            complexity = random.uniform(0.0, 1.0)

        # Base parameter sampling with increments (like supervised data generation)
        params = self._sample_base_parameters()

        # Apply curriculum scaling
        params = self._apply_curriculum_scaling(params, complexity)

        return ScenarioParameters(
            num_robots=params['num_robots'],
            num_targets=params['num_targets'],
            adversarial_ratio=params['adversarial_ratio'],
            false_positive_rate=params['false_positive_rate'],
            false_negative_rate=params['false_negative_rate'],
            world_size=params['world_size'],
            proximal_range=self.proximal_range,
            episode_length=params['episode_length'],
            difficulty_level=complexity  # Low complexity = low difficulty, high complexity = high difficulty
        )

    def _sample_base_parameters(self) -> Dict:
        """Sample base parameters with increments like supervised data generation"""

        # Sample num_robots
        num_robots = random.randint(self.num_robots_range[0], self.num_robots_range[1])

        # Sample num_targets with increment of 5
        min_targets, max_targets = self.num_targets_range
        target_steps = (max_targets - min_targets) // 5
        if target_steps > 0:
            target_step = random.randint(0, target_steps)
            num_targets = min_targets + (target_step * 5)
        else:
            num_targets = min_targets

        # Sample adversarial_ratio with increment of 0.1
        min_adv, max_adv = self.adversarial_ratio_range
        adv_steps = int((max_adv - min_adv) / 0.1)
        if adv_steps > 0:
            adv_step = random.randint(0, adv_steps)
            adversarial_ratio = round(min_adv + (adv_step * 0.1), 1)
        else:
            adversarial_ratio = min_adv

        # Sample false_positive_rate with increment of 0.1
        min_fp, max_fp = self.false_positive_rate_range
        fp_steps = int((max_fp - min_fp) / 0.1)
        if fp_steps > 0:
            fp_step = random.randint(0, fp_steps)
            false_positive_rate = round(min_fp + (fp_step * 0.1), 1)
        else:
            false_positive_rate = min_fp

        # Sample false_negative_rate with increment of 0.1
        min_fn, max_fn = self.false_negative_rate_range
        fn_steps = int((max_fn - min_fn) / 0.1)
        if fn_steps > 0:
            fn_step = random.randint(0, fn_steps)
            false_negative_rate = round(min_fn + (fn_step * 0.1), 1)
        else:
            false_negative_rate = min_fn

        # Sample world_size with increment of 10
        min_world, max_world = self.world_size_range
        min_size, max_size = min_world[0], max_world[0]
        size_steps = int((max_size - min_size) / 10)
        if size_steps > 0:
            size_step = random.randint(0, size_steps)
            world_size = min_size + (size_step * 10)
        else:
            world_size = min_size

        # Use fixed episode length
        episode_length = self.episode_length

        return {
            'num_robots': num_robots,
            'num_targets': num_targets,
            'adversarial_ratio': adversarial_ratio,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'world_size': (world_size, world_size),
            'episode_length': episode_length
        }

    def _apply_curriculum_scaling(self, params: Dict, complexity: float) -> Dict:
        """
        Apply curriculum learning scaling to parameters

        Args:
            complexity: Current complexity level (0.0 = start with fewer robots/targets, 1.0 = end with lots)
        """

        if not self.curriculum_learning:
            return params

        # Start with MIN robots/targets and gradually increase to larger scenarios
        robot_range_size = self.num_robots_range[1] - self.num_robots_range[0]
        target_range_size = self.num_targets_range[1] - self.num_targets_range[0]
        world_range_size = self.world_size_range[1][0] - self.world_size_range[0][0]

        # complexity = 0.0: use min values (fewer robots/targets)
        # complexity = 1.0: use max values (lots of robots/targets)
        curriculum_robots = int(self.num_robots_range[0] + complexity * robot_range_size)
        curriculum_targets = int(self.num_targets_range[0] + complexity * target_range_size)
        curriculum_world = self.world_size_range[0][0] + complexity * world_range_size

        # Ensure at least minimum values
        curriculum_robots = max(self.num_robots_range[0], curriculum_robots)
        curriculum_targets = max(self.num_targets_range[0], curriculum_targets)
        max_world = self.world_size_range[1][0]
        curriculum_world = max(self.world_size_range[0][0], min(curriculum_world, max_world))

        # Override base parameters with curriculum-adjusted values
        params['num_robots'] = curriculum_robots
        params['num_targets'] = curriculum_targets
        params['world_size'] = (curriculum_world, curriculum_world)

        # Keep adversarial parameters as randomly sampled (no curriculum scaling)
        # Only num_robots and num_targets follow curriculum progression

        # Use fixed episode length (no curriculum scaling for episode length)
        params['episode_length'] = self.episode_length

        return params

    def create_scenario_environment(self, params: ScenarioParameters) -> Tuple[SimulationEnvironment, Dict]:
        """
        Create a simulation environment from scenario parameters

        Args:
            params: Scenario parameters

        Returns:
            Tuple of (simulation_environment, ground_truth_labels)
        """
        # Create simulation environment
        sim_env = SimulationEnvironment(
            num_robots=params.num_robots,
            num_targets=params.num_targets,
            adversarial_ratio=params.adversarial_ratio,
            world_size=params.world_size,
            false_positive_rate=params.false_positive_rate,
            false_negative_rate=params.false_negative_rate,
            proximal_range=params.proximal_range,
            fov_range=self.fov_range,
            fov_angle=self.fov_angle
        )

        # Extract ground truth labels
        ground_truth = self._extract_ground_truth(sim_env)

        return sim_env, ground_truth

    def _extract_ground_truth(self, sim_env: SimulationEnvironment) -> Dict:
        """Extract ground truth labels from simulation environment"""
        ground_truth = {
            'adversarial_agents': [],
            'false_tracks': [],
            'scenario_params': None
        }

        # Extract adversarial robot IDs
        adversarial_agents = []
        for robot in sim_env.robots:
            if hasattr(robot, 'is_adversarial') and robot.is_adversarial:
                adversarial_agents.append(robot.id)
        ground_truth['adversarial_agents'] = adversarial_agents

        # Extract false positive track IDs from shared_fp_objects
        false_tracks = []
        for fp_obj in sim_env.shared_fp_objects:
            # False tracks are generated with format: "{robot_id}_fp_obj_{fp_obj.id}"
            # We need to find all possible combinations with existing robots
            fp_object_id = f"fp_obj_{fp_obj.id}"
            for robot in sim_env.robots:
                # Only include tracks that would be visible to this robot
                if robot.is_in_fov(fp_obj.position):
                    fp_track_key = f"{robot.id}_{fp_object_id}"
                    false_tracks.append(fp_track_key)
        ground_truth['false_tracks'] = false_tracks

        return ground_truth

    def get_curriculum_stats(self) -> Dict:
        """Get current curriculum learning statistics"""
        if not self.curriculum_learning:
            return {
                'curriculum_enabled': False,
                'current_level': None,
                'difficulty': None,
                'avg_performance': None,
                'episodes_in_level': None,
                'levels_total': len(self.level_complexities)
            }

        avg_perf = float(np.mean(self.curriculum_window)) if self.curriculum_window else None

        return {
            'curriculum_enabled': True,
            'current_level': self.curriculum_level,
            'difficulty': float(self.level_complexities[self.curriculum_level]),
            'avg_performance': avg_perf,
            'episodes_in_level': self.episodes_in_level,
            'levels_total': len(self.level_complexities)
        }

    def reset_curriculum(self):
        """Reset curriculum learning to the easiest difficulty."""
        self.curriculum_level = 0
        self.curriculum_window.clear()
        self.episodes_in_level = 0

    def set_complexity(self, complexity: float):
        """Manually set target difficulty (0.0 easy, 1.0 hard)."""
        if not self.curriculum_learning:
            return

        bounded = max(0.0, min(1.0, complexity))
        closest_idx = int(np.argmin(np.abs(self.level_complexities - bounded)))
        self.curriculum_level = closest_idx
        self.curriculum_window.clear()
        self.episodes_in_level = 0

    def update_curriculum(self, performance_metric: float) -> bool:
        """Update curriculum progression based on recent performance.

        Returns True if difficulty was increased.
        """
        if not self.curriculum_learning or performance_metric is None:
            return False

        self.curriculum_window.append(performance_metric)
        self.episodes_in_level += 1

        if self.curriculum_level >= len(self.level_complexities) - 1:
            return False  # Already at max difficulty

        if self.episodes_in_level < self.curriculum_min_episodes:
            return False

        if len(self.curriculum_window) < self.curriculum_window.maxlen:
            return False

        avg_perf = float(np.mean(self.curriculum_window))
        if avg_perf >= self.curriculum_progress_threshold:
            self.curriculum_level += 1
            self.curriculum_window.clear()
            self.episodes_in_level = 0
            print(f"Curriculum advanced to level {self.curriculum_level} (difficulty={self.level_complexities[self.curriculum_level]:.2f})")
            return True

        return False


# Example usage and testing
if __name__ == "__main__":
    print("Testing RL Scenario Generator")

    # Create scenario generator
    generator = RLScenarioGenerator(
        curriculum_learning=True,
        num_robots_range=(3, 8),
        num_targets_range=(10, 30),
        adversarial_ratio_range=(0.0, 0.6)
    )

    print("\n=== Testing Scenario Generation ===")

    # Generate scenarios for different episodes to show curriculum progression
    print("Curriculum progression with staged difficulty:")
    for episode in [0, 500, 1000, 2000]:
        params = generator.sample_scenario_parameters(episode)
        stats = generator.get_curriculum_stats()

        print(f"\nEpisode {episode}:")
        print(f"  Curriculum level: {stats['current_level']} / {stats['levels_total'] - 1}")
        print(f"  Difficulty: {stats['difficulty']:.3f}")
        print(f"  Robots: {params.num_robots} (range: {generator.num_robots_range})")
        print(f"  Targets: {params.num_targets} (range: {generator.num_targets_range})")
        print(f"  Adversarial ratio: {params.adversarial_ratio:.1f}")
        print(f"  False positive rate: {params.false_positive_rate:.1f}")
        print(f"  Episode length: {params.episode_length}")

    # Test scenario creation
    print("\n=== Testing Scenario Creation ===")
    params = generator.sample_scenario_parameters(0)
    sim_env, ground_truth = generator.create_scenario_environment(params)

    print(f"Created scenario with {len(sim_env.robots)} robots")
    print(f"Adversarial agents: {ground_truth['adversarial_agents']}")
    print(f"False tracks: {len(ground_truth['false_tracks'])}")

    # Test curriculum stats
    print("\n=== Curriculum Stats ===")
    stats = generator.get_curriculum_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nScenario generation tests completed!")
