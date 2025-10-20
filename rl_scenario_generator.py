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
    robot_density: float
    target_density: float
    target_density_multiplier: float
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
                 robot_density_range: Tuple[float, float] = (0.0005, 0.0030),
                 target_density_multiplier_range: Union[float, Tuple[float, float]] = (2.0, 2.0),
                 adversarial_ratio_range: Tuple[float, float] = (0.2, 0.5),
                 false_positive_rate_range: Tuple[float, float] = (0.1, 0.7),
                 false_negative_rate_range: Tuple[float, float] = (0.0, 0.3),
                 world_size: float = 100.0,
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
            robot_density_range: Range of robot densities (min, max) per unit area
            target_density_multiplier_range: Range or fixed multiplier applied to sampled robot density
            adversarial_ratio_range: Range of adversarial robot ratios (min, max)
            false_positive_rate_range: Range of false positive rates (min, max)
            false_negative_rate_range: Range of false negative rates (min, max)
            world_size: Side length of fixed square world (meters)
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
        self.robot_density_range = robot_density_range
        if isinstance(target_density_multiplier_range, tuple):
            self.target_density_multiplier_range = target_density_multiplier_range
        else:
            self.target_density_multiplier_range = (target_density_multiplier_range,
                                                    target_density_multiplier_range)
        self.adversarial_ratio_range = adversarial_ratio_range
        self.false_positive_rate_range = false_positive_rate_range
        self.false_negative_rate_range = false_negative_rate_range
        self.world_size = (world_size, world_size)
        self.world_area = self.world_size[0] * self.world_size[1]
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
        self.curriculum_completed = False

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
        elif self.curriculum_completed:
            complexity = random.uniform(0.0, 1.0)
        else:
            complexity = random.uniform(0.0, 1.0)

        # Base parameter sampling with increments (like supervised data generation)
        params = self._sample_base_parameters()

        # Apply curriculum scaling
        params = self._apply_curriculum_scaling(params, complexity)

        return ScenarioParameters(
            robot_density=params['robot_density'],
            target_density=params['target_density'],
            target_density_multiplier=params['target_density_multiplier'],
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

    @staticmethod
    def _sample_density(range_tuple: Tuple[float, float], step: float) -> float:
        min_val, max_val = range_tuple
        min_val = float(min_val)
        max_val = float(max_val)
        if max_val <= min_val + 1e-9:
            return min_val
        step = max(step, 1e-6)
        steps = max(1, int(round((max_val - min_val) / step)))
        idx = random.randint(0, steps)
        return round(min_val + idx * step, 8)

    def _sample_base_parameters(self) -> Dict:
        """Sample base parameters with increments like supervised data generation"""

        robot_density = self._sample_density(self.robot_density_range, 0.0001)
        target_multiplier = self._sample_density(self.target_density_multiplier_range, 0.1)
        target_density = round(robot_density * target_multiplier, 8)
        num_robots = max(1, int(round(robot_density * self.world_area)))
        num_targets = max(1, int(round(target_density * self.world_area)))

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

        # Use fixed episode length
        episode_length = self.episode_length

        return {
            'robot_density': robot_density,
            'target_density': target_density,
            'target_density_multiplier': target_multiplier,
            'num_robots': num_robots,
            'num_targets': num_targets,
            'adversarial_ratio': adversarial_ratio,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'world_size': self.world_size,
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

        # Start with lower densities and gradually increase
        min_robot_density, max_robot_density = self.robot_density_range
        min_target_multiplier, max_target_multiplier = self.target_density_multiplier_range

        curriculum_robot_density = min_robot_density + complexity * (max_robot_density - min_robot_density)
        curriculum_target_multiplier = min_target_multiplier + complexity * (max_target_multiplier - min_target_multiplier)
        curriculum_target_density = round(curriculum_robot_density * curriculum_target_multiplier, 8)

        params['robot_density'] = float(curriculum_robot_density)
        params['target_density_multiplier'] = float(curriculum_target_multiplier)
        params['target_density'] = float(curriculum_target_density)
        params['num_robots'] = max(1, int(round(params['robot_density'] * self.world_area)))
        params['num_targets'] = max(1, int(round(params['target_density'] * self.world_area)))
        params['world_size'] = self.world_size
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
            world_size=params.world_size,
            robot_density=params.robot_density,
            target_density=params.target_density,
            adversarial_ratio=params.adversarial_ratio,
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
            'scenario_params': None,
            'true_objects': []
        }

        # Extract adversarial robot IDs
        adversarial_agents = []
        for robot in sim_env.robots:
            if hasattr(robot, 'is_adversarial') and robot.is_adversarial:
                adversarial_agents.append(robot.id)
        ground_truth['adversarial_agents'] = adversarial_agents

        # Record all legitimate ground truth objects
        true_objects = []
        for gt_obj in sim_env.ground_truth_objects:
            true_objects.append(f"gt_obj_{gt_obj.id}")
        ground_truth['true_objects'] = true_objects

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
                'levels_total': len(self.level_complexities),
                'completed': self.curriculum_completed
            }

        avg_perf = float(np.mean(self.curriculum_window)) if self.curriculum_window else None

        return {
            'curriculum_enabled': True,
            'current_level': self.curriculum_level,
            'difficulty': float(self.level_complexities[self.curriculum_level]),
            'avg_performance': avg_perf,
            'episodes_in_level': self.episodes_in_level,
            'levels_total': len(self.level_complexities),
            'completed': self.curriculum_completed
        }

    def reset_curriculum(self):
        """Reset curriculum learning to the easiest difficulty."""
        self.curriculum_level = 0
        self.curriculum_window.clear()
        self.episodes_in_level = 0
        self.curriculum_completed = False
        self.curriculum_learning = True

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
        if performance_metric is None:
            return False

        if self.curriculum_completed:
            return False

        if not self.curriculum_learning:
            self.curriculum_window.append(performance_metric)
            return False

        self.curriculum_window.append(performance_metric)
        self.episodes_in_level += 1

        if self.curriculum_level >= len(self.level_complexities) - 1:
            if len(self.curriculum_window) == self.curriculum_window.maxlen:
                avg_perf = float(np.mean(self.curriculum_window))
                if avg_perf >= self.curriculum_progress_threshold:
                    self.curriculum_completed = True
                    self.curriculum_learning = False
                    print("Curriculum completed. Switching to full scenario sampling.")
                    return True
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
        robot_density_range=(0.0004, 0.0008),
        target_density_multiplier_range=(3.0, 4.0),
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
        print(f"  Robots: {params.num_robots} (density: {params.robot_density:.6f})")
        print(f"  Target multiplier: {params.target_density_multiplier:.2f}")
        print(f"  Targets: {params.num_targets} (density: {params.target_density:.6f})")
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
