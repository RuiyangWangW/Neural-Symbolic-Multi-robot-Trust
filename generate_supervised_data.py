#!/usr/bin/env python3
"""
Data Generation for Supervised Trust Learning

This script generates training data from simulation environments with ground truth labels
for whether each robot/track is trustworthy. Uses ground truth trust assignment based on
adversarial labels to create realistic trust distributions.
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict, Union
import pickle
from dataclasses import dataclass
import random
from pathlib import Path

# Import existing simulation components
from simulation_environment import SimulationEnvironment
from supervised_trust_gnn import TrustFeatureCalculator, EgoGraphBuilder


@dataclass
class SupervisedDataSample:
    """
    Single data sample for supervised learning
    """
    x_dict: Dict  # Node features dictionary
    edge_index_dict: Dict  # Edge indices dictionary
    agent_labels: torch.Tensor  # Binary trust labels for agents [num_agents, 1]
    track_labels: torch.Tensor  # Binary trust labels for tracks [num_tracks, 1]
    timestep: int
    episode: int
    ego_robot_id: str  # ID of the ego robot for this sample


class SupervisedDataGenerator:
    """
    Generates supervised learning data from simulation environment using ground truth trust assignment
    """

    def __init__(self,
                 robot_density: Union[float, Tuple[float, float]] = 0.0005,
                 target_density_multiplier: Union[float, Tuple[float, float]] = 2.0,
                 adversarial_ratio: Union[float, Tuple[float, float]] = 0.5,
                 world_size: Tuple[float, float] = (100.0, 100.0),
                 false_positive_rate: Union[float, Tuple[float, float]] = 0.5,
                 false_negative_rate: Union[float, Tuple[float, float]] = 0.0,
                 proximal_range: float = 50.0,
                 fov_range: float = 50.0,
                 fov_angle: float = np.pi/3,
                 max_steps_per_episode: int = 100):
        """
        Initialize data generator with ground truth trust assignment

        Args:
            robot_density: Robot density (float) or range (min, max) used to sample populations
            target_density_multiplier: Multiplier (float) or range (min, max) applied to sampled robot density
            adversarial_ratio: Fraction of robots that are adversarial (float) or range (min, max)
            world_size: Size of simulation world (tuple) or range ((min_x, min_y), (max_x, max_y))
            false_positive_rate: Rate of false positive detections (float) or range (min, max)
            false_negative_rate: Rate of false negative detections (float) or range (min, max)
            proximal_range: Proximal sensing range (fixed value)
            fov_range: Field of view range (kept constant)
            fov_angle: Field of view angle (kept constant)
            max_steps_per_episode: Maximum steps per episode
        """
        self.max_steps_per_episode = max_steps_per_episode

        # Store parameter ranges for sampling
        self.robot_density_range = robot_density if isinstance(robot_density, tuple) else (robot_density, robot_density)
        self.target_density_multiplier_range = (target_density_multiplier
                                                if isinstance(target_density_multiplier, tuple)
                                                else (target_density_multiplier, target_density_multiplier))
        self.adversarial_ratio_range = adversarial_ratio if isinstance(adversarial_ratio, tuple) else (adversarial_ratio, adversarial_ratio)
        self.world_size = world_size

        self.false_positive_rate_range = false_positive_rate if isinstance(false_positive_rate, tuple) else (false_positive_rate, false_positive_rate)
        self.false_negative_rate_range = false_negative_rate if isinstance(false_negative_rate, tuple) else (false_negative_rate, false_negative_rate)
        self.proximal_range = proximal_range  # Fixed value, no range
        self.fov_range = fov_range
        self.fov_angle = fov_angle

        # Initialize simulation environment (will be recreated for each episode with sampled parameters)
        self.sim_env = None

        # Initialize enhanced feature calculator for proper neural symbolic features
        self.feature_calculator = TrustFeatureCalculator()

        # Initialize ego graph builder (will be updated for each episode)
        self.ego_graph_builder = None

    def _assign_ground_truth_trust(self, robots: List):
        """
        Assign trust values based on ground truth labels with controlled noise.

        80% of the time: Perfect ground truth trust
        - Legitimate robots/tracks: trust randomly from 0.7 to 1.0
        - Adversarial robots/false positive tracks: trust randomly from 0.0 to 0.3

        20% of the time: Random noise trust
        - Any entity: trust randomly from 0.0 to 1.0 (regardless of ground truth)

        This creates a more comprehensive dataset for supervised learning.
        """
        import random

        for robot in robots:
            # 80% chance of ground truth, 20% chance of random noise
            if random.random() < 0.8:
                # Ground truth trust assignment
                if robot.is_adversarial:
                    # Adversarial robot: low trust (0.0 to 0.3)
                    trust_value = random.uniform(0.0, 0.3)
                else:
                    # Legitimate robot: high trust (0.7 to 1.0)
                    trust_value = random.uniform(0.7, 1.0)
            else:
                # Random noise: any value between 0 and 1
                trust_value = random.uniform(0.0, 1.0)

            # Calculate confidence based on certainty
            # High confidence when trust is close to 0 or 1
            # Use distance from 0.5 as measure of certainty
            certainty = abs(trust_value - 0.5) * 2  # 0 to 1 scale
            # Map certainty to κ (kappa): higher certainty = higher κ = higher confidence
            # κ range: [5, 30] where 5 = low confidence, 30 = high confidence
            kappa = 5 + certainty * 25

            # Calculate alpha and beta from trust and kappa
            # mean = alpha/(alpha+beta) = trust_value
            # alpha + beta = kappa
            alpha = trust_value * kappa
            beta = (1.0 - trust_value) * kappa

            # Ensure minimum values
            alpha = max(1.0, alpha)
            beta = max(1.0, beta)

            # Assign to robot
            robot.trust_alpha = alpha
            robot.trust_beta = beta

            # Also assign trust to all tracks from this robot
            for track in robot.get_all_tracks():
                # 80% chance of ground truth, 20% chance of random noise
                if random.random() < 0.8:
                    # Ground truth trust assignment for tracks
                    # Determine if track is false positive using same logic as label generation
                    track_id_str = str(track.object_id)
                    is_false_positive = track_id_str.startswith('fp_obj_')

                    # Track trust should be based on whether it's a false positive,
                    # NOT on whether the robot is adversarial
                    # (adversarial robots can detect real objects too!)
                    if is_false_positive:
                        # False positive track: low trust
                        track_trust_value = random.uniform(0.0, 0.3)
                    else:
                        # True positive track: high trust
                        track_trust_value = random.uniform(0.7, 1.0)
                else:
                    # Random noise: any value between 0 and 1
                    track_trust_value = random.uniform(0.0, 1.0)

                # Calculate confidence for track
                track_certainty = abs(track_trust_value - 0.5) * 2
                track_kappa = 5 + track_certainty * 25

                track_alpha = track_trust_value * track_kappa
                track_beta = (1.0 - track_trust_value) * track_kappa

                track_alpha = max(1.0, track_alpha)
                track_beta = max(1.0, track_beta)

                track.trust_alpha = track_alpha
                track.trust_beta = track_beta

    def _simulate_step_with_rl(self, step_idx: int):
        """
        Execute a single environment step and apply ground truth trust assignment.
        """
        frame_data = self.sim_env.step()

        for robot in self.sim_env.robots:
            robot.update_current_timestep_tracks()

        # Assign ground truth trust values instead of using RL trust system
        self._assign_ground_truth_trust(self.sim_env.robots)

        return frame_data

    @staticmethod
    def _sample_density(range_tuple: Tuple[float, float], step: float) -> float:
        """Sample a density value within a range using discrete increments for reproducibility."""
        min_val, max_val = range_tuple
        min_val = float(min_val)
        max_val = float(max_val)

        if max_val <= min_val + 1e-9:
            return min_val

        step = max(step, 1e-6)
        steps = max(1, int(round((max_val - min_val) / step)))
        idx = random.randint(0, steps)
        return round(min_val + idx * step, 8)

    def _sample_episode_parameters(self) -> Dict:
        """
        Sample parameters for a single episode from the specified ranges with specific increments

        Returns:
            Dictionary containing sampled parameters
        """
        # Sample density-driven population sizes
        robot_density = self._sample_density(self.robot_density_range, step=0.0001)
        target_multiplier = self._sample_density(self.target_density_multiplier_range, step=0.1)
        target_density = round(robot_density * target_multiplier, 8)
        area = self.world_size[0] * self.world_size[1]
        num_robots = max(1, int(round(robot_density * area)))
        num_targets = max(1, int(round(target_density * area)))

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
            'proximal_range': self.proximal_range  # Fixed value
        }

    def _create_simulation_environment(self, params: Dict) -> None:
        """
        Create a new simulation environment with the given parameters

        Args:
            params: Dictionary of sampled parameters
        """
        self.sim_env = SimulationEnvironment(
            world_size=self.world_size,
            robot_density=params['robot_density'],
            target_density=params['target_density'],
            adversarial_ratio=params['adversarial_ratio'],
            false_positive_rate=params['false_positive_rate'],
            false_negative_rate=params['false_negative_rate'],
            proximal_range=params['proximal_range'],
            fov_range=self.fov_range,
            fov_angle=self.fov_angle
        )

        # Update ego graph builder with fixed proximal range
        self.ego_graph_builder = EgoGraphBuilder(proximal_range=params['proximal_range'])

    def _generate_labels_from_ego_graph(self, ego_robot, ego_graph: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate labels for ALL robots and tracks in the ego graph.

        IMPORTANT: Since we're using ground truth trust assignment for supervised learning,
        ALL robots and tracks in the ego graph have correct trust values and should receive labels.
        This ensures features and labels are properly aligned.

        Args:
            ego_robot: The ego robot for this graph
            ego_graph: The ego graph with robot and track data

        Returns:
            Tuple of (agent_labels, track_labels) where 1 = trustworthy, 0 = adversarial
            - agent_labels: [num_agents, 1] tensor for ALL robots in ego graph
            - track_labels: [num_tracks, 1] tensor for ALL tracks in ego graph
        """
        # Generate labels for ALL robots in ego graph
        # NOTE: proximal_robots already INCLUDES ego robot (at index 0)
        agent_labels = []

        # Get all proximal robots (includes ego robot)
        proximal_robots = getattr(ego_graph, '_proximal_robots', [])

        # Labels for ALL proximal robots (ego robot is already first in this list)
        for robot in proximal_robots:
            label = 1.0 if not robot.is_adversarial else 0.0
            agent_labels.append(label)

        agent_labels = torch.tensor(agent_labels, dtype=torch.float32).unsqueeze(1)

        # Generate labels for ALL tracks in ego graph
        track_labels = []
        if 'track' in ego_graph.x_dict and ego_graph.x_dict['track'].shape[0] > 0:
            # Get all proximal robot tracks
            proximal_robot_tracks = {}
            for robot in proximal_robots:
                robot_tracks = robot.get_all_current_tracks()
                proximal_robot_tracks[robot.id] = robot_tracks

            # Get tracks in same order as ego graph builder
            fused_tracks, individual_tracks, _ = self.ego_graph_builder._perform_track_fusion(
                proximal_robots, proximal_robot_tracks)
            all_tracks = fused_tracks + individual_tracks

            ground_truth_objects = getattr(self.sim_env, 'ground_truth_objects', [])

            # Create labels for ALL tracks (not just ego-owned)
            for track in all_tracks[:ego_graph.x_dict['track'].shape[0]]:
                track_id_str = str(track.object_id)

                # Check if track is ground truth
                if track_id_str.startswith('fp_obj_'):
                    is_ground_truth = False
                elif track_id_str.startswith('gt_obj_'):
                    numeric_id = track_id_str.replace('gt_obj_', '')
                    is_ground_truth = any(str(getattr(obj, 'id', '')) == numeric_id for obj in ground_truth_objects)
                else:
                    is_ground_truth = any(str(getattr(obj, 'id', '')) == track_id_str for obj in ground_truth_objects)

                # Ground truth track = 1, False positive = 0
                label = 1.0 if is_ground_truth else 0.0
                track_labels.append(label)

        track_labels = torch.tensor(track_labels, dtype=torch.float32).unsqueeze(1) if track_labels else torch.empty(0, 1, dtype=torch.float32)

        return agent_labels, track_labels

    def generate_episode_data(self, episode_idx: int = 0, step_interval: int = 10,
                             ego_sample_ratio: float = 0.2) -> Tuple[List[SupervisedDataSample], Dict]:
        """
        Generate supervised data for one episode using RL trust algorithm

        Args:
            episode_idx: Episode index for tracking
            step_interval: Sample ego graphs every N steps (default: 10)
            ego_sample_ratio: Proportion of ego graphs to sample at each timestep (default: 0.2)

        Returns:
            Tuple of (list of data samples for the episode, episode parameters)
        """
        # Sample new parameters for this episode
        episode_params = self._sample_episode_parameters()

        print(f"Generating data for episode {episode_idx}...")
        print(f"  Parameters: {episode_params}")
        print(f"  Sampling: every {step_interval} steps, {ego_sample_ratio*100:.0f}% of ego graphs per step")

        # Create simulation environment with sampled parameters
        self._create_simulation_environment(episode_params)

        # Log initial randomized state for verification
        if len(self.sim_env.robots) > 0:
            robot0_pos = self.sim_env.robots[0].start_position[:2]
            robot0_goal = self.sim_env.robots[0].goal_position[:2]
            adv_ids = [r.id for r in self.sim_env.robots if r.is_adversarial]
            print(f"  Randomized: Robot0 start=({robot0_pos[0]:.1f}, {robot0_pos[1]:.1f}), " +
                  f"goal=({robot0_goal[0]:.1f}, {robot0_goal[1]:.1f}), adversarial={adv_ids}")

        episode_data = []

        for step in range(self.max_steps_per_episode):
            try:
                self._simulate_step_with_rl(step)
            except Exception as e:
                print(f"Warning: RL trust step failed at step {step}: {e}")
                break

            # Only sample ego graphs at specified intervals
            if step % step_interval != 0:
                continue

            robots = self.sim_env.robots
            if not robots:
                print(f"Warning: No robots at step {step}")
                continue

            # Sample a subset of robots for ego graphs
            num_robots_to_sample = max(1, int(len(robots) * ego_sample_ratio))
            sampled_robots = random.sample(robots, num_robots_to_sample)

            # Generate ego graphs for sampled robots
            for ego_robot in sampled_robots:
                try:
                    # Build ego-graph for this robot
                    ego_graph = self.ego_graph_builder.build_ego_graph(ego_robot, robots)
                    if ego_graph is None:
                        continue

                    # Generate labels ONLY for ego robot and its tracks (corrected MDP)
                    agent_labels, track_labels = self._generate_labels_from_ego_graph(ego_robot, ego_graph)

                    # Create clean data sample
                    sample = SupervisedDataSample(
                        x_dict=ego_graph.x_dict.copy(),
                        edge_index_dict=ego_graph.edge_index_dict.copy(),
                        agent_labels=agent_labels,
                        track_labels=track_labels,
                        timestep=step,
                        episode=episode_idx,
                        ego_robot_id=ego_robot.id
                    )

                    # Only save samples that have tracks (skip empty track graphs)
                    if 'track' in sample.x_dict and sample.x_dict['track'].shape[0] > 0:
                        episode_data.append(sample)

                except Exception as e:
                    print(f"Warning: Error generating ego graph for robot {ego_robot.id} at step {step}: {e}")
                    continue
        print(f"Generated {len(episode_data)} samples for episode {episode_idx}")
        return episode_data, episode_params

    def generate_dataset(self,
                        num_episodes: int = 10,
                        save_path: str = "supervised_trust_dataset.pkl",
                        log_path: str = None,
                        step_interval: int = 10,
                        ego_sample_ratio: float = 0.2) -> Tuple[List[SupervisedDataSample], List[Dict]]:
        """
        Generate complete dataset with multiple episodes using diverse parameters

        Args:
            num_episodes: Number of episodes to generate
            save_path: Path to save the dataset
            log_path: Optional path to save generation log (default: save_path.replace('.pkl', '.log'))
            step_interval: Sample ego graphs every N steps (default: 10)
            ego_sample_ratio: Proportion of ego graphs to sample at each timestep (default: 0.2)

        Returns:
            Tuple of (list of all data samples, list of episode parameters)
        """
        # Set up logging
        if log_path is None:
            log_path = save_path.replace('.pkl', '_generation.log')

        import sys
        import datetime

        # Create log file and tee output
        log_file = open(log_path, 'w')

        def log_print(*args, **kwargs):
            """Print to both console and log file"""
            message = ' '.join(str(arg) for arg in args)
            print(message, **kwargs)
            log_file.write(message + '\n')
            log_file.flush()

        log_print(f"="*80)
        log_print(f"Supervised Dataset Generation Log")
        log_print(f"="*80)
        log_print(f"Start time: {datetime.datetime.now()}")
        log_print(f"Output: {save_path}")
        log_print(f"Log: {log_path}")
        log_print(f"")

        log_print(f"🔄 Generating supervised dataset with {num_episodes} episodes...")
        log_print(f"📊 Parameter ranges:")
        log_print(f"   - Robot density: {self.robot_density_range}")
        log_print(f"   - Target density multiplier: {self.target_density_multiplier_range}")
        derived_min = round(self.robot_density_range[0] * self.target_density_multiplier_range[0], 6)
        derived_max = round(self.robot_density_range[1] * self.target_density_multiplier_range[1], 6)
        log_print(f"   - Target density (derived): ({derived_min}, {derived_max})")
        log_print(f"   - Adversarial ratio: {self.adversarial_ratio_range}")
        log_print(f"   - False positive rate: {self.false_positive_rate_range}")
        log_print(f"   - False negative rate: {self.false_negative_rate_range}")
        log_print(f"   - World size (square): {self.world_size[0]} x {self.world_size[1]}")
        log_print(f"   - Proximal range (fixed): {self.proximal_range}")
        log_print(f"⏱️  Max steps per episode: {self.max_steps_per_episode}")
        log_print(f"📊 Sampling strategy:")
        log_print(f"   - Step interval: every {step_interval} steps")
        log_print(f"   - Ego sample ratio: {ego_sample_ratio*100:.0f}% of robots per sampled step")
        log_print(f"   - Expected samples per episode: ~{(self.max_steps_per_episode // step_interval) * ego_sample_ratio * 5:.0f} (assuming ~5 robots)")

        log_print(f"🧠 Using ground truth trust assignment:")
        log_print(f"   - Legitimate robots/tracks: trust ∈ [0.7, 1.0]")
        log_print(f"   - Adversarial robots/tracks: trust ∈ [0.0, 0.3]")
        log_print(f"   - Confidence: Higher when trust is closer to 0 or 1")
        log_print(f"")

        all_data = []
        all_episode_params = []

        for episode in range(num_episodes):
            try:
                episode_data, episode_params = self.generate_episode_data(
                    episode, step_interval=step_interval, ego_sample_ratio=ego_sample_ratio
                )
                all_data.extend(episode_data)
                all_episode_params.append(episode_params)

                # Progress update
                if (episode + 1) % 5 == 0:
                    log_print(f"✅ Completed {episode + 1}/{num_episodes} episodes. Total samples: {len(all_data)}")

            except Exception as e:
                log_print(f"⚠️ Error generating episode {episode}: {e}")
                continue

        log_print(f"🎉 Dataset generation complete!")
        log_print(f"📈 Total samples before balancing: {len(all_data)}")

        # ========================================================================
        # BALANCED SAMPLING: Equal adversarial and legitimate robot samples
        # ========================================================================
        log_print(f"\n" + "="*80)
        log_print("BALANCED SAMPLING")
        log_print("="*80)

        # Separate samples by adversarial vs legitimate
        adversarial_samples = []
        legitimate_samples = []

        for sample in all_data:
            # agent_labels: 1.0 = legitimate, 0.0 = adversarial
            if sample.agent_labels.shape[0] > 0:
                agent_label = float(sample.agent_labels[0, 0])
                if agent_label == 0.0:
                    adversarial_samples.append(sample)
                else:
                    legitimate_samples.append(sample)
            else:
                # No agent label (shouldn't happen, but keep sample)
                legitimate_samples.append(sample)

        log_print(f"Adversarial robot samples: {len(adversarial_samples)}")
        log_print(f"Legitimate robot samples:  {len(legitimate_samples)}")

        # Balance: keep ALL adversarial, sample equal number of legitimate
        num_adversarial = len(adversarial_samples)
        num_legitimate = len(legitimate_samples)

        if num_legitimate > num_adversarial:
            # Randomly sample legitimate samples to match adversarial count
            import random
            random.seed(42)  # Reproducible sampling
            sampled_legitimate = random.sample(legitimate_samples, num_adversarial)

            log_print(f"\nBalancing dataset:")
            log_print(f"  Keeping all {num_adversarial} adversarial samples")
            log_print(f"  Sampling {num_adversarial} out of {num_legitimate} legitimate samples")
            log_print(f"  Final ratio: 50% adversarial, 50% legitimate")

            legitimate_samples = sampled_legitimate
        elif num_adversarial > num_legitimate:
            log_print(f"\nWarning: More adversarial ({num_adversarial}) than legitimate ({num_legitimate})!")
            log_print(f"  Keeping all samples (cannot balance)")
        else:
            log_print(f"\nAlready balanced: {num_adversarial} adversarial, {num_legitimate} legitimate")

        # Merge balanced samples
        all_data = adversarial_samples + legitimate_samples
        random.shuffle(all_data)  # Shuffle to mix adversarial and legitimate

        log_print(f"\nBalanced dataset:")
        log_print(f"  Total samples: {len(all_data)} (was {len(adversarial_samples) + num_legitimate})")
        log_print(f"  Adversarial: {len(adversarial_samples)} ({100*len(adversarial_samples)/len(all_data):.1f}%)")
        log_print(f"  Legitimate:  {len(legitimate_samples)} ({100*len(legitimate_samples)/len(all_data):.1f}%)")
        log_print("="*80)

        # Calculate statistics
        agent_samples = sum(1 for sample in all_data if sample.agent_labels.shape[0] > 0)
        track_samples = sum(1 for sample in all_data if sample.track_labels.shape[0] > 0)

        # Check for agent co-detection edges
        codetection_edge_samples = sum(1 for sample in all_data
                                    if ('agent', 'co_detection', 'agent') in sample.edge_index_dict)

        if all_data:
            avg_agents_per_sample = np.mean([sample.agent_labels.shape[0] for sample in all_data])
            avg_tracks_per_sample = np.mean([sample.track_labels.shape[0] for sample in all_data])

            # Feature dimensions after alpha/beta removal
            sample_agent_features = all_data[0].x_dict['agent'].shape[1] if 'agent' in all_data[0].x_dict else 0

            # Find a sample with tracks to get track feature dimensions
            sample_track_features = 0
            for sample in all_data:
                if 'track' in sample.x_dict and sample.x_dict['track'].shape[0] > 0:
                    sample_track_features = sample.x_dict['track'].shape[1]
                    break
        else:
            avg_agents_per_sample = 0
            avg_tracks_per_sample = 0
            sample_agent_features = 0
            sample_track_features = 0

        # Calculate parameter diversity statistics
        param_stats = {}
        if all_episode_params:
            param_stats = {
                'num_robots': {
                    'min': min(p['num_robots'] for p in all_episode_params),
                    'max': max(p['num_robots'] for p in all_episode_params),
                    'avg': np.mean([p['num_robots'] for p in all_episode_params])
                },
                'num_targets': {
                    'min': min(p['num_targets'] for p in all_episode_params),
                    'max': max(p['num_targets'] for p in all_episode_params),
                    'avg': np.mean([p['num_targets'] for p in all_episode_params])
                },
                'robot_density': {
                    'min': min(p['robot_density'] for p in all_episode_params),
                    'max': max(p['robot_density'] for p in all_episode_params),
                    'avg': np.mean([p['robot_density'] for p in all_episode_params])
                },
                'target_density_multiplier': {
                    'min': min(p['target_density_multiplier'] for p in all_episode_params),
                    'max': max(p['target_density_multiplier'] for p in all_episode_params),
                    'avg': np.mean([p['target_density_multiplier'] for p in all_episode_params])
                },
                'target_density': {
                    'min': min(p['target_density'] for p in all_episode_params),
                    'max': max(p['target_density'] for p in all_episode_params),
                    'avg': np.mean([p['target_density'] for p in all_episode_params])
                },
                'adversarial_ratio': {
                    'min': min(p['adversarial_ratio'] for p in all_episode_params),
                    'max': max(p['adversarial_ratio'] for p in all_episode_params),
                    'avg': np.mean([p['adversarial_ratio'] for p in all_episode_params])
                },
                'false_positive_rate': {
                    'min': min(p['false_positive_rate'] for p in all_episode_params),
                    'max': max(p['false_positive_rate'] for p in all_episode_params),
                    'avg': np.mean([p['false_positive_rate'] for p in all_episode_params])
                },
                'false_negative_rate': {
                    'min': min(p['false_negative_rate'] for p in all_episode_params),
                    'max': max(p['false_negative_rate'] for p in all_episode_params),
                    'avg': np.mean([p['false_negative_rate'] for p in all_episode_params])
                },
                'world_size': {
                    'min': min(p['world_size'][0] for p in all_episode_params),  # Use x dimension (same as y for square)
                    'max': max(p['world_size'][0] for p in all_episode_params),
                    'avg': np.mean([p['world_size'][0] for p in all_episode_params])
                },
                'proximal_range': {
                    'min': min(p['proximal_range'] for p in all_episode_params),
                    'max': max(p['proximal_range'] for p in all_episode_params),
                    'avg': np.mean([p['proximal_range'] for p in all_episode_params])
                }
            }

        log_print(f"\n📊 Dataset Statistics:")
        log_print(f"   - Samples with agents: {agent_samples}")
        log_print(f"   - Samples with tracks: {track_samples}")
        log_print(f"   - Samples with agent co-detection edges: {codetection_edge_samples}")
        log_print(f"   - Avg agents per sample: {avg_agents_per_sample:.1f}")
        log_print(f"   - Avg tracks per sample: {avg_tracks_per_sample:.1f}")
        log_print(f"   - Agent feature dimensions: {sample_agent_features} (alpha/beta removed)")
        log_print(f"   - Track feature dimensions: {sample_track_features} (alpha/beta removed)")

        if all_episode_params:
            log_print(f"\n📈 Parameter Diversity:")
            for param_name, stats in param_stats.items():
                log_print(f"   - {param_name}: {stats['min']:.3f} - {stats['max']:.3f} (avg: {stats['avg']:.3f})")

        # Save dataset with parameters
        dataset_package = {
            'samples': all_data,
            'episode_parameters': all_episode_params,
            'parameter_ranges': {
                'robot_density': self.robot_density_range,
                'target_density_multiplier': self.target_density_multiplier_range,
                'target_density_derived_range': (
                    round(self.robot_density_range[0] * self.target_density_multiplier_range[0], 6),
                    round(self.robot_density_range[1] * self.target_density_multiplier_range[1], 6)
                ),
                'adversarial_ratio': self.adversarial_ratio_range,
                'false_positive_rate': self.false_positive_rate_range,
                'false_negative_rate': self.false_negative_rate_range,
                'world_size': self.world_size,
                'proximal_range': self.proximal_range  # Fixed value
            },
            'statistics': {
                'total_samples': len(all_data),
                'agent_samples': agent_samples,
                'track_samples': track_samples,
                'codetection_edge_samples': codetection_edge_samples,
                'avg_agents_per_sample': avg_agents_per_sample,
                'avg_tracks_per_sample': avg_tracks_per_sample,
                'parameter_diversity': param_stats if all_episode_params else {}
            }
        }

        log_print(f"\n💾 Saving dataset to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(dataset_package, f)
        log_print(f"✅ Dataset saved successfully!")

        log_print(f"\n" + "="*80)
        log_print(f"Completion time: {datetime.datetime.now()}")
        log_print(f"="*80)

        # Close log file
        log_file.close()

        return all_data, all_episode_params


def main():
    """Generate supervised learning dataset with diverse parameters"""
    import argparse

    def parse_range(value):
        """Parse range argument in format 'min,max' or single value"""
        if ',' in value:
            min_val, max_val = value.split(',')
            return (float(min_val), float(max_val))
        else:
            val = float(value)
            return (val, val)

    parser = argparse.ArgumentParser(description='Generate supervised trust learning dataset with diverse parameters')
    parser.add_argument('--episodes', type=int, default=50000,
                       help='Number of episodes to generate (default: 50000)')
    parser.add_argument('--robot-density', type=str, default='0.0005,0.0020',
                       help='Robot density range in robots per square unit (default: 0.0005,0.0020)')
    parser.add_argument('--target-density-multiplier', type=str, default='2.0',
                       help='Target density multiplier applied to sampled robot density (default: 2.0)')
    parser.add_argument('--adversarial-ratio', type=str, default='0.2,0.5',
                       help='Adversarial robot ratio: single value or range "min,max" (default: 0.2,0.5)')
    parser.add_argument('--false-positive-rate', type=str, default='0.1,0.7',
                       help='False positive rate: single value or range "min,max" (default: 0.1,0.7)')
    parser.add_argument('--false-negative-rate', type=str, default='0.0,0.3',
                       help='False negative rate: single value or range "min,max" (default: 0.0,0.3)')
    parser.add_argument('--world-size', type=float, default=100.0,
                       help='Side length of the square world (fixed, default: 100.0)')
    parser.add_argument('--proximal-range', type=float, default=50.0,
                       help='Proximal sensing range: fixed value (default: 50.0)')
    parser.add_argument('--steps', type=int, default=100,
                       help='Max steps per episode (default: 100)')
    parser.add_argument('--step-interval', type=int, default=10,
                       help='Sample ego graphs every N steps to reduce duplicates (default: 10)')
    parser.add_argument('--ego-sample-ratio', type=float, default=0.2,
                       help='Proportion of ego graphs to sample at each timestep (default: 0.2 = 20%%)')
    parser.add_argument('--output', type=str, default='supervised_trust_dataset.pkl',
                       help='Output file path (default: supervised_trust_dataset.pkl)')
    args = parser.parse_args()

    # Parse parameter ranges
    robot_density_range = parse_range(args.robot_density)
    target_density_multiplier_range = parse_range(args.target_density_multiplier)
    adversarial_ratio_range = parse_range(args.adversarial_ratio)
    false_positive_rate_range = parse_range(args.false_positive_rate)
    false_negative_rate_range = parse_range(args.false_negative_rate)
    world_size_value = float(args.world_size)
    world_size = (world_size_value, world_size_value)

    print(f"🎯 Using parameter ranges:")
    print(f"   - Robot density: {robot_density_range}")
    print(f"   - Target density multiplier: {target_density_multiplier_range}")
    derived_min = round(robot_density_range[0] * target_density_multiplier_range[0], 6)
    derived_max = round(robot_density_range[1] * target_density_multiplier_range[1], 6)
    print(f"   - Target density (derived): ({derived_min}, {derived_max})")
    print(f"   - Adversarial ratio: {adversarial_ratio_range}")
    print(f"   - False positive rate: {false_positive_rate_range}")
    print(f"   - False negative rate: {false_negative_rate_range}")
    print(f"   - World size (square): {world_size_value} x {world_size_value}")
    print(f"   - Proximal range (fixed): {args.proximal_range}")

    # Create data generator
    generator = SupervisedDataGenerator(
        robot_density=robot_density_range,
        target_density_multiplier=target_density_multiplier_range,
        adversarial_ratio=adversarial_ratio_range,
        world_size=world_size,
        false_positive_rate=false_positive_rate_range,
        false_negative_rate=false_negative_rate_range,
        proximal_range=args.proximal_range,
        max_steps_per_episode=args.steps
    )

    # Generate dataset
    dataset, episode_params = generator.generate_dataset(
        num_episodes=args.episodes,
        save_path=args.output,
        step_interval=args.step_interval,
        ego_sample_ratio=args.ego_sample_ratio
    )

    print(f"\n🎯 Dataset generation complete!")
    print(f"📈 {len(dataset)} samples from {len(episode_params)} episodes saved to {args.output}")
    print(f"🧠 Trust assigned using ground truth labels:")
    print(f"   - Legitimate robots/tracks: trust ∈ [0.7, 1.0]")
    print(f"   - Adversarial robots/tracks: trust ∈ [0.0, 0.3]")
    print(f"   - Confidence: Higher when trust is closer to 0 or 1")
    print(f"🔗 Agent co-detection edges included (robots detecting same objects)")
    print(f"📉 Alpha/beta features removed from node features")
    print(f"🎲 Parameter diversity across episodes for more robust training")


if __name__ == "__main__":
    main()
