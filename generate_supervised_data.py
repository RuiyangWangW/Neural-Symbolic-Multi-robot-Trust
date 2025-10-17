#!/usr/bin/env python3
"""
Data Generation for Supervised Trust Learning

This script generates training data from simulation environments with ground truth labels
for whether each robot/track is trustworthy. Uses the RL trust update system for realistic
trust updates and adds agent-to-agent trustworthiness comparison edges.
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
from rl_trust_system import RLTrustSystem


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
    Generates supervised learning data from simulation environment using the RL trust update system
    """

    def __init__(self,
                 robot_density: Union[float, Tuple[float, float]] = 0.0005,
                 target_density: Union[float, Tuple[float, float]] = 0.0020,
                 adversarial_ratio: Union[float, Tuple[float, float]] = 0.5,
                 world_size: Tuple[float, float] = (100.0, 100.0),
                 false_positive_rate: Union[float, Tuple[float, float]] = 0.5,
                 false_negative_rate: Union[float, Tuple[float, float]] = 0.0,
                 proximal_range: float = 50.0,
                 fov_range: float = 50.0,
                 fov_angle: float = np.pi/3,
                 max_steps_per_episode: int = 100,
                 gnn_model_path: str = "supervised_trust_model.pth",
                 rl_model_path: str = "rl_trust_model.pth",
                 device: str = 'cpu',
                 trust_step_size: float = 1.0,
                 trust_strength_cap: float = 100.0):
        """
        Initialize data generator

        Args:
            num_robots: Number of robots (int) or range (min, max)
            num_targets: Number of ground truth objects (int) or range (min, max)
            adversarial_ratio: Fraction of robots that are adversarial (float) or range (min, max)
            world_size: Size of simulation world (tuple) or range ((min_x, min_y), (max_x, max_y))
            false_positive_rate: Rate of false positive detections (float) or range (min, max)
            false_negative_rate: Rate of false negative detections (float) or range (min, max)
            proximal_range: Proximal sensing range (fixed value)
            fov_range: Field of view range (kept constant)
            fov_angle: Field of view angle (kept constant)
            max_steps_per_episode: Maximum steps per episode
            gnn_model_path: Path to trained (or untrained) ego-graph evidence model
            rl_model_path: Path to trained (or untrained) RL updater model
            device: Torch device for trust models
            trust_step_size: Global step size multiplier for trust updates
            trust_strength_cap: Maximum combined alpha+beta strength before normalization
        """
        self.max_steps_per_episode = max_steps_per_episode

        # Store parameter ranges for sampling
        self.robot_density_range = robot_density if isinstance(robot_density, tuple) else (robot_density, robot_density)
        self.target_density_range = target_density if isinstance(target_density, tuple) else (target_density, target_density)
        self.adversarial_ratio_range = adversarial_ratio if isinstance(adversarial_ratio, tuple) else (adversarial_ratio, adversarial_ratio)
        self.world_size = world_size

        self.false_positive_rate_range = false_positive_rate if isinstance(false_positive_rate, tuple) else (false_positive_rate, false_positive_rate)
        self.false_negative_rate_range = false_negative_rate if isinstance(false_negative_rate, tuple) else (false_negative_rate, false_negative_rate)
        self.proximal_range = proximal_range  # Fixed value, no range
        self.fov_range = fov_range
        self.fov_angle = fov_angle

        self.device = device
        self.gnn_model_path = Path(gnn_model_path) if gnn_model_path else None
        self.rl_model_path = Path(rl_model_path) if rl_model_path else None
        self.trust_step_size = trust_step_size
        self.trust_strength_cap = trust_strength_cap

        # Initialize simulation environment (will be recreated for each episode with sampled parameters)
        self.sim_env = None

        # Initialize enhanced feature calculator for proper neural symbolic features
        self.feature_calculator = TrustFeatureCalculator()

        # Initialize ego graph builder (will be updated for each episode)
        self.ego_graph_builder = None

        # Initialize RL trust system (preferred trust updater)
        self.rl_trust_system = None
        self._initialize_rl_trust_system()

    def _initialize_rl_trust_system(self) -> None:
        """Set up the RL trust update system, falling back to fresh weights if checkpoints are absent"""
        evidence_path = str(self.gnn_model_path) if self.gnn_model_path and self.gnn_model_path.exists() else None
        updater_path = str(self.rl_model_path) if self.rl_model_path and self.rl_model_path.exists() else None

        if self.gnn_model_path and not self.gnn_model_path.exists():
            print(f"ℹ️ GNN evidence model '{self.gnn_model_path}' not found. Initializing with fresh weights.")
        if self.rl_model_path and not self.rl_model_path.exists():
            print(f"ℹ️ RL updater model '{self.rl_model_path}' not found. Initializing with fresh weights.")

        try:
            self.rl_trust_system = RLTrustSystem(
                evidence_model_path=evidence_path,
                updater_model_path=updater_path,
                device=self.device,
                step_size=self.trust_step_size,
                include_critic=False,
                strength_cap=self.trust_strength_cap,
            )
            gnn_label = evidence_path if evidence_path else "fresh initialization"
            rl_label = updater_path if updater_path else "fresh initialization"
            print(f"✅ RL trust system ready (GNN={gnn_label}, updater={rl_label})")
        except Exception as e:
            print(f"⚠️ Failed to initialize RL trust system: {e}")
            self.rl_trust_system = None
            raise

    def _simulate_step_with_rl(self, step_idx: int):
        """
        Execute a single environment step and apply RL trust updates, mirroring compare_trust_methods.
        """
        frame_data = self.sim_env.step()

        for robot in self.sim_env.robots:
            robot.update_current_timestep_tracks()

        if self.rl_trust_system is None:
            self._initialize_rl_trust_system()

        if self.rl_trust_system is None:
            raise RuntimeError("RL trust system unavailable after initialization attempt")

        self.rl_trust_system.update_trust(self.sim_env.robots)

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
        target_density = self._sample_density(self.target_density_range, step=0.0001)
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

    def _generate_labels_from_ego_graph(self, ego_graph: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate labels directly from ego graph's robot and track information

        Args:
            ego_graph: The ego graph with robot and track data

        Returns:
            Tuple of (agent_labels, track_labels) where 1 = trustworthy, 0 = adversarial
        """
        # Generate agent labels from proximal robots (stored in ego graph during construction)
        agent_labels = []
        proximal_robots = getattr(ego_graph, '_proximal_robots', [])
        for robot in proximal_robots:
            # Honest robot = 1, Adversarial robot = 0
            label = 1.0 if not robot.is_adversarial else 0.0
            agent_labels.append(label)

        agent_labels = torch.tensor(agent_labels, dtype=torch.float32).unsqueeze(1) if agent_labels else torch.empty(0, 1, dtype=torch.float32)

        # Generate track labels from track objects (stored in ego graph during construction)
        track_labels = []
        if 'track' in ego_graph.x_dict and ego_graph.x_dict['track'].shape[0] > 0:
            # Get tracks in the same order as ego graph construction
            proximal_robot_tracks = {}
            for robot in proximal_robots:
                robot_tracks = robot.get_all_current_tracks()
                proximal_robot_tracks[robot.id] = robot_tracks

            # Get tracks in same order as ego graph builder
            fused_tracks, individual_tracks, _ = self.ego_graph_builder._perform_track_fusion(
                proximal_robots, proximal_robot_tracks)
            all_tracks = fused_tracks + individual_tracks

            ground_truth_objects = getattr(self.sim_env, 'ground_truth_objects', [])

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

    def generate_episode_data(self, episode_idx: int = 0) -> Tuple[List[SupervisedDataSample], Dict]:
        """
        Generate supervised data for one episode using RL trust algorithm

        Args:
            episode_idx: Episode index for tracking

        Returns:
            Tuple of (list of data samples for the episode, episode parameters)
        """
        # Sample new parameters for this episode
        episode_params = self._sample_episode_parameters()

        print(f"Generating data for episode {episode_idx}...")
        print(f"  Parameters: {episode_params}")

        # Create simulation environment with sampled parameters
        self._create_simulation_environment(episode_params)

        episode_data = []

        for step in range(self.max_steps_per_episode):
            try:
                self._simulate_step_with_rl(step)
            except Exception as e:
                print(f"Warning: RL trust step failed at step {step}: {e}")
                break

            robots = self.sim_env.robots
            if not robots:
                print(f"Warning: No robots at step {step}")
                continue

            # Generate ego graphs for each robot (same as RL training loop)
            for ego_robot in robots:
                try:
                    # Build ego-graph for this robot
                    ego_graph = self.ego_graph_builder.build_ego_graph(ego_robot, robots)
                    if ego_graph is None:
                        continue

                    # Generate labels directly from ego graph data
                    agent_labels, track_labels = self._generate_labels_from_ego_graph(ego_graph)

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
                        save_path: str = "supervised_trust_dataset.pkl") -> Tuple[List[SupervisedDataSample], List[Dict]]:
        """
        Generate complete dataset with multiple episodes using diverse parameters

        Args:
            num_episodes: Number of episodes to generate
            save_path: Path to save the dataset

        Returns:
            Tuple of (list of all data samples, list of episode parameters)
        """
        print(f"🔄 Generating supervised dataset with {num_episodes} episodes...")
        print(f"📊 Parameter ranges:")
        print(f"   - Robot density: {self.robot_density_range}")
        print(f"   - Target density: {self.target_density_range}")
        print(f"   - Adversarial ratio: {self.adversarial_ratio_range}")
        print(f"   - False positive rate: {self.false_positive_rate_range}")
        print(f"   - False negative rate: {self.false_negative_rate_range}")
        print(f"   - World size (square): {self.world_size[0]} x {self.world_size[1]}")
        print(f"   - Proximal range (fixed): {self.proximal_range}")
        print(f"⏱️  Max steps per episode: {self.max_steps_per_episode}")
        gnn_label = str(self.gnn_model_path) if self.gnn_model_path and self.gnn_model_path.exists() else ("fresh initialization" if self.gnn_model_path is None else f"{self.gnn_model_path} (fresh init)")
        rl_label = str(self.rl_model_path) if self.rl_model_path and self.rl_model_path.exists() else ("fresh initialization" if self.rl_model_path is None else f"{self.rl_model_path} (fresh init)")
        print(f"🧠 Using RL trust system for updates (GNN: {gnn_label}, updater: {rl_label})")

        all_data = []
        all_episode_params = []

        for episode in range(num_episodes):
            try:
                episode_data, episode_params = self.generate_episode_data(episode)
                all_data.extend(episode_data)
                all_episode_params.append(episode_params)

                # Progress update
                if (episode + 1) % 5 == 0:
                    print(f"✅ Completed {episode + 1}/{num_episodes} episodes. Total samples: {len(all_data)}")

            except Exception as e:
                print(f"⚠️ Error generating episode {episode}: {e}")
                continue

        print(f"🎉 Dataset generation complete!")
        print(f"📈 Total samples: {len(all_data)}")

        # Calculate statistics
        agent_samples = sum(1 for sample in all_data if sample.agent_labels.shape[0] > 0)
        track_samples = sum(1 for sample in all_data if sample.track_labels.shape[0] > 0)

        # Check for agent comparison edges
        comparison_edge_samples = sum(1 for sample in all_data
                                    if ('agent', 'more_trustworthy_than', 'agent') in sample.edge_index_dict)

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

        print(f"📊 Dataset Statistics:")
        print(f"   - Samples with agents: {agent_samples}")
        print(f"   - Samples with tracks: {track_samples}")
        print(f"   - Samples with agent comparison edges: {comparison_edge_samples}")
        print(f"   - Avg agents per sample: {avg_agents_per_sample:.1f}")
        print(f"   - Avg tracks per sample: {avg_tracks_per_sample:.1f}")
        print(f"   - Agent feature dimensions: {sample_agent_features} (alpha/beta removed)")
        print(f"   - Track feature dimensions: {sample_track_features} (alpha/beta removed)")

        if all_episode_params:
            print(f"📈 Parameter Diversity:")
            for param_name, stats in param_stats.items():
                print(f"   - {param_name}: {stats['min']:.3f} - {stats['max']:.3f} (avg: {stats['avg']:.3f})")

        # Save dataset with parameters
        dataset_package = {
            'samples': all_data,
            'episode_parameters': all_episode_params,
            'parameter_ranges': {
                'robot_density': self.robot_density_range,
                'target_density': self.target_density_range,
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
                'comparison_edge_samples': comparison_edge_samples,
                'avg_agents_per_sample': avg_agents_per_sample,
                'avg_tracks_per_sample': avg_tracks_per_sample,
                'parameter_diversity': param_stats if all_episode_params else {}
            }
        }

        print(f"💾 Saving dataset to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(dataset_package, f)
        print(f"✅ Dataset saved successfully!")

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
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to generate (default: 10)')
    parser.add_argument('--robot-density', type=str, default='0.0005,0.0030',
                       help='Robot density range in robots per square unit (default: 0.0005,0.0030)')
    parser.add_argument('--target-density', type=str, default='0.0010,0.0050',
                       help='Target density range in objects per square unit (default: 0.0010,0.0050)')
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
    parser.add_argument('--output', type=str, default='supervised_trust_dataset.pkl',
                       help='Output file path (default: supervised_trust_dataset.pkl)')
    parser.add_argument('--gnn-model', type=str, default='trained_gnn_model.pth',
                        help='Path to the trained GNN evidence model (default: trained_gnn_model.pth)')
    parser.add_argument('--rl-model', type=str, default='rl_trust_model.pth',
                        help='Path to the trained RL updater model (default: rl_trust_model.pth)')
    args = parser.parse_args()

    # Parse parameter ranges
    robot_density_range = parse_range(args.robot_density)
    target_density_range = parse_range(args.target_density)
    adversarial_ratio_range = parse_range(args.adversarial_ratio)
    false_positive_rate_range = parse_range(args.false_positive_rate)
    false_negative_rate_range = parse_range(args.false_negative_rate)
    world_size_value = float(args.world_size)
    world_size = (world_size_value, world_size_value)

    print(f"🎯 Using parameter ranges:")
    print(f"   - Robot density: {robot_density_range}")
    print(f"   - Target density: {target_density_range}")
    print(f"   - Adversarial ratio: {adversarial_ratio_range}")
    print(f"   - False positive rate: {false_positive_rate_range}")
    print(f"   - False negative rate: {false_negative_rate_range}")
    print(f"   - World size (square): {world_size_value} x {world_size_value}")
    print(f"   - Proximal range (fixed): {args.proximal_range}")

    # Create data generator
    generator = SupervisedDataGenerator(
        robot_density=robot_density_range,
        target_density=target_density_range,
        adversarial_ratio=adversarial_ratio_range,
        world_size=world_size,
        false_positive_rate=false_positive_rate_range,
        false_negative_rate=false_negative_rate_range,
        proximal_range=args.proximal_range,
        max_steps_per_episode=args.steps,
        gnn_model_path=args.gnn_model,
        rl_model_path=args.rl_model
    )

    # Generate dataset
    dataset, episode_params = generator.generate_dataset(
        num_episodes=args.episodes,
        save_path=args.output
    )

    print(f"\n🎯 Dataset generation complete!")
    print(f"📈 {len(dataset)} samples from {len(episode_params)} episodes saved to {args.output}")
    gnn_summary = str(generator.gnn_model_path) if generator.gnn_model_path and generator.gnn_model_path.exists() else ("fresh initialization" if generator.gnn_model_path is None else f"{generator.gnn_model_path} (fresh init)")
    rl_summary = str(generator.rl_model_path) if generator.rl_model_path and generator.rl_model_path.exists() else ("fresh initialization" if generator.rl_model_path is None else f"{generator.rl_model_path} (fresh init)")
    print(f"🧠 Trust updates applied using RL trust system (GNN: {gnn_summary}, updater: {rl_summary})")
    print(f"🔗 Agent comparison edges included")
    print(f"📉 Alpha/beta features removed from node features")
    print(f"🎲 Parameter diversity across episodes for more robust training")


if __name__ == "__main__":
    main()
