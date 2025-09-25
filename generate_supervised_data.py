#!/usr/bin/env python3
"""
Data Generation for Supervised Trust Learning

This script generates training data from simulation environments with ground truth labels
for whether each robot/track is trustworthy. Uses the paper trust algorithm for realistic
trust updates and adds agent-to-agent trustworthiness comparison edges.
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import List, Tuple, Dict, Union
import pickle
from dataclasses import dataclass
import random

# Import existing simulation components
from simulation_environment import SimulationEnvironment
from neural_symbolic_trust_algorithm import NeuralSymbolicTrustAlgorithm
from paper_trust_algorithm import PaperTrustAlgorithm
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
    Generates supervised learning data from simulation environment using paper trust algorithm
    """

    def __init__(self,
                 num_robots: Union[int, Tuple[int, int]] = 5,
                 num_targets: Union[int, Tuple[int, int]] = 20,
                 adversarial_ratio: Union[float, Tuple[float, float]] = 0.5,
                 world_size: Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]] = (60, 60),
                 false_positive_rate: Union[float, Tuple[float, float]] = 0.5,
                 false_negative_rate: Union[float, Tuple[float, float]] = 0.0,
                 proximal_range: float = 50.0,
                 fov_range: float = 50.0,
                 fov_angle: float = np.pi/3,
                 max_steps_per_episode: int = 100):
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
        """
        self.max_steps_per_episode = max_steps_per_episode

        # Store parameter ranges for sampling
        self.num_robots_range = num_robots if isinstance(num_robots, tuple) else (num_robots, num_robots)
        self.num_targets_range = num_targets if isinstance(num_targets, tuple) else (num_targets, num_targets)
        self.adversarial_ratio_range = adversarial_ratio if isinstance(adversarial_ratio, tuple) else (adversarial_ratio, adversarial_ratio)

        # Handle world_size range - check if it's a range of tuples or a single tuple
        if isinstance(world_size, tuple) and len(world_size) == 2 and isinstance(world_size[0], tuple):
            # Range format: ((min_x, min_y), (max_x, max_y))
            self.world_size_range = world_size
        else:
            # Single size format: (x, y)
            self.world_size_range = (world_size, world_size)

        self.false_positive_rate_range = false_positive_rate if isinstance(false_positive_rate, tuple) else (false_positive_rate, false_positive_rate)
        self.false_negative_rate_range = false_negative_rate if isinstance(false_negative_rate, tuple) else (false_negative_rate, false_negative_rate)
        self.proximal_range = proximal_range  # Fixed value, no range
        self.fov_range = fov_range
        self.fov_angle = fov_angle

        # Initialize simulation environment (will be recreated for each episode with sampled parameters)
        self.sim_env = None

        # Initialize paper trust algorithm for realistic trust updates
        self.paper_trust_algorithm = PaperTrustAlgorithm()

        # Initialize enhanced feature calculator for proper neural symbolic features
        self.feature_calculator = TrustFeatureCalculator()

        # Initialize ego graph builder (will be updated for each episode)
        self.ego_graph_builder = None

    def _sample_episode_parameters(self) -> Dict:
        """
        Sample parameters for a single episode from the specified ranges with specific increments

        Returns:
            Dictionary containing sampled parameters
        """
        # Sample integer parameters
        num_robots = random.randint(self.num_robots_range[0], self.num_robots_range[1])

        # Sample num_targets with increment of 5 from lower bound
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

        # Sample world_size with increment of 10 (square world: x = y)
        min_world, max_world = self.world_size_range

        # Sample square world size (same for both x and y)
        min_size, max_size = min_world[0], max_world[0]  # Use x dimension for both
        size_steps = int((max_size - min_size) / 10)
        if size_steps > 0:
            size_step = random.randint(0, size_steps)
            world_size = min_size + (size_step * 10)
        else:
            world_size = min_size

        return {
            'num_robots': num_robots,
            'num_targets': num_targets,
            'adversarial_ratio': adversarial_ratio,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'world_size': (world_size, world_size),  # Square world
            'proximal_range': self.proximal_range  # Fixed value
        }

    def _create_simulation_environment(self, params: Dict) -> None:
        """
        Create a new simulation environment with the given parameters

        Args:
            params: Dictionary of sampled parameters
        """
        self.sim_env = SimulationEnvironment(
            num_robots=params['num_robots'],
            num_targets=params['num_targets'],
            adversarial_ratio=params['adversarial_ratio'],
            world_size=params['world_size'],
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
        Generate supervised data for one episode using paper trust algorithm

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

        # Reset simulation environment
        self.sim_env.reset()
        episode_data = []

        for step in range(self.max_steps_per_episode):
            try:
                self.sim_env.step()

                # Check if simulation should end (can be enhanced with specific conditions)
                if step >= self.max_steps_per_episode:
                    break

            except Exception as e:
                print(f"Warning: Error during simulation step {step}: {e}")
                break
            
            # Get current robots
            robots = self.sim_env.robots
            for robot in robots:
                robot.update_current_timestep_tracks()
            if not robots:
                print(f"Warning: No robots at step {step}")
                continue
            
            # Build global state from current simulation state
            global_state = self._build_global_state()
            if global_state is None:
                print(f"Warning: No global state at step {step}")
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
            
            try:
                trust_updates = self.paper_trust_algorithm.update_trust(robots, self.sim_env)
                print(f"  Step {step}: Applied trust updates to {len(trust_updates)} robots")
            except Exception as e:
                print(f"Warning: Trust update failed at step {step}: {e}")
        print(f"Generated {len(episode_data)} samples for episode {episode_idx}")
        return episode_data, episode_params

    def _build_global_state(self):
        """
        Build global state from current simulation environment

        Returns:
            HeteroData representing the global state
        """
        try:
            robots = self.sim_env.robots
            if not robots:
                return None

            # Get all tracks from all robots
            all_tracks = []
            for robot in robots:
                robot_tracks = robot.get_all_current_tracks()
                all_tracks.extend(robot_tracks)

            # Build global graph using ego graph builder with all robots
            # Allow building graph even with no tracks initially
            global_graph = self.ego_graph_builder._build_multi_robot_graph(
                robots, [], all_tracks, {}  # No fused tracks, all individual
            )
            return global_graph

        except Exception as e:
            print(f"Warning: Failed to build global state: {e}")
            return None

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
        print(f"   - Robots: {self.num_robots_range}")
        print(f"   - Targets: {self.num_targets_range}")
        print(f"   - Adversarial ratio: {self.adversarial_ratio_range}")
        print(f"   - False positive rate: {self.false_positive_rate_range}")
        print(f"   - False negative rate: {self.false_negative_rate_range}")
        print(f"   - World size (square): {self.world_size_range[0][0]} to {self.world_size_range[1][0]}")
        print(f"   - Proximal range (fixed): {self.proximal_range}")
        print(f"⏱️  Max steps per episode: {self.max_steps_per_episode}")
        print(f"🧠 Using paper trust algorithm for realistic trust updates")

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
                'num_robots': self.num_robots_range,
                'num_targets': self.num_targets_range,
                'adversarial_ratio': self.adversarial_ratio_range,
                'false_positive_rate': self.false_positive_rate_range,
                'false_negative_rate': self.false_negative_rate_range,
                'world_size': self.world_size_range,
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
    parser.add_argument('--robots', type=str, default='3,10',
                       help='Number of robots: single value or range "min,max" (default: 3,10)')
    parser.add_argument('--targets', type=str, default='10,30',
                       help='Number of targets: single value or range "min,max" (default: 10,30)')
    parser.add_argument('--adversarial-ratio', type=str, default='0.2,0.5',
                       help='Adversarial robot ratio: single value or range "min,max" (default: 0.2,0.5)')
    parser.add_argument('--false-positive-rate', type=str, default='0.1,0.7',
                       help='False positive rate: single value or range "min,max" (default: 0.1,0.7)')
    parser.add_argument('--false-negative-rate', type=str, default='0.0,0.3',
                       help='False negative rate: single value or range "min,max" (default: 0.0,0.3)')
    parser.add_argument('--world-size', type=str, default='50,100',
                       help='Square world size: single value or range "min,max" (default: 50,100)')
    parser.add_argument('--proximal-range', type=float, default=50.0,
                       help='Proximal sensing range: fixed value (default: 50.0)')
    parser.add_argument('--steps', type=int, default=100,
                       help='Max steps per episode (default: 100)')
    parser.add_argument('--output', type=str, default='supervised_trust_dataset.pkl',
                       help='Output file path (default: supervised_trust_dataset.pkl)')

    args = parser.parse_args()

    # Parse parameter ranges
    robots_range = parse_range(args.robots)
    targets_range = parse_range(args.targets)
    adversarial_ratio_range = parse_range(args.adversarial_ratio)
    false_positive_rate_range = parse_range(args.false_positive_rate)
    false_negative_rate_range = parse_range(args.false_negative_rate)
    world_size_range = parse_range(args.world_size)

    # Convert robot and target ranges to integers
    robots_range = (int(robots_range[0]), int(robots_range[1]))
    targets_range = (int(targets_range[0]), int(targets_range[1]))

    # Convert world size range to square format: ((min, min), (max, max))
    world_size_range = ((world_size_range[0], world_size_range[0]),
                       (world_size_range[1], world_size_range[1]))

    print(f"🎯 Using parameter ranges:")
    print(f"   - Robots: {robots_range}")
    print(f"   - Targets: {targets_range}")
    print(f"   - Adversarial ratio: {adversarial_ratio_range}")
    print(f"   - False positive rate: {false_positive_rate_range}")
    print(f"   - False negative rate: {false_negative_rate_range}")
    print(f"   - World size (square): {world_size_range[0][0]} to {world_size_range[1][0]}")
    print(f"   - Proximal range (fixed): {args.proximal_range}")

    # Create data generator
    generator = SupervisedDataGenerator(
        num_robots=robots_range,
        num_targets=targets_range,
        adversarial_ratio=adversarial_ratio_range,
        world_size=world_size_range,
        false_positive_rate=false_positive_rate_range,
        false_negative_rate=false_negative_rate_range,
        proximal_range=args.proximal_range,
        max_steps_per_episode=args.steps
    )

    # Generate dataset
    dataset, episode_params = generator.generate_dataset(
        num_episodes=args.episodes,
        save_path=args.output
    )

    print(f"\n🎯 Dataset generation complete!")
    print(f"📈 {len(dataset)} samples from {len(episode_params)} episodes saved to {args.output}")
    print(f"🧠 Trust updates applied using paper trust algorithm")
    print(f"🔗 Agent comparison edges included")
    print(f"📉 Alpha/beta features removed from node features")
    print(f"🎲 Parameter diversity across episodes for more robust training")


if __name__ == "__main__":
    main()