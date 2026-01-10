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
from supervised_trust_gnn import EgoGraphBuilder


@dataclass
class SupervisedDataSample:
    """
    Single data sample for supervised learning (structure-only, no node features)

    The full ego graph is saved (all robots and tracks) for GNN input.
    Cross-validation constraints are applied via:
    - ego_has_cross_validation: Only samples where ego robot has co_detection/contradicts edges
    - meaningful_track_indices: Tracks detected by ego AND with edges to >=2 robots (cross-validation)

    During training, loss is computed only for:
    - Ego robot (index 0) if ego_has_cross_validation=True
    - Tracks with indices in meaningful_track_indices
    """
    edge_index_dict: Dict  # Edge indices dictionary (full ego graph structure)
    agent_labels: torch.Tensor  # Binary trust labels for ALL agents [num_agents, 1]
    track_labels: torch.Tensor  # Binary trust labels for ALL tracks [num_tracks, 1]
    num_agents: int  # Number of agent nodes in full ego graph
    num_tracks: int  # Number of track nodes in full ego graph
    ego_has_cross_validation: bool  # Does ego robot have co_detection or contradicts edges?
    meaningful_track_indices: List[int]  # Indices of tracks for loss (ego-detected + >=2 robot edges)
    timestep: int
    episode: int
    ego_robot_id: str  # ID of the ego robot for this sample
    # Pre-computed triplets for faster training
    agent_triplets: torch.Tensor = None  # [num_agents, max_edges, 8] - pre-computed triplet sequences
    agent_triplet_mask: torch.Tensor = None  # [num_agents, max_edges] - True for padding
    track_triplets: torch.Tensor = None  # [num_tracks, max_edges, 8] - pre-computed triplet sequences
    track_triplet_mask: torch.Tensor = None  # [num_tracks, max_edges] - True for padding


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
                 fov_range: float = 80.0,
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

        # Sample adversarial_ratio with increment of 0.05
        min_adv, max_adv = self.adversarial_ratio_range
        adv_steps = int((max_adv - min_adv) / 0.05)
        if adv_steps > 0:
            adv_step = random.randint(0, adv_steps)
            adversarial_ratio = round(min_adv + (adv_step * 0.05), 2)
        else:
            adversarial_ratio = min_adv

        # Sample false_positive_rate with increment of 0.05
        min_fp, max_fp = self.false_positive_rate_range
        fp_steps = int((max_fp - min_fp) / 0.05)
        if fp_steps > 0:
            fp_step = random.randint(0, fp_steps)
            false_positive_rate = round(min_fp + (fp_step * 0.05), 2)
        else:
            false_positive_rate = min_fp

        # Sample false_negative_rate with increment of 0.05
        min_fn, max_fn = self.false_negative_rate_range
        fn_steps = int((max_fn - min_fn) / 0.05)
        if fn_steps > 0:
            fn_step = random.randint(0, fn_steps)
            false_negative_rate = round(min_fn + (fn_step * 0.05), 2)
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

    def _check_ego_cross_validation(self, ego_graph: HeteroData) -> bool:
        """
        Check if ego robot (index 0) has cross-validation with other robots.

        Cross-validation means ego robot has at least one of:
        - co_detection edge with another robot (both detect same object)
        - contradicts edge with another robot (ego detects something other doesn't)

        Args:
            ego_graph: Ego graph with edge_index_dict

        Returns:
            True if ego robot has cross-validation, False otherwise
        """
        edge_index_dict = ego_graph.edge_index_dict

        # Check for co_detection edges: (agent, co_detection, agent)
        if ('agent', 'co_detection', 'agent') in edge_index_dict:
            co_detection_edges = edge_index_dict[('agent', 'co_detection', 'agent')]
            if co_detection_edges.numel() > 0:
                # Check if ego robot (index 0) is in any co_detection edge
                ego_in_co_detection = (co_detection_edges[0] == 0).any() or (co_detection_edges[1] == 0).any()
                if ego_in_co_detection:
                    return True

        # Check for contradicts edges: (agent, contradicts, agent)
        if ('agent', 'contradicts', 'agent') in edge_index_dict:
            contradicts_edges = edge_index_dict[('agent', 'contradicts', 'agent')]
            if contradicts_edges.numel() > 0:
                # Check if ego robot (index 0) is in any contradicts edge
                ego_in_contradicts = (contradicts_edges[0] == 0).any() or (contradicts_edges[1] == 0).any()
                if ego_in_contradicts:
                    return True

        return False

    def _identify_meaningful_tracks(self, ego_robot, ego_graph: HeteroData, num_tracks: int) -> List[int]:
        """
        Identify meaningful tracks for loss computation.

        A track is meaningful if:
        1. It's currently detected by ego robot (in get_all_current_tracks())
        2. It has edges to at least one OTHER robot besides ego (cross-validation)
           - Ego robot is always at index 0
           - Track must have edges to >= 2 robots total (ego + at least 1 other)

        Args:
            ego_robot: The ego robot
            ego_graph: Ego graph with track nodes and edges
            num_tracks: Total number of tracks in ego graph

        Returns:
            List of track indices that meet both criteria
        """
        meaningful_indices = []

        # Get tracks currently detected by ego robot
        ego_current_tracks = ego_robot.get_all_current_tracks()
        # IMPORTANT: Match by object_id, not track_id, because track fusion changes track_id
        ego_object_ids = set(track.object_id for track in ego_current_tracks)

        # Get fused and individual tracks from ego graph (in same order as track nodes)
        if hasattr(ego_graph, '_fused_tracks') and hasattr(ego_graph, '_individual_tracks'):
            all_tracks = ego_graph._fused_tracks + ego_graph._individual_tracks
        else:
            # Fallback: can't identify tracks without stored track list
            return []

        # Check each track for both criteria
        for track_idx, track in enumerate(all_tracks[:num_tracks]):
            # Criterion 1: Is this track currently detected by ego robot?
            # Match by object_id (not track_id) since fusion changes track_id
            if track.object_id not in ego_object_ids:
                continue

            # Criterion 2: Does this track have edges to at least one OTHER robot?
            # Count total robots with edges - should be >= 2 (ego + at least 1 other)
            num_robots_with_edges = self._count_robots_with_edges_to_track(ego_graph, track_idx)
            if num_robots_with_edges >= 2:
                meaningful_indices.append(track_idx)

        return meaningful_indices

    def _count_robots_with_edges_to_track(self, ego_graph: HeteroData, track_idx: int) -> int:
        """
        Count how many robots have edges to a specific track.

        Args:
            ego_graph: Ego graph with edge_index_dict
            track_idx: Index of the track node

        Returns:
            Number of unique robots with edges to this track
        """
        edge_index_dict = ego_graph.edge_index_dict
        robots_with_edges = set()

        # Check agent->track edges (in_fov_and_observed, in_fov_only)
        agent_to_track_edge_types = [
            ('agent', 'in_fov_and_observed', 'track'),
            ('agent', 'in_fov_only', 'track')
        ]

        for edge_type in agent_to_track_edge_types:
            if edge_type in edge_index_dict:
                edges = edge_index_dict[edge_type]
                if edges.numel() > 0:
                    # edges[0] is source (agent), edges[1] is target (track)
                    # Find all agents that have edges to this track
                    agents_to_this_track = edges[0][edges[1] == track_idx]
                    robots_with_edges.update(agents_to_this_track.tolist())

        return len(robots_with_edges)

    def _extract_triplets_for_storage(self, node_type: str, num_nodes: int, edge_index_dict: Dict):
        """
        Extract symbolic triplets for all nodes of a given type (for pre-computation during dataset generation).

        For each node, we find all outgoing edges and encode them as triplets:
        τ = (source_type, edge_relation, target_type)

        Args:
            node_type: 'agent' or 'track'
            num_nodes: Number of nodes of this type
            edge_index_dict: Dictionary of edge indices

        Returns:
            Tuple of (triplets, mask)
            - triplets: [num_nodes, max_edges, 8] - symbolic triplet representations
            - mask: [num_nodes, max_edges] - True for padding positions
        """
        # Define all edge types (same as in SupervisedTrustGNN)
        edge_types = [
            ('agent', 'in_fov_and_observed', 'track'),
            ('track', 'observed_and_in_fov_by', 'agent'),
            ('agent', 'in_fov_only', 'track'),
            ('track', 'in_fov_only_by', 'agent'),
            ('agent', 'co_detection', 'agent'),
            ('agent', 'contradicts', 'agent'),
        ]
        edge_type_to_idx = {edge_type: i for i, edge_type in enumerate(edge_types)}

        # Collect all edges where this node type is the source
        node_edges = []  # List of (node_idx, edge_list) pairs

        for node_idx in range(num_nodes):
            edge_list = []

            # Iterate through all edge types
            for edge_type in edge_types:
                src_type, relation, dst_type = edge_type

                # Check if this edge type originates from our node_type
                if src_type == node_type and edge_type in edge_index_dict:
                    edge_index = edge_index_dict[edge_type]

                    if edge_index.numel() > 0:
                        # Find edges where source is node_idx
                        mask = (edge_index[0] == node_idx)
                        if mask.any():
                            # Get target nodes for these edges
                            target_nodes = edge_index[1][mask]

                            # Create triplet for each edge
                            for target_idx in target_nodes:
                                # Encode triplet: (src_type, relation, dst_type)
                                # src_type: 0=agent, 1=track (1 bit)
                                src_bit = 1.0 if src_type == 'track' else 0.0

                                # relation: one-hot over 6 edge types (6 bits)
                                relation_onehot = [0.0] * 6
                                relation_idx = edge_type_to_idx[edge_type]
                                relation_onehot[relation_idx] = 1.0

                                # dst_type: 0=agent, 1=track (1 bit)
                                dst_bit = 1.0 if dst_type == 'track' else 0.0

                                # Concatenate: [src_bit, relation_onehot (6), dst_bit] = 8 dims
                                triplet = [src_bit] + relation_onehot + [dst_bit]
                                edge_list.append(triplet)

            node_edges.append(edge_list)

        # Find maximum number of edges across all nodes
        max_edges = max(len(edges) for edges in node_edges) if node_edges else 1
        max_edges = max(max_edges, 1)  # At least 1 to avoid empty tensors

        # Pad all edge lists to max_edges
        triplets_list = []
        mask_list = []

        for edges in node_edges:
            num_edges = len(edges)

            if num_edges > 0:
                # Pad with zeros
                padded_edges = edges + [[0.0] * 8] * (max_edges - num_edges)
                # Mask: False for real edges, True for padding
                mask = [False] * num_edges + [True] * (max_edges - num_edges)
            else:
                # No edges: all padding
                padded_edges = [[0.0] * 8] * max_edges
                mask = [True] * max_edges

            triplets_list.append(padded_edges)
            mask_list.append(mask)

        # Convert to tensors
        triplets = torch.tensor(triplets_list, dtype=torch.float32)  # [num_nodes, max_edges, 8]
        mask = torch.tensor(mask_list, dtype=torch.bool)  # [num_nodes, max_edges]

        return triplets, mask

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
        # Get all proximal robot tracks
        proximal_robot_tracks = {}
        for robot in proximal_robots:
            robot_tracks = robot.get_all_current_tracks()
            proximal_robot_tracks[robot.id] = robot_tracks

        # Get tracks in same order as ego graph builder
        fused_tracks, individual_tracks, _ = self.ego_graph_builder._perform_track_fusion(
            proximal_robots, proximal_robot_tracks)
        all_tracks = fused_tracks + individual_tracks

        if len(all_tracks) > 0:
            ground_truth_objects = getattr(self.sim_env, 'ground_truth_objects', [])

            # Create labels for ALL tracks (not just ego-owned)
            for track in all_tracks:
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

    def generate_episode_data(self, episode_idx: int = 0, step_interval: int = 10) -> Tuple[List[SupervisedDataSample], Dict]:
        """
        Generate supervised data for one episode using RL trust algorithm

        Args:
            episode_idx: Episode index for tracking
            step_interval: Sample ego graphs every N steps (default: 10)

        Returns:
            Tuple of (list of data samples for the episode, episode parameters)
        """
        # Sample new parameters for this episode
        episode_params = self._sample_episode_parameters()

        print(f"Generating data for episode {episode_idx}...")
        print(f"  Parameters: {episode_params}")
        print(f"  Sampling: every {step_interval} steps, 20% of robots")

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

        # Track filtering statistics
        filtered_no_cross_validation = 0
        filtered_no_meaningful_tracks = 0

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

            # Sample 20% of robots at each timestep to reduce dataset size
            num_robots_to_sample = max(1, int(len(robots) * 0.2))

            # IMPORTANT: Resample strategy to handle filtering
            # When a robot is filtered out, try to sample a replacement from remaining robots
            saved_robot_ids = set()  # Track which robots we've successfully saved at this timestep
            available_robots = list(robots)  # Pool of robots we can sample from
            random.shuffle(available_robots)  # Shuffle for random sampling

            # Try to get num_robots_to_sample valid samples
            robot_idx = 0
            while len(saved_robot_ids) < num_robots_to_sample and robot_idx < len(available_robots):
                ego_robot = available_robots[robot_idx]
                robot_idx += 1

                # Skip if we've already saved this robot at this timestep
                if ego_robot.id in saved_robot_ids:
                    continue

                try:
                    # Build ego-graph for this robot
                    ego_graph = self.ego_graph_builder.build_ego_graph(ego_robot, robots)
                    if ego_graph is None:
                        continue

                    # Generate labels for ALL nodes in ego graph
                    agent_labels, track_labels = self._generate_labels_from_ego_graph(ego_robot, ego_graph)

                    num_agents = agent_labels.shape[0]
                    num_tracks = track_labels.shape[0]

                    # Check cross-validation: Does ego robot have co_detection or contradicts edges?
                    ego_has_cross_validation = self._check_ego_cross_validation(ego_graph)

                    # Skip samples where ego robot has no cross-validation
                    # Continue to next robot in available pool (resampling)
                    if not ego_has_cross_validation:
                        filtered_no_cross_validation += 1
                        continue

                    # REMOVED: Filter for adversarial with 0 contradicts
                    # Reason: Train-test mismatch! During inference, we don't know which robots
                    # are adversarial, so we can't apply this filter. Model must learn to handle
                    # adversarial robots with 0 contradicts, even though they're hard to classify.
                    #
                    # if not self._check_adversarial_has_contradicts(ego_robot, ego_graph):
                    #     filtered_adversarial_no_contradicts += 1
                    #     continue

                    # Identify meaningful tracks: ego-detected tracks with edges to >=2 robots (cross-validation)
                    meaningful_track_indices = self._identify_meaningful_tracks(
                        ego_robot, ego_graph, num_tracks
                    )

                    # Skip if no meaningful tracks (rare but possible)
                    if len(meaningful_track_indices) == 0:
                        filtered_no_meaningful_tracks += 1
                        continue

                    # Pre-compute triplet sequences for faster training
                    agent_triplets, agent_triplet_mask = self._extract_triplets_for_storage(
                        'agent', num_agents, ego_graph.edge_index_dict
                    )
                    track_triplets, track_triplet_mask = self._extract_triplets_for_storage(
                        'track', num_tracks, ego_graph.edge_index_dict
                    )

                    # Create data sample with full ego graph + meaningful track indices + pre-computed triplets
                    sample = SupervisedDataSample(
                        edge_index_dict=ego_graph.edge_index_dict.copy(),
                        agent_labels=agent_labels,
                        track_labels=track_labels,
                        num_agents=num_agents,
                        num_tracks=num_tracks,
                        ego_has_cross_validation=ego_has_cross_validation,
                        meaningful_track_indices=meaningful_track_indices,
                        timestep=step,
                        episode=episode_idx,
                        ego_robot_id=ego_robot.id,
                        agent_triplets=agent_triplets,
                        agent_triplet_mask=agent_triplet_mask,
                        track_triplets=track_triplets,
                        track_triplet_mask=track_triplet_mask
                    )

                    episode_data.append(sample)

                    # Mark this robot as saved for this timestep
                    # This prevents duplicate samples and enables resampling for filtered robots
                    saved_robot_ids.add(ego_robot.id)

                except Exception as e:
                    print(f"Warning: Error generating ego graph for robot {ego_robot.id} at step {step}: {e}")
                    continue

            # If we exhausted all robots without getting enough valid samples, that's OK
            # The while loop exits and we continue to next sampling step
            # This handles cases where many robots are filtered out

        # Report filtering statistics
        total_filtered = (filtered_no_cross_validation +
                         filtered_no_meaningful_tracks)
        print(f"Generated {len(episode_data)} samples for episode {episode_idx}")
        if total_filtered > 0:
            print(f"  Filtered out {total_filtered} samples:")
            print(f"    - No cross-validation: {filtered_no_cross_validation}")
            print(f"    - No meaningful tracks: {filtered_no_meaningful_tracks}")
        return episode_data, episode_params

    def generate_dataset(self,
                        num_episodes: int = 10,
                        save_path: str = "supervised_trust_dataset.pkl",
                        log_path: str = None,
                        step_interval: int = 10) -> Tuple[List[SupervisedDataSample], List[Dict]]:
        """
        Generate complete dataset with multiple episodes using diverse parameters

        Args:
            num_episodes: Number of episodes to generate
            save_path: Path to save the dataset
            log_path: Optional path to save generation log (default: save_path.replace('.pkl', '.log'))
            step_interval: Sample ego graphs every N steps (default: 10)

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
        log_print(f"   - Robot sampling: 20% of robots at each sampled timestep")
        log_print(f"   - Expected samples per episode: ~{(self.max_steps_per_episode // step_interval) * 2:.0f} (assuming ~10 robots, 20% = 2 robots)")

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
                    episode, step_interval=step_interval
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

        # Import random for shuffling (needed regardless of balancing)
        import random as random_module
        random_module.seed(42)  # Reproducible sampling

        if num_adversarial > num_legitimate:
            # More adversarial samples: sample down to match legitimate count
            sampled_adversarial = random_module.sample(adversarial_samples, num_legitimate)

            log_print(f"\nBalancing dataset:")
            log_print(f"  Warning: Cross-validation filtering favored adversarial robots ({num_adversarial} adv vs {num_legitimate} legit)")
            log_print(f"  Reason: Adversarial robots have more contradicts edges (false positives)")
            log_print(f"  Sampling {num_legitimate} out of {num_adversarial} adversarial samples")
            log_print(f"  Keeping all {num_legitimate} legitimate samples")
            log_print(f"  Final ratio: 50% adversarial, 50% legitimate")

            adversarial_samples = sampled_adversarial
        elif num_legitimate > num_adversarial:
            # More legitimate samples: sample down to match adversarial count
            sampled_legitimate = random_module.sample(legitimate_samples, num_adversarial)

            log_print(f"\nBalancing dataset:")
            log_print(f"  Keeping all {num_adversarial} adversarial samples")
            log_print(f"  Sampling {num_adversarial} out of {num_legitimate} legitimate samples")
            log_print(f"  Final ratio: 50% adversarial, 50% legitimate")

            legitimate_samples = sampled_legitimate
        else:
            log_print(f"\nAlready balanced: {num_adversarial} adversarial, {num_legitimate} legitimate")

        # Merge balanced samples
        all_data = adversarial_samples + legitimate_samples
        random_module.shuffle(all_data)  # Shuffle to mix adversarial and legitimate

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
            avg_agents_per_sample = np.mean([sample.num_agents for sample in all_data])
            avg_tracks_per_sample = np.mean([sample.num_tracks for sample in all_data])
        else:
            avg_agents_per_sample = 0
            avg_tracks_per_sample = 0

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
        log_print(f"   - Structure-only learning: No node features (graph edges only)")

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
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Number of episodes to generate (default: 2000)')
    parser.add_argument('--robot-density', type=str, default='0.0005,0.0020',
                       help='Robot density range in robots per square unit (default: 0.0005,0.0020)')
    parser.add_argument('--target-density-multiplier', type=str, default='2.0',
                       help='Target density multiplier applied to sampled robot density (default: 2.0)')
    parser.add_argument('--adversarial-ratio', type=str, default='0.1,0.3',
                       help='Adversarial robot ratio: single value or range "min,max" (default: 0.1,0.3)')
    parser.add_argument('--false-positive-rate', type=str, default='0.1,0.3',
                       help='False positive rate: single value or range "min,max" (default: 0.1,0.3)')
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
        step_interval=args.step_interval
    )

    print(f"\n🎯 Dataset generation complete!")
    print(f"📈 {len(dataset)} samples from {len(episode_params)} episodes saved to {args.output}")
    print(f"🧠 Trust assigned using ground truth labels:")
    print(f"   - Legitimate robots/tracks: trust ∈ [0.7, 1.0]")
    print(f"   - Adversarial robots/tracks: trust ∈ [0.0, 0.3]")
    print(f"   - Confidence: Higher when trust is closer to 0 or 1")
    print(f"🔗 Agent co-detection edges included (robots detecting same objects)")
    print(f"🏗️  Structure-only learning: Graph edges only, no node features")
    print(f"🎲 Parameter diversity across episodes for more robust training")


if __name__ == "__main__":
    main()
