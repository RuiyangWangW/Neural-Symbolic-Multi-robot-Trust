"""
RL Trust Environment

This module contains the RLTrustEnvironment class that wraps the simulation environment
for reinforcement learning training of the trust GNN.
"""

import numpy as np
import torch
import random
import math
from typing import Dict, List, Optional
from simulation_environment import SimulationEnvironment
# Robot and Track classes are imported through simulation_environment
# This maintains consistency with paper_trust_algorithm.py approach


class RLTrustEnvironment:
    """
    Environment wrapper for RL training
    
    This class follows the same trust update approach as paper_trust_algorithm.py:
    - Uses Robot.update_trust() and Track.update_trust() methods
    - Uses Robot.trust_value and Track.trust_value properties  
    - Maintains consistency with the Robot/Track class interfaces
    """
    
    def _get_robots_list(self):
        """Get robots as a consistent list format"""
        if not hasattr(self.sim_env, 'robots') or not self.sim_env.robots:
            return []
        return self.sim_env.robots  # SimulationEnvironment.robots is already a List[Robot]
    
    def __init__(self, num_robots=5, num_targets=20, adversarial_ratio=0.5, max_steps_per_episode=100, 
                 world_size=(60, 60), false_positive_rate=0.5, false_negative_rate=0.0, 
                 movement_speed=1.0, proximal_range=100.0, fov_range=50.0, fov_angle=np.pi/3):

        # Store all parameters (now configurable from training script)
        self.num_robots = num_robots
        self.num_targets = num_targets
        self.adversarial_ratio = adversarial_ratio
        self.world_size = world_size
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.movement_speed = movement_speed
        self.proximal_range = proximal_range
        self.fov_range = fov_range
        self.fov_angle = fov_angle
        
        # Store max steps per episode
        self.max_steps_per_episode = max_steps_per_episode
        
        # Multi-ego training mode - each robot serves as ego at each timestep
        self.accumulated_robot_updates = {}  # robot_id -> list of (delta_alpha, delta_beta)
        self.accumulated_track_updates = {}  # track_id -> list of (delta_alpha, delta_beta)
    
    def _calculate_beta_std(self, alpha: float, beta: float) -> float:
        """
        Calculate the standard deviation of a Beta(alpha, beta) distribution.
        
        The standard deviation of Beta(α, β) is: σ = √(αβ/((α+β)²(α+β+1)))
        This represents the uncertainty/confidence in the trust estimate.
        
        Args:
            alpha: Alpha parameter of Beta distribution
            beta: Beta parameter of Beta distribution
            
        Returns:
            Standard deviation of the Beta distribution
        """
        alpha_plus_beta = alpha + beta
        variance = (alpha * beta) / ((alpha_plus_beta ** 2) * (alpha_plus_beta + 1))
        return math.sqrt(variance)
    
    def reset(self):
        """Reset environment for new episode"""
        # Clear and reset all trust distributions for fresh start each episode
        self._previous_trust_distributions = {}
        
        # Clear multi-ego accumulation structures
        self.accumulated_robot_updates.clear()
        self.accumulated_track_updates.clear()
        
        # FIXED SCENARIO: Set consistent random seed BEFORE creating SimulationEnvironment
        # This ensures identical scenarios across all episodes
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)
        
        # Create new simulation environment (since it doesn't have reset method)
        # Step once to generate initial observations
        print(f"   🌍 Creating SimulationEnvironment: {self.num_robots} robots, {self.num_targets} targets")
        self.sim_env = SimulationEnvironment(
            num_robots=self.num_robots,
            num_targets=self.num_targets,
            adversarial_ratio=self.adversarial_ratio,
            world_size=self.world_size,
            proximal_range=self.proximal_range,
            fov_range=self.fov_range,
            fov_angle=self.fov_angle,
        )
        print(f"   📊 SimulationEnvironment created with {len(self.sim_env.robots)} robots")
        
        self.step_count = 0
        
        # Verify clean start
        self._verify_clean_trust_reset()
        
        try:
            self.sim_env.step()
            for robot in self.sim_env.robots:
                robot.update_current_timestep_tracks()
        except Exception as e:
            print(f"   ⚠️ Warning: Initial simulation step failed: {e}")
            
        # Re-seed for consistency 
        np.random.seed(42)
        random.seed(42)
        
        # Initialize with fresh initial trust values for this episode
        for robot in self.sim_env.robots:
            self._previous_trust_distributions[robot.id] = {
                'alpha': 1.0,
                'beta': 1.0
            }
        
        return self._get_current_state()
    
    def _verify_clean_trust_reset(self):
        """Verify that all trust values start fresh for the episode (consistent with paper_trust_algorithm.py)"""
        for robot in self.sim_env.robots:
            # Check robot trust values using properties (same as paper algorithm)
            if robot.trust_alpha != 1.0 or robot.trust_beta != 1.0:
                robot.trust_alpha = 1.0
                robot.trust_beta = 1.0
            
            # Also reset all track trust values to default
            for track in robot.get_all_tracks():
                if track.trust_alpha != 1.0 or track.trust_beta != 1.0:
                    track.trust_alpha = 1.0
                    track.trust_beta = 1.0
        
        # Ensure no persistent data structures exist
        persistent_attrs = ['_previous_trust_distributions']
        for attr in persistent_attrs:
            if hasattr(self, attr) and getattr(self, attr):
                # print(f"  ⚠️  WARNING: {attr} not properly cleared!")
                setattr(self, attr, {})
    
    
    def _get_current_state(self):
        """Get current state with multi-robot track fusion (no single ego robot)"""
        # Get all robots
        robots = self._get_robots_list()
        
        if not robots:
            return None
        
        # Use all robots equally - no single ego robot
        # Step 1: Generate individual track lists for each robot
        individual_robot_tracks = self._generate_individual_robot_tracks(robots)
        
        # Step 2: Perform track fusion between robots
        fused_tracks, individual_tracks, track_fusion_map = self._perform_track_fusion(
            robots, individual_robot_tracks)
        
        # Step 3: Build graph with all tracks and proper edges
        graph_data = self._build_multi_robot_graph(
            robots, fused_tracks, individual_tracks, track_fusion_map)
        
        return graph_data
    
    def _generate_individual_robot_tracks(self, robots):
        """Generate individual track lists for each robot using Robot.get_all_tracks()"""
        individual_robot_tracks = {}
        
        total_tracks = 0
        for robot in robots:
            # Use the robot's built-in track management
            robot_tracks = robot.get_all_current_tracks()  # Returns List[Track]
            individual_robot_tracks[robot.id] = robot_tracks
            total_tracks += len(robot_tracks)
            
        return individual_robot_tracks
    
    
    def _perform_track_fusion(self, robots, individual_robot_tracks):
        """Perform track fusion between robots with proper trust inheritance"""
        fusion_distance_threshold = 5.0
        
        fused_tracks = []
        individual_tracks = []
        track_fusion_map = {}
        
        # Collect all tracks from all robots
        all_tracks = []
        for robot_id, tracks in individual_robot_tracks.items():
            for track in tracks:
                all_tracks.append((robot_id, track))

        # Group tracks by object_id - this is the cleaner approach
        object_to_tracks = {}
        for robot_id, track in all_tracks:
            object_id = track.object_id
            if object_id not in object_to_tracks:
                object_to_tracks[object_id] = []
            object_to_tracks[object_id].append((robot_id, track))
        
        # Process each object group
        for object_id, tracks_list in object_to_tracks.items():
            if len(tracks_list) > 1:
                # Multiple robots see the same object - create fused track
                fused_track = self._create_fused_track(tracks_list, robots)
                fused_tracks.append(fused_track)
                
                # Map all constituent tracks to the fused track
                for robot_id, track in tracks_list:
                    track_fusion_map[track.track_id] = fused_track.track_id
            else:
                # Only one robot sees this object - keep as individual track
                robot_id, individual_track = tracks_list[0]
                individual_tracks.append(individual_track)
                track_fusion_map[individual_track.track_id] = individual_track.track_id
        
        return fused_tracks, individual_tracks, track_fusion_map
    
    def _create_fused_track(self, tracks_to_fuse, all_robots):
        """Create a fused track with proper trust inheritance"""
        # Use highest trust robot's track as primary track
        robot_trusts = {}
        for robot in all_robots:
            robot_trusts[robot.id] = robot.trust_value  # Use Robot.trust_value property
        
        # Find track from highest trust robot
        best_track = None
        best_robot_trust = -1
        for robot_id, track in tracks_to_fuse:
            if robot_trusts[robot_id] > best_robot_trust:
                best_robot_trust = robot_trusts[robot_id]
                best_track = track
        
        primary_track = best_track
        trust_alpha = primary_track.trust_alpha
        trust_beta = primary_track.trust_beta  
        
        # Create fused track ID
        contributing_robots = sorted([r for r, t in tracks_to_fuse])
        fused_id = f"fused_{'_'.join(map(str, contributing_robots))}_{primary_track.object_id}"
        
        # Average position and velocity from all contributing tracks
        positions = np.array([t.position for r, t in tracks_to_fuse])
        velocities = np.array([t.velocity for r, t in tracks_to_fuse])
        
        avg_position = np.mean(positions, axis=0)
        avg_velocity = np.mean(velocities, axis=0)
        
        # Create fused track using the Track class from robot_track_classes
        from robot_track_classes import Track
        fused_track = Track(
            track_id=fused_id,
            robot_id=primary_track.robot_id,  # Keep the primary robot's ID
            object_id=primary_track.object_id,
            position=avg_position,
            velocity=avg_velocity,
            trust_alpha=trust_alpha,
            trust_beta=trust_beta,
            timestamp=primary_track.timestamp
        )
        
        return fused_track
    
    def _build_multi_robot_graph(self, robots, fused_tracks, individual_tracks, track_fusion_map):
        """Build graph with all robots and tracks, including proper edge relationships"""
        from torch_geometric.data import HeteroData
        import torch
        
        graph_data = HeteroData()
        all_tracks = fused_tracks + individual_tracks
                
        # Create agent nodes for all robots with rich neural-symbolic features
        agent_nodes = {}
        agent_features = []
        for i, robot in enumerate(robots):
            agent_nodes[robot.id] = i
            
            # Compute trust-based predicates using Robot.trust_value property
            robot_trust = robot.trust_value
            
            # Feature 1: HighConfidence(robot) - robot confidence > 5.0
            # Calculate confidence as inverse of Beta distribution standard deviation
            robot_std = self._calculate_beta_std(robot.trust_alpha, robot.trust_beta)
            robot_confidence = 1.0 / (robot_std + 1e-6)  # Inverse std with epsilon for stability
            high_confidence_pred = 1.0 if robot_confidence > 5.0 else 0.0
            
            # Feature 2: HighlyTrusted(robot) - confident positive classification
            highly_trusted_pred = 1.0 if robot_trust > 0.5 else 0.0
            
            # Feature 3: Suspicious(robot) - likely adversarial
            suspicious_pred = 1.0 if robot_trust < 0.5 else 0.0
            
            # Feature 4: HighConnectivity(robot) - observes many tracks
            robot_track_count = sum(1 for track in all_tracks if self._robot_observes_track(robot, track, fused_tracks, individual_tracks, track_fusion_map))
            high_connectivity_pred = 1.0 if robot_track_count >= 3 else 0.0
            
            agent_features.append([
                high_confidence_pred,    # Feature 1
                highly_trusted_pred,     # Feature 2  
                suspicious_pred,         # Feature 3
                high_connectivity_pred,  # Feature 4
                robot.trust_alpha,       # Feature 5: Alpha parameter
                robot.trust_beta,        # Feature 6: Beta parameter
            ])
        
        graph_data['agent'].x = torch.tensor(agent_features, dtype=torch.float)
        graph_data.agent_nodes = agent_nodes
        
        # Create track nodes for all tracks with rich neural-symbolic features  
        track_nodes = {}
        track_features = []
        for i, track in enumerate(all_tracks):
            track_nodes[track.track_id] = i  # Use track.track_id instead of track.id
            
            # Compute trust and confidence based predicates using Track.trust_value property
            track_trust = track.trust_value
            
            # Feature 1: HighConfidence(track) - track confidence > 5.0
            # Calculate confidence as inverse of Beta distribution standard deviation
            track_std = self._calculate_beta_std(track.trust_alpha, track.trust_beta)
            track_confidence = 1.0 / (track_std + 1e-6)  # Inverse std with epsilon for stability
            high_confidence_pred = 1.0 if track_confidence > 5.0 else 0.0

            # Feature 2: highly_trusted_pred(track) - basic trust > 0.8
            highly_trusted_pred = 1.0 if track_trust > 0.5 else 0.0
            
            # Feature 3: LikelyFalsePositive(track) - suspicious patterns
            suspicious_pred = 1.0 if (track_trust < 0.5) else 0.0

            # Feature 4: MultiRobotObserved(track) - observed by multiple robots (fused)
            multi_robot_pred = 1.0 if track not in individual_tracks else 0.0
            
            # Feature 5: WellObserved(track) - track has been observed/updated many times
            well_observed_pred = 1.0 if track.observation_count > 50 else 0.0
            
            track_features.append([
                high_confidence_pred,    # Feature 1
                highly_trusted_pred,     # Feature 2
                multi_robot_pred,        # Feature 3  
                suspicious_pred,         # Feature 4
                well_observed_pred,      # Feature 5
                track.trust_alpha,       # Feature 6: Alpha parameter
                track.trust_beta,        # Feature 7: Beta parameter
            ])
        
        graph_data['track'].x = torch.tensor(track_features, dtype=torch.float)
        graph_data.track_nodes = track_nodes
        
        # Build edges with precise semantics:
        # - in_fov_and_observed: Robot observes a track that is within its FoV  
        # - in_fov_only: Track is in robot's FoV but robot does not observe it
        in_fov_and_observed_edges = []  # (agent, track) - robot observes track AND it's in FoV
        observed_and_in_fov_by_edges = []  # (track, agent) - track observed by robot AND in its FoV
        in_fov_only_edges = []  # (agent, track) - track in robot's FoV but NOT observed by robot
        in_fov_only_by_edges = []  # (track, agent) - track in robot's FoV but NOT observed by robot
        
        # CRITICAL: Create edges to maintain same connectivity as original
        observed_count = 0
        fov_only_count = 0
                
        # Debug tracking
        total_possible_observations = 0
        actual_observations = 0
        
        for robot in robots:
            robot_idx = agent_nodes[robot.id]
            
            for track in all_tracks:
                track_idx = track_nodes[track.track_id]
                total_possible_observations += 1
                
                # Check if robot observes this track (based on ownership/contribution)
                observes_track = self._robot_observes_track(robot, track, fused_tracks, individual_tracks, track_fusion_map)
                
                # Check if track is in robot's field of view (distance/angle based)
                in_fov_by_distance = self._track_in_robot_fov(robot, track)
                
                # Now categorize based on precise semantics
                if observes_track:
                    # Robot observes the track (and it must be in some form of FoV)
                    in_fov_and_observed_edges.append([robot_idx, track_idx])
                    observed_and_in_fov_by_edges.append([track_idx, robot_idx])
                    observed_count += 1
                    actual_observations += 1
                elif in_fov_by_distance:
                    # Track is in FoV by distance but robot doesn't observe it
                    in_fov_only_edges.append([robot_idx, track_idx])
                    in_fov_only_by_edges.append([track_idx, robot_idx])
                    fov_only_count += 1
        
        # Convert to tensors and create proper edge structure
        edge_types = [
            ('agent', 'in_fov_and_observed', 'track'),
            ('track', 'observed_and_in_fov_by', 'agent'),
            ('agent', 'in_fov_only', 'track'),
            ('track', 'in_fov_only_by', 'agent'),
        ]
        
        edge_data = [in_fov_and_observed_edges, observed_and_in_fov_by_edges, in_fov_only_edges, in_fov_only_by_edges]
        
        for edge_type, edges in zip(edge_types, edge_data):
            if edges:
                # Convert list of [src, dst] pairs to tensor format [2, num_edges]
                edge_tensor = torch.tensor(edges, dtype=torch.long).T
                graph_data[edge_type].edge_index = edge_tensor
            else:
                graph_data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Add x_dict and edge_index_dict properties that the GNN expects
        graph_data.x_dict = {
            'agent': graph_data['agent'].x,
            'track': graph_data['track'].x
        }
        
        graph_data.edge_index_dict = {}
        for edge_type in edge_types:
            if hasattr(graph_data[edge_type], 'edge_index'):
                graph_data.edge_index_dict[edge_type] = graph_data[edge_type].edge_index
        
        # Store additional information for trust updates
        graph_data._current_robots = robots
        graph_data._fused_tracks = fused_tracks
        graph_data._individual_tracks = individual_tracks
        graph_data._track_fusion_map = track_fusion_map
        return graph_data
    
    def _robot_observes_track(self, robot, track, fused_tracks, individual_tracks, track_fusion_map):
        """Determine if robot observes this track (based on fusion logic)"""
        # Check individual tracks first
        if track in individual_tracks:
            return track.robot_id == robot.id
        
        # Check if this is a fused track and robot contributed to it
        if track in fused_tracks:
            # Debug: Show the mapping logic
            found_contribution = False
            for original_id, fused_id in track_fusion_map.items():
                if fused_id == track.track_id:
                    # Check if this robot contributed to the fused track
                    if original_id.startswith(f"{robot.id}_"):
                        found_contribution = True
                        break
            
            return found_contribution
        
        return False
    
    def _track_in_robot_fov(self, robot, track):
        """Determine if track is in robot's field of view using Robot's built-in method"""
        return robot.is_in_fov(track.position)  # Use Robot class's is_in_fov method
   
    def step(self, all_actions, step_count):
        """Apply all actions from multi-ego training to the environment"""
        
        # Get current state BEFORE applying updates
        current_state = self._get_current_state()
        self.step_count = step_count

        # Clear accumulated updates for this step
        self.accumulated_robot_updates.clear()
        self.accumulated_track_updates.clear()
        
        # Apply actions from all ego robots
        if all_actions and current_state:
            robots = self._get_robots_list()
            
            for i, ego_actions in enumerate(all_actions):
                if i < len(robots):
                    ego_robot_id = robots[i].id
                    try:
                        # Accumulate trust updates from this ego robot's actions
                        self._accumulate_trust_updates(ego_actions, current_state, ego_robot_id)
                    except Exception as e:
                        print(f"Error applying actions for ego robot {ego_robot_id}: {e}")
                        continue
            
            # Apply all accumulated updates at once
            self._apply_accumulated_trust_updates()

        np.random.seed(42 + self.step_count)  # Different seed per step, but deterministic
        random.seed(42 + self.step_count)
            
        # Advance simulation FIRST
        self.sim_env.step() 

        for robot in self.sim_env.robots:
            robot.update_current_timestep_tracks()
            
        # Get final state after all updates applied
        next_state = self._get_current_state()
        
        # Prepare simulation step data - no single ego robot  
        robots = self._get_robots_list()
        ground_truth_objects = []
        if hasattr(self.sim_env, 'ground_truth_objects'):
            ground_truth_objects = self.sim_env.ground_truth_objects
        
        simulation_step_data = {
            'all_robots': robots,  # Use all robots instead of ego/proximal split
            'ground_truth_objects': ground_truth_objects
        }
        
        # Store current trust distributions for reward computation
        current_trust_distributions = {}
        if hasattr(current_state, '_current_robots'):
            for robot in current_state._current_robots:
                # Use Robot's trust_value property (same as paper_trust_algorithm.py)
                trust_value = robot.trust_value
                current_trust_distributions[robot.id] = {
                    'trust': trust_value,
                    'alpha': robot.trust_alpha,
                    'beta': robot.trust_beta
                }
        
        simulation_step_data['current_trust_distributions'] = current_trust_distributions
        
        # Check if episode is done
        num_objects = len(ground_truth_objects)
        num_robots = len(robots)
        
        done = (num_robots == 0 or 
                num_objects == 0 or
                self.step_count >= self.max_steps_per_episode-1)
        reward = self._compute_sparse_reward(next_state, simulation_step_data, done)

        return next_state, reward, done, {
            'step_count': self.step_count,
            'num_robots': num_robots,
            'num_objects': num_objects,
            'training_mode': f"multi_ego_{len(all_actions)}_robots",
            'trust_updates_applied': True,
            'successful_ego_robots': len(all_actions),
            'reward_components': {
                'total': reward
            }
        }
    
    def _accumulate_trust_updates(self, actions, robot_state, robot_id):
        """Accumulate trust updates from this robot's perspective (don't apply yet)"""
        if not actions or not robot_state:
            return
            
        # Accumulate robot trust updates
        if 'agent' in actions and hasattr(robot_state, 'agent_nodes'):
            agent_actions = actions['agent']
            
            for robot_id, node_idx in robot_state.agent_nodes.items():
                if node_idx < agent_actions['value'].shape[0]:
                    # Convert GNN action to PSM updates
                    psm_value = agent_actions['value'][node_idx].item()
                    psm_confidence = agent_actions['confidence'][node_idx].item()
                    
                    delta_alpha = psm_confidence * psm_value 
                    delta_beta = psm_confidence * (1.0 - psm_value)
                    
                    # Store update from this robot's perspective
                    if robot_id not in self.accumulated_robot_updates:
                        self.accumulated_robot_updates[robot_id] = []
                    
                    self.accumulated_robot_updates[robot_id].append((delta_alpha, delta_beta))
        
        # Accumulate track trust updates
        if 'track' in actions and hasattr(robot_state, 'track_nodes'):
            track_actions = actions['track']
            
            for track_id, node_idx in robot_state.track_nodes.items():
                if node_idx < track_actions['value'].shape[0]:
                    # Convert GNN action to PSM updates
                    psm_value = track_actions['value'][node_idx].item()
                    psm_confidence = track_actions['confidence'][node_idx].item()
                    
                    delta_alpha = psm_confidence * psm_value 
                    delta_beta = psm_confidence * (1.0 - psm_value)
                    
                    # Store update from this robot's perspective
                    if track_id not in self.accumulated_track_updates:
                        self.accumulated_track_updates[track_id] = []
                    
                    self.accumulated_track_updates[track_id].append((delta_alpha, delta_beta))
    
    def _apply_accumulated_trust_updates(self):
        """Apply all accumulated trust updates from multiple ego perspectives"""
        
        #print(f"\n🔄 [STEP {self.step_count}] Applying Trust Updates:")
        
        # Apply accumulated robot updates using Robot.update_trust() method
        robots = self._get_robots_list()
        
        for robot in robots:
            if robot.id in self.accumulated_robot_updates:
                updates = self.accumulated_robot_updates[robot.id]
                
                # Store previous trust values for debugging
                prev_alpha = robot.trust_alpha
                prev_beta = robot.trust_beta
                prev_trust = robot.trust_value
                
                # Sum all delta updates from different robot perspectives
                total_delta_alpha = sum(delta_alpha for delta_alpha, delta_beta in updates)
                total_delta_beta = sum(delta_beta for delta_alpha, delta_beta in updates)
                
                # Apply accumulated updates using Robot's built-in method (same as paper_trust_algorithm.py)
                robot.update_trust(total_delta_alpha, total_delta_beta)
                
                # Debug print robot trust update
                #is_adversarial = getattr(robot, 'is_adversarial', False)
                #adv_marker = "🟥" if is_adversarial else "🟩"
                #print(f"  {adv_marker} Robot {robot.id}: {prev_trust:.3f} → {robot.trust_value:.3f} " +
                #      f"(α: {prev_alpha:.2f}→{robot.trust_alpha:.2f}, β: {prev_beta:.2f}→{robot.trust_beta:.2f}) " +
                #      f"Δ=(+{total_delta_alpha:.3f}, +{total_delta_beta:.3f})")
        
        # Apply accumulated track updates - need to handle fused vs individual tracks properly
        self._apply_accumulated_track_updates()
    
    def _apply_accumulated_track_updates(self):
        """
        Apply accumulated track updates with proper fused track propagation.
        
        For fused tracks: PSM updates are applied to each constituent local track
        For individual tracks: PSM updates are applied directly to the local track
        """
        if not self.accumulated_track_updates:
            return
            
        # Get the current state to access track fusion mapping
        current_state = self._get_current_state()
        if not current_state or not hasattr(current_state, '_track_fusion_map'):
            return
        
        track_fusion_map = current_state._track_fusion_map
        
        # Apply updates for each accumulated track
        for track_id, updates in self.accumulated_track_updates.items():
            # Sum all delta updates from different robot perspectives
            total_delta_alpha = sum(delta_alpha for delta_alpha, delta_beta in updates)
            total_delta_beta = sum(delta_beta for delta_alpha, delta_beta in updates)
            
            # Determine if this is a fused track or individual track
            if track_id.startswith('fused_'):
                # This is a fused track - propagate updates to constituent local tracks
                self._propagate_fused_track_updates(track_id, total_delta_alpha, total_delta_beta, track_fusion_map)
            else:
                # This is an individual track - apply update directly
                self._apply_individual_track_update(track_id, total_delta_alpha, total_delta_beta)
    
    def _propagate_fused_track_updates(self, fused_track_id, delta_alpha, delta_beta, track_fusion_map):
        """Propagate fused track updates to all constituent local tracks"""
        # Find all original track IDs that were fused into this fused track
        constituent_track_ids = []
        for original_track_id, mapped_fused_id in track_fusion_map.items():
            if mapped_fused_id == fused_track_id:
                constituent_track_ids.append(original_track_id)
        
        # Apply the same PSM update to each constituent local track
        for original_track_id in constituent_track_ids:
            self._apply_individual_track_update(original_track_id, delta_alpha, delta_beta)
    
    def _apply_individual_track_update(self, track_id, delta_alpha, delta_beta):
        """Apply PSM update to a specific local track in robot's track list"""
        # Parse track ID format: "{robot_id}_{object_type}_obj_{object_id}"
        if "_" in track_id and ("_gt_obj_" in track_id or "_fp_obj_" in track_id):
            try:
                if "_gt_obj_" in track_id:
                    robot_id_str, remainder = track_id.split("_gt_obj_")
                    object_id = "gt_obj_" + remainder
                elif "_fp_obj_" in track_id:
                    robot_id_str, remainder = track_id.split("_fp_obj_") 
                    object_id = "fp_obj_" + remainder
                else:
                    return  # Unknown object type
                    
                robot_id = int(robot_id_str)
                
            except (ValueError, IndexError):
                return  # Failed to parse
        else:
            return  # Track ID doesn't match expected format
        
        # Find the robot and apply update to its track
        robots = self._get_robots_list()
        for robot in robots:
            if robot.id == robot_id:
                # Find the track for this object
                track = robot.get_track(object_id)
                if track is not None:
                    # Store previous trust values for debugging
                    #prev_alpha = track.trust_alpha
                    #prev_beta = track.trust_beta
                    #prev_trust = track.trust_value
                    
                    # Apply PSM update using Track's built-in method
                    track.update_trust(delta_alpha, delta_beta)
                    
                    # Debug print track trust update
                    #is_false_positive = object_id.startswith('fp_')
                    #track_marker = "🔴" if is_false_positive else "🟢"
                    #print(f"    {track_marker} R{robot_id} Track {object_id}: {prev_trust:.3f} → {track.trust_value:.3f} " +
                    #      f"(α: {prev_alpha:.2f}→{track.trust_alpha:.2f}, β: {prev_beta:.2f}→{track.trust_beta:.2f}) " +
                    #      f"Δ=(+{delta_alpha:.3f}, +{delta_beta:.3f})")
                break
    
    def _compute_final_episode_reward(self, final_state, simulation_step_data):
        """
        Compute final episode reward based on correct identification of:
        1. Adversarial robots (trust < 0.5)
        2. False positive tracks (trust < 0.5)
        """
        # Extract simulation data
        all_robots = simulation_step_data.get('all_robots', [])
        ground_truth_objects = simulation_step_data.get('ground_truth_objects', [])
        
        final_reward = 0.0
        
        # === FINAL ROBOT IDENTIFICATION REWARD ===
        robot_identification_bonus = 0.0
        total_robots = 0
        correct_robot_identifications = 0
        adversarial_robots = 0
        legitimate_robots = 0
        correctly_identified_adversarial = 0
        correctly_identified_legitimate = 0
        
        for robot in all_robots:
            total_robots += 1
            final_trust = robot.trust_value  # Use Robot.trust_value property
            is_adversarial = getattr(robot, 'is_adversarial', False)
            
            if is_adversarial:
                adversarial_robots += 1
                if final_trust < 0.3:
                    correctly_identified_adversarial += 1
                    correct_robot_identifications += 1
            else:
                legitimate_robots += 1
                if final_trust >= 0.7:
                    correctly_identified_legitimate += 1
                    correct_robot_identifications += 1
        
        if total_robots > 0:
            robot_accuracy = correct_robot_identifications / total_robots
            robot_identification_bonus = robot_accuracy * 10.0  # Up to 10 points for perfect robot identification
            
            # PENALTIES for wrong robot classifications
            wrong_robot_identifications = total_robots - correct_robot_identifications
            robot_penalty = wrong_robot_identifications * -1.0  # Penalty for each wrong classification

            # Extra penalty for trusting adversarial robots (very dangerous!)
            trusted_adversaries = adversarial_robots - correctly_identified_adversarial
            adversarial_penalty = trusted_adversaries * -5.0  # Higher penalty for security risks

            robot_identification_bonus += robot_penalty + adversarial_penalty
            
            # Additional bonus for balanced classification
            if adversarial_robots > 0 and legitimate_robots > 0:
                adv_accuracy = correctly_identified_adversarial / adversarial_robots
                leg_accuracy = correctly_identified_legitimate / legitimate_robots
                balanced_accuracy = (adv_accuracy + leg_accuracy) / 2.0
                robot_identification_bonus += balanced_accuracy * 5.0  # Up to 5 bonus points for balanced accuracy
        else:
            robot_accuracy = 0.0  # No robots to classify
        
        # === OBJECT-CENTRIC TRACK REWARD WITH PROPER FUSION ===
        track_identification_bonus = 0.0
        
        # Collect all tracks from all robots using Robot.get_all_tracks()
        all_tracks = []
        for robot in all_robots:
            robot_tracks = robot.get_all_tracks()  # Use Robot class method
            all_tracks.extend(robot_tracks)
        
        # Separate into ground truth and false positive lists
        ground_truth_tracks = []
        false_positive_tracks = []
        
        for track in all_tracks:
            if track.object_id is not None:  # Track class always has object_id, trust_alpha, trust_beta
                
                # Check if it's a legitimate object by comparing to ground truth
                # Extract numeric part from track object_id (e.g., "gt_obj_0" -> "0")
                track_id_str = str(track.object_id)
                if track_id_str.startswith('gt_obj_'):
                    numeric_id = track_id_str.replace('gt_obj_', '')
                else:
                    numeric_id = track_id_str
                
                is_false_positive = not any(str(getattr(obj, 'id', '')) == numeric_id for obj in ground_truth_objects)
                
                if is_false_positive:
                    false_positive_tracks.append(track)
                else:
                    ground_truth_tracks.append(track)
        
        # Perform object_id fusion for ground truth tracks
        object_trust_groups = {}
        for track in ground_truth_tracks:
            object_id = track.object_id
            track_trust = track.trust_value  # Use Track.trust_value property
            if object_id not in object_trust_groups:
                object_trust_groups[object_id] = []
            object_trust_groups[object_id].append(track_trust)
        
        # Perform object_id fusion for false positive tracks
        false_positive_groups = {}
        for track in false_positive_tracks:
            object_id = track.object_id
            track_trust = track.trust_value  # Use Track.trust_value property
            if object_id not in false_positive_groups:
                false_positive_groups[object_id] = []
            false_positive_groups[object_id].append(track_trust)
        
        total_tracks_processed = len(all_tracks)
        
        # Evaluate per-object trust (mean of all tracks for each object)
        total_objects = 0
        correct_object_identifications = 0
        
        # Legitimate objects (should be trusted)
        legitimate_objects = len(object_trust_groups)
        correctly_identified_legit_objects = 0
        for object_id, trust_values in object_trust_groups.items():
            total_objects += 1
            avg_object_trust = sum(trust_values) / len(trust_values)
            if avg_object_trust >= 0.7:  # Object correctly trusted
                correctly_identified_legit_objects += 1
                correct_object_identifications += 1
        
        # False positive objects (should be distrusted) 
        false_positive_objects = len(false_positive_groups)
        correctly_identified_fp_objects = 0
        for object_id, trust_values in false_positive_groups.items():
            total_objects += 1
            avg_object_trust = sum(trust_values) / len(trust_values)
            if avg_object_trust < 0.3:  # False positive correctly distrusted
                correctly_identified_fp_objects += 1
                correct_object_identifications += 1
        
        # Calculate object-centric reward
        if total_objects > 0:
            object_accuracy = correct_object_identifications / total_objects
            track_identification_bonus = correct_object_identifications * 8.0 
            
            # PENALTIES for wrong object classifications
            wrong_object_identifications = total_objects - correct_object_identifications
            object_penalty = wrong_object_identifications * -1.0  # Penalty for each wrong object classification

            # Extra penalty for trusting false positive objects (degrades sensor fusion)
            trusted_false_positive_objects = len(false_positive_groups) - correctly_identified_fp_objects
            false_positive_penalty = trusted_false_positive_objects * -1.0 # Higher penalty for trusting bad data

            track_identification_bonus += max(-10.0, object_penalty + false_positive_penalty)
            
            # Additional bonus for balanced object classification
            if len(false_positive_groups) > 0 and len(object_trust_groups) > 0:
                fp_accuracy = correctly_identified_fp_objects / len(false_positive_groups)
                legit_accuracy = correctly_identified_legit_objects / len(object_trust_groups) 
                balanced_object_accuracy = (fp_accuracy + legit_accuracy) / 2.0
                track_identification_bonus += balanced_object_accuracy * 3.0  # Up to 3 bonus points
        else:
            object_accuracy = 0.0  # No objects to classify
        
        final_reward = robot_identification_bonus + track_identification_bonus
        
        # Log detailed classification metrics including observation counts
        track_obs_counts = []
        well_observed_count = 0
        for robot in all_robots:
            for track in robot.get_all_tracks():
                track_obs_counts.append(track.observation_count)
                if track.observation_count > 10:
                    well_observed_count += 1
        
        avg_obs_count = sum(track_obs_counts) / len(track_obs_counts) if track_obs_counts else 0
        
        print(f"  🏆 [FINAL REWARD] Episode Complete: {final_reward:.2f} points (assessed {total_tracks_processed} robot tracks)")
        print(f"    📊 Track Observations: Avg={avg_obs_count:.1f}, Well-observed={well_observed_count}/{len(track_obs_counts)} tracks")
        print(f"    🤖 Robots: {correct_robot_identifications}/{total_robots} correct ({robot_accuracy:.2%} accuracy)")
        if adversarial_robots > 0:
            adv_acc = correctly_identified_adversarial / adversarial_robots
            print(f"      - Adversarial: {correctly_identified_adversarial}/{adversarial_robots} ({adv_acc:.2%})")
        if legitimate_robots > 0:
            leg_acc = correctly_identified_legitimate / legitimate_robots
            print(f"      - Legitimate: {correctly_identified_legitimate}/{legitimate_robots} ({leg_acc:.2%})")
        
        # Simple object-centric track logging
        print(f"    📦 Objects: {correct_object_identifications}/{total_objects} correct ({object_accuracy:.2%} accuracy) [from robot tracks]")
        if len(false_positive_groups) > 0:
            fp_acc = correctly_identified_fp_objects / len(false_positive_groups)
            print(f"      - False Positive Objects: {correctly_identified_fp_objects}/{len(false_positive_groups)} ({fp_acc:.2%})")
        if len(object_trust_groups) > 0:
            legit_acc = correctly_identified_legit_objects / len(object_trust_groups)
            print(f"      - Legitimate Objects: {correctly_identified_legit_objects}/{len(object_trust_groups)} ({legit_acc:.2%})")
        
        print(f"      - Total: {len(object_trust_groups)} legitimate objects, {len(false_positive_groups)} false positive objects")
        print(f"      - Processed {total_tracks_processed} robot tracks from {len(all_robots)} robots")
        
        return final_reward
    
    def _compute_sparse_reward(self, next_state, simulation_step_data, done):
        """
        Balanced reward structure with controlled scaling to prevent explosion:
        - Immediate reward for trust updates in correct direction (scale: ±2)
        - Small final episode reward when episode ends (scale: ±3)
        - Total reward clamped to reasonable range
        """
        
        # === IMMEDIATE TRUST DIRECTION REWARD ===
        immediate_reward = self._compute_trust_direction_reward(next_state, simulation_step_data)
        
        # === FINAL EPISODE REWARD ===
        final_reward = 0.0
        if done:
            # Meaningful final reward for classification accuracy
            episode_classification_score = self._compute_final_episode_reward(next_state, simulation_step_data)
            # Make final reward substantial so it's clearly visible
            final_reward = episode_classification_score * 0.1  # Full weight for final episode reward
        
        total_reward = immediate_reward + final_reward
                
        # Final safety clamp to prevent any extreme values

        return total_reward

    def _compute_trust_direction_reward(self, next_state, simulation_step_data):
        """
        Compute immediate reward for trust updates moving in correct direction
        Positive reward for trust moving toward ground truth, negative for moving away
        """
        
        direction_reward = 0.0
        
        # Extract simulation data
        all_robots = simulation_step_data.get('all_robots', [])
        
        # === ROBOT TRUST DIRECTION REWARD ===
        for robot in all_robots:
            # Current trust value after update
            current_trust = robot.trust_value  # Use Robot.trust_value property
            is_adversarial = getattr(robot, 'is_adversarial', False)
            
            # Get previous trust value from stored distributions
            robot_id = robot.id
            prev_trust = None
            if hasattr(self, '_previous_trust_distributions') and robot_id in self._previous_trust_distributions:
                prev_data = self._previous_trust_distributions[robot_id]
                prev_trust = prev_data['alpha'] / (prev_data['alpha'] + prev_data['beta'])
            
            if prev_trust is not None:
                # Determine correct trust direction
                if is_adversarial:
                    # Adversarial robots should have LOW trust (target: 0.1)
                    target_trust = 0.1
                else:
                    # Legitimate robots should have HIGH trust (target: 0.9)
                    target_trust = 0.9
                
                # Calculate trust movement direction
                prev_distance = abs(prev_trust - target_trust)
                current_distance = abs(current_trust - target_trust)
                
                # Reward for moving closer to target, penalty for moving away
                if current_distance < prev_distance:
                    # Moving in correct direction
                    improvement = prev_distance - current_distance
                    direction_reward += improvement  
                elif current_distance > prev_distance:
                    # Moving in wrong direction
                    degradation = current_distance - prev_distance
                    direction_reward -= degradation * 2
        
        # === TRACK TRUST DIRECTION REWARD ===
        # Get current tracks from all robots in the next state
        current_tracks = []
        if hasattr(next_state, '_current_robots'):
            for robot in next_state._current_robots:
                current_tracks.extend(robot.get_all_tracks())
        
        for track in current_tracks:
            current_track_trust = track.trust_value  # Use Track.trust_value property
            track_id = track.track_id
            
            # Get previous track trust
            prev_track_trust = None
            if hasattr(self, '_previous_trust_distributions') and f"track_{track_id}" in self._previous_trust_distributions:
                prev_data = self._previous_trust_distributions[f"track_{track_id}"]
                prev_track_trust = prev_data['alpha'] / (prev_data['alpha'] + prev_data['beta'])
            
            if prev_track_trust is not None:
                # For tracks, use ground truth to determine target trust
                track_id_str = str(track.object_id)
                
                if track_id_str.startswith('gt_obj_'):
                    # Ground truth object - should be trusted (high trust)
                    target_track_trust = 0.9
                elif track_id_str.startswith('shared_fp_') or track_id_str.startswith('fp_'):
                    # False positive object - should be distrusted (low trust)
                    target_track_trust = 0.1
                else:
                    # Unknown type - skip
                    continue

                # Calculate track trust movement direction
                prev_distance = abs(prev_track_trust - target_track_trust)
                current_distance = abs(current_track_trust - target_track_trust)
                
                if current_distance < prev_distance:
                    improvement = prev_distance - current_distance
                    direction_reward += improvement * 0.1 
                elif current_distance > prev_distance:
                    degradation = current_distance - prev_distance
                    direction_reward -= degradation * 0.2
        
        # Store current trust distributions for next step comparison
        if not hasattr(self, '_previous_trust_distributions'):
            self._previous_trust_distributions = {}
        
        # Store robot trust distributions
        for robot in all_robots:
            self._previous_trust_distributions[robot.id] = {
                'alpha': robot.trust_alpha,
                'beta': robot.trust_beta
            }
        
        # Store track trust distributions
        for track in current_tracks:
            self._previous_trust_distributions[f"track_{track.track_id}"] = {
                'alpha': track.trust_alpha,
                'beta': track.trust_beta
            }
        
        return direction_reward
    
    def _find_track_by_id(self, track_id, final_state):
        """Find track object by ID - simplified for new Robot/Track structure"""
        # Check all robots for tracks with matching track_id
        if hasattr(final_state, '_current_robots'):
            for robot in final_state._current_robots:
                for track in robot.get_all_tracks():
                    if track.track_id == track_id:
                        return track
        
        return None  # Track object not found
    
    def get_current_state(self):
        """
        Get the current state as a graph representation for multi-ego training
        Returns the graph with trust distributions as node features
        """
        return self._get_current_state()