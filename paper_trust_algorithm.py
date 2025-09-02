#!/usr/bin/env python3
"""
Paper's Trust-Based Sensor Fusion Algorithm Implementation

Based on: "Trust-Based Assured Sensor Fusion in Distributed Aerial Autonomy"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from trust_algorithm import TrustAlgorithm
from robot_track_classes import Robot, Track

class DataAggregator:
    """Centralized data aggregator for weighted object state estimation"""
    
    def __init__(self, robots: List[Robot]):
        self.robots = robots

    def compute_weighted_object_state(self, track_group: List[Track]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute weighted position and velocity for fused track group"""
        if not track_group:
            return None, None

        weights, positions, velocities = [], [], []
        for track in track_group:
            source_robot = next((r for r in self.robots if r.id == track.robot_id), None)
            if source_robot is None:
                continue
            agent_trust = source_robot.trust_value
            track_trust = track.trust_value

            weight = agent_trust * track_trust * getattr(track, 'confidence', 1.0)
            weights.append(weight)
            positions.append(track.position)
            velocities.append(track.velocity)

        if not weights:
            return None, None

        weights = np.array(weights)
        positions = np.array(positions)
        velocities = np.array(velocities)
        weights = weights / np.sum(weights)
        
        fused_pos = np.sum(positions * weights[:, np.newaxis], axis=0)
        fused_vel = np.sum(velocities * weights[:, np.newaxis], axis=0)
        
        return fused_pos, fused_vel

    def fuse_tracks_for_ego_robot(
        self,
        ego_robot_id: int,
        fusion_threshold: float = 2.0,
        trust_threshold: float = 0.4
    ) -> Dict[str, Track]:
        """
        Paper-compliant fusion approach:
        1) Fuse ego robot's tracks with proximal tracks weighted by agent trust
        2) Create one fused track per detected object
        3) Weight contributions by agent trust values
        
        Returns:
            Dict mapping object_id to fused Track for tracking which tracks are fused
        """
        # Get ego robot and its CURRENT TIMESTEP tracks
        ego_robot = next((r for r in self.robots if r.id == ego_robot_id), None)
        if ego_robot is None:
            return {}
            
        ego_object_tracks = ego_robot.get_all_tracks()
        if not ego_object_tracks:
            return {}

        fused_track_map: Dict[str, Track] = {}  # object_id -> fused track

        # For each object detected by ego robot
        for ego_track in ego_object_tracks:
            # Trust gate for ego track
            ego_trust = ego_track.trust_value
            if ego_trust < trust_threshold:
                continue

            # Skip if we already processed this object
            if ego_track.object_id in fused_track_map:
                continue

            # 1) Find proximal tracks for same object (within distance threshold)
            proximal_tracks_for_object: List[Track] = [ego_track]  # Start with ego track
            
            # Check all other robots for tracks of the same object
            for other_robot in self.robots:
                if other_robot.id == ego_robot_id:
                    continue
                    
                agent_trust = other_robot.trust_value
                if agent_trust < trust_threshold:
                    continue
                
                # Find tracks from this robot for the same object (ALL tracks)
                for other_track in other_robot.get_all_tracks():
                    if (other_track.object_id == ego_track.object_id and 
                        np.linalg.norm(ego_track.position - other_track.position) <= fusion_threshold):
                        track_trust = other_track.trust_value
                        # Use agent trust as additional weight (as per paper)
                        if track_trust >= trust_threshold:
                            proximal_tracks_for_object.append(other_track)

            # 2) Compute agent-trust-weighted fused state
            fused_pos, fused_vel = self.compute_weighted_object_state(proximal_tracks_for_object)
            if fused_pos is None or fused_vel is None:
                # Fallback to ego track
                fused_track_map[ego_track.object_id] = ego_track
                continue

            # 3) Create fused track with agent-trust weighting
            track_id = f"{ego_robot_id}_fused_{ego_track.object_id}"
            fused_track = Track(
                track_id=track_id,
                robot_id=ego_robot_id,
                object_id=ego_track.object_id,
                position=fused_pos,
                velocity=fused_vel,
                trust_alpha=ego_track.trust_alpha,
                trust_beta=ego_track.trust_beta,
                timestamp=ego_track.timestamp
            )
            # Set additional attributes for simulation compatibility
            fused_track_map[ego_track.object_id] = fused_track

        return fused_track_map


class TrustEstimator:
    """Implements the original paper's trust estimation using assignment-based PSM generation"""
    
    def __init__(self, negative_bias: float = 10.0, negative_threshold: float = 0.3):
        # Equation 6 parameters for negatively weighted updates
        self.negative_bias = negative_bias  # B^c_n parameter
        self.negative_threshold = negative_threshold  # T^c_n parameter
    
    def update_agent_trust(self, robot: Robot, psms: List[Tuple[float, float]]) -> None:
        """Update robot/agent trust using Beta distribution updates from paper equation (5) and (6)"""
        for value, confidence in psms:
            # Paper's equation (6): Negatively weighted updates
            # ω_{j,k} = B^c_n if v_{j,k} < T^c_n, 1.0 otherwise
            if value < self.negative_threshold:
                weight = self.negative_bias
            else:
                weight = 1.0
            
            # Apply weighting to confidence for negative evidence
            weighted_confidence = weight * confidence
            
            # Paper's equation (5) with equation (6) weighting: Beta-Bernoulli conjugate update
            delta_alpha = confidence * value
            delta_beta = weighted_confidence * (1 - value)
            
            robot.update_trust(delta_alpha, delta_beta)
            
    
    def update_track_trust(self, track: Track, psms: List[Tuple[float, float]]) -> None:
        """Update track trust using Beta distribution updates from paper equation (5) and (6)"""
        for value, confidence in psms:
            # Paper's equation (6): Negatively weighted updates
            # ω_{j,k} = B^c_n if v_{j,k} < T^c_n, 1.0 otherwise
            if value < self.negative_threshold:
                weight = self.negative_bias
            else:
                weight = 1.0
            
            # Apply weighting to confidence for negative evidence
            weighted_confidence = weight * confidence
            
            # Paper's equation (5) with equation (6) weighting: Beta-Bernoulli conjugate update
            delta_alpha = confidence * value
            delta_beta = weighted_confidence * (1 - value)
            
            track.update_trust(delta_alpha, delta_beta)
        
    
    def get_expected_trust(self, alpha: float, beta: float) -> float:
        """Calculate expected value E[trust] = alpha / (alpha + beta)"""
        return alpha / (alpha + beta)
    
    def get_trust_variance(self, alpha: float, beta: float) -> float:
        """Calculate variance V[trust] = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))"""
        denominator = (alpha + beta) ** 2 * (alpha + beta + 1)
        return (alpha * beta) / denominator

class PaperTrustAlgorithm:
    """Implementation of the original paper's trust-based sensor fusion algorithm"""
    
    def __init__(self, negative_bias: float = 3.0, negative_threshold: float = 0.3):
        self.trust_estimator = TrustEstimator(negative_bias, negative_threshold)
        self.data_aggregator: Optional[DataAggregator] = None
    
    def update_trust(self, robots: List[Robot], environment = None) -> Dict[int, Dict]:
        """Update trust values using the paper's assignment-based PSM generation"""
        
        # Use all-time accumulated tracks for fusion (reverted from current timestep)
        self.data_aggregator = DataAggregator(robots)
        
        trust_updates = {}
        
        # Store PSMs for each robot to be applied later
        robot_psms = {robot.id: [] for robot in robots}
        all_track_psms = {}
        
        # OUTER LOOP: For each ego robot, create ego fused tracks from ALL tracks
        for ego_robot in robots:
            # Use all tracks for trust evaluation
            ego_robot_tracks = ego_robot.get_all_tracks()
            
            # Create ego fused tracks by fusing ego tracks with nearby tracks from other robots
            if not ego_robot_tracks:
                continue
                
            # Create ego robot's fused tracks (weighted by agent trust as per paper)
            ego_fused_track_map = self.data_aggregator.fuse_tracks_for_ego_robot(ego_robot.id)
            
            if not ego_fused_track_map:
                continue
            
            # Convert fused track map to list for PSM generation
            ego_fused_tracks = list(ego_fused_track_map.values())
            
            # INNER LOOP: For each proximal robot, compare its raw tracks with ego fused tracks
            # Filter proximal robots by distance if environment is available
            if environment:
                proximal_robots_in_range = environment.get_proximal_robots(ego_robot)
            else:
                # Fallback: consider all other robots as proximal
                proximal_robots_in_range = [r for r in robots if r.id != ego_robot.id]
            
            for proximal_robot in proximal_robots_in_range:
                    
                # Use proximal robot's ALL tracks for trust evaluation
                proximal_robot_tracks = proximal_robot.get_all_tracks()
                if not proximal_robot_tracks:
                    continue
                
                # Generate PSMs based on assignment between ego fused tracks and proximal robot tracks
                self._generate_assignment_psms_paper_style(
                    ego_fused_tracks, proximal_robot_tracks, ego_robot, proximal_robot, 
                    robot_psms, all_track_psms)
        
        # Apply collected PSMs to update robot trust
        for robot in robots:
            if robot_psms[robot.id]:
                self.trust_estimator.update_agent_trust(robot, robot_psms[robot.id])
            
            # Update track trust for all tracks that have PSMs (use robots' local tracks directly)
            for track in robot.get_all_tracks():
                if track.track_id in all_track_psms:
                    self.trust_estimator.update_track_trust(track, all_track_psms[track.track_id])
            
            # ALWAYS include ALL robots in trust_updates to maintain continuous trust evolution
            trust_updates[robot.id] = {
                'alpha': robot.trust_alpha,
                'beta': robot.trust_beta,
                'mean_trust': robot.trust_value,
                'num_agent_psms': len(robot_psms[robot.id]),
                'track_trust_info': {
                    track.track_id: {
                        'alpha': track.trust_alpha,
                        'beta': track.trust_beta,
                        'mean_trust': track.trust_value,
                        'num_psms': len(all_track_psms.get(track.track_id, [])),
                        'object_id': track.object_id
                    }
                    for track in robot.get_all_tracks()
                }
            }
        
        return trust_updates
    
    def _generate_assignment_psms_paper_style(self, ego_fused_tracks: List[Track], proximal_tracks: List[Track],
                                            ego_robot: Robot, proximal_robot: Robot,
                                            robot_psms: Dict, all_track_psms: Dict):
        """Generate PSMs using linear sum assignment as per paper"""
        
        if not ego_fused_tracks or not proximal_tracks:
            return
            
        # Create cost matrix for assignment based on spatial distance
        cost_matrix = self._compute_assignment_cost_matrix(ego_fused_tracks, proximal_tracks)
        
        # Perform linear sum assignment (Hungarian algorithm)
        ego_indices, proximal_indices = linear_sum_assignment(cost_matrix)
        
        # Track which proximal tracks were assigned
        assigned_proximal_indices = set(proximal_indices)
        
        # Process assigned track pairs
        assignment_threshold = 5.0  
        for ego_idx, prox_idx in zip(ego_indices, proximal_indices):
            cost = cost_matrix[ego_idx, prox_idx]
            
            if cost <= assignment_threshold:  # Valid assignment
                ego_track = ego_fused_tracks[ego_idx]
                proximal_track = proximal_tracks[prox_idx]
                
                # MATCH: Assigned tracks - positive evidence
                self._generate_positive_psm_paper_style(
                    ego_track, proximal_track, proximal_robot, robot_psms, all_track_psms
                )
            else:
                # Assignment cost too high - treat as unassigned
                assigned_proximal_indices.discard(prox_idx)
        
        # Process unassigned proximal tracks (false positives)
        for prox_idx, proximal_track in enumerate(proximal_tracks):
            if prox_idx not in assigned_proximal_indices:
                if ego_robot.is_in_fov(proximal_track.position):
                    # MISMATCH: Unassigned proximal track - negative evidence (false positive)
                    self._generate_negative_psm_paper_style(
                        proximal_track, proximal_robot, robot_psms, all_track_psms
                    )
    
    def _compute_assignment_cost_matrix(self, ego_tracks: List[Track], proximal_tracks: List[Track]) -> np.ndarray:
        """Compute cost matrix for linear sum assignment based on spatial distance"""
        n_ego = len(ego_tracks)
        n_prox = len(proximal_tracks)
        
        cost_matrix = np.zeros((n_ego, n_prox))
        
        for i, ego_track in enumerate(ego_tracks):
            for j, prox_track in enumerate(proximal_tracks):
                # Use Euclidean distance as the primary cost
                spatial_distance = np.linalg.norm(ego_track.position - prox_track.position)
                cost_matrix[i, j] = spatial_distance
        
        return cost_matrix
    
    def _generate_positive_psm_paper_style(self, ego_fused_track: Track, proximal_track: Track,
                                         proximal_robot: Robot, robot_psms: Dict, all_track_psms: Dict):
        """Generate positive PSM when proximal track matches ego fused track"""
        # Calculate ego fused track trust statistics (used in agent PSM)
        ego_fused_track_expected_trust = self.trust_estimator.get_expected_trust(
            ego_fused_track.trust_alpha, ego_fused_track.trust_beta)
        ego_fused_track_trust_variance = self.trust_estimator.get_trust_variance(
            ego_fused_track.trust_alpha, ego_fused_track.trust_beta)
        
        # Calculate agent trust statistics (used in track PSM)
        agent_expected_trust = proximal_robot.trust_value
        
        # Agent PSM: value = E[ego_fused_track_trust], confidence = 1-V[ego_fused_track_trust] (positive)
        agent_value = ego_fused_track_expected_trust
        agent_confidence = 1.0 - ego_fused_track_trust_variance  # Ensure positive confidence
        robot_psms[proximal_robot.id].append((agent_value, agent_confidence))
        
        # Track PSM: value = 1 (match), confidence = E[agent_trust]
        track_value = 1.0
        track_confidence = agent_expected_trust
        
        if proximal_track.track_id not in all_track_psms:
            all_track_psms[proximal_track.track_id] = []
        all_track_psms[proximal_track.track_id].append((track_value, track_confidence))
    
    def _generate_negative_psm_paper_style(self, proximal_track: Track, proximal_robot: Robot,
                                         robot_psms: Dict, all_track_psms: Dict):
        """Generate negative PSM when proximal track does not match any ego fused track but is in ego's FOV"""
        # Calculate proximal track trust statistics (used in agent PSM)
        proximal_track_expected_trust = self.trust_estimator.get_expected_trust(
            proximal_track.trust_alpha, proximal_track.trust_beta)
        proximal_track_trust_variance = self.trust_estimator.get_trust_variance(
            proximal_track.trust_alpha, proximal_track.trust_beta)
        
        # Calculate agent trust statistics (used in track PSM)
        agent_expected_trust = proximal_robot.trust_value
        
        # Agent PSM: value = E[ego_fused_track_trust], confidence = 1-V[ego_fused_track_trust] (positive)        
        agent_value = proximal_track_expected_trust
        agent_confidence = 1.0 - proximal_track_trust_variance  # Higher confidence when track trust is certain 
        robot_psms[proximal_robot.id].append((agent_value, agent_confidence))
        
        # Track PSM: value = 0 (no match/false positive), confidence = E[agent_trust]
        track_value = 0.0
        track_confidence = agent_expected_trust
        
        if proximal_track.track_id not in all_track_psms:
            all_track_psms[proximal_track.track_id] = []
        all_track_psms[proximal_track.track_id].append((track_value, track_confidence))
    
    def get_expected_trust(self, alpha: float, beta: float) -> float:
        """Calculate expected value E[trust] = alpha / (alpha + beta)"""
        return self.trust_estimator.get_expected_trust(alpha, beta)
    
    def get_trust_variance(self, alpha: float, beta: float) -> float:
        """Calculate variance of trust distribution"""
        return self.trust_estimator.get_trust_variance(alpha, beta)