#!/usr/bin/env python3
"""
Simplified Trust-Based Sensor Fusion Algorithm

Simplified version using:
- Current-time detections only (no historical tracks)
- Direct object ID matching (no Hungarian assignment)
- Agent-trust-weighted fusion

Based on: "Trust-Based Assured Sensor Fusion in Distributed Aerial Autonomy"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
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
        Simplified fusion using ONLY current-time detections:
        1) Use ego robot's CURRENT tracks only
        2) Match other robots' tracks by object ID (no Hungarian assignment needed)
        3) Weight contributions by agent trust values

        Returns:
            Dict mapping object_id to fused Track
        """
        # Get ego robot
        ego_robot = next((r for r in self.robots if r.id == ego_robot_id), None)
        if ego_robot is None:
            return {}

        # Use ONLY current-time detected tracks
        ego_current_tracks = ego_robot.get_all_tracks()

        if not ego_current_tracks:
            return {}

        fused_track_map: Dict[str, Track] = {}  # object_id -> fused track

        # For each object ego robot currently observes
        for ego_track in ego_current_tracks:
            # Skip if we already processed this object
            if ego_track.object_id in fused_track_map:
                continue

            # 1) Find tracks from other robots for the SAME object ID
            proximal_tracks_for_object: List[Track] = [ego_track]

            # Check all other robots for CURRENT tracks with matching object ID
            for other_robot in self.robots:
                if other_robot.id == ego_robot_id:
                    continue

                # Get current tracks only
                other_current_tracks = other_robot.get_all_tracks()

                # Match by object ID directly (no Hungarian needed!)
                for other_track in other_current_tracks:
                    if other_track.object_id == ego_track.object_id:
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
        """Update trust values using simplified object ID-based PSM generation"""

        # Use current-time tracks only
        self.data_aggregator = DataAggregator(robots)

        trust_updates = {}

        # Store PSMs for each robot to be applied later
        robot_psms = {robot.id: [] for robot in robots}
        all_track_psms = {}

        # OUTER LOOP: For each ego robot, create ego fused tracks from CURRENT tracks
        for ego_robot in robots:
            # Use CURRENT tracks only
            ego_robot_tracks = ego_robot.get_all_tracks()

            if not ego_robot_tracks:
                continue

            # Create ego robot's fused tracks (weighted by agent trust)
            ego_fused_track_map = self.data_aggregator.fuse_tracks_for_ego_robot(ego_robot.id)

            if not ego_fused_track_map:
                continue

            # INNER LOOP: For each proximal robot, compare its CURRENT tracks with ego fused tracks
            # Filter proximal robots by distance if environment is available
            if environment:
                proximal_robots_in_range = environment.get_proximal_robots(ego_robot)
            else:
                # Fallback: consider all other robots as proximal
                proximal_robots_in_range = [r for r in robots if r.id != ego_robot.id]

            for proximal_robot in proximal_robots_in_range:

                # Use proximal robot's CURRENT tracks only
                proximal_robot_tracks = proximal_robot.get_all_tracks()
                if not proximal_robot_tracks:
                    continue

                # Generate PSMs based on OBJECT ID matching (no Hungarian assignment!)
                self._generate_psms_by_object_id(
                    ego_fused_track_map, proximal_robot_tracks, ego_robot, proximal_robot,
                    robot_psms, all_track_psms)
        
        # Apply collected PSMs to update robot trust
        for robot in robots:
            if robot_psms[robot.id]:
                self.trust_estimator.update_agent_trust(robot, robot_psms[robot.id])
            
            # Update track trust for all CURRENT tracks that have PSMs
            track_updates_count = 0
            for track in robot.get_all_tracks():
                if track.track_id in all_track_psms:
                    self.trust_estimator.update_track_trust(track, all_track_psms[track.track_id])
                    track_updates_count += 1
            # Debug: show how many tracks were updated
            # if track_updates_count > 0:
            #     print(f"  Robot {robot.id}: updated trust for {track_updates_count} tracks")
            
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
    
    def _generate_psms_by_object_id(self, ego_fused_track_map: Dict[str, Track],
                                    proximal_tracks: List[Track],
                                    ego_robot: Robot, proximal_robot: Robot,
                                    robot_psms: Dict, all_track_psms: Dict):
        """
        Generate PSMs using direct object ID matching (simplified - no Hungarian needed!)

        Logic:
        - If proximal track's object_id exists in ego_fused_track_map: MATCH (positive PSM)
        - If proximal track's object_id NOT in ego_fused_track_map and in ego's FoV: MISMATCH (negative PSM)
        """

        if not ego_fused_track_map or not proximal_tracks:
            return

        # Process each proximal track
        for proximal_track in proximal_tracks:
            obj_id = proximal_track.object_id

            # Check if this object exists in ego's fused tracks
            if obj_id in ego_fused_track_map:
                # MATCH: Same object ID - positive evidence
                ego_fused_track = ego_fused_track_map[obj_id]
                self._generate_positive_psm(
                    ego_fused_track, proximal_track, proximal_robot, robot_psms, all_track_psms
                )
            else:
                # Object not in ego's fused tracks - check if it's in ego's FoV
                if ego_robot.is_in_fov(proximal_track.position):
                    # MISMATCH: Proximal robot sees object that ego doesn't - negative evidence (false positive)
                    self._generate_negative_psm(
                        proximal_track, proximal_robot, robot_psms, all_track_psms
                    )
    
    def _generate_positive_psm(self, ego_fused_track: Track, proximal_track: Track,
                               proximal_robot: Robot, robot_psms: Dict, all_track_psms: Dict):
        """Generate positive PSM when proximal track matches ego fused track by object ID"""
        # Calculate ego fused track trust statistics (used in agent PSM)
        ego_fused_track_expected_trust = self.trust_estimator.get_expected_trust(
            ego_fused_track.trust_alpha, ego_fused_track.trust_beta)
        ego_fused_track_trust_variance = self.trust_estimator.get_trust_variance(
            ego_fused_track.trust_alpha, ego_fused_track.trust_beta)

        # Calculate agent trust statistics (used in track PSM)
        agent_expected_trust = proximal_robot.trust_value

        # Agent PSM: value = E[ego_fused_track_trust], confidence = 1-V[ego_fused_track_trust]
        agent_value = ego_fused_track_expected_trust
        agent_confidence = 1.0 - ego_fused_track_trust_variance
        robot_psms[proximal_robot.id].append((agent_value, agent_confidence))

        # Track PSM: value = 1 (match), confidence = E[agent_trust]
        track_value = 1.0
        track_confidence = agent_expected_trust

        if proximal_track.track_id not in all_track_psms:
            all_track_psms[proximal_track.track_id] = []
        all_track_psms[proximal_track.track_id].append((track_value, track_confidence))

    def _generate_negative_psm(self, proximal_track: Track, proximal_robot: Robot,
                               robot_psms: Dict, all_track_psms: Dict):
        """Generate negative PSM when proximal track is a false positive (not in ego's fused tracks but in ego's FoV)"""
        # Calculate proximal track trust statistics (used in agent PSM)
        proximal_track_expected_trust = self.trust_estimator.get_expected_trust(
            proximal_track.trust_alpha, proximal_track.trust_beta)
        proximal_track_trust_variance = self.trust_estimator.get_trust_variance(
            proximal_track.trust_alpha, proximal_track.trust_beta)

        # Calculate agent trust statistics (used in track PSM)
        agent_expected_trust = proximal_robot.trust_value

        # Agent PSM: value = E[proximal_track_trust], confidence = 1-V[proximal_track_trust]
        agent_value = proximal_track_expected_trust
        agent_confidence = 1.0 - proximal_track_trust_variance
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