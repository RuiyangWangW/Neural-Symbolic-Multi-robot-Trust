#!/usr/bin/env python3
"""
Simplified Trust-Based Sensor Fusion Algorithm

Simplified version using:
- Current-time detections only (no historical tracks)
- Direct object ID matching against ego's own reported tracks (no Hungarian
  assignment, no position-based fusion - proximal reports are deduplicated by
  object_id so each distinct object contributes at most one PSM per ego robot)

Based on: "Trust-Based Assured Sensor Fusion in Distributed Aerial Autonomy"
"""

import numpy as np
from typing import List, Dict, Tuple
from trust_algorithm import TrustAlgorithm
from robot_track_classes import Robot, Track


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

    def update_trust(self, robots: List[Robot], environment = None) -> Dict[int, Dict]:
        """Update trust values using direct object ID matching against ego's own reports"""

        trust_updates = {}

        # Store PSMs for each robot to be applied later
        robot_psms = {robot.id: [] for robot in robots}
        all_track_psms = {}

        # OUTER LOOP: For each ego robot, compare its own reported tracks against what
        # proximal robots (fused by object ID, so each object counts once) report
        for ego_robot in robots:
            if not ego_robot.reported_tracks:
                continue

            # Filter proximal robots by distance if environment is available
            if environment:
                proximal_robots_in_range = environment.get_proximal_robots(ego_robot)
            else:
                # Fallback: consider all other robots as proximal
                proximal_robots_in_range = [r for r in robots if r.id != ego_robot.id]

            # Fuse proximal robots' reported tracks by object ID so each distinct object
            # is only counted once, regardless of how many proximal robots report it
            proximal_tracks_by_object: Dict[str, Track] = {}
            for proximal_robot in proximal_robots_in_range:
                for track in proximal_robot.get_reported_tracks_list():
                    proximal_tracks_by_object.setdefault(track.object_id, track)

            if not proximal_tracks_by_object:
                continue

            # Generate PSMs based on direct object ID matching against ego's own reports
            self._generate_psms_by_object_id(
                proximal_tracks_by_object, ego_robot, robot_psms, all_track_psms)

        # Apply collected PSMs to update robot trust
        for robot in robots:
            if robot_psms[robot.id]:
                self.trust_estimator.update_agent_trust(robot, robot_psms[robot.id])
            
            # NEW ARCHITECTURE: Update track trust and forward to all_tracks
            track_updates_count = 0
            for track in robot.get_reported_tracks_list():
                if track.track_id in all_track_psms:
                    # Get trust deltas before update
                    old_alpha = track.trust_alpha
                    old_beta = track.trust_beta

                    # Update the reported track
                    self.trust_estimator.update_track_trust(track, all_track_psms[track.track_id])

                    # Forward to all_tracks (persistent storage)
                    delta_alpha = track.trust_alpha - old_alpha
                    delta_beta = track.trust_beta - old_beta
                    robot.forward_trust_update_to_all_tracks(
                        object_id=track.object_id,
                        delta_alpha=delta_alpha,
                        delta_beta=delta_beta
                    )
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
                    for track in robot.get_reported_tracks_list()
                }
            }
        
        return trust_updates
    
    def _generate_psms_by_object_id(self, proximal_tracks_by_object: Dict[str, Track],
                                    ego_robot: Robot,
                                    robot_psms: Dict, all_track_psms: Dict):
        """
        Generate PSMs by matching ego's own reported tracks directly against the set of
        distinct objects proximal neighbors report (already fused/deduplicated by object_id,
        one entry per object regardless of how many proximal robots reported it).

        All evidence is applied to ego_robot's own agent trust and ego_robot's own track,
        never to a proximal robot - we're asking "does ego's own reporting agree with what
        its neighbors report?"

        Logic (one PSM per distinct proximal object, not per proximal robot):
        - If proximal object_id is also in ego_robot.reported_tracks: MATCH (positive PSM)
          -> ego correctly reported something a neighbor corroborates
        - If proximal object_id is NOT in ego_robot.reported_tracks but is in ego's FoV:
          MISMATCH (negative PSM) -> ego failed to report something a neighbor sees in ego's FoV
        """
        for obj_id, proximal_track in proximal_tracks_by_object.items():
            ego_track = ego_robot.reported_tracks.get(obj_id)

            if ego_track is not None:
                # MATCH: ego already reported this object - positive evidence for ego
                self._generate_positive_psm(
                    ego_robot, ego_track, robot_psms, all_track_psms
                )
            elif ego_robot.is_in_fov(proximal_track.position):
                # MISMATCH: Ego should have seen this (neighbor sees it, it's in ego's
                # FoV) but didn't report it - negative evidence for ego
                self._generate_negative_psm(
                    ego_robot, robot_psms
                )

    def _generate_positive_psm(self, ego_robot: Robot, ego_track: Track,
                               robot_psms: Dict, all_track_psms: Dict):
        """Generate positive PSM for ego_robot when a proximal neighbor corroborates ego's own track"""
        # Calculate ego's own track trust statistics (used in agent PSM)
        ego_track_expected_trust = self.trust_estimator.get_expected_trust(
            ego_track.trust_alpha, ego_track.trust_beta)
        ego_track_trust_variance = self.trust_estimator.get_trust_variance(
            ego_track.trust_alpha, ego_track.trust_beta)

        # Calculate ego's own agent trust statistics (used in track PSM confidence)
        agent_expected_trust = ego_robot.trust_value

        # Agent PSM (for ego_robot): value = E[ego_track_trust], confidence = 1-V[ego_track_trust]
        agent_value = ego_track_expected_trust
        agent_confidence = 1.0 - ego_track_trust_variance
        robot_psms[ego_robot.id].append((agent_value, agent_confidence))

        # Track PSM (for ego's own track): value = 1 (match), confidence = E[ego agent_trust]
        track_value = 1.0
        track_confidence = agent_expected_trust

        if ego_track.track_id not in all_track_psms:
            all_track_psms[ego_track.track_id] = []
        all_track_psms[ego_track.track_id].append((track_value, track_confidence))

    def _generate_negative_psm(self, ego_robot: Robot, robot_psms: Dict):
        """
        Generate negative PSM for ego_robot when it fails to report an object a proximal
        neighbor sees in ego's FoV.

        Ego never created a track for this object (that's the whole point of the mismatch),
        so there is no ego track to apply a track-level PSM to - only ego's agent trust is
        penalized here.
        """
        # Agent PSM (for ego_robot): value = 0 (ego missed a real detection), full confidence
        agent_value = 0.0
        agent_confidence = 1.0
        robot_psms[ego_robot.id].append((agent_value, agent_confidence))
    
    def get_expected_trust(self, alpha: float, beta: float) -> float:
        """Calculate expected value E[trust] = alpha / (alpha + beta)"""
        return self.trust_estimator.get_expected_trust(alpha, beta)
    
    def get_trust_variance(self, alpha: float, beta: float) -> float:
        """Calculate variance of trust distribution"""
        return self.trust_estimator.get_trust_variance(alpha, beta)