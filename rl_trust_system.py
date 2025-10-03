#!/usr/bin/env python3
"""
Main RL Trust Update System

Implements the exact ego-sweep procedure from the framework:
1. Loop over all robots as ego
2. Build ego graph, run GNN → per-node scores
3. Updater outputs step scales for nodes in this ego graph
4. Accumulate deltas using confidence, cross-weights, step scales
5. Apply globally with forgetting, caps, budgets
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from robot_track_classes import Robot, Track
from rl_evidence import EvidenceExtractor, NodeScores
from rl_updater import LearnableUpdater, UpdateDecision


@dataclass
class TrustDeltas:
    """Accumulated trust deltas for global application"""
    robot_deltas: Dict[int, Tuple[float, float]]    # robot_id -> (delta_alpha, delta_beta)
    track_deltas: Dict[str, Tuple[float, float]]    # track_id -> (delta_alpha, delta_beta)


class RLTrustSystem:
    """Main RL trust update system"""

    def __init__(self,
                 evidence_model_path: str,
                 updater_model_path: str = None,
                 device: str = 'cpu',
                 rho_min: float = 0.2,
                 c_min: float = 0.2,
                 step_size: float = 0.1,
                 strength_cap: float = 50.0,
):

        # Components
        self.evidence_extractor = EvidenceExtractor(evidence_model_path, device)
        self.updater = LearnableUpdater(updater_model_path, device)

        # Hyperparameters
        self.rho_min = rho_min
        self.c_min = c_min
        self.step_size = step_size
        self.strength_cap = strength_cap

    def compute_confidence(self, score: float, is_track: bool = False, maturity: float = 1.0) -> float:
        """Compute confidence from GNN score and track maturity"""
        # Factor 1: prediction sharpness
        pred_conf = 2 * abs(score - 0.5)

        if is_track:
            # Factor 2: track maturity for tracks only
            final_conf = pred_conf * maturity
            return max(self.c_min, final_conf)  # Floor for new tracks
        else:
            return pred_conf

    def get_detections_this_step(self, all_robots: List[Robot]) -> Tuple[Dict[int, List[str]], Dict[str, List[int]]]:
        """
        Get D_i(t) and R_j(t): who detected what this step

        Returns:
            (robot_detections, track_detectors)
            robot_detections[robot_id] = list of track_ids robot detected
            track_detectors[track_id] = list of robot_ids that detected track
        """
        robot_detections = {}  # D_i(t)
        track_detectors = defaultdict(list)  # R_j(t)

        for robot in all_robots:
            current_tracks = robot.get_current_timestep_tracks()
            track_ids = [track.track_id for track in current_tracks]

            if track_ids:  # Only if robot detected something
                robot_detections[robot.id] = track_ids
                for track_id in track_ids:
                    track_detectors[track_id].append(robot.id)

        return robot_detections, dict(track_detectors)

    def compute_cross_weights(self,
                             robot_detections: Dict[int, List[str]],
                             track_detectors: Dict[str, List[int]],
                             all_robots: List[Robot]) -> Tuple[Dict[int, float], Dict[str, float]]:
        """
        Compute cross-weights ρ_i^fromTracks(t) and ρ_j^fromAgents(t)
        """
        # Build robot and track lookup
        robot_lookup = {robot.id: robot for robot in all_robots}
        track_lookup = {}
        for robot in all_robots:
            for track in robot.get_current_timestep_tracks():
                track_lookup[track.track_id] = track

        # Pre-compute FoV visibility for each robot to extend detections
        robot_visible_tracks = {}
        for robot in all_robots:
            visible = []
            for track in track_lookup.values():
                if robot.is_in_fov(track.position):
                    visible.append(track.track_id)
            robot_visible_tracks[robot.id] = visible

        rho_robot = {}  # ρ_i^fromTracks(t)
        rho_track = {}  # ρ_j^fromAgents(t)

        # Gate robot i by tracks it detected or has in FoV
        for robot in all_robots:
            detected_track_ids = set(robot_detections.get(robot.id, []))
            fov_track_ids = set(robot_visible_tracks.get(robot.id, []))
            observed_track_ids = detected_track_ids | fov_track_ids
            if observed_track_ids:
                track_means = []
                for track_id in observed_track_ids:
                    if track_id in track_lookup:
                        t_j = track_lookup[track_id].trust_value
                        track_means.append(max(self.rho_min, np.clip(2 * t_j, 0, 1)))
                if track_means:
                    rho_robot[robot.id] = np.mean(track_means)

        # Gate track j by robots that detected it or have it in FoV
        for track_id, track in track_lookup.items():
            detecting_robot_ids = set(track_detectors.get(track_id, []))
            fov_robot_ids = {robot.id for robot in all_robots if track_id in robot_visible_tracks.get(robot.id, [])}
            observer_ids = detecting_robot_ids | fov_robot_ids
            if observer_ids:
                robot_means = []
                for robot_id in observer_ids:
                    if robot_id in robot_lookup:
                        m_i = robot_lookup[robot_id].trust_value
                        robot_means.append(max(self.rho_min, np.clip(2 * m_i, 0, 1)))
                if robot_means:
                    rho_track[track_id] = np.mean(robot_means)

        return rho_robot, rho_track

    def ego_sweep_step(self, all_robots: List[Robot], precomputed_decisions: Dict = None) -> TrustDeltas:
        """
        Perform one ego-sweep step following the exact procedure

        Args:
            all_robots: List of all robots in the simulation
            precomputed_decisions: Optional dict of {ego_robot_id: UpdateDecision} to use instead of sampling
        """
        # Initialize accumulators
        robot_deltas = defaultdict(lambda: [0.0, 0.0])
        track_deltas = defaultdict(lambda: [0.0, 0.0])

        # Get global detection info
        robot_detections, track_detectors = self.get_detections_this_step(all_robots)
        rho_robot, rho_track = self.compute_cross_weights(robot_detections, track_detectors, all_robots)

        # Build track lookup for easy access
        track_lookup = {}
        for robot in all_robots:
            for track in robot.get_current_timestep_tracks():
                track_lookup[track.track_id] = track

        # Loop over all robots as ego
        for ego_robot in all_robots:
            # Get GNN scores for this ego graph
            scores = self.evidence_extractor.get_scores(ego_robot, all_robots)

            if not scores.agent_scores and not scores.track_scores:
                continue  # No scores from GNN

            # Get ego graph nodes (robots and tracks in this ego graph)
            ego_robot_ids = list(scores.agent_scores.keys())
            ego_track_ids = list(scores.track_scores.keys())

            ego_robots = [robot for robot in all_robots if robot.id in ego_robot_ids]
            ego_tracks = [track_lookup[track_id] for track_id in ego_track_ids if track_id in track_lookup]

            # Build observer lists (detections + FoV)
            track_observers = {}
            for track in ego_tracks:
                detectors = set(track_detectors.get(track.track_id, []))
                fov_watchers = {robot.id for robot in ego_robots if robot.is_in_fov(track.position)}
                observers = detectors | fov_watchers
                if observers:
                    track_observers[track.track_id] = sorted(observers)

            observer_robot_ids = set(robot_detections.keys())
            for observers in track_observers.values():
                observer_robot_ids.update(observers)

            participating_robots = [robot for robot in ego_robots if robot.id in observer_robot_ids]
            participating_tracks = [track for track in ego_tracks if track.track_id in track_observers]

            if not participating_robots and not participating_tracks:
                continue  # No participating nodes in this ego graph

            # Get step scales from updater (use precomputed if available)
            if precomputed_decisions and ego_robot.id in precomputed_decisions:
                step_decision = precomputed_decisions[ego_robot.id]
            else:
                step_decision = self.updater.get_step_scales(
                    ego_robots, ego_tracks, participating_robots, participating_tracks,
                    scores.agent_scores, scores.track_scores, track_observers
                )

            # Accumulate robot deltas
            for robot in participating_robots:
                robot_id = robot.id
                if robot_id not in step_decision.robot_steps or robot_id not in rho_robot:
                    continue  # Skip if no step scale or cross-weight

                agent_score = scores.agent_scores.get(robot_id, 0.5)
                confidence = self.compute_confidence(agent_score, is_track=False)
                step_scale = step_decision.robot_steps[robot_id]
                cross_weight = rho_robot[robot_id]

                # Scale agent_score by (step_scale × confidence × cross_weight)
                #effective_weight = step_scale * confidence * cross_weight
                effective_weight = step_scale
                # Accumulate pseudo-counts
                robot_deltas[robot_id][0] += effective_weight * agent_score
                robot_deltas[robot_id][1] += effective_weight * (1 - agent_score)

            # Accumulate track deltas
            for track in participating_tracks:
                track_id = track.track_id
                if track_id not in step_decision.track_steps or track_id not in rho_track:
                    continue  # Skip if no step scale or cross-weight

                track_score = scores.track_scores.get(track_id, 0.5)
                maturity = min(1.0, track.observation_count / 10.0)
                confidence = self.compute_confidence(track_score, is_track=True, maturity=maturity)
                step_scale = step_decision.track_steps[track_id]
                cross_weight = rho_track[track_id]

                # Scale track_score by (step_scale × confidence × cross_weight)
                #effective_weight = step_scale * confidence * cross_weight
                effective_weight = step_scale
                # Accumulate pseudo-counts
                track_deltas[track_id][0] += effective_weight * track_score
                track_deltas[track_id][1] += effective_weight * (1 - track_score)

        # Convert to final format
        final_robot_deltas = {k: tuple(v) for k, v in robot_deltas.items()}
        final_track_deltas = {k: tuple(v) for k, v in track_deltas.items()}

        return TrustDeltas(final_robot_deltas, final_track_deltas)

    def apply_deltas_globally(self, deltas: TrustDeltas, all_robots: List[Robot]):
        """Apply accumulated deltas globally with step size, forgetting, and caps"""

        # Build lookup for tracks
        track_lookup = {}
        for robot in all_robots:
            for track in robot.get_current_timestep_tracks():
                track_lookup[track.track_id] = track

        # Apply robot deltas
        for robot_id, (delta_alpha, delta_beta) in deltas.robot_deltas.items():
            robot = next((r for r in all_robots if r.id == robot_id), None)
            if robot:
                robot.trust_alpha += self.step_size * delta_alpha
                robot.trust_beta += self.step_size * delta_beta

        # Apply track deltas
        for track_id, (delta_alpha, delta_beta) in deltas.track_deltas.items():
            track = track_lookup.get(track_id)
            if track:
                track.trust_alpha += self.step_size * delta_alpha
                track.trust_beta += self.step_size * delta_beta

        # Apply strength caps
        for robot in all_robots:

            # Strength cap
            total = robot.trust_alpha + robot.trust_beta
            if total > self.strength_cap:
                scale = self.strength_cap / total
                robot.trust_alpha *= scale
                robot.trust_beta *= scale

        for robot in all_robots:
            for track in robot.get_current_timestep_tracks():

                # Strength cap
                total = track.trust_alpha + track.trust_beta
                if total > self.strength_cap:
                    scale = self.strength_cap / total
                    track.trust_alpha *= scale
                    track.trust_beta *= scale

    def update_trust(self, all_robots: List[Robot], precomputed_decisions: Dict = None):
        """
        Main entry point: perform one complete trust update step

        Args:
            all_robots: List of all robots in the simulation
            precomputed_decisions: Optional dict of {ego_robot_id: UpdateDecision} to use instead of sampling
        """
        # Ego-sweep to accumulate deltas
        deltas = self.ego_sweep_step(all_robots, precomputed_decisions)

        # Apply globally
        self.apply_deltas_globally(deltas, all_robots)

    def get_adversarial_flags(self, all_robots: List[Robot], robot_threshold: float = 0.3, track_threshold: float = 0.3) -> Tuple[List[int], List[str]]:
        """
        Flag adversarial robots and false tracks based on trust thresholds

        Returns:
            (adversarial_robot_ids, false_track_ids)
        """
        adversarial_robots = []
        false_tracks = []

        for robot in all_robots:
            if robot.trust_value < robot_threshold:
                adversarial_robots.append(robot.id)

            for track in robot.get_current_timestep_tracks():
                if track.trust_value < track_threshold:
                    false_tracks.append(track.track_id)

        return adversarial_robots, false_tracks
