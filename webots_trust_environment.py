#!/usr/bin/env python3
"""
Webots Trust-Based Sensor Fusion Environment

Extends WebotsSimulationEnvironment to support adversarial robots with
false positive and false negative detections for trust algorithm testing.

Configures robots with SPOT dual-camera FoV (use_spot_fov=True) to match
the actual Webots simulation camera geometry.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Set, Optional
from pathlib import Path
from collections import defaultdict

from webots_simulation_environment import WebotsSimulationEnvironment
from robot_track_classes import Robot, Track


class WebotsTrustEnvironment(WebotsSimulationEnvironment):
    """
    Webots simulation environment with adversarial robot support.

    Uses filtered Webots data as ground truth and adds adversarial behavior:
    - False Positive (FP): Adversarial robots detect non-existent objects
    - False Negative (FN): Adversarial robots miss real objects
    """

    def __init__(self,
                 webots_data_path: str = "webots_sim_filtered_corrected",
                 adversarial_robot_ids: Optional[List[str]] = None,
                 false_positive_rate: float = 0.2,
                 false_negative_rate: float = 0.1,
                 allow_fp_codetection: bool = False,
                 random_seed: Optional[int] = None):
        """
        Initialize Webots trust environment.

        Args:
            webots_data_path: Path to filtered Webots simulation data
            adversarial_robot_ids: List of robot names to mark as adversarial (e.g., ['SPOT_0', 'SPOT_2'])
                                  If None, no robots are adversarial
            false_positive_rate: Rate of FP detections relative to ground truth objects
            false_negative_rate: Probability of missing a real detection (0.0 to 1.0)
            allow_fp_codetection: If True, multiple adversarial robots can codetect same FP
            random_seed: Random seed for reproducibility
        """
        # Initialize base environment
        super().__init__(webots_data_path)

        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Adversarial configuration
        self.adversarial_robot_ids = set(adversarial_robot_ids) if adversarial_robot_ids else set()
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.allow_fp_codetection = allow_fp_codetection

        # Robot trust management
        self.robots: Dict[str, Robot] = {}
        self._initialize_robots()

        # False positive objects management
        # FP objects are stored with their angle offset and updated each timestep to follow assigned robot
        self.fp_objects: List[Dict] = []  # List of FP object definitions with world positions
        self.fp_object_assignments: Dict[int, str] = {}  # Maps FP object ID to assigned robot name
        self.fp_object_angles: Dict[int, float] = {}  # Maps FP object ID to angle offset (like simulation_environment)
        self.next_fp_id = 0

        self._initialize_fp_objects()

        # Current timestep
        self.current_timestep = 0

        print(f"\nWebots Trust Environment Initialized:")
        print(f"  Total robots: {len(self.robot_data)}")
        print(f"  Adversarial robots: {len(self.adversarial_robot_ids)}")
        print(f"  Legitimate robots: {len(self.robot_data) - len(self.adversarial_robot_ids)}")
        print(f"  Ground truth objects: {len(self.ground_truth_objects)}")
        print(f"  False positive rate: {self.false_positive_rate}")
        print(f"  False negative rate: {self.false_negative_rate}")
        print(f"  FP co-detection: {self.allow_fp_codetection}")
        print(f"  Total FP objects: {len(self.fp_objects)}")

    def _initialize_robots(self):
        """Initialize Robot objects for trust tracking"""
        for robot_name in self.robot_data.keys():
            # Create Robot object with SPOT dual-camera FoV and occupancy grid for line-of-sight
            robot = Robot(
                robot_id=robot_name,
                position=np.array([0.0, 0.0, 0.0]),  # Will be updated each timestep
                velocity=np.array([0.0, 0.0, 0.0]),
                trust_alpha=1.0,
                trust_beta=1.0,
                fov_range=20.0,  # From camera specs
                fov_angle=np.pi/4,  # 45 degrees
                use_spot_fov=True,  # Enable SPOT dual-camera FoV for Webots
                occ_grid=self.occ_grid,  # Pass occupancy grid for line-of-sight checking
                grid_resolution=self.resolution,
                grid_xmin=self.grid_xmin,
                grid_ymin=self.grid_ymin
            )

            # Add additional attributes
            robot.proximal_range = 100.0  # Large range to include all robots

            # Mark as adversarial if in adversarial list
            robot.is_adversarial = robot_name in self.adversarial_robot_ids

            self.robots[robot_name] = robot

    def _initialize_fp_objects(self):
        """
        Initialize false positive objects for adversarial robots.

        Each adversarial robot gets assigned FP objects based on false_positive_rate.
        FP objects store a persistent angle offset and are updated each timestep like in simulation_environment.py.
        """
        adversarial_robots = [name for name in self.adversarial_robot_ids]

        if not adversarial_robots:
            return

        # Calculate number of FP objects per adversarial robot
        num_ground_truth = len(self.ground_truth_objects)
        total_fp_objects = int(self.false_positive_rate * num_ground_truth * 2)

        # Ensure at least one FP object per adversarial robot
        total_fp_objects = max(len(adversarial_robots), total_fp_objects)

        # Camera/FoV parameters
        fov_angle = np.pi / 4  # 45 degrees
        fov_range = 20.0       # 20 meters

        # Create FP objects distributed across adversarial robots
        for i in range(total_fp_objects):
            # Assign to adversarial robot (round-robin)
            assigned_robot = adversarial_robots[i % len(adversarial_robots)]

            # Generate a persistent random angle offset within FoV (like simulation_environment.py)
            angle_offset = random.uniform(-fov_angle/2, fov_angle/2)

            fp_obj = {
                'id': self.next_fp_id,
                'gid': f'FP:{self.next_fp_id}',
                'assigned_robot': assigned_robot,
                'angle_offset': angle_offset,
                'object_type': 'false_positive',
                'position': np.array([0.0, 0.0, 0.3])  # Will be updated in _update_fp_positions()
            }

            self.fp_objects.append(fp_obj)
            self.fp_object_assignments[self.next_fp_id] = assigned_robot
            self.fp_object_angles[self.next_fp_id] = angle_offset
            self.next_fp_id += 1

        # Initialize FP positions at timestep 0
        self._update_fp_positions(0)

    def _update_fp_positions(self, timestep: int):
        """
        Update FP object positions to follow their assigned robots.

        FP objects maintain a fixed angle offset relative to their assigned robot's heading.
        The position is adjusted to ensure the object stays within both geometric FoV AND
        line-of-sight (not blocked by walls/obstacles). If blocked, the position is moved
        closer to the robot recursively until a clear line-of-sight is found.

        Args:
            timestep: Current timestep
        """
        fov_range = 20.0  # FoV range from camera specs

        for fp_obj in self.fp_objects:
            assigned_robot_name = fp_obj['assigned_robot']

            if assigned_robot_name not in self.robots:
                continue

            robot = self.robots[assigned_robot_name]
            angle_offset = fp_obj['angle_offset']

            # Calculate robot heading from orientation
            # In Webots, robot.orientation is already the heading (yaw angle)
            robot_heading = robot.orientation

            # Calculate initial target position: 50% of FoV range, at designated angle
            initial_distance = fov_range * 0.5
            target_angle = robot_heading + angle_offset

            # Try to find a position that's both in FoV and has line-of-sight
            # Start at initial distance and move closer if blocked
            max_attempts = 20
            distance_step = initial_distance / max_attempts

            target_position = None
            for attempt in range(max_attempts):
                # Try progressively closer distances
                current_distance = initial_distance - (attempt * distance_step)

                if current_distance <= 0.5:  # Don't get too close to robot
                    current_distance = 0.5

                test_position = robot.position + np.array([
                    current_distance * np.cos(target_angle),
                    current_distance * np.sin(target_angle),
                    0.3  # Height (0.3m above ground)
                ])

                # Check if this position is visible (both FoV and line-of-sight)
                if robot.is_in_fov(test_position):
                    target_position = test_position
                    break

            # If no valid position found (shouldn't happen), use closest position
            if target_position is None:
                target_position = robot.position + np.array([
                    0.5 * np.cos(target_angle),
                    0.5 * np.sin(target_angle),
                    0.3
                ])

            # Store in global position (this allows other robots to check is_in_fov)
            fp_obj['position'] = target_position

    def get_robot_detections(self, robot_name: str, timestep: int) -> List[Track]:
        """
        Get detections for a robot at a given timestep with adversarial behavior applied.

        Args:
            robot_name: Name of the robot
            timestep: Current timestep

        Returns:
            List of Track objects (including FP and with FN applied)
        """
        robot = self.robots[robot_name]

        # Get robot position for this timestep
        robot_positions = self.get_robot_positions(timestep)
        if robot_name not in robot_positions:
            return []

        robot_x, robot_y, robot_yaw = robot_positions[robot_name]
        robot.position = np.array([robot_x, robot_y, 0.0])
        robot.orientation = robot_yaw

        # Get ground truth detections from Webots data
        robot_timestep_data = self.robot_data[robot_name][timestep]
        visible = robot_timestep_data.get('visible', {})
        union_detections = visible.get('union', [])

        tracks = []

        if robot.is_adversarial:
            # Adversarial robot: apply FN to real detections and add FP
            tracks = self._generate_adversarial_detections(
                robot_name, union_detections, timestep
            )
        else:
            # Legitimate robot: return clean detections
            tracks = self._generate_legitimate_detections(
                robot_name, union_detections, timestep
            )

        return tracks

    def _generate_legitimate_detections(self, robot_name: str,
                                       union_detections: List[Dict],
                                       timestep: int) -> List[Track]:
        """Generate clean detections for legitimate robots"""
        tracks = []
        robot = self.robots[robot_name]

        for detection in union_detections:
            obj_gid = detection['gid']
            obj_pos = detection['pos']
            position = np.array([obj_pos['x'], obj_pos['y'], obj_pos['z']])
            velocity = np.array([0.0, 0.0, 0.0])  # Webots doesn't provide velocity

            # Check if track already exists in robot's local tracks
            existing_track = robot.get_track(obj_gid)
            if existing_track:
                # Update existing track with new observation
                existing_track.update_state(position, velocity, float(timestep))
                track = existing_track
            else:
                # Create new track (constant track_id without timestep)
                track = Track(
                    track_id=f"{robot_name}_{obj_gid}",
                    robot_id=robot_name,
                    object_id=obj_gid,
                    position=position,
                    velocity=velocity,
                    trust_alpha=1.0,
                    trust_beta=1.0,
                    timestamp=float(timestep)
                )
                # Add new track to robot's local tracks
                robot.add_track(track)

            tracks.append(track)

        return tracks

    def _generate_adversarial_detections(self, robot_name: str,
                                        union_detections: List[Dict],
                                        timestep: int) -> List[Track]:
        """Generate adversarial detections with FP and FN"""
        tracks = []
        robot = self.robots[robot_name]

        # 1. Apply False Negatives: randomly drop real detections
        for detection in union_detections:
            # Skip detection with probability = false_negative_rate
            if random.random() < self.false_negative_rate:
                continue  # False negative - miss this detection

            obj_gid = detection['gid']
            obj_pos = detection['pos']
            position = np.array([obj_pos['x'], obj_pos['y'], obj_pos['z']])
            velocity = np.array([0.0, 0.0, 0.0])

            # Check if track already exists in robot's local tracks
            existing_track = robot.get_track(obj_gid)
            if existing_track:
                # Update existing track with new observation
                existing_track.update_state(position, velocity, float(timestep))
                track = existing_track
            else:
                # Create new track (constant track_id without timestep)
                track = Track(
                    track_id=f"{robot_name}_{obj_gid}",
                    robot_id=robot_name,
                    object_id=obj_gid,
                    position=position,
                    velocity=velocity,
                    trust_alpha=1.0,
                    trust_beta=1.0,
                    timestamp=float(timestep)
                )
                # Add new track to robot's local tracks
                robot.add_track(track)

            tracks.append(track)

        # 2. Add False Positives: detect FP objects that are in this robot's FoV
        for fp_obj in self.fp_objects:
            assigned_robot = fp_obj['assigned_robot']

            # Determine if this robot can detect this FP object
            if assigned_robot == robot_name:
                # This robot is assigned to this FP object - can detect
                can_detect = True
            elif self.allow_fp_codetection and robot_name in self.adversarial_robot_ids:
                # FP codetection enabled and this is an adversarial robot - can detect
                can_detect = True
            else:
                # Not assigned and codetection disabled - skip
                can_detect = False

            # Check if FP object is in robot's FoV
            if can_detect and robot.is_in_fov(fp_obj['position']):
                obj_gid = fp_obj['gid']
                fp_position = fp_obj['position']
                velocity = np.array([0.0, 0.0, 0.0])

                # Check if track already exists in robot's local tracks
                existing_track = robot.get_track(obj_gid)
                if existing_track:
                    # Update existing track with new observation
                    existing_track.update_state(fp_position, velocity, float(timestep))
                    track = existing_track
                else:
                    # Create new track (constant track_id without timestep)
                    track = Track(
                        track_id=f"{robot_name}_{obj_gid}",
                        robot_id=robot_name,
                        object_id=obj_gid,
                        position=fp_position,
                        velocity=velocity,
                        trust_alpha=1.0,
                        trust_beta=1.0,
                        timestamp=float(timestep)
                    )
                    # Add new track to robot's local tracks
                    robot.add_track(track)

                tracks.append(track)

        return tracks

    def step(self, timestep: int):
        """
        Advance simulation to a specific timestep and update all robot detections.

        Args:
            timestep: Target timestep
        """
        if timestep >= self.num_timesteps:
            raise ValueError(f"Timestep {timestep} exceeds maximum {self.num_timesteps}")

        self.current_timestep = timestep

        # Update each robot's position and orientation FIRST
        robot_positions = self.get_robot_positions(timestep)

        for robot_name, robot in self.robots.items():
            if robot_name not in robot_positions:
                continue

            robot_x, robot_y, robot_yaw = robot_positions[robot_name]
            robot.position = np.array([robot_x, robot_y, 0.0])
            robot.orientation = robot_yaw
            robot.current_timestep = float(timestep)

        # THEN update FP object positions to follow their assigned robots
        # (needs updated robot positions/orientations)
        self._update_fp_positions(timestep)

        # Now generate detections for each robot
        for robot_name, robot in self.robots.items():
            # Clear current timestep tracks first
            robot.current_timestep_tracks.clear()

            # Get detections for this robot (this handles track creation/update internally)
            tracks = self.get_robot_detections(robot_name, timestep)

            # Mark all detected tracks as current for this timestep
            for track in tracks:
                robot.current_timestep_tracks[track.object_id] = track

    def get_proximal_robots(self, ego_robot: Robot) -> List[Robot]:
        """
        Get robots within proximal range of the ego robot.

        Args:
            ego_robot: The ego robot

        Returns:
            List of robots within proximal_range of ego_robot
        """
        proximal_robots = []
        proximal_range = getattr(ego_robot, 'proximal_range', 100.0)

        for robot in self.robots.values():
            if robot.id != ego_robot.id:  # Exclude the ego robot itself
                distance = np.linalg.norm(ego_robot.position - robot.position)
                if distance <= proximal_range:
                    proximal_robots.append(robot)

        return proximal_robots

    def get_state_dict(self) -> Dict:
        """Get current state for trust algorithm processing"""
        return {
            'timestep': self.current_timestep,
            'robot_positions': {
                name: robot.position.tolist()
                for name, robot in self.robots.items()
            },
            'robot_tracks': {
                name: [
                    {
                        'object_id': track.object_id,
                        'position': track.position.tolist(),
                        'confidence': track.confidence,
                        'trust_alpha': track.trust_alpha,
                        'trust_beta': track.trust_beta
                    }
                    for track in robot.tracks.values()
                ]
                for name, robot in self.robots.items()
            },
            'adversarial_robots': list(self.adversarial_robot_ids),
            'ground_truth_object_count': len(self.ground_truth_objects),
            'fp_object_count': len(self.fp_objects)
        }

    def get_robot_trust_scores(self) -> Dict[str, float]:
        """Get current trust scores for all robots"""
        trust_scores = {}
        for robot_name, robot in self.robots.items():
            trust_mean = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
            trust_scores[robot_name] = trust_mean
        return trust_scores

    def print_status(self, timestep: Optional[int] = None):
        """Print current status of the environment"""
        if timestep is None:
            timestep = self.current_timestep

        print(f"\n=== Timestep {timestep} ===")

        # Group robots by type
        leg_robots = [name for name, r in self.robots.items() if not r.is_adversarial]
        adv_robots = [name for name, r in self.robots.items() if r.is_adversarial]

        print(f"Legitimate robots: {len(leg_robots)}")
        for robot_name in leg_robots[:3]:  # Show first 3
            robot = self.robots[robot_name]
            trust = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
            print(f"  {robot_name}: Trust={trust:.3f}, Tracks={len(robot.local_tracks)}")

        if len(leg_robots) > 3:
            print(f"  ... and {len(leg_robots) - 3} more")

        print(f"\nAdversarial robots: {len(adv_robots)}")
        for robot_name in adv_robots:
            robot = self.robots[robot_name]
            trust = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
            print(f"  {robot_name}: Trust={trust:.3f}, Tracks={len(robot.local_tracks)}")


if __name__ == "__main__":
    # Test the environment
    print("Testing WebotsTrustEnvironment...")

    # Create environment with 2 adversarial robots
    env = WebotsTrustEnvironment(
        webots_data_path="webots_sim_filtered_corrected",
        adversarial_robot_ids=['SPOT_1', 'SPOT_3'],
        false_positive_rate=0.2,
        false_negative_rate=0.1,
        allow_fp_codetection=False,
        random_seed=42
    )

    # Test a few timesteps
    for t in [0, 10, 20]:
        env.step(t)
        env.print_status(t)

    print("\n✓ Test complete!")
