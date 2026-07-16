#!/usr/bin/env python3
"""
Webots Trust-Based Sensor Fusion Environment

Extends WebotsSimulationEnvironment to support adversarial robots, synchronized with
the main simulation pipeline's current architecture (robot_types.py's LegitimateRobot/
AdversarialRobot, robot_track_classes.py's reported_tracks/all_tracks track model).

Legitimate and adversarial robots are real LegitimateRobot/AdversarialRobot instances.
The Webots replay data supplies each timestep's TRUE (optimal) object positions - the
cross-robot-averaged union of what every robot's real camera actually recorded
(get_ground_truth_object_positions) - exactly like the live ground_truth_objects list
simulation_environment.py's realistic mode consumes. No real robot's sensor is perfect, so
every robot (legitimate and adversarial alike) runs its own DetectorSensor over that true
position set each timestep (_feed_real_detections): is_in_fov gating (SPOT dual-camera +
occupancy-grid line-of-sight), sensor_fn_rate missed-detection sampling, Gaussian
position/velocity noise (noise_std), and transient sensor_fp_* clutter - the same
realistic-sensor layer applied on top of ground truth in the synthetic pipeline.

Adversarial robots run the actual objective-driven MILP policy
(AdversarialRobot._run_optimized_policy_on_current_tracks, mode='optimized', "aggressive"
delta_plus=delta_minus=3.0 by default - see optimized_policy_benchmark.py) against their
real detections plus assigned persistent FP objects, exactly like the synthetic
simulation_environment.py pipeline - just fed from replay data instead of a live sensor
model.

Configures robots with SPOT dual-camera FoV (use_spot_fov=True) and the loaded occupancy
grid for line-of-sight checking, to match the actual Webots simulation camera geometry.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Set, Optional

from webots_simulation_environment import WebotsSimulationEnvironment
from robot_track_classes import Robot, Track
from robot_types import LegitimateRobot, AdversarialRobot
from simulation_environment import GroundTruthObject


class WebotsTrustEnvironment(WebotsSimulationEnvironment):
    """
    Webots simulation environment with adversarial robot support.

    Uses filtered Webots data as ground truth and adds adversarial behavior via the real
    AdversarialRobot MILP policy (mode='optimized'):
    - Persistent False Positive (FP) objects: fabricated objects adversarial robots may
      choose to report, following their assigned robot's heading like
      simulation_environment.py's _update_fp_objects
    - Ground Truth (GT) suppression: adversarial robots may choose not to report a real
      detection they actually saw, driven by the same cost-benefit MILP objective
    """

    def __init__(self,
                 webots_data_path: str = "webots_sim_filtered_corrected",
                 adversarial_robot_ids: Optional[List[str]] = None,
                 adversarial_fp_injection_rate: float = 0.2,
                 allow_fp_codetection: bool = True,
                 delta_plus: float = 3.0,
                 delta_minus: float = 3.0,
                 fov_range: float = 20.0,
                 fov_angle: float = np.pi / 4,
                 proximal_range: float = 100.0,
                 noise_std: float = 1.0,
                 sensor_fp_rate: float = 0.05,
                 sensor_fn_rate: float = 0.05,
                 random_seed: Optional[int] = None):
        """
        Initialize Webots trust environment.

        Args:
            webots_data_path: Path to filtered Webots simulation data
            adversarial_robot_ids: List of robot names to mark as adversarial (e.g., ['SPOT_0', 'SPOT_2'])
                                  If None, no robots are adversarial
            adversarial_fp_injection_rate: Rate determining number of persistent adversarial
                FP objects (num_fp_objects = rate * num_gt_objects * 2, at least one per
                adversarial robot) - matches simulation_environment.py's convention
            allow_fp_codetection: If True, any adversarial robot may report FP objects
                assigned to OTHER adversarial robots (collusion), not just its own
            delta_plus: Corroboration factor (FP-gain coefficient) in the MILP objective.
                Default 3.0 ("aggressive") matches the training distribution used by
                generate_supervised_data.py - see AdversarialRobot._estimate_objective_change
                in robot_types.py for the full tradeoff.
            delta_minus: Dilution factor (GT-suppression coefficient) in the same objective.
                Default 3.0 ("aggressive"), matching delta_plus.
            fov_range: Field of view range in meters (from SPOT camera specs)
            fov_angle: Field of view angle in radians (per SPOT camera, before dual-camera
                combination - see Robot.is_in_spot_dual_camera_fov)
            proximal_range: Communication/proximal range for ego-graph construction and
                neighbor information sharing
            noise_std: Standard deviation (meters) of Gaussian position noise DetectorSensor
                adds on top of each timestep's true (replay) object position - matches
                detector_sensor.py's convention (velocity noise is noise_std * 0.1), applied
                identically to legitimate and adversarial robots' own sensors.
            sensor_fp_rate: Transient sensor false positive rate (DetectorSensor clutter
                objects, distinct from persistent adversarial FP objects) - same default as
                LegitimateRobot/AdversarialRobot elsewhere in the codebase.
            sensor_fn_rate: Sensor false negative rate - probability a real object within
                is_in_fov is nonetheless missed by the sensor, same default as
                LegitimateRobot/AdversarialRobot elsewhere in the codebase.
            random_seed: Random seed for reproducibility
        """
        # Initialize base environment
        super().__init__(webots_data_path)

        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Adversarial configuration
        self.adversarial_robot_ids: Set[str] = set(adversarial_robot_ids) if adversarial_robot_ids else set()
        self.adversarial_fp_injection_rate = adversarial_fp_injection_rate
        self.allow_fp_codetection = allow_fp_codetection
        self.delta_plus = delta_plus
        self.delta_minus = delta_minus
        self.fov_range = fov_range
        self.fov_angle = fov_angle
        self.proximal_range = proximal_range
        self.noise_std = noise_std
        self.sensor_fp_rate = sensor_fp_rate
        self.sensor_fn_rate = sensor_fn_rate

        # Robot trust management (real LegitimateRobot/AdversarialRobot instances)
        self.robots: Dict[str, Robot] = {}
        self._initialize_robots()

        # Persistent adversarial FP objects (GroundTruthObject instances, reused from
        # simulation_environment.py so AdversarialRobot's MILP policy needs no changes)
        self.shared_fp_objects: List[GroundTruthObject] = []
        self.fp_object_assignments: Dict[int, str] = {}  # fp id -> assigned robot name
        self.fp_object_angles: Dict[int, float] = {}  # fp id -> persistent angle offset
        self.next_fp_id = 0

        self._initialize_fp_objects()

        # Current timestep
        self.current_timestep = 0

        print(f"\nWebots Trust Environment Initialized:")
        print(f"  Total robots: {len(self.robot_data)}")
        print(f"  Adversarial robots: {len(self.adversarial_robot_ids)}")
        print(f"  Legitimate robots: {len(self.robot_data) - len(self.adversarial_robot_ids)}")
        print(f"  Ground truth objects: {len(self.ground_truth_objects)}")
        print(f"  Adversarial mode: optimized (aggressive), delta_plus={self.delta_plus}, delta_minus={self.delta_minus}")
        print(f"  FP co-detection: {self.allow_fp_codetection}")
        print(f"  Total FP objects: {len(self.shared_fp_objects)}")

    def _initialize_robots(self):
        """Initialize LegitimateRobot/AdversarialRobot instances for trust tracking."""
        for robot_name in self.robot_data.keys():
            is_adversarial = robot_name in self.adversarial_robot_ids

            if is_adversarial:
                robot = AdversarialRobot(
                    robot_id=robot_name,
                    position=np.array([0.0, 0.0, 0.0]),  # Updated every timestep
                    velocity=np.array([0.0, 0.0, 0.0]),
                    fov_range=self.fov_range,
                    fov_angle=self.fov_angle,
                    mode='optimized',
                    sensor_fp_rate=self.sensor_fp_rate,
                    sensor_fn_rate=self.sensor_fn_rate,
                    delta_plus=self.delta_plus,
                    delta_minus=self.delta_minus,
                )
            else:
                robot = LegitimateRobot(
                    robot_id=robot_name,
                    position=np.array([0.0, 0.0, 0.0]),
                    velocity=np.array([0.0, 0.0, 0.0]),
                    fov_range=self.fov_range,
                    fov_angle=self.fov_angle,
                    mode='realistic',  # Webots replay supplies the "optimal" true detections
                                        # (see _feed_real_detections) - mode='realistic' gives
                                        # this robot its own DetectorSensor to layer FN/noise/
                                        # transient-FP sensor imperfection on top, same as
                                        # AdversarialRobot always has one.
                    sensor_fp_rate=self.sensor_fp_rate,
                    sensor_fn_rate=self.sensor_fn_rate,
                )

            # SPOT dual-camera FoV + occupancy grid for line-of-sight, not exposed as
            # LegitimateRobot/AdversarialRobot constructor params - set post-construction
            robot.use_spot_fov = True
            robot.occ_grid = self.occ_grid
            robot.grid_resolution = self.resolution
            robot.grid_xmin = self.grid_xmin
            robot.grid_ymin = self.grid_ymin
            robot.grid_height, robot.grid_width = self.occ_grid.shape

            robot.proximal_range = self.proximal_range

            self.robots[robot_name] = robot

    def _initialize_fp_objects(self):
        """
        Initialize persistent false positive objects for adversarial robots.

        Mirrors simulation_environment.py's FP object generation: each FP object is
        assigned to one adversarial robot and follows it at a fixed angle offset within
        that robot's FoV (see _update_fp_object_positions).
        """
        adversarial_robots = list(self.adversarial_robot_ids)
        if not adversarial_robots:
            return

        num_ground_truth = len(self.ground_truth_objects)
        total_fp_objects = int(self.adversarial_fp_injection_rate * num_ground_truth * 2)
        total_fp_objects = max(len(adversarial_robots), total_fp_objects)

        for i in range(total_fp_objects):
            assigned_robot = adversarial_robots[i % len(adversarial_robots)]
            angle_offset = random.uniform(-self.fov_angle / 2, self.fov_angle / 2)

            fp_obj = GroundTruthObject(
                id=self.next_fp_id,
                position=np.array([0.0, 0.0, 0.3]),  # Set by _update_fp_object_positions
                velocity=np.array([0.0, 0.0, 0.0]),
                object_type='false_positive',
                movement_pattern='stationary',
                spawn_time=0.0,
            )

            self.shared_fp_objects.append(fp_obj)
            self.fp_object_assignments[self.next_fp_id] = assigned_robot
            self.fp_object_angles[self.next_fp_id] = angle_offset
            self.next_fp_id += 1

        self._update_fp_object_positions()

    def _update_fp_object_positions(self):
        """
        Update FP object positions to follow their assigned robots.

        Each FP object maintains a fixed angle offset relative to its assigned robot's
        current heading. Distance starts at 50% of FoV range and is reduced until a
        position with clear line-of-sight (via the occupancy grid) is found, matching the
        original webots_trust_environment.py behavior.
        """
        for fp_obj in self.shared_fp_objects:
            assigned_robot_name = self.fp_object_assignments[fp_obj.id]
            if assigned_robot_name not in self.robots:
                continue

            robot = self.robots[assigned_robot_name]
            angle_offset = self.fp_object_angles[fp_obj.id]
            target_angle = robot.orientation + angle_offset

            initial_distance = self.fov_range * 0.5
            max_attempts = 20
            distance_step = initial_distance / max_attempts

            target_position = None
            for attempt in range(max_attempts):
                current_distance = max(initial_distance - attempt * distance_step, 0.5)
                test_position = robot.position + np.array([
                    current_distance * np.cos(target_angle),
                    current_distance * np.sin(target_angle),
                    0.3
                ])
                if robot.is_in_fov(test_position):
                    target_position = test_position
                    break

            if target_position is None:
                target_position = robot.position + np.array([
                    0.5 * np.cos(target_angle),
                    0.5 * np.sin(target_angle),
                    0.3
                ])

            fp_obj.position = target_position

    def _feed_real_detections(self, robot_name: str, timestep: int):
        """
        Populate a robot's current_timestep_tracks by running its own realistic
        DetectorSensor over this timestep's TRUE object positions - the Webots replay
        supplies the "optimal" (perfect) detections (the union of what every robot's real
        camera recorded, cross-robot-averaged per object via
        get_ground_truth_object_positions), exactly like the live ground_truth_objects
        list simulation_environment.py's _generate_realistic_detections consumes. No real
        robot's sensor is actually optimal, so DetectorSensor.generate_detections is
        layered on top for every robot (legitimate and adversarial alike, matching how
        AdversarialRobot always owns a DetectorSensor and LegitimateRobot gets one too here
        via mode='realistic') to apply:
        - is_in_fov gating (SPOT dual-camera + occupancy-grid line-of-sight)
        - sensor_fn_rate: probabilistic missed detections of real objects
        - Gaussian position/velocity noise (noise_std)
        - transient sensor_fp_* clutter objects near the robot

        Ground-truth object_ids are recovered from each detection's 'gt_object' field
        (the real gid, e.g. 'DEF:Woodbox_0') rather than DetectorSensor's default
        f"gt_obj_{id}" wrapping, so downstream DEF:-prefix classification still works.

        Args:
            robot_name: Name of the robot
            timestep: Current timestep index
        """
        robot = self.robots[robot_name]
        robot.clear_timestep_specific_tracks()

        # True (optimal) object positions for this timestep, cross-robot-averaged from the
        # real recording - the "live ground_truth_objects list" DetectorSensor expects.
        gt_positions = self.get_ground_truth_object_positions(timestep)
        ground_truth_objects = [
            GroundTruthObject(
                id=gid,
                position=np.asarray(pos, dtype=float),
                velocity=np.zeros(3),  # Webots replay has no recorded velocity
                object_type='unknown',
                movement_pattern='stationary',
                spawn_time=0.0,
            )
            for gid, pos in gt_positions.items()
        ]

        detected_gt_objects, detected_sensor_fp_objects = robot.detector_sensor.generate_detections(
            robot_position=robot.position,
            robot_orientation=robot.orientation,
            robot_fov_range=robot.fov_range,
            robot_fov_angle=robot.fov_angle,
            is_in_fov_func=robot.is_in_fov,
            ground_truth_objects=ground_truth_objects,
            time=float(timestep),
            noise_std=self.noise_std,
            world_size=(self.grid_xmax - self.grid_xmin, self.grid_ymax - self.grid_ymin),
        )

        for detection in detected_gt_objects:
            robot.add_sensor_detection(
                object_id=detection['gt_object'].id,  # real gid, e.g. 'DEF:Woodbox_0'
                position=detection['position'],
                velocity=detection['velocity'],
                timestamp=float(timestep)
            )

        for detection in detected_sensor_fp_objects:
            robot.add_sensor_detection(
                object_id=detection['object_id'],  # 'sensor_fp_{robot_id}_{fp_id}'
                position=detection['position'],
                velocity=detection['velocity'],
                timestamp=float(timestep)
            )

    def step(self, timestep: int):
        """
        Advance simulation to a specific timestep: update robot poses, FP object
        positions, feed real detections, run each robot's reporting policy (legitimate:
        pass-through; adversarial: real MILP policy), then exchange neighbor information -
        matching simulation_environment.py's step() ordering exactly (detection generation
        BEFORE neighbor communication, so adversarial robots decide what to report using
        T-1 neighbor info, then share T's reports for use in T+1).

        Args:
            timestep: Target timestep
        """
        if timestep >= self.num_timesteps:
            raise ValueError(f"Timestep {timestep} exceeds maximum {self.num_timesteps}")

        self.current_timestep = timestep

        # Start new timestep for all robots (clears current_timestep_tracks)
        for robot in self.robots.values():
            robot.start_new_timestep(float(timestep))

        # Update robot positions/orientations from Webots data FIRST
        robot_positions = self.get_robot_positions(timestep)
        for robot_name, robot in self.robots.items():
            if robot_name not in robot_positions:
                continue
            robot_x, robot_y, robot_yaw = robot_positions[robot_name]
            robot.position = np.array([robot_x, robot_y, 0.0])
            robot.orientation = robot_yaw

        # THEN update FP object positions to follow their assigned robots
        self._update_fp_object_positions()

        # DETECTION GENERATION PHASE: feed real detections, then let each robot decide
        # what to report (adversarial robots use T-1 neighbor_information to run the MILP
        # policy - see receive_neighbor_information below)
        for robot_name, robot in self.robots.items():
            if robot_name not in robot_positions:
                continue

            self._feed_real_detections(robot_name, timestep)

            if robot.is_adversarial:
                assigned_fps = (
                    self.shared_fp_objects if self.allow_fp_codetection
                    else [fp for fp in self.shared_fp_objects
                          if self.fp_object_assignments.get(fp.id) == robot_name]
                )
                robot._assigned_fp_objects_cache = assigned_fps
                robot._run_optimized_policy_on_current_tracks(float(timestep))
            else:
                robot.set_reported_tracks_from_current()

            # Ensure every reported object_id has an all_tracks entry, independent of
            # whether any trust algorithm runs an update on it this step - see
            # simulation_environment.py's generate_detections, which this pipeline
            # otherwise bypasses (Webots feeds detections directly, not through it).
            robot.register_reported_tracks_in_all_tracks()

        # NEIGHBOR COMMUNICATION PHASE: share this timestep's reports with proximal
        # neighbors, for use in NEXT timestep's adversarial decisions
        for robot in self.robots.values():
            robot.clear_neighbor_information()

        for ego_robot in self.robots.values():
            proximal_robots = self.get_proximal_robots(ego_robot)
            for neighbor_robot in proximal_robots:
                ego_robot.receive_neighbor_information(neighbor_robot)

    def get_proximal_robots(self, ego_robot: Robot) -> List[Robot]:
        """
        Get robots within proximal range of the ego robot.

        Args:
            ego_robot: The ego robot

        Returns:
            List of robots within proximal_range of ego_robot
        """
        proximal_robots = []
        proximal_range = getattr(ego_robot, 'proximal_range', self.proximal_range)

        for robot in self.robots.values():
            if robot.id != ego_robot.id:
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
                        'trust_alpha': track.trust_alpha,
                        'trust_beta': track.trust_beta
                    }
                    for track in robot.get_reported_tracks_list()
                ]
                for name, robot in self.robots.items()
            },
            'adversarial_robots': list(self.adversarial_robot_ids),
            'ground_truth_object_count': len(self.ground_truth_objects),
            'fp_object_count': len(self.shared_fp_objects)
        }

    def get_robot_trust_scores(self) -> Dict[str, float]:
        """Get current trust scores for all robots"""
        return {name: robot.trust_value for name, robot in self.robots.items()}

    def print_status(self, timestep: Optional[int] = None):
        """Print current status of the environment"""
        if timestep is None:
            timestep = self.current_timestep

        print(f"\n=== Timestep {timestep} ===")

        leg_robots = [name for name, r in self.robots.items() if not r.is_adversarial]
        adv_robots = [name for name, r in self.robots.items() if r.is_adversarial]

        print(f"Legitimate robots: {len(leg_robots)}")
        for robot_name in leg_robots[:3]:
            robot = self.robots[robot_name]
            print(f"  {robot_name}: Trust={robot.trust_value:.3f}, Reported={len(robot.reported_tracks)}")

        if len(leg_robots) > 3:
            print(f"  ... and {len(leg_robots) - 3} more")

        print(f"\nAdversarial robots: {len(adv_robots)}")
        for robot_name in adv_robots:
            robot = self.robots[robot_name]
            print(f"  {robot_name}: Trust={robot.trust_value:.3f}, Reported={len(robot.reported_tracks)}")


if __name__ == "__main__":
    # Test the environment
    print("Testing WebotsTrustEnvironment...")

    # Create environment with 2 adversarial robots running the optimized aggressive policy
    env = WebotsTrustEnvironment(
        webots_data_path="webots_sim_filtered_corrected",
        adversarial_robot_ids=['SPOT_1', 'SPOT_3'],
        adversarial_fp_injection_rate=0.2,
        allow_fp_codetection=True,
        delta_plus=3.0,
        delta_minus=3.0,
        random_seed=42
    )

    # Test a few timesteps
    for t in [0, 10, 20]:
        env.step(t)
        env.print_status(t)

    print("\n✓ Test complete!")
