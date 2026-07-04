#!/usr/bin/env python3
"""
Robot Type Definitions for Multi-Robot Trust Simulation

This module defines different robot types with explicit behavior modes:
- Legitimate Robots: Optimal vs Realistic nominal detectors
- Adversarial Robots: Normal, Optimized, and Deceptive modes
"""

import numpy as np
import random
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from robot_track_classes import Robot, Track, is_position_in_fov_2d
from detector_sensor import DetectorSensor


class LegitimateRobot(Robot):
    """
    Legitimate robot with configurable detection modes.

    Modes:
    - 'optimal': Perfect detector (no FP/FN) - used for data collection
    - 'realistic': Natural noisy detector with small FP/FN rates (uses DetectorSensor)
    """

    def __init__(self,
                 robot_id: int,
                 position: np.ndarray,
                 velocity: np.ndarray,
                 fov_range: float,
                 fov_angle: float,
                 mode: str = 'optimal',
                 sensor_fp_rate: float = 0.05,
                 sensor_fn_rate: float = 0.05):
        """
        Initialize a legitimate robot.

        Args:
            robot_id: Unique robot identifier
            position: Initial position [x, y, z]
            velocity: Initial velocity [vx, vy, vz]
            fov_range: Field of view range
            fov_angle: Field of view angle (radians)
            mode: Detection mode ('optimal' or 'realistic')
            sensor_fp_rate: Sensor false positive rate (only for 'realistic' mode, transient)
            sensor_fn_rate: Sensor false negative rate (only for 'realistic' mode, transient)
        """
        super().__init__(robot_id, position, velocity, fov_range=fov_range, fov_angle=fov_angle)

        self.is_adversarial = False
        self.mode = mode

        # For realistic mode: use DetectorSensor class
        if mode == 'realistic':
            self.detector_sensor = DetectorSensor(
                robot_id=robot_id,
                sensor_fp_rate=sensor_fp_rate,
                sensor_fn_rate=sensor_fn_rate
            )
        else:
            self.detector_sensor = None  # Perfect detector has no sensor noise

        # Local information state: I_i(t) = (Z_i^nat(t), {(x_j(t), A_j(t), Z_j(t))}_{r_j in N_i(t)})
        self.neighbor_information: Dict[int, Dict] = {}  # neighbor_id -> {position, fov_range, fov_angle, orientation, tracks}

    def receive_neighbor_information(self, neighbor_robot: Robot):
        """
        Receive and store information from a neighboring robot.

        NEW ARCHITECTURE:
        - Stores neighbor FoV/position info in neighbor_information (for geometry checks)
        - Stores what neighbor REPORTED in last_reported_tracks (for supporting/contradicting counts)

        Args:
            neighbor_robot: A neighboring robot to receive information from
        """
        # Store neighbor's position and FoV parameters (for geometric checks)
        self.neighbor_information[neighbor_robot.id] = {
            'position': neighbor_robot.position.copy(),
            'fov_range': neighbor_robot.fov_range,
            'fov_angle': neighbor_robot.fov_angle,
            'orientation': neighbor_robot.orientation,
        }

        # Store what neighbor REPORTED (object IDs only, for supporting/contradicting counts)
        self.update_neighbor_reported_tracks(
            neighbor_id=neighbor_robot.id,
            reported_object_ids=neighbor_robot.get_reported_object_ids()
        )

    def clear_neighbor_information(self):
        """Clear stored neighbor information (called at start of new timestep)."""
        self.neighbor_information.clear()

    def get_neighbor_count(self) -> int:
        """Get the number of neighbors currently stored."""
        return len(self.neighbor_information)


    def generate_detections(self,
                          ground_truth_objects: List,
                          time: float,
                          noise_std: float = 0.0,
                          world_size: Tuple[float, float] = (100.0, 100.0)) -> List[Track]:
        """
        Generate detections based on the robot's mode.

        Args:
            ground_truth_objects: List of ground truth objects in the environment
            time: Current simulation time
            noise_std: Standard deviation for position noise (only for realistic mode)
            world_size: Size of the simulation world (for FP generation)

        Returns:
            List of Track objects representing detections
        """
        if self.mode == 'optimal':
            return self._generate_optimal_detections(ground_truth_objects, time)
        elif self.mode == 'realistic':
            return self._generate_realistic_detections(ground_truth_objects, time, noise_std, world_size)
        else:
            raise ValueError(f"Unknown legitimate robot mode: {self.mode}")

    def _generate_optimal_detections(self,
                                    ground_truth_objects: List,
                                    time: float) -> List[Track]:
        """
        Optimal nominal detector: Perfect detections with no FP/FN.

        NEW ARCHITECTURE:
        - Populates current_timestep_tracks with raw sensor detections
        - Inherits trust from all_tracks if object was seen before
        - Sets reported_tracks = current_timestep_tracks (no manipulation)
        """
        # Clear timestep-specific tracks
        self.clear_timestep_specific_tracks()

        detections = []

        for gt_obj in ground_truth_objects:
            if self.is_in_fov(gt_obj.position):
                object_id = f"gt_obj_{gt_obj.id}"

                # Perfect detection - no noise
                noisy_pos = gt_obj.position.copy()
                noisy_vel = gt_obj.velocity.copy()

                # Add to current_timestep_tracks (automatically inherits trust from all_tracks)
                track = self.add_sensor_detection(
                    object_id=object_id,
                    position=noisy_pos,
                    velocity=noisy_vel,
                    timestamp=time
                )

                detections.append(track)

        # Legitimate robot: reported_tracks = current_timestep_tracks
        self.set_reported_tracks_from_current()

        return detections

    def _generate_realistic_detections(self,
                                      ground_truth_objects: List,
                                      time: float,
                                      noise_std: float,
                                      world_size: Tuple[float, float]) -> List[Track]:
        """
        Realistic nominal detector: Natural noisy detector with small FP/FN rates.

        NEW ARCHITECTURE:
        - Uses DetectorSensor to model realistic sensor behavior
        - Populates current_timestep_tracks with detected GT and sensor FP objects
        - Inherits trust from all_tracks if object was seen before
        - Sets reported_tracks = current_timestep_tracks (no manipulation)
        """
        # Clear timestep-specific tracks
        self.clear_timestep_specific_tracks()

        detections = []

        # Use DetectorSensor to generate realistic detections
        detected_gt_objects, detected_sensor_fp_objects = self.detector_sensor.generate_detections(
            robot_position=self.position,
            robot_orientation=self.orientation,
            robot_fov_range=self.fov_range,
            robot_fov_angle=self.fov_angle,
            is_in_fov_func=self.is_in_fov,
            ground_truth_objects=ground_truth_objects,
            time=time,
            noise_std=noise_std,
            world_size=world_size
        )

        # Add detected GT objects to current_timestep_tracks
        for detection in detected_gt_objects:
            object_id = detection['object_id']
            noisy_pos = detection['position']
            noisy_vel = detection['velocity']

            # Add to current_timestep_tracks (automatically inherits trust from all_tracks)
            track = self.add_sensor_detection(
                object_id=object_id,
                position=noisy_pos,
                velocity=noisy_vel,
                timestamp=time
            )

            detections.append(track)

        # Add transient sensor FP objects to current_timestep_tracks
        for detection in detected_sensor_fp_objects:
            object_id = detection['object_id']
            noisy_pos = detection['position']
            noisy_vel = detection['velocity']

            # Add to current_timestep_tracks (automatically inherits trust from all_tracks)
            track = self.add_sensor_detection(
                object_id=object_id,
                position=noisy_pos,
                velocity=noisy_vel,
                timestamp=time
            )

            detections.append(track)

        # Legitimate robot: reported_tracks = current_timestep_tracks
        self.set_reported_tracks_from_current()

        return detections


class AdversarialRobot(Robot):
    """
    Adversarial robot with multiple attack modes.

    ALL adversarial robots use realistic detector sensor (DetectorSensor) as baseline,
    then add adversarial manipulations on top.

    Modes:
    - 'normal': Random adversarial FP/FN manipulations + sensor noise
    - 'optimized': Strategic attacks (policy-based) + sensor noise
    - 'deceptive': Optimized attacks + trust manipulation + sensor noise
    """

    def __init__(self,
                 robot_id: int,
                 position: np.ndarray,
                 velocity: np.ndarray,
                 fov_range: float,
                 fov_angle: float,
                 mode: str = 'normal',
                 adversarial_fp_injection_rate: float = 0.5,
                 adversarial_fn_suppression_rate: float = 0.0,
                 sensor_fp_rate: float = 0.05,
                 sensor_fn_rate: float = 0.05,
                 delta_plus: float = 3.0,
                 delta_minus: float = 1.0):
        """
        Initialize an adversarial robot.

        Args:
            robot_id: Unique robot identifier
            position: Initial position [x, y, z]
            velocity: Initial velocity [vx, vy, vz]
            fov_range: Field of view range
            fov_angle: Field of view angle (radians)
            mode: Attack mode ('normal', 'optimized', or 'deceptive')
            adversarial_fp_injection_rate: Rate determining number of persistent FP objects injected
                                          (num_fp_objects = rate × num_gt_objects)
            adversarial_fn_suppression_rate: Rate of transient FN suppression (only for 'normal' mode)
            sensor_fp_rate: Sensor false positive rate (transient, baseline sensor noise)
            sensor_fn_rate: Sensor false negative rate (transient, baseline sensor noise)
            delta_plus: Corroboration factor in the optimized/deceptive MILP objective - scales
                how much reporting a persistent FP object is worth (FP gain) relative to its
                contradiction cost. Higher values make the policy report FPs more readily/more
                often, even with fewer supporting neighbors. Only used by 'optimized' and
                'deceptive' modes (see _estimate_objective_change). Default 3.0 keeps contested
                FP objects (ones a legitimate neighbor could contradict) from saturating to
                "always reported" - values >=5.0 make FP reporting unconditional regardless of
                contradicting neighbors, while 1.4-2.0 leaves a real mixed strategy.
            delta_minus: Dilution factor in the same objective - scales how much suppressing a
                GT object is worth (GT-suppression gain). Only used by 'optimized' and
                'deceptive' modes.
        """
        super().__init__(robot_id, position, velocity, fov_range=fov_range, fov_angle=fov_angle)

        self.is_adversarial = True
        self.mode = mode

        # Adversarial manipulation parameters (on top of sensor noise)
        self.adversarial_fp_injection_rate = adversarial_fp_injection_rate
        self.adversarial_fn_suppression_rate = adversarial_fn_suppression_rate

        # ALL adversarial robots use realistic detector sensor
        self.detector_sensor = DetectorSensor(
            robot_id=robot_id,
            sensor_fp_rate=sensor_fp_rate,
            sensor_fn_rate=sensor_fn_rate
        )

        # NEW: Objective-driven policy parameters (cost-benefit analysis, 'optimized'/'deceptive' modes)
        self.use_objective_policy = True  # Enable objective-based decisions for optimized mode
        self.delta_plus = delta_plus    # Corroboration factor for FP reports
        self.delta_minus = delta_minus  # Dilution factor for GT suppression
        self.max_adversarial_operations = 5  # Maximum adversarial operations per timestep

        # Local information state: I_i(t) = (Z_i^nat(t), {(x_j(t), A_j(t), Z_j(t))}_{r_j in N_i(t)})
        self.neighbor_information: Dict[int, Dict] = {}  # neighbor_id -> {position, fov_range, fov_angle, orientation, tracks}

    def receive_neighbor_information(self, neighbor_robot: Robot):
        """
        Receive and store information from a neighboring robot.

        NEW ARCHITECTURE:
        - Stores neighbor FoV/position info in neighbor_information (for geometry checks)
        - Stores what neighbor REPORTED in last_reported_tracks (for supporting/contradicting counts)

        Args:
            neighbor_robot: A neighboring robot to receive information from
        """
        # Store neighbor's position and FoV parameters (for geometric checks)
        self.neighbor_information[neighbor_robot.id] = {
            'position': neighbor_robot.position.copy(),
            'fov_range': neighbor_robot.fov_range,
            'fov_angle': neighbor_robot.fov_angle,
            'orientation': neighbor_robot.orientation,
        }

        # Store what neighbor REPORTED (object IDs only, for supporting/contradicting counts)
        self.update_neighbor_reported_tracks(
            neighbor_id=neighbor_robot.id,
            reported_object_ids=neighbor_robot.get_reported_object_ids()
        )

    def clear_neighbor_information(self):
        """Clear stored neighbor information (called at start of new timestep)."""
        self.neighbor_information.clear()

    def get_neighbor_count(self) -> int:
        """Get the number of neighbors currently stored."""
        return len(self.neighbor_information)


    def generate_detections(self,
                          ground_truth_objects: List,
                          time: float,
                          noise_std: float = 0.0,
                          world_size: Tuple[float, float] = (100.0, 100.0),
                          neighbor_robots: Optional[List[Robot]] = None,
                          assigned_fp_objects: Optional[List] = None) -> List[Track]:
        """
        Generate detections based on the robot's attack mode.

        All modes use DetectorSensor for baseline sensor noise, then apply adversarial manipulations.

        Args:
            ground_truth_objects: List of ground truth objects
            time: Current simulation time
            noise_std: Standard deviation for position noise
            world_size: Size of the simulation world
            neighbor_robots: List of neighboring robots (for optimized/deceptive modes, deprecated)
            assigned_fp_objects: List of assigned persistent adversarial FP objects (used by all modes)

        Returns:
            List of Track objects representing detections
        """
        # Cache assigned FP objects for use in optimized/deceptive modes
        if assigned_fp_objects is not None:
            self._assigned_fp_objects_cache = assigned_fp_objects

        if self.mode == 'normal':
            return self._generate_normal_adversarial_detections(
                ground_truth_objects, time, noise_std, assigned_fp_objects, world_size
            )
        elif self.mode == 'optimized':
            return self._generate_optimized_adversarial_detections(
                ground_truth_objects, time, noise_std, world_size, neighbor_robots
            )
        elif self.mode == 'deceptive':
            return self._generate_deceptive_adversarial_detections(
                ground_truth_objects, time, noise_std, world_size, neighbor_robots
            )
        else:
            raise ValueError(f"Unknown adversarial robot mode: {self.mode}")

    def _generate_normal_adversarial_detections(self,
                                               ground_truth_objects: List,
                                               time: float,
                                               noise_std: float,
                                               assigned_fp_objects: Optional[List],
                                               world_size: Tuple[float, float]) -> List[Track]:
        """
        Normal adversarial mode: Random adversarial FP/FN manipulations + sensor noise.

        NEW TRACK ARCHITECTURE:
        1. Populate current_timestep_tracks with sensor detections
        2. Apply adversarial manipulations (random FP injection, FN suppression)
        3. Populate reported_tracks with manipulated tracks
        """
        # Clear timestep-specific tracks
        self.clear_timestep_specific_tracks()

        # STEP 1: Populate current_timestep_tracks with sensor detections
        detected_gt_objects, detected_sensor_fp_objects = self.detector_sensor.generate_detections(
            robot_position=self.position,
            robot_orientation=self.orientation,
            robot_fov_range=self.fov_range,
            robot_fov_angle=self.fov_angle,
            is_in_fov_func=self.is_in_fov,
            ground_truth_objects=ground_truth_objects,
            time=time,
            noise_std=noise_std,
            world_size=world_size
        )

        # Add all sensor detections to current_timestep_tracks
        for detection in detected_gt_objects:
            self.add_sensor_detection(
                object_id=detection['object_id'],
                position=detection['position'],
                velocity=detection['velocity'],
                timestamp=time
            )

        for detection in detected_sensor_fp_objects:
            self.add_sensor_detection(
                object_id=detection['object_id'],
                position=detection['position'],
                velocity=detection['velocity'],
                timestamp=time
            )

        # STEP 2: Apply adversarial manipulations to create reported_tracks

        # 2a: Add GT objects that pass FN suppression
        for detection in detected_gt_objects:
            # Apply random FN suppression
            if random.random() >= self.adversarial_fn_suppression_rate:
                object_id = detection['object_id']
                track = self.current_timestep_tracks[object_id]
                self.reported_tracks[object_id] = track

        # 2b: Add sensor FP detections (all pass through)
        for detection in detected_sensor_fp_objects:
            object_id = detection['object_id']
            track = self.current_timestep_tracks[object_id]
            self.reported_tracks[object_id] = track

        # 2c: Add persistent adversarial FP injections
        if assigned_fp_objects is not None:
            for fp_obj in assigned_fp_objects:
                if self.is_in_fov(fp_obj.position):
                    object_id = f"fp_obj_{fp_obj.id}"

                    # Inherit trust from all_tracks if seen before
                    if object_id in self.all_tracks:
                        trust_alpha = self.all_tracks[object_id].trust_alpha
                        trust_beta = self.all_tracks[object_id].trust_beta
                    else:
                        trust_alpha = 1.0
                        trust_beta = 1.0

                    # Create track for FP injection
                    track = Track(
                        track_id=f"{self.id}_{object_id}",
                        robot_id=self.id,
                        object_id=object_id,
                        position=fp_obj.position.copy(),
                        velocity=fp_obj.velocity.copy(),
                        trust_alpha=trust_alpha,
                        trust_beta=trust_beta,
                        timestamp=time
                    )

                    self.reported_tracks[object_id] = track

        # Return list of reported tracks
        return list(self.reported_tracks.values())

    def _generate_optimized_adversarial_detections(self,
                                                  ground_truth_objects: List,
                                                  time: float,
                                                  noise_std: float,
                                                  world_size: Tuple[float, float],
                                                  neighbor_robots: Optional[List[Robot]]) -> List[Track]:
        """
        Optimized adversarial mode: Objective-driven binary decisions per object.

        NEW TRACK ARCHITECTURE:
        1. Populate current_timestep_tracks with sensor detections (inherits trust from all_tracks)
        2. Run adversarial policy to decide what to report (FP injection, GT suppression)
        3. Populate reported_tracks with manipulated tracks
        4. Return reported_tracks for communication to neighbors

        Decision based on maximizing adversarial objective:
        J_adv = alpha * (FP trust) - beta * (GT trust)
        """
        # Clear timestep-specific tracks
        self.clear_timestep_specific_tracks()

        # STEP 1: Populate current_timestep_tracks with sensor detections
        detected_gt_objects, detected_sensor_fp_objects = self.detector_sensor.generate_detections(
            robot_position=self.position,
            robot_orientation=self.orientation,
            robot_fov_range=self.fov_range,
            robot_fov_angle=self.fov_angle,
            is_in_fov_func=self.is_in_fov,
            ground_truth_objects=ground_truth_objects,
            time=time,
            noise_std=noise_std,
            world_size=world_size
        )

        # Add all sensor detections to current_timestep_tracks
        # (inherits trust from all_tracks if object was seen before)
        for detection in detected_gt_objects:
            self.add_sensor_detection(
                object_id=detection['object_id'],
                position=detection['position'],
                velocity=detection['velocity'],
                timestamp=time
            )

        for detection in detected_sensor_fp_objects:
            self.add_sensor_detection(
                object_id=detection['object_id'],
                position=detection['position'],
                velocity=detection['velocity'],
                timestamp=time
            )

        # STEP 2: Run adversarial policy to decide what to report
        # Input: current_timestep_tracks (sensor detections) + FP objects in FoV
        # Output: reported_tracks (manipulated detections)

        # ===== Collect FP objects in FoV =====
        fp_objects_in_fov = []
        if hasattr(self, '_assigned_fp_objects_cache'):
            for fp_obj in self._assigned_fp_objects_cache:
                if self.is_in_fov(fp_obj.position):
                    object_id = f"fp_obj_{fp_obj.id}"
                    fp_objects_in_fov.append({
                        'id': fp_obj.id,
                        'object_id': object_id,
                        'position': fp_obj.position.copy(),
                        'velocity': fp_obj.velocity.copy(),
                        'fp_obj': fp_obj
                    })

        # ===== MULTI-OBJECT INTEGER OPTIMIZATION =====
        # Decide which objects to report (maximize adversarial objective)

        if self.use_objective_policy:
            # Collect all objects for policy evaluation
            all_objects = []

            # Add persistent FP objects (adversarial knows these are FPs)
            for hyp in fp_objects_in_fov:
                all_objects.append({
                    'type': 'fp',
                    'object_id': hyp['object_id'],
                    'data': hyp,
                })

            # Add ALL sensor detections from current_timestep_tracks
            # (Robot treats these as GT-like objects - doesn't know which are real)
            for object_id, track in self.current_timestep_tracks.items():
                all_objects.append({
                    'type': 'gt',
                    'object_id': object_id,
                    'data': {
                        'position': track.position,
                        'velocity': track.velocity,
                        'object_id': object_id,
                    },
                })

            # Solve integer optimization problem using scipy.optimize.milp
            n_objects = len(all_objects)
            if n_objects > 0:
                # Compute objective coefficients for each object
                # For binary variable x_i:
                #   x_i = 1 means REPORT the object
                #   x_i = 0 means DON'T REPORT (ignore/suppress) the object
                #
                # Objective: maximize sum_i (c_i * x_i)
                # where c_i = objective_change(report) - objective_change(don't_report)

                objective_coeffs = []
                for obj in all_objects:
                    # Binary decision: report or ignore
                    is_fp = (obj['type'] == 'fp')

                    delta_report = self._estimate_objective_change(obj['data'], 'report', is_fp=is_fp)
                    delta_ignore = self._estimate_objective_change(obj['data'], 'ignore', is_fp=is_fp)

                    # Coefficient for x_i in the objective function
                    # If x_i = 1 (report), we get delta_report
                    # If x_i = 0 (don't report), we get delta_ignore
                    # Linear form: delta_ignore + x_i * (delta_report - delta_ignore)
                    # To maximize, we use coefficient (delta_report - delta_ignore)
                    objective_coeffs.append(delta_report - delta_ignore)

                # Use scipy.optimize.milp to solve the integer program
                from scipy.optimize import milp, Bounds, LinearConstraint
                import numpy as np

                # Objective: maximize c^T x  ->  minimize -c^T x
                c = -np.array(objective_coeffs)

                # Bounds: 0 <= x_i <= 1 for all binary variables
                bounds = Bounds(lb=0, ub=1)

                # Integrality: all variables are binary (0 or 1)
                integrality = np.ones(n_objects)

                # Add constraint: limit number of adversarial operations per timestep
                #
                # ADVERSARIAL OPERATIONS:
                # - FP object: Reporting (x_i = 1) is adversarial
                # - GT object: Suppressing (x_i = 0) is adversarial
                #
                # CONSTRAINT FORMULATION:
                # We want: (# FP reports) + (# GT suppressions) <= max_operations
                #
                # Count FP reports: sum(x_i for i in FP_indices)
                # Count GT suppressions: sum(1 - x_i for i in GT_indices) = num_GT - sum(x_i for i in GT_indices)
                #
                # Combined: sum(x_i for FP) + num_GT - sum(x_i for GT) <= max_operations
                # Rearrange: sum(x_i for FP) - sum(x_i for GT) <= max_operations - num_GT
                #
                # This is a linear constraint: A @ x <= b where:
                # - A[i] = +1 for FP objects (reporting is adversarial)
                # - A[i] = -1 for GT objects (NOT reporting is adversarial)
                # - b = max_operations - num_GT

                # Count object types
                num_gt = sum(1 for obj in all_objects if obj['type'] == 'gt')

                # Build constraint matrix A
                A_constraint = np.zeros((1, n_objects))
                for i, obj in enumerate(all_objects):
                    if obj['type'] == 'fp':
                        A_constraint[0, i] = 1.0   # Coefficient for FP objects
                    else:  # obj['type'] == 'gt'
                        A_constraint[0, i] = -1.0  # Coefficient for GT objects

                # Right-hand side of constraint
                b_constraint = np.array([self.max_adversarial_operations - num_gt])

                constraint = LinearConstraint(A_constraint, -np.inf, b_constraint)

                # Solve the MILP with the constraint
                result = milp(c=c, bounds=bounds, constraints=constraint, integrality=integrality)

                if result.success:
                    best_actions = tuple(result.x > 0.5)  # Convert to boolean tuple

                    # Debug: Verify constraint satisfaction
                    if hasattr(self, '_debug_constraints') and self._debug_constraints:
                        # Count actual adversarial operations
                        num_fp_reports = sum(1 for i, x in enumerate(best_actions)
                                           if all_objects[i]['type'] == 'fp' and x)
                        num_gt_suppressions = sum(1 for i, x in enumerate(best_actions)
                                                 if all_objects[i]['type'] == 'gt' and not x)
                        num_adversarial_ops = num_fp_reports + num_gt_suppressions

                        num_fp = sum(1 for obj in all_objects if obj['type'] == 'fp')
                        constraint_value = A_constraint @ result.x
                        print(f"  [Robot {self.id}] Objects: {num_fp} FP, {num_gt} GT | "
                              f"Attacks: {num_fp_reports} FP + {num_gt_suppressions} GT = {num_adversarial_ops}/{self.max_adversarial_operations} | "
                              f"Constraint: {constraint_value[0]:.2f} <= {b_constraint[0]:.2f}")
                else:
                    # Fallback: use greedy approach if optimization fails
                    best_actions = tuple(objective_coeffs[i] > 0 for i in range(n_objects))

                # STEP 3: Populate reported_tracks based on policy decisions
                for i, should_report in enumerate(best_actions):
                    obj = all_objects[i]

                    if should_report:
                        object_id = obj['object_id']
                        data = obj['data']

                        # Inherit trust from all_tracks if this object was seen before
                        if object_id in self.all_tracks:
                            trust_alpha = self.all_tracks[object_id].trust_alpha
                            trust_beta = self.all_tracks[object_id].trust_beta
                        else:
                            trust_alpha = 1.0
                            trust_beta = 1.0

                        # Create track for reported_tracks
                        track = Track(
                            track_id=f"{self.id}_{object_id}",
                            robot_id=self.id,
                            object_id=object_id,
                            position=data['position'].copy(),
                            velocity=data['velocity'].copy(),
                            trust_alpha=trust_alpha,
                            trust_beta=trust_beta,
                            timestamp=time
                        )

                        # Add to reported_tracks
                        self.reported_tracks[object_id] = track

                        # Update FP object's last supported time if applicable
                        if obj['type'] == 'fp':
                            data['fp_obj'].last_supported_time = time

        # Return list of reported tracks
        return list(self.reported_tracks.values())

    def _generate_deceptive_adversarial_detections(self,
                                                  ground_truth_objects: List,
                                                  time: float,
                                                  noise_std: float,
                                                  world_size: Tuple[float, float],
                                                  neighbor_robots: Optional[List[Robot]]) -> List[Track]:
        """
        Deceptive adversarial mode: Optimized attacks + trust/confidence manipulation.

        This mode extends optimized attacks by also manipulating the trust/confidence
        values associated with local tracks to deceive the trust inference system.

        Adversarial track lies strategy (matching compute_object_metrics_with_adversarial_lies):
        - GT objects: lie low (random uniform 0.0 to 0.2)
        - FP objects: lie high (random uniform 0.8 to 1.0)

        This affects the weighted averaging calculation: sum(robot_trust * track_trust) / sum(robot_trust)
        """
        # First, generate tracks using optimized strategy
        tracks = self._generate_optimized_adversarial_detections(
            ground_truth_objects, time, noise_std, world_size, neighbor_robots
        )

        # Then, manipulate trust/confidence values to deceive the system
        for track in tracks:
            # Convert trust value to Beta distribution parameters that produce desired mean
            # For a Beta(alpha, beta), mean = alpha / (alpha + beta)

            # Match the FP/GT prefix classification used by the canonical reference
            # implementation (comprehensive_trust_benchmark.py's
            # compute_object_metrics_with_adversarial_lies): FP-like objects include
            # persistent adversarial FPs (fp_obj_/fp_/adv_fp_) AND transient sensor FPs
            # (sensor_fp_) - both are false detections that should be reported as
            # credible under this deception strategy, not just the adversarially-injected
            # ones.
            is_fp_like = (track.object_id.startswith('fp_obj_') or
                          track.object_id.startswith('fp_') or
                          track.object_id.startswith('adv_fp_') or
                          track.object_id.startswith('sensor_fp_'))

            if is_fp_like:
                # False positive: lie high (0.8 to 1.0)
                # This makes FP objects appear highly credible
                target_trust = random.uniform(0.8, 1.0)
                # Use Beta parameters that give high trust
                track.trust_alpha = target_trust * 10  # e.g., 8.0 to 10.0
                track.trust_beta = (1.0 - target_trust) * 10  # e.g., 0.0 to 2.0
            elif track.object_id.startswith('gt_'):
                # Real object (GT): lie low (0.0 to 0.2)
                # This makes GT objects appear less credible
                target_trust = random.uniform(0.0, 0.2)
                # Use Beta parameters that give low trust
                track.trust_alpha = target_trust * 10  # e.g., 0.0 to 2.0
                track.trust_beta = (1.0 - target_trust) * 10  # e.g., 8.0 to 10.0
            else:
                # Other objects: maintain normal confidence
                track.trust_alpha = 1.0
                track.trust_beta = 1.0

        return tracks

    def _is_position_in_fov(self,
                           position: np.ndarray,
                           robot_pos: np.ndarray,
                           robot_orientation: float,
                           fov_range: float,
                           fov_angle: float) -> bool:
        """
        Check if a position is within a robot's field of view.

        Delegates to the shared is_position_in_fov_2d (same 2D distance + angle-wedge
        geometry as Robot.is_in_fov's default mode) rather than duplicating the math,
        so the two can't silently drift apart again.

        Args:
            position: Position to check [x, y, z]
            robot_pos: Robot position [x, y, z]
            robot_orientation: Robot heading angle (radians)
            fov_range: Field of view range
            fov_angle: Field of view angle (radians)

        Returns:
            True if position is in FoV, False otherwise
        """
        distance = np.linalg.norm((position - robot_pos)[:2])
        if distance < 0.1:
            return False

        return is_position_in_fov_2d(position, robot_pos, robot_orientation, fov_range, fov_angle)

    def _count_neighbors_in_fov_and_supporting(self, object_data: Dict) -> tuple:
        """
        Count neighbors who can see this object AND whether they support it.

        Args:
            object_data: Dictionary with object information (must have 'position' and 'object_id')

        Returns:
            (supporting_count, contradicting_count)
            - supporting: neighbors who can see AND report the object (exact match by object_id)
            - contradicting: neighbors who can see but DON'T report the object

        Raises:
            AssertionError: If object_data is missing required fields
        """
        # Validate required fields
        assert 'position' in object_data, "object_data must contain 'position' field"
        assert 'object_id' in object_data, "object_data must contain 'object_id' field"

        obj_position = object_data['position']
        object_id = object_data['object_id']

        # Validate object_id is not None (critical for matching)
        assert object_id is not None, f"object_id cannot be None! object_data: {object_data}"

        supporting = 0
        contradicting = 0

        # NEW: Use last_reported_tracks instead of neighbor_information['tracks']
        for neighbor_id, neighbor_info in self.neighbor_information.items():
            neighbor_pos = neighbor_info.get('position')
            neighbor_orientation = neighbor_info.get('orientation', 0.0)
            neighbor_fov_range = neighbor_info.get('fov_range')
            neighbor_fov_angle = neighbor_info.get('fov_angle')

            # Check if object is in neighbor's FoV
            in_fov = self._is_position_in_fov(
                obj_position, neighbor_pos, neighbor_orientation,
                neighbor_fov_range, neighbor_fov_angle
            )

            if not in_fov:
                continue  # Neighbor can't see it, irrelevant for cost/benefit

            # Check if neighbor reported this object (from last timestep)
            # Use last_reported_tracks which stores object_ids the neighbor reported
            reported_object_ids = self.last_reported_tracks.get(neighbor_id, [])

            if object_id in reported_object_ids:
                # Neighbor reported this object (exact match by object_id)
                supporting += 1
            else:
                # Neighbor can see but didn't report this object
                contradicting += 1

        return supporting, contradicting

    def _estimate_objective_change(self,
                                    object_data: Dict,
                                    action: str,
                                    is_fp: bool) -> float:
        """
        Estimate expected change in adversarial objective using cost-benefit analysis
        with delta factors (corroboration/dilution).

        NEW LOGIC WITH DELTA FACTORS:
        - For FP objects:
          * Report gain = δ+ * (1 + supporting neighbors)
          * Report cost = (contradicting neighbors)
          * Reward = δ+ * report_gain - report_cost
        - For GT objects:
          * Ignore gain = δ- * (1 + contradicting neighbors)
          * Ignore cost = (supporting neighbors)
          * Reward = δ- * ignore_gain - ignore_cost

        Args:
            object_data: Dictionary with object information (must have 'position')
            action: Action to evaluate - binary choice:
                    'report' = report the object (FP or GT)
                    'ignore' = don't report the object (FP or GT)
            is_fp: True if object is FP, False if GT

        Returns:
            Expected change in J_adv (can be negative!)
        """

        # Count neighbors who can see this object
        supporting, contradicting = self._count_neighbors_in_fov_and_supporting(object_data)

        # Delta factors (corroboration/dilution effects)
        # δ+ amplifies trust gain when neighbors corroborate (more supporting neighbors)
        # δ- amplifies trust reduction when neighbors dilute (fewer supporting neighbors)
        delta_plus = self.delta_plus
        delta_minus = self.delta_minus

        # Calculate cost-benefit based on action and object type
        # UNIFIED OBJECTIVE: (Delta * Trust Impact) - (Consistency Cost)

        if is_fp:
            # FP Object: Want to MAXIMIZE trust
            if action == 'report':
                # Report Gain = δ+ * (1 + supporting neighbors)
                # The +1 represents the adversary's own contribution to FP trust
                report_gain = 1.0 + supporting

                # Report Cost = contradicting neighbors who can see but don't report
                report_cost = contradicting

                # Net Objective Change = δ+ * gain - cost
                objective_change = delta_plus * report_gain - report_cost

            else:  # ignore
                # Ignoring FP = no trust gain, no cost
                objective_change = 0.0

        else:
            # GT Object: Want to MINIMIZE trust (suppress)
            if action == 'report':
                # Reporting GT = honest behavior, neutral for adversarial objective
                objective_change = 0.0

            else:  # ignore/suppress
                # Ignore Gain = δ- * (1 + contradicting neighbors who can see but don't report)
                # The +1 represents the adversary's own contribution to GT suppression
                ignore_gain = 1.0 + contradicting

                # Ignore Cost = supporting neighbors who DO report the GT
                ignore_cost = supporting

                # Net Objective Change = δ- * gain - cost
                objective_change = delta_minus * ignore_gain - ignore_cost

        return objective_change
