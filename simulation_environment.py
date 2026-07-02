#!/usr/bin/env python3
"""
Refactored Neural-Symbolic Trust-Based Sensor Fusion Simulation Environment

This simulation environment supports pluggable trust algorithms for comparison
between the original paper's method and neural symbolic approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
from scipy.stats import beta
from scipy.spatial.distance import euclidean

# Import the clean robot and track classes
from robot_track_classes import Robot, Track
# Import specialized robot types
from robot_types import LegitimateRobot, AdversarialRobot


@dataclass
class GroundTruthObject:
    """Ground truth object with dynamic movement"""
    id: int
    position: np.ndarray     # [x, y, z]
    velocity: np.ndarray     # [vx, vy, vz]
    object_type: str         # 'vehicle', 'person', etc.
    movement_pattern: str    # 'linear', 'random_walk', 'circular', 'stationary'
    spawn_time: float        # When object first appeared

    # Movement parameters
    base_speed: float = 1.0
    turn_probability: float = 0.02  # For random walk

    # Circular motion parameters (only used for circular movement)
    circular_center: np.ndarray = None  # Center of circular motion
    circular_radius: float = 0.0        # Radius of circular motion
    direction_change_time: float = 5.0  # For direction changes

    # FP object tracking (only used for FP objects in shared_fp_objects)
    first_detected_time: Optional[float] = None  # When ANY robot first reported it
    last_supported_time: Optional[float] = None  # When it was last reported by any robot


class SimulationEnvironment:
    """Multi-robot simulation environment with pluggable trust algorithms"""
    
    def __init__(self,
                 world_size: Tuple[float, float] = (100.0, 100.0),
                 robot_density: float = 0.0005,
                 target_density: float = 0.002,
                 adversarial_ratio: float = 0.3,
                 proximal_range: float = 80.0,
                 fov_range: float = 50.0,
                 fov_angle: float = np.pi/3,
                 adversarial_fp_injection_rate: float = 0.5,
                 adversarial_fn_suppression_rate: float = 0.0,
                 sensor_fp_rate: float = 0.05,
                 sensor_fn_rate: float = 0.05,
                 allow_fp_codetection: bool = False,
                 num_robots: Optional[int] = None,
                 num_targets: Optional[int] = None,
                 legitimate_mode: str = 'optimal',
                 adversarial_mode: str = 'normal',
                 delta_plus: float = 5.0,
                 delta_minus: float = 1.0):
        """
        Initialize simulation environment

        Args:
            world_size: Size of the simulation world (fixed for density-based configuration)
            robot_density: Robots per unit area (used when num_robots not provided)
            target_density: Ground-truth targets per unit area (used when num_targets not provided)
            adversarial_ratio: Fraction of robots that are adversarial
            proximal_range: Maximum distance for robots to be considered proximal
            fov_range: Field of view range for robots (default: 30.0)
            fov_angle: Field of view angle in radians (default: π/3, 60 degrees)
            adversarial_fp_injection_rate: Rate determining number of persistent adversarial FP objects
                                          (num_fp_objects = rate × num_gt_objects)
            adversarial_fn_suppression_rate: Rate of transient adversarial FN suppression (only for 'normal' mode)
            sensor_fp_rate: Sensor false positive rate (transient, for realistic detectors)
            sensor_fn_rate: Sensor false negative rate (transient, for realistic detectors)
            allow_fp_codetection: If True, allows adversarial robots to co-detect FP objects (default: False)
            legitimate_mode: Mode for legitimate robots ('optimal' or 'realistic')
            adversarial_mode: Mode for adversarial robots ('normal', 'optimized', or 'deceptive')
            delta_plus: Corroboration factor (FP-gain coefficient) in the 'optimized'/'deceptive'
                MILP cost-benefit objective. Higher values push adversarial robots to report
                persistent FP objects more readily/more often. Only affects 'optimized'/'deceptive'
                modes (see AdversarialRobot._estimate_objective_change in robot_types.py).
            delta_minus: Dilution factor (GT-suppression coefficient) in the same objective.
                Only affects 'optimized'/'deceptive' modes.
        """
        self.world_size = world_size
        self.area = self.world_size[0] * self.world_size[1]

        if num_robots is not None:
            self.num_robots = max(1, int(num_robots))
            self.robot_density = self.num_robots / self.area
        else:
            self.robot_density = max(1e-6, float(robot_density))
            self.num_robots = max(1, int(round(self.robot_density * self.area)))

        if num_targets is not None:
            self.num_targets = max(1, int(num_targets))
            self.target_density = self.num_targets / self.area
        else:
            self.target_density = max(1e-6, float(target_density))
            self.num_targets = max(1, int(round(self.target_density * self.area)))

        self.adversarial_ratio = adversarial_ratio
        self.proximal_range = proximal_range
        self.fov_range = fov_range
        self.fov_angle = fov_angle
        self.adversarial_fp_injection_rate = adversarial_fp_injection_rate
        self.adversarial_fn_suppression_rate = adversarial_fn_suppression_rate
        self.sensor_fp_rate = sensor_fp_rate
        self.sensor_fn_rate = sensor_fn_rate
        self.allow_fp_codetection = allow_fp_codetection
        self.legitimate_mode = legitimate_mode
        self.adversarial_mode = adversarial_mode
        self.delta_plus = delta_plus
        self.delta_minus = delta_minus

        self.robots: List[Robot] = []
        self.ground_truth_objects: List[GroundTruthObject] = []
        # Tracks are stored locally in each Robot instance (not centralized here)

        self.time = 0.0
        self.dt = 0.1  # 10Hz simulation rate
        self.next_object_id = 0  # For generating unique object IDs
        
        # FP objects management - each FP object is assigned to a specific adversarial robot
        self.shared_fp_objects: List[GroundTruthObject] = []  # All FP objects
        self.fp_object_assignments: Dict[int, int] = {}  # Maps FP object ID to assigned robot ID
        self.fp_object_angles: Dict[int, float] = {}  # Maps FP object ID to its random angle offset within FoV
        self.next_shared_fp_id = 0  # Global FP ID counter
        
        self._initialize_environment()
    
    def reset(self):
        """Reset the simulation to initial state"""
        # Clear existing state
        self.robots.clear()
        self.ground_truth_objects.clear()
        self.shared_fp_objects.clear()
        self.fp_object_assignments.clear()
        self.fp_object_angles.clear()

        # Reset time and counters
        self.time = 0.0
        self.next_object_id = 0
        self.next_shared_fp_id = 0
        
        # Re-initialize environment with new random positions and adversarial assignments
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize robots and targets in the environment"""
        # Initialize robots at random positions
        # Ensure at least 1 adversarial robot if adversarial_ratio > 0
        num_adversarial = int(self.num_robots * self.adversarial_ratio)
        if self.adversarial_ratio > 0 and num_adversarial == 0:
            num_adversarial = 1  # Guarantee at least 1 adversarial robot
        adversarial_ids = set(random.sample(range(self.num_robots), num_adversarial))
        
        for i in range(self.num_robots):

            # Start position (home base)
            start_pos = np.array([
                random.uniform(5, self.world_size[0] - 5),
                random.uniform(5, self.world_size[1] - 5),
                random.uniform(15, 25)  # Altitude
            ])

            # Generate random goal position for patrol
            goal_pos = np.array([
                random.uniform(5, self.world_size[0] - 5),
                random.uniform(5, self.world_size[1] - 5),
                start_pos[2]  # Keep same altitude
            ])

            # Ensure goal is at reasonable distance from start
            min_patrol_distance = min(self.world_size) * 0.3
            max_attempts = 20
            attempts = 0

            while np.linalg.norm(goal_pos[:2] - start_pos[:2]) < min_patrol_distance and attempts < max_attempts:
                goal_pos = np.array([
                    random.uniform(5, self.world_size[0] - 5),
                    random.uniform(5, self.world_size[1] - 5),
                    start_pos[2]
                ])
                attempts += 1

            # Create robot with specialized type
            is_adversarial = (i in adversarial_ids)
            if is_adversarial:
                robot = AdversarialRobot(
                    robot_id=i,
                    position=start_pos.copy(),
                    velocity=np.array([0.0, 0.0, 0.0]),
                    fov_range=self.fov_range,
                    fov_angle=self.fov_angle,
                    mode=self.adversarial_mode,
                    adversarial_fp_injection_rate=self.adversarial_fp_injection_rate,
                    adversarial_fn_suppression_rate=self.adversarial_fn_suppression_rate,
                    sensor_fp_rate=self.sensor_fp_rate,
                    sensor_fn_rate=self.sensor_fn_rate,
                    delta_plus=self.delta_plus,
                    delta_minus=self.delta_minus
                )
                # Note: Persistent FP objects are now managed globally in shared_fp_objects
                # and assigned to adversaries in generate_detections() via assigned_fp_objects parameter
            else:
                robot = LegitimateRobot(
                    robot_id=i,
                    position=start_pos.copy(),
                    velocity=np.array([0.0, 0.0, 0.0]),
                    fov_range=self.fov_range,
                    fov_angle=self.fov_angle,
                    mode=self.legitimate_mode,
                    sensor_fp_rate=self.sensor_fp_rate,
                    sensor_fn_rate=self.sensor_fn_rate
                )

            # Set additional attributes for patrol behavior
            robot.start_position = start_pos
            robot.goal_position = goal_pos
            robot.patrol_speed = random.uniform(1.5, 2.5)
            robot.orientation = 0.0
            robot.is_returning_home = False
            robot.position_tolerance = 3.0

            # Initialize velocity and orientation toward goal
            self._update_robot_navigation(robot)

            self.robots.append(robot)
        
        # Initialize ground truth objects
        for i in range(self.num_targets):
            obj = self._create_ground_truth_object(self.time)
            self.ground_truth_objects.append(obj)

        # Initialize persistent adversarial false positive objects - each adversarial robot gets at least one
        # Number of FP objects = adversarial_fp_injection_rate * number of ground truth objects
        adversarial_robots = [r for r in self.robots if r.is_adversarial]
        num_adversarial = len(adversarial_robots)

        # Only create FP objects if there are adversarial robots
        if num_adversarial > 0:
            # Calculate total FP objects needed based on adversarial FP injection rate
            num_fp_objects_needed = int(self.adversarial_fp_injection_rate * len(self.ground_truth_objects))

            # Ensure we have at least one FP object per adversarial robot
            num_fp_objects = max(num_adversarial, num_fp_objects_needed)

            # Assign FP objects to adversarial robots
            # First, assign one to each adversarial robot
            robot_assignments = list(adversarial_robots)  # Start with one per robot

            # If we need more FP objects, randomly assign the extras
            if num_fp_objects > num_adversarial:
                extra_fp_needed = num_fp_objects - num_adversarial
                robot_assignments.extend(random.choices(adversarial_robots, k=extra_fp_needed))

            # Create FP objects and assign them to robots
            for robot in robot_assignments:
                # Generate a random angle offset within the FoV cone for this FP object
                # This ensures FP objects are randomly distributed across the robot's FoV
                random_angle_offset = random.uniform(-self.fov_angle * 0.45, self.fov_angle * 0.45)

                fp_obj = self._create_fp_object_for_robot(robot, self.time, random_angle_offset)
                self.shared_fp_objects.append(fp_obj)
                self.fp_object_assignments[fp_obj.id] = robot.id
                self.fp_object_angles[fp_obj.id] = random_angle_offset  # Store for persistent tracking

        # Initialize trust algorithm
        print(f"Simulation initialized with {len(self.robots)} robots and {len(self.ground_truth_objects)} objects")
        print(f"  - Ground truth objects: {len(self.ground_truth_objects)}")
        print(f"  - Persistent adversarial FP objects: {len(self.shared_fp_objects)} (adversarial_fp_injection_rate={self.adversarial_fp_injection_rate:.2f})")
        print(f"  - Adversarial robots: {num_adversarial}")
        print(f"  - FP objects per adversarial robot: avg={len(self.shared_fp_objects)/max(1,num_adversarial):.2f}")
    
    def _create_fp_object_for_robot(self, robot: Robot, spawn_time: float, angle_offset: float) -> GroundTruthObject:
        """Create a false positive object assigned to a specific adversarial robot

        The FP object is positioned in the robot's FoV at a random angle and will
        maintain this relative position, staying visible only to this specific robot.

        Args:
            robot: The robot this FP object is assigned to
            spawn_time: Time of creation
            angle_offset: Random angle offset within FoV (radians, relative to robot heading)
        """
        obj_id = self.next_shared_fp_id
        self.next_shared_fp_id += 1

        # Position the FP object in the robot's FoV
        # Place it at a random distance within the FoV
        distance = random.uniform(self.fov_range * 0.3, self.fov_range * 0.7)  # 30-70% of FoV range

        # Calculate position using robot heading and the provided random angle offset
        robot_heading = np.arctan2(robot.velocity[1], robot.velocity[0]) if np.linalg.norm(robot.velocity) > 0.1 else 0
        angle = robot_heading + angle_offset

        position = robot.position + np.array([
            distance * np.cos(angle),
            distance * np.sin(angle),
            0.0
        ])

        # Ensure position is within world bounds
        position[0] = np.clip(position[0], 5, self.world_size[0] - 5)
        position[1] = np.clip(position[1], 5, self.world_size[1] - 5)

        # Use random_walk movement pattern - object wanders slowly
        movement_pattern = 'random_walk'
        base_speed = random.uniform(0.3, 1.0)  # Slow movement
        velocity = np.array([0.0, 0.0, 0.0])  # Will be updated by random walk

        # Object type
        object_types = ['vehicle', 'person', 'animal']
        object_type = random.choice(object_types)

        return GroundTruthObject(
            id=obj_id,
            position=position,
            velocity=velocity,
            object_type=object_type,
            movement_pattern=movement_pattern,
            spawn_time=spawn_time,
            base_speed=base_speed,
            turn_probability=random.uniform(0.1, 0.2),  # Higher turn probability to stay nearby
            direction_change_time=random.uniform(2, 5),
            circular_center=None,
            circular_radius=0.0
        )

    def _create_ground_truth_object(self, spawn_time: float) -> GroundTruthObject:
        """Create a new ground truth object with random properties"""
        obj_id = self.next_object_id
        self.next_object_id += 1
        
        # Random position
        position = np.array([
            random.uniform(5, self.world_size[0] - 5),
            random.uniform(5, self.world_size[1] - 5),
            0.0  # Ground level
        ])
        
        # Choose movement pattern
        movement_patterns = ['linear', 'random_walk', 'circular', 'stationary']
        movement_pattern = random.choice(movement_patterns)
        
        # Set initial velocity based on movement pattern
        base_speed = random.uniform(0.5, 2.5)
        
        if movement_pattern == 'linear':
            # Random direction, constant velocity
            angle = random.uniform(0, 2*np.pi)
            velocity = np.array([
                base_speed * np.cos(angle),
                base_speed * np.sin(angle),
                0.0
            ])
        elif movement_pattern == 'circular':
            # Set up circular motion parameters
            potential_centers = [
                (self.world_size[0] * 0.2, self.world_size[1] * 0.2),
                (self.world_size[0] * 0.8, self.world_size[1] * 0.2),
                (self.world_size[0] * 0.2, self.world_size[1] * 0.8),
                (self.world_size[0] * 0.8, self.world_size[1] * 0.8),
            ]
            center_x, center_y = random.choice(potential_centers)
            circular_center = np.array([center_x, center_y, 0.0])
            circular_radius = random.uniform(5, 12)
            
            # Set initial position on the circle
            angle = random.uniform(0, 2 * np.pi)
            position = circular_center + circular_radius * np.array([
                np.cos(angle), np.sin(angle), 0.0
            ])
            
            velocity = np.array([base_speed, 0.0, 0.0])
        else:  # random_walk or stationary
            velocity = np.array([0.0, 0.0, 0.0])
        
        # Object type
        object_types = ['vehicle', 'person', 'animal']
        object_type = random.choice(object_types)

        # Prepare circular motion parameters if needed
        circular_center_param = circular_center if movement_pattern == 'circular' else None
        circular_radius_param = circular_radius if movement_pattern == 'circular' else 0.0

        return GroundTruthObject(
            id=obj_id,
            position=position,
            velocity=velocity,
            object_type=object_type,
            movement_pattern=movement_pattern,
            spawn_time=spawn_time,
            base_speed=base_speed,
            turn_probability=random.uniform(0.01, 0.05),
            direction_change_time=random.uniform(3, 8),
            circular_center=circular_center_param,
            circular_radius=circular_radius_param
        )
    
    def _update_ground_truth_objects(self):
        """Update positions and states of all ground truth objects"""
        # Update object based on movement pattern
        for obj in self.ground_truth_objects:
            if obj.movement_pattern == 'stationary':
                pass
                
            elif obj.movement_pattern == 'linear':
                # Move in straight line, bounce off walls
                new_pos = obj.position + obj.velocity * self.dt
                
                # Bounce off boundaries
                if new_pos[0] <= 0 or new_pos[0] >= self.world_size[0]:
                    obj.velocity[0] *= -1
                if new_pos[1] <= 0 or new_pos[1] >= self.world_size[1]:
                    obj.velocity[1] *= -1
                
                obj.position += obj.velocity * self.dt
                obj.position[0] = np.clip(obj.position[0], 0, self.world_size[0])
                obj.position[1] = np.clip(obj.position[1], 0, self.world_size[1])
                
            elif obj.movement_pattern == 'random_walk':
                # Random walk with occasional direction changes
                if random.random() < obj.turn_probability:
                    angle = random.uniform(0, 2*np.pi)
                    obj.velocity = np.array([
                        obj.base_speed * np.cos(angle),
                        obj.base_speed * np.sin(angle),
                        0.0
                    ])
                
                new_pos = obj.position + obj.velocity * self.dt
                if (new_pos[0] <= 0 or new_pos[0] >= self.world_size[0] or 
                    new_pos[1] <= 0 or new_pos[1] >= self.world_size[1]):
                    angle = random.uniform(0, 2*np.pi)
                    obj.velocity = np.array([
                        obj.base_speed * np.cos(angle),
                        obj.base_speed * np.sin(angle),
                        0.0
                    ])
                
                obj.position += obj.velocity * self.dt
                obj.position[0] = np.clip(obj.position[0], 0, self.world_size[0])
                obj.position[1] = np.clip(obj.position[1], 0, self.world_size[1])
                
            elif obj.movement_pattern == 'circular':
                # Circular movement
                if obj.circular_center is not None:
                    center = obj.circular_center
                    radius = obj.circular_radius
                    angular_velocity = obj.base_speed / radius
                    
                    angle = angular_velocity * (self.time - obj.spawn_time)
                    obj.position = center + radius * np.array([
                        np.cos(angle), np.sin(angle), 0.0
                    ])

        # Update false positive objects (same dynamics as ground truth objects)
        self._update_fp_objects()
    
    def _update_fp_objects(self):
        """Update FP object positions to keep them perfectly in their assigned robot's FoV

        Since we know the robot's position and heading at each timestep, we simply
        position the FP object at a fixed offset relative to the robot, guaranteeing
        it's ALWAYS in the FoV. No complex tracking or recovery modes needed.
        """
        robot_lookup = {r.id: r for r in self.robots}

        for fp_obj in self.shared_fp_objects:
            # Get the robot this FP object is assigned to
            assigned_robot_id = self.fp_object_assignments.get(fp_obj.id)
            if assigned_robot_id is None or assigned_robot_id not in robot_lookup:
                continue  # Skip if assignment is invalid

            robot = robot_lookup[assigned_robot_id]

            # Get this FP object's persistent random angle offset
            # This angle was randomly assigned at initialization and stays constant
            angle_offset = self.fp_object_angles.get(fp_obj.id, 0.0)

            # Calculate robot heading
            robot_heading = np.arctan2(robot.velocity[1], robot.velocity[0]) if np.linalg.norm(robot.velocity) > 0.1 else 0

            # Calculate target position: 50% of FoV range, at designated angle
            target_distance = self.fov_range * 0.5
            target_angle = robot_heading + angle_offset
            target_position = robot.position + np.array([
                target_distance * np.cos(target_angle),
                target_distance * np.sin(target_angle),
                0.0
            ])

            # Keep within world bounds
            target_position[0] = np.clip(target_position[0], 1, self.world_size[0] - 1)
            target_position[1] = np.clip(target_position[1], 1, self.world_size[1] - 1)

            # Calculate velocity as the displacement from current to target position
            # This makes the FP object smoothly track the robot
            displacement = target_position - fp_obj.position
            fp_obj.velocity = displacement / self.dt  # Velocity needed to reach target in one timestep

            # Directly set position to target (guarantees FP stays in FoV)
            fp_obj.position = target_position.copy()

    def _update_fp_object_timestamps(self):
        """Update FP object timestamps based on which robots are currently reporting them.

        This tracks when FP objects are first detected and last supported by ANY robot
        (legitimate or adversarial) based on their current tracks.
        """
        for fp_obj in self.shared_fp_objects:
            # Check all robots' tracks to see if any robot is reporting this FP object.
            # Match by object_id (not position) - the FP's identity is f"fp_obj_{fp_obj.id}"
            # (see robot_types.py), which every reporting robot uses consistently. Matching
            # by position instead would incorrectly match unrelated tracks (e.g. a genuine
            # GT object or a different FP) that merely happen to pass within tolerance.
            fp_object_id = f"fp_obj_{fp_obj.id}"
            is_currently_supported = False

            for robot in self.robots:
                if fp_object_id in robot.reported_tracks:
                    is_currently_supported = True
                    break

            # Update timestamps if this FP is being reported
            if is_currently_supported:
                # Set first_detected_time if this is the first time it's been detected
                if fp_obj.first_detected_time is None:
                    fp_obj.first_detected_time = self.time

                # Always update last_supported_time
                fp_obj.last_supported_time = self.time

    def _update_robot_navigation(self, robot: Robot):
        """Update robot's velocity and orientation based on current patrol target"""
        # Determine current target (goal or home)
        if robot.is_returning_home:
            target = robot.start_position
        else:
            target = robot.goal_position
        
        # Calculate direction vector
        direction = target - robot.position
        distance = np.linalg.norm(direction)
        
        if distance > robot.position_tolerance:
            # Normalize direction and set velocity
            direction_normalized = direction / distance
            robot.update_state(robot.position, direction_normalized * robot.patrol_speed)
            robot.orientation = np.arctan2(direction_normalized[1], direction_normalized[0])
        else:
            # Reached target, stop and switch direction
            robot.update_state(robot.position, np.array([0.0, 0.0, 0.0]))
            robot.is_returning_home = not robot.is_returning_home
    
    
    def generate_detections(self, robot: Robot, noise_std: float = 1.0) -> List[Track]:
        """
        Generate detections for a robot using its specialized detection method.

        Delegates to robot-specific detection methods which handle all mode-specific logic.
        """
        if robot.is_adversarial:
            # Adversarial robot: pass assigned FP objects for all modes.
            # If allow_fp_codetection is True, any adversarial robot may also report FP
            # objects assigned to OTHER adversarial robots, enabling collusion on the same
            # fabricated object. The FP object is still geometrically glued to its
            # originally assigned robot's position/heading (_update_fp_objects), so this
            # only matters when robot.is_in_fov() (checked downstream in
            # robot_types.py's detection generation) actually passes for a non-owning
            # robot - matching webots_trust_environment.py's precedent.
            if self.allow_fp_codetection:
                assigned_fps = self.shared_fp_objects
            else:
                assigned_fps = [fp for fp in self.shared_fp_objects
                              if self.fp_object_assignments.get(fp.id) == robot.id]
            return robot.generate_detections(
                ground_truth_objects=self.ground_truth_objects,
                time=self.time,
                noise_std=noise_std,
                world_size=self.world_size,
                neighbor_robots=None,  # Will use neighbor_information instead
                assigned_fp_objects=assigned_fps
            )
        else:
            # Legitimate robot: no FP objects needed
            return robot.generate_detections(
                ground_truth_objects=self.ground_truth_objects,
                time=self.time,
                noise_std=noise_std,
                world_size=self.world_size
            )

    # Legacy detection functions removed - detection is now handled by robot-specific methods
    # See robot.generate_detections() in robot_types.py for current implementation

    def get_proximal_robots(self, ego_robot: 'Robot') -> List['Robot']:
        """Get robots within proximal range of the ego robot"""
        proximal_robots = []
        
        for robot in self.robots:
            if robot.id != ego_robot.id:  # Exclude the ego robot itself
                distance = np.linalg.norm(ego_robot.position - robot.position)
                if distance <= self.proximal_range:
                    proximal_robots.append(robot)
        
        return proximal_robots
    
    def step(self) -> Dict:
        """Execute one simulation step"""
        # Start new timestep for all robots
        for robot in self.robots:
            robot.start_new_timestep(self.time)

        # Update ground truth objects
        self._update_ground_truth_objects()

        # Note: FP objects now follow their assigned robots automatically via _update_fp_objects()
        # No need to update persistent_false_hypotheses separately for optimized/deceptive modes

        # Update robot positions and navigation
        for robot in self.robots:
            self._update_robot_navigation(robot)
            robot.position += robot.velocity * self.dt

            # Boundary conditions
            if robot.position[0] < 0 or robot.position[0] > self.world_size[0]:
                robot.velocity[0] *= -1
                robot.position[0] = np.clip(robot.position[0], 0, self.world_size[0])
            if robot.position[1] < 0 or robot.position[1] > self.world_size[1]:
                robot.velocity[1] *= -1
                robot.position[1] = np.clip(robot.position[1], 0, self.world_size[1])

        # DETECTION GENERATION PHASE
        # Generate detections for each robot FIRST
        # Adversarial robots use neighbor_information from PREVIOUS timestep (T-1)
        # to make strategic decisions about what to report in current timestep (T)
        for robot in self.robots:
            self.generate_detections(robot)

        # NEIGHBOR COMMUNICATION PHASE: Share information with proximal robots
        # This happens AFTER detection generation so that:
        # 1. Adversaries use T-1 neighbor info to decide what to report in T
        # 2. Then we share T's reports for use in T+1

        # Step 1: Clear previous neighbor information (from T-1)
        for robot in self.robots:
            if hasattr(robot, 'clear_neighbor_information'):
                robot.clear_neighbor_information()

        # Step 2: Share current timestep tracks with all proximal neighbors
        for ego_robot in self.robots:
            # Get proximal robots for this ego robot
            proximal_robots = self.get_proximal_robots(ego_robot)

            # Receive information from each proximal neighbor
            if hasattr(ego_robot, 'receive_neighbor_information'):
                for neighbor_robot in proximal_robots:
                    # Each robot shares what it just reported in THIS timestep
                    # This will be stored and used for decision-making in NEXT timestep
                    ego_robot.receive_neighbor_information(neighbor_robot)

        # Update FP object timestamps based on robot detections
        self._update_fp_object_timestamps()

        self.time += self.dt
        
        return {
            'world_size': self.world_size,
            'time': self.time,
            'robot_states': [(r.id, r.position.tolist(), r.velocity.tolist(), r.is_adversarial, r.fov_range, r.fov_angle) 
                           for r in self.robots],
            'tracks': {robot.id: [(t.object_id, t.position.tolist(), getattr(t, 'confidence', 1.0), t.trust_alpha, t.trust_beta) 
                                 for t in robot.get_all_tracks()] 
                      for robot in self.robots},
            'ground_truth': {
                'objects': [(obj.id, obj.position.tolist(), obj.velocity.tolist(), 
                           obj.object_type, obj.movement_pattern) for obj in self.ground_truth_objects],
                'adversarial_robots': [r.id for r in self.robots if r.is_adversarial],
                'legitimate_robots': [r.id for r in self.robots if not r.is_adversarial],
                'num_objects': len(self.ground_truth_objects)
            },
            'shared_fp_objects': [(fp_obj.id, fp_obj.position.tolist(), fp_obj.velocity.tolist(), 
                                  fp_obj.object_type, fp_obj.movement_pattern) for fp_obj in self.shared_fp_objects],
            'trust_updates': {
                str(robot.id): {
                    'mean_trust': robot.trust_alpha / (robot.trust_alpha + robot.trust_beta),
                    'trust_alpha': robot.trust_alpha,
                    'trust_beta': robot.trust_beta
                } for robot in self.robots
            }
        }
    
    def visualize_current_state(self, save_path: Optional[str] = None):
        """Visualize current simulation state matching create_simulation_gif.py format"""
        _, ax = plt.subplots(figsize=(12, 10))
        
        # Set up the plot to match GIF visualization
        ax.set_xlim(0, self.world_size[0])
        ax.set_ylim(0, self.world_size[1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        
        # Title with trust info (matching GIF format)
        leg_trusts = [r.trust_alpha/(r.trust_alpha+r.trust_beta) for r in self.robots if not r.is_adversarial]
        adv_trusts = [r.trust_alpha/(r.trust_alpha+r.trust_beta) for r in self.robots if r.is_adversarial]
        
        title_parts = [f'Trust-Based Sensor Fusion - t={self.time:.1f}s']
        if leg_trusts:
            title_parts.append(f'LEG Trust: {np.mean(leg_trusts):.2f}')
        if adv_trusts:
            title_parts.append(f'ADV Trust: {np.mean(adv_trusts):.2f}')
        
        ax.set_title(' | '.join(title_parts), fontsize=14, fontweight='bold')
        
        # Plot ground truth objects (blue stars, same as GIF)
        for obj in self.ground_truth_objects:
            ax.scatter(obj.position[0], obj.position[1],
                      c='blue', marker='*', s=300, alpha=0.9,
                      edgecolors='black', linewidth=2, 
                      label='Ground Truth' if obj.id == 0 else '')
            
            # Add GT object ID label
            ax.text(obj.position[0], obj.position[1] + 1.2, f'GT{obj.id}', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Plot robots with FOV (matching GIF format exactly)
        for i, robot in enumerate(self.robots):
            pos = robot.position[:2]  # Only x, y coordinates
            trust_val = robot.trust_alpha/(robot.trust_alpha+robot.trust_beta)
            
            # Robot color and type (matching GIF)
            if robot.is_adversarial:
                color = 'red'
                robot_type = 'ADV'
            else:
                color = 'green'
                robot_type = 'LEG'
            
            # Robot size based on trust (matching GIF)
            robot_size = 100 + trust_val * 200
            
            # All robots use square marker (matching GIF)
            ax.scatter(pos[0], pos[1], c=color, marker='s', s=robot_size,
                      edgecolors='white', linewidth=2, alpha=0.9,
                      label=f'{robot_type} Robot' if (robot_type == 'LEG' and i == 0) or (robot_type == 'ADV' and robot.is_adversarial and not any(r.is_adversarial and r.id < robot.id for r in self.robots)) else '')
            
            # Robot ID text (matching GIF)
            ax.text(pos[0], pos[1] - 1.5, f'R{robot.id}\n{trust_val:.2f}', 
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
            
            # Field of View (FOV) visualization (matching GIF)
            fov_angles = np.linspace(robot.orientation - robot.fov_angle/2,
                                   robot.orientation + robot.fov_angle/2, 30)
            
            # FOV triangle
            fov_x = [pos[0]]
            fov_y = [pos[1]]
            for angle in fov_angles:
                fov_x.append(pos[0] + robot.fov_range * np.cos(angle))
                fov_y.append(pos[1] + robot.fov_range * np.sin(angle))
            fov_x.append(pos[0])
            fov_y.append(pos[1])
            
            # FOV fill (matching GIF transparency)
            fov_alpha = 0.15 if robot.is_adversarial else 0.1
            ax.fill(fov_x, fov_y, color=color, alpha=fov_alpha)
            ax.plot(fov_x, fov_y, color=color, linewidth=2, alpha=0.6)
            
        # Plot shared false positive objects (orange stars, same as GT but different color)
        for fp_obj in self.shared_fp_objects:
            fp_pos = fp_obj.position[:2]  # Only x, y coordinates
            
            # FP Objects: Use same star symbol as GT objects but orange color
            ax.scatter(fp_pos[0], fp_pos[1], 
                      c='orange', marker='*', s=300, alpha=0.9, 
                      edgecolors='black', linewidth=2)
            
            # Add FP object ID label (similar to GT objects)
            ax.text(fp_pos[0], fp_pos[1] + 1.2, f'FP{fp_obj.id}', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Add consistent legend (matching GIF format)
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=12, label='Ground Truth Objects'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', markeredgecolor='black', 
                   markeredgewidth=2, markersize=12, label='False Positive Objects'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Legitimate Robots'), 
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Adversarial Robots'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
        
        # Add info box (matching GIF format)
        info_text = f"Ground Truth Objects: {len(self.ground_truth_objects)}\n"
        
        # Count FP objects
        total_fp_objects = len(self.shared_fp_objects)
        info_text += f"False Positive Objects: {total_fp_objects}\n"
        
        # Add robot trust info
        if leg_trusts:
            info_text += f"Avg LEG Trust: {np.mean(leg_trusts):.2f}\n"
        if adv_trusts:
            info_text += f"Avg ADV Trust: {np.mean(adv_trusts):.2f}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    """Main simulation runner demonstrating different trust algorithms"""
    print("Initializing Refactored Neural-Symbolic Trust-Based Sensor Fusion Simulation")
    
    # Import paper trust algorithm
    from paper_trust_algorithm import PaperTrustAlgorithm
    
    # Test with paper's algorithm first
    print("\n=== Testing with Paper's Trust Algorithm ===")
    env_paper = SimulationEnvironment(
        world_size=(100.0, 100.0),
        robot_density=0.0005,
        target_density=0.0020
    )
    
    # Initialize paper trust algorithm
    paper_algo = PaperTrustAlgorithm()
    
    # Storage for GIF creation
    simulation_frames = []
     
    # Run simulation with trust algorithm integration
    for step in range(500):
        # Step the simulation
        frame_data = env_paper.step()
        
        # Update current timestep tracks for all robots after step
        for robot in env_paper.robots:
            robot.update_current_timestep_tracks()
        
        # Apply trust algorithm updates using the new simplified interface
        try:
            paper_algo.update_trust(env_paper.robots, env_paper)
        except Exception as e:
            # Handle any trust algorithm errors gracefully
            print(f"Trust algorithm error at step {step}: {e}")
        
        simulation_frames.append(frame_data)
        
    env_paper.visualize_current_state('paper_algorithm_state.png')
    
    # Save collected simulation data for GIF creation
    import json
    with open('trust_simulation_data.json', 'w') as f:
        json.dump(simulation_frames, f, indent=2)
    print(f"Saved {len(simulation_frames)} frames of simulation data to trust_simulation_data.json")
    
    # Print comparison summary
    print("\n=== Comparison Summary ===")
    print("Paper Algorithm Trust Levels:")
    for robot in env_paper.robots:
        trust_mean = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
        robot_type = "ADVERSARIAL" if robot.is_adversarial else "LEGITIMATE"
        print(f"  Robot {robot.id} ({robot_type}): Trust={trust_mean:.3f}")

if __name__ == "__main__":
    main()
