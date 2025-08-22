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
import json

# Import the trust algorithm interfaces
from trust_algorithm import TrustAlgorithm, RobotState, Track
from paper_trust_algorithm import PaperTrustAlgorithm
from neural_symbolic_trust_algorithm import NeuralSymbolicTrustAlgorithm


@dataclass
class GroundTruthObject:
    """Ground truth object with dynamic movement"""
    id: int
    position: np.ndarray     # [x, y, z]
    velocity: np.ndarray     # [vx, vy, vz]
    object_type: str         # 'vehicle', 'person', etc.
    movement_pattern: str    # 'linear', 'random_walk', 'circular', 'stationary'
    spawn_time: float        # When object first appeared
    lifespan: float          # How long object exists (-1 for permanent)
    
    # Movement parameters
    base_speed: float = 1.0
    turn_probability: float = 0.02  # For random walk
    
    # Circular motion parameters (only used for circular movement)
    circular_center: np.ndarray = None  # Center of circular motion
    circular_radius: float = 0.0        # Radius of circular motion
    direction_change_time: float = 5.0  # For direction changes


class SimulationEnvironment:
    """Multi-robot simulation environment with pluggable trust algorithms"""
    
    def __init__(self, num_robots: int = 5, num_targets: int = 10, 
                 world_size: Tuple[float, float] = (100.0, 100.0),
                 adversarial_ratio: float = 0.3,
                 trust_algorithm: Optional[TrustAlgorithm] = None):
        """
        Initialize simulation environment
        
        Args:
            num_robots: Number of robots in the simulation
            num_targets: Number of ground truth objects
            world_size: Size of the simulation world
            adversarial_ratio: Fraction of robots that are adversarial
            trust_algorithm: Trust algorithm to use (defaults to paper's method)
        """
        self.num_robots = num_robots
        self.num_targets = num_targets
        self.world_size = world_size
        self.adversarial_ratio = adversarial_ratio
        
        # Initialize trust algorithm (default to paper's method)
        if trust_algorithm is None:
            self.trust_algorithm = PaperTrustAlgorithm()
        else:
            self.trust_algorithm = trust_algorithm
        
        self.robots: List[RobotState] = []
        self.ground_truth_objects: List[GroundTruthObject] = []
        self.tracks: Dict[int, List[Track]] = {}  # Robot ID -> list of tracks
        self.raw_tracks: Dict[int, List[Track]] = {}  # Robot ID -> raw unfiltered tracks
        
        # Per-robot object track management
        self.robot_object_tracks: Dict[int, Dict[str, Track]] = {}  # robot_id -> {object_id: Track}
        
        self.time = 0.0
        self.dt = 0.1  # 10Hz simulation rate
        self.next_object_id = 0  # For generating unique object IDs
        
        # FP objects management for adversarial robots
        self.fp_objects_by_robot: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
        self.fp_next_id_by_robot: Dict[int, int] = {}
        
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize robots and targets in the environment"""
        # Initialize robots at random positions
        num_adversarial = int(self.num_robots * self.adversarial_ratio)
        adversarial_ids = set(random.sample(range(self.num_robots), num_adversarial))
        
        for i in range(self.num_robots):
            # Initialize per-robot object tracking structures
            self.robot_object_tracks[i] = {}
            self.fp_objects_by_robot[i] = {}
            self.fp_next_id_by_robot[i] = 0
            
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
            
            robot = RobotState(
                id=i,
                position=start_pos.copy(),
                velocity=np.array([0.0, 0.0, 0.0]),
                orientation=0.0,
                fov_range=20.0,
                fov_angle=np.pi/3,  # 60 degrees
                is_adversarial=(i in adversarial_ids),
                start_position=start_pos,
                goal_position=goal_pos,
                patrol_speed=random.uniform(1.5, 2.5)
            )
            
            # Initialize velocity and orientation toward goal
            self._update_robot_navigation(robot)
            
            self.robots.append(robot)
            self.tracks[i] = []
            self.raw_tracks[i] = []
        
        # Initialize ground truth objects
        for i in range(self.num_targets):
            obj = self._create_ground_truth_object(self.time)
            self.ground_truth_objects.append(obj)
        
        # Initialize trust algorithm
        self.trust_algorithm.initialize(self.robots)
        
        print(f"Simulation initialized with {len(self.robots)} robots and {len(self.ground_truth_objects)} objects")
        print(f"Using trust algorithm: {type(self.trust_algorithm).__name__}")
    
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
        
        # Lifespan
        if random.random() < 0.7:  # 70% permanent objects
            lifespan = -1
        else:  # 30% temporary objects
            lifespan = random.uniform(10, 30)
        
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
            lifespan=lifespan,
            base_speed=base_speed,
            turn_probability=random.uniform(0.01, 0.05),
            direction_change_time=random.uniform(3, 8),
            circular_center=circular_center_param,
            circular_radius=circular_radius_param
        )
    
    def _update_ground_truth_objects(self):
        """Update positions and states of all ground truth objects"""
        objects_to_remove = []
        
        for obj in self.ground_truth_objects:
            # Check if object should be removed (lifespan expired)
            if obj.lifespan > 0 and (self.time - obj.spawn_time) > obj.lifespan:
                objects_to_remove.append(obj)
                continue
            
            # Update object based on movement pattern
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
        
        # Remove expired objects
        for obj in objects_to_remove:
            self.ground_truth_objects.remove(obj)
        
        # Occasionally spawn new objects
        if random.random() < 0.05:  # 5% chance per step
            if len(self.ground_truth_objects) < self.num_targets * 2:
                new_obj = self._create_ground_truth_object(self.time)
                self.ground_truth_objects.append(new_obj)
    
    def _update_robot_navigation(self, robot: RobotState):
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
            robot.velocity = direction_normalized * robot.patrol_speed
            robot.orientation = np.arctan2(direction_normalized[1], direction_normalized[0])
        else:
            # Reached target, stop and switch direction
            robot.velocity = np.array([0.0, 0.0, 0.0])
            robot.is_returning_home = not robot.is_returning_home
    
    def is_in_fov(self, robot: RobotState, target_pos: np.ndarray) -> bool:
        """Check if target is within robot's field of view"""
        # Calculate relative position
        rel_pos = target_pos - robot.position
        distance = np.linalg.norm(rel_pos[:2])  # 2D distance
        
        if distance > robot.fov_range:
            return False
        
        # Check angle constraint
        target_angle = np.arctan2(rel_pos[1], rel_pos[0])
        angle_diff = abs(target_angle - robot.orientation)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Wrap around
        
        return angle_diff <= robot.fov_angle / 2
    
    def generate_detections(self, robot: RobotState, noise_std: float = 1.0) -> List[Track]:
        """Generate detections for a robot based on whether it's legitimate or adversarial"""
        if robot.is_adversarial:
            return self._generate_adversarial_detections(robot, noise_std)
        else:
            return self._generate_legitimate_detections(robot, noise_std)
    
    def _generate_legitimate_detections(self, robot: RobotState, noise_std: float) -> List[Track]:
        """Generate perfect detections for legitimate robots"""
        detections = []
        
        for gt_obj in self.ground_truth_objects:
            if self.is_in_fov(robot, gt_obj.position):
                object_id = f"gt_obj_{gt_obj.id}"
                
                # Legitimate robots detect with minimal noise
                noisy_pos = gt_obj.position + np.random.normal(0, noise_std * 0.5, 3)
                noisy_vel = gt_obj.velocity + np.random.normal(0, 0.2, 3)
                covariance = np.eye(6) * (noise_std * 0.5)**2
                
                # Get or initialize track trust for this robot-object pair
                if object_id in self.trust_algorithm.robot_object_trust[robot.id]:
                    trust_alpha, trust_beta = self.trust_algorithm.robot_object_trust[robot.id][object_id]
                else:
                    # Initialize with neutral Beta(1,1) prior
                    trust_alpha = 1.0
                    trust_beta = 1.0
                    self.trust_algorithm.robot_object_trust[robot.id][object_id] = (trust_alpha, trust_beta)
                
                # Create track ID
                track_key = f"{robot.id}_{object_id}"
                
                track = Track(
                    id=track_key,
                    position=noisy_pos,
                    velocity=noisy_vel,
                    covariance=covariance,
                    confidence=random.uniform(0.85, 0.98),
                    timestamp=self.time,
                    source_robot=robot.id,
                    trust_alpha=trust_alpha,
                    trust_beta=trust_beta,
                    object_id=object_id
                )
                
                # Update per-robot object track registry
                self.robot_object_tracks[robot.id][object_id] = track
                detections.append(track)
        
        return detections
    
    def _generate_adversarial_detections(self, robot: RobotState, noise_std: float) -> List[Track]:
        """Generate malicious detections for adversarial robots"""
        tracks = []
        
        # Parameters for adversarial behavior
        false_negative_rate = 0.0
        false_positive_rate = 1.0
        position_noise_std = noise_std * 0.5
        velocity_noise_std = 0.2
        
        # True objects in FOV
        true_objects_in_fov = [gt for gt in self.ground_truth_objects if self.is_in_fov(robot, gt.position)]
        
        # Generate tracks for true objects (with possible false negatives)
        for gt_obj in true_objects_in_fov:
            if random.random() < false_negative_rate:
                continue
            
            object_id = f"gt_obj_{gt_obj.id}"
            noisy_pos = gt_obj.position + np.random.normal(0, position_noise_std, 3)
            noisy_vel = gt_obj.velocity + np.random.normal(0, velocity_noise_std, 3)
            covariance = np.eye(6) * position_noise_std**2
            
            # Get or initialize track trust for this robot-object pair
            if object_id in self.trust_algorithm.robot_object_trust[robot.id]:
                trust_alpha, trust_beta = self.trust_algorithm.robot_object_trust[robot.id][object_id]
            else:
                trust_alpha = 1.0
                trust_beta = 1.0
                self.trust_algorithm.robot_object_trust[robot.id][object_id] = (trust_alpha, trust_beta)
            
            track_key = f"{robot.id}_{object_id}"
            
            track = Track(
                id=track_key,
                position=noisy_pos,
                velocity=noisy_vel,
                covariance=covariance,
                confidence=random.uniform(0.85, 0.98),
                timestamp=self.time,
                source_robot=robot.id,
                trust_alpha=trust_alpha,
                trust_beta=trust_beta,
                object_id=object_id
            )
            
            self.robot_object_tracks[robot.id][object_id] = track
            tracks.append(track)
        
        # Generate persistent false positives
        target_num_fp_in_fov = int(len(true_objects_in_fov) * false_positive_rate)
        fp_store = self.fp_objects_by_robot[robot.id]
        
        # Count current FPs in FOV
        fp_in_fov = [fp_id for fp_id, state in fp_store.items() 
                     if self.is_in_fov(robot, state['position'])]
        current_fp_in_fov = len(fp_in_fov)
        
        # Cleanup: Remove excess FPs, prioritizing out-of-FOV ones
        max_total_fps = target_num_fp_in_fov + 3  # Allow some FPs to exist outside FOV for persistence
        if len(fp_store) > max_total_fps:
            # Prioritize removing FPs that are out of FOV first
            fp_out_of_fov = [fp_id for fp_id, state in fp_store.items() 
                            if not self.is_in_fov(robot, state['position'])]
            to_drop = fp_out_of_fov[:len(fp_store) - max_total_fps]
            
            # If we still need to drop more, remove some in-FOV ones
            if len(to_drop) < len(fp_store) - max_total_fps:
                remaining_to_drop = len(fp_store) - max_total_fps - len(to_drop)
                to_drop.extend(random.sample(fp_in_fov, min(remaining_to_drop, len(fp_in_fov))))
            
            for oid in to_drop:
                fp_store.pop(oid, None)
        
        # Recalculate FP counts after cleanup
        fp_in_fov = [fp_id for fp_id, state in fp_store.items() 
                     if self.is_in_fov(robot, state['position'])]
        current_fp_in_fov = len(fp_in_fov)
        
        # Spawn new FPs if needed
        while current_fp_in_fov < target_num_fp_in_fov:
            # Create FP in robot's FOV
            angle = robot.orientation + random.uniform(-robot.fov_angle/2, robot.fov_angle/2)
            distance = random.uniform(3, robot.fov_range)
            new_pos = robot.position + np.array([
                distance*np.cos(angle), 
                distance*np.sin(angle), 
                random.uniform(-5, 5)
            ])
            
            # Check it's not too close to true objects
            if all(np.linalg.norm(new_pos - gt.position) >= 3.0 for gt in true_objects_in_fov):
                speed = random.uniform(0.2, 1.5)
                theta = random.uniform(0, 2*np.pi)
                vel = np.array([speed*np.cos(theta), speed*np.sin(theta), random.uniform(-0.2, 0.2)])
                
                fp_id = f"fp_{robot.id}_{self.fp_next_id_by_robot[robot.id]}"
                self.fp_next_id_by_robot[robot.id] += 1
                fp_store[fp_id] = {'position': new_pos, 'velocity': vel}
                current_fp_in_fov += 1
        
        # Update FP dynamics and emit tracks for FPs in FOV
        for fp_id, state in list(fp_store.items()):
            pos = state['position']
            vel = state['velocity']
            
            # Random steering
            if random.random() < 0.1:
                steer = random.uniform(-np.pi/6, np.pi/6)
                rot = np.array([[np.cos(steer), -np.sin(steer), 0],
                                [np.sin(steer),  np.cos(steer), 0],
                                [0, 0, 1]])
                vel = rot @ vel
                state['velocity'] = vel
            
            # Propagate position
            pos = pos + vel * self.dt
            
            # Boundary handling
            if pos[0] <= 0 or pos[0] >= self.world_size[0]: vel[0] *= -1
            if pos[1] <= 0 or pos[1] >= self.world_size[1]: vel[1] *= -1
            pos[0] = np.clip(pos[0], 0, self.world_size[0])
            pos[1] = np.clip(pos[1], 0, self.world_size[1])
            
            state['position'] = pos
            state['velocity'] = vel
            
            # Emit track only if in FOV
            if self.is_in_fov(robot, pos):
                fp_track_key = f"{robot.id}_{fp_id}"
                
                # Get or initialize track trust for this robot-FP pair
                if fp_id in self.trust_algorithm.robot_object_trust[robot.id]:
                    trust_alpha, trust_beta = self.trust_algorithm.robot_object_trust[robot.id][fp_id]
                else:
                    trust_alpha = 1.0
                    trust_beta = 1.0
                    self.trust_algorithm.robot_object_trust[robot.id][fp_id] = (trust_alpha, trust_beta)
                
                track = Track(
                    id=fp_track_key,
                    position=pos.copy(),
                    velocity=vel.copy(),
                    covariance=np.eye(6) * 1e-6,
                    confidence=random.uniform(0.95, 0.99),
                    timestamp=self.time,
                    source_robot=robot.id,
                    trust_alpha=trust_alpha,
                    trust_beta=trust_beta,
                    object_id=fp_id
                )
                
                self.robot_object_tracks[robot.id][fp_id] = track
                tracks.append(track)
        
        return tracks
    
    def step(self) -> Dict:
        """Execute one simulation step"""
        # Update ground truth objects
        self._update_ground_truth_objects()
        
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
        
        # Generate detections for each robot
        for robot in self.robots:
            raw_detections = self.generate_detections(robot)
            self.raw_tracks[robot.id] = raw_detections
            
            # Convert per-robot object tracks to list format
            self.tracks[robot.id] = list(self.robot_object_tracks[robot.id].values())
        
        # Update trust using the pluggable trust algorithm
        trust_updates = self.trust_algorithm.update_trust(
            self.robots, self.tracks, self.robot_object_tracks, self.time
        )
        
        self.time += self.dt
        
        return {
            'time': self.time,
            'robot_states': [(r.id, r.position.tolist(), r.velocity.tolist(), r.is_adversarial) 
                           for r in self.robots],
            'tracks': {robot_id: [(t.id, t.position.tolist(), t.confidence, t.trust_alpha, t.trust_beta) 
                                for t in tracks] 
                      for robot_id, tracks in self.tracks.items()},
            'trust_updates': trust_updates,
            'trust_algorithm': type(self.trust_algorithm).__name__,
            'ground_truth': {
                'objects': [(obj.id, obj.position.tolist(), obj.velocity.tolist(), 
                           obj.object_type, obj.movement_pattern) for obj in self.ground_truth_objects],
                'adversarial_robots': [r.id for r in self.robots if r.is_adversarial],
                'legitimate_robots': [r.id for r in self.robots if not r.is_adversarial],
                'num_objects': len(self.ground_truth_objects)
            },
            'fp_objects': {robot_id: [(fp_id, state['position'].tolist(), state['velocity'].tolist()) 
                                     for fp_id, state in fp_store.items()]
                          for robot_id, fp_store in self.fp_objects_by_robot.items() if fp_store}
        }
    
    def collect_training_data(self, num_steps: int = 1000) -> List[Dict]:
        """Collect training data for neural symbolic learning"""
        training_data = []
        
        for _ in range(num_steps):
            step_data = self.step()
            training_data.append(step_data)
        
        return training_data
    
    def save_data(self, data: List[Dict], filename: str):
        """Save collected data to file"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} simulation steps to {filename}")
    
    def switch_trust_algorithm(self, new_algorithm: TrustAlgorithm):
        """Switch to a different trust algorithm"""
        print(f"Switching from {type(self.trust_algorithm).__name__} to {type(new_algorithm).__name__}")
        self.trust_algorithm = new_algorithm
        self.trust_algorithm.initialize(self.robots)
    
    def visualize_current_state(self, save_path: Optional[str] = None):
        """Visualize current simulation state matching create_simulation_gif.py format"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
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
        
        title_parts = [f'Trust-Based Sensor Fusion ({type(self.trust_algorithm).__name__}) - t={self.time:.1f}s']
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
            
            # Plot false positive objects (orange stars, same as GT but different color)
            if robot.id in self.fp_objects_by_robot:
                for fp_id, fp_state in self.fp_objects_by_robot[robot.id].items():
                    fp_pos = fp_state['position'][:2]  # Only x, y coordinates
                    
                    # FP Objects: Use same star symbol as GT objects but orange color
                    ax.scatter(fp_pos[0], fp_pos[1], 
                              c='orange', marker='*', s=300, alpha=0.9, 
                              edgecolors='black', linewidth=2)
                    
                    # Add FP object ID label (similar to GT objects)
                    fp_id_clean = fp_id.replace(f'fp_{robot.id}_', '')
                    ax.text(fp_pos[0], fp_pos[1] + 1.2, f'FP{fp_id_clean}', 
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
        total_fp_objects = sum(len(fp_store) for fp_store in self.fp_objects_by_robot.values())
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
    
    # Test with paper's algorithm first
    print("\n=== Testing with Paper's Trust Algorithm ===")
    paper_algorithm = PaperTrustAlgorithm()
    env_paper = SimulationEnvironment(
        num_robots=5, num_targets=20, world_size=(50.0, 50.0), 
        trust_algorithm=paper_algorithm
    )
    
    # Collect some data (using 500 steps like original)
    paper_data = env_paper.collect_training_data(num_steps=500)
    env_paper.save_data(paper_data, 'paper_trust_data.json')
    env_paper.visualize_current_state('paper_algorithm_state.png')
    
    """
    # Test with neural symbolic algorithm
    print("\n=== Testing with Neural Symbolic Trust Algorithm ===")
    neural_algorithm = NeuralSymbolicTrustAlgorithm()
    env_neural = SimulationEnvironment(
        num_robots=5, num_targets=20, world_size=(50.0, 50.0), 
        trust_algorithm=neural_algorithm
    )
    
    # Collect data with neural symbolic algorithm
    neural_data = env_neural.collect_training_data(num_steps=500)
    env_neural.save_data(neural_data, 'neural_symbolic_trust_data.json')
    env_neural.visualize_current_state('neural_symbolic_algorithm_state.png')
    """
    # Print comparison summary
    print("\n=== Comparison Summary ===")
    print("Paper Algorithm Trust Levels:")
    for robot in env_paper.robots:
        trust_mean = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
        robot_type = "ADVERSARIAL" if robot.is_adversarial else "LEGITIMATE"
        print(f"  Robot {robot.id} ({robot_type}): Trust={trust_mean:.3f}")
    """
    print("\nNeural Symbolic Algorithm Trust Levels:")
    for robot in env_neural.robots:
        trust_mean = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
        robot_type = "ADVERSARIAL" if robot.is_adversarial else "LEGITIMATE"
        print(f"  Robot {robot.id} ({robot_type}): Trust={trust_mean:.3f}")
    
    print("\nRefactoring completed successfully!")
    print("You can now:")
    print("1. Compare performance between different trust algorithms")
    print("2. Train neural symbolic models on paper algorithm data")
    print("3. Implement custom trust algorithms by extending TrustAlgorithm")
    """

if __name__ == "__main__":
    main()