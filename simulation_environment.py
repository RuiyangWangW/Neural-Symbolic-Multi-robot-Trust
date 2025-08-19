#!/usr/bin/env python3
"""
Neural-Symbolic Trust-Based Sensor Fusion Simulation Environment

This simulation environment replicates the multi-robot trust-based sensor fusion 
approach from the original paper, designed to collect data for training neural 
symbolic methods to replace hand-designed logical rules.

Based on: "Trust-Based Assured Sensor Fusion in Distributed Aerial Autonomy"
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
from scipy.stats import beta
from scipy.spatial.distance import euclidean
# from scipy.optimize import linear_sum_assignment  # No longer needed with simplified approach
import json
from collections import Counter


@dataclass
class RobotState:
    """Robot state including position, velocity, and sensor capabilities"""
    id: int
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    orientation: float    # heading angle
    fov_range: float     # sensor range
    fov_angle: float     # field of view angle (radians)
    is_adversarial: bool = False  # Ground truth: is this robot adversarial?
    trust_alpha: float = 1.0  # Beta distribution alpha parameter
    trust_beta: float = 1.0   # Beta distribution beta parameter
    
    # Patrol pattern attributes
    start_position: np.ndarray = None  # Home base position
    goal_position: np.ndarray = None   # Current patrol goal
    patrol_speed: float = 2.0          # Movement speed
    is_returning_home: bool = False    # Direction of patrol
    position_tolerance: float = 3.0    # Distance tolerance to consider goal reached


@dataclass
class Track:
    """Object track with state and covariance"""
    id: int
    position: np.ndarray     # [x, y, z]
    velocity: np.ndarray     # [vx, vy, vz]
    covariance: np.ndarray   # state covariance matrix
    confidence: float        # detection confidence
    timestamp: float
    source_robot: int
    
    # Track trust parameters (Beta distribution)
    trust_alpha: float = 1.0
    trust_beta: float = 1.0

    # NEW: persistent fused object identity
    object_id: Optional[str] = None

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

@dataclass
class DataAggregator:
    """Centralized data aggregator for weighted object state estimation"""
    robots: List[RobotState]
    tracks_by_robot: Dict[int, List[Track]]

    # Simple object trust store: object_id -> {'alpha': float, 'beta': float}
    object_trust: Dict[str, Dict] = None

    def __post_init__(self):
        if self.object_trust is None:
            self.object_trust = {}

    def get_or_create_object_trust(self, object_id: str, seed_alpha: float = 1.0, seed_beta: float = 1.0) -> Tuple[float, float]:
        """Get or create trust parameters for an object ID"""
        if object_id not in self.object_trust:
            self.object_trust[object_id] = {
                'alpha': float(seed_alpha),
                'beta': float(seed_beta)
            }
        return self.object_trust[object_id]['alpha'], self.object_trust[object_id]['beta']
    
    def update_object_trust(self, object_id: str, alpha: float, beta: float):
        """Update trust parameters for an object"""
        if object_id not in self.object_trust:
            self.object_trust[object_id] = {}
        self.object_trust[object_id]['alpha'] = alpha
        self.object_trust[object_id]['beta'] = beta

    def select_object_id_from_group(self, group_tracks: List[Track]) -> Optional[str]:
        """Select object ID from group - simply use most common ID"""
        group_ids = [t.object_id for t in group_tracks if t.object_id]
        if not group_ids:
            return None
            
        # Use most frequent object ID
        counts = Counter(group_ids)
        most_common_id = counts.most_common(1)[0][0]
        return most_common_id

    def get_object_id_and_trust(self, group_tracks: List[Track]) -> Tuple[str, float, float]:
        """Get object ID and trust from group tracks - uses direct object IDs from tracks"""
        # Simply use the object ID from tracks (already assigned during detection)
        object_id = self.select_object_id_from_group(group_tracks)
        
        if object_id:
            # Get or create trust for this object
            alpha, beta = self.get_or_create_object_trust(
                object_id, 
                seed_alpha=group_tracks[0].trust_alpha, 
                seed_beta=group_tracks[0].trust_beta
            )
            return object_id, alpha, beta
        else:
            # Fallback - shouldn't happen with new approach
            return f"unknown_obj_{random.randint(1000, 9999)}", 1.0, 1.0

    def compute_weighted_object_state(self, track_group: List[Track]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute weighted position and velocity for fused track group"""
        if not track_group:
            return None, None

        weights, positions, velocities = [], [], []
        for track in track_group:
            source_robot = next((r for r in self.robots if r.id == track.source_robot), None)
            if source_robot is None:
                continue
            agent_trust = source_robot.trust_alpha / (source_robot.trust_alpha + source_robot.trust_beta)
            track_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)

            weight = agent_trust * track_trust * track.confidence
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
    ) -> List[Track]:
        """
        Simplified fusion approach using direct object IDs:
        1) Distance-based grouping of tracks within fusion threshold
        2) Use object IDs directly from tracks (assigned during detection)
        3) One fused track per object ID
        """
        ego_tracks = self.tracks_by_robot.get(ego_robot_id, [])
        if not ego_tracks:
            return []

        fused_tracks: List[Track] = []
        processed_object_ids = set()  # Ensure one track per object

        for ego_track in ego_tracks:
            # Trust gate for ego track
            ego_trust = ego_track.trust_alpha / (ego_track.trust_alpha + ego_track.trust_beta)
            if ego_trust < trust_threshold:
                continue

            # Skip if we already processed this object
            if ego_track.object_id in processed_object_ids:
                continue

            # 1) Distance-based grouping with tracks of same object ID
            group: List[Track] = [ego_track]
            for other_robot_id, other_tracks in self.tracks_by_robot.items():
                if other_robot_id == ego_robot_id:
                    continue
                for ot in other_tracks:
                    # Only group tracks with same object ID that are close enough
                    if (ot.object_id == ego_track.object_id and 
                        np.linalg.norm(ego_track.position - ot.position) <= fusion_threshold):
                        t_trust = ot.trust_alpha / (ot.trust_alpha + ot.trust_beta)
                        if t_trust >= trust_threshold:
                            group.append(ot)

            # 2) Compute fused state (position + velocity)
            fused_pos, fused_vel = self.compute_weighted_object_state(group)
            if fused_pos is None or fused_vel is None:
                # Fallback to ego track
                fused_tracks.append(ego_track)
                processed_object_ids.add(ego_track.object_id)
                continue

            # 3) Get object trust (create if needed)
            object_id, obj_alpha, obj_beta = self.get_object_id_and_trust(group)
            processed_object_ids.add(object_id)

            # 4) Create fused track
            track_id = f"{ego_robot_id}:{object_id}"
            fused_tracks.append(Track(
                id=track_id,
                position=fused_pos,
                velocity=fused_vel,
                covariance=ego_track.covariance,
                confidence=min(1.0, sum(t.confidence for t in group) / len(group)),
                timestamp=ego_track.timestamp,
                source_robot=ego_robot_id,
                trust_alpha=obj_alpha,
                trust_beta=obj_beta,
                object_id=object_id
            ))

        return fused_tracks

class TrustEstimator:
    """Implements the original paper's trust estimation using assignment-based PSM generation"""
    
    def __init__(self, negative_bias: float = 10.0, negative_threshold: float = 0.3):
        # Equation 6 parameters for negatively weighted updates
        self.negative_bias = negative_bias  # B^c_n parameter
        self.negative_threshold = negative_threshold  # T^c_n parameter
    
    def update_agent_trust(self, robot: RobotState, psms: List[Tuple[float, float]]) -> None:
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
            
            robot.trust_alpha += delta_alpha
            robot.trust_beta += delta_beta
            
            # Ensure alpha and beta stay positive
            robot.trust_alpha = robot.trust_alpha
            robot.trust_beta = robot.trust_beta
    
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
            
            track.trust_alpha += delta_alpha
            track.trust_beta += delta_beta
            
            # Ensure alpha and beta stay positive
            track.trust_alpha = max(0.1, track.trust_alpha)
            track.trust_beta = max(0.1, track.trust_beta)
    
    def get_expected_trust(self, alpha: float, beta: float) -> float:
        """Calculate expected value E[trust] = alpha / (alpha + beta)"""
        return alpha / (alpha + beta)
    
    def get_trust_variance(self, alpha: float, beta: float) -> float:
        """Calculate variance V[trust] = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))"""
        denominator = (alpha + beta) ** 2 * (alpha + beta + 1)
        return (alpha * beta) / denominator


class SimulationEnvironment:
    """Multi-robot simulation environment for trust-based sensor fusion"""
    
    def __init__(self, num_robots: int = 5, num_targets: int = 10, 
                 world_size: Tuple[float, float] = (100.0, 100.0),
                 adversarial_ratio: float = 0.3):
        self.num_robots = num_robots
        self.num_targets = num_targets
        self.world_size = world_size
        self.adversarial_ratio = adversarial_ratio  # Fraction of robots that are adversarial
        self.robots: List[RobotState] = []
        self.ground_truth_objects: List[GroundTruthObject] = []  # Dynamic ground truth objects
        self.tracks: Dict[int, List[Track]] = {}  # Robot ID -> list of tracks (filtered/merged)
        self.raw_tracks: Dict[int, List[Track]] = {}  # Robot ID -> raw unfiltered tracks for PSM generation
        self.time = 0.0
        self.dt = 0.1  # 10Hz simulation rate
        self.next_object_id = 0  # For generating unique object IDs
        
        # Track trust registry to maintain trust distributions over time
        self.track_trust_registry: Dict[str, Tuple[float, float]] = {}  # track_id -> (alpha, beta)
        
        # Centralized data aggregator (per paper architecture)
        self.data_aggregator: DataAggregator = None
        
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize robots and targets in the environment"""
        # Initialize robots at random positions
        num_adversarial = int(self.num_robots * self.adversarial_ratio)
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
            
            # Ensure goal is at reasonable distance from start (adaptive to world size)
            min_patrol_distance = min(self.world_size) * 0.3  # 30% of smaller world dimension
            max_attempts = 20  # Prevent infinite loops
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
                velocity=np.array([0.0, 0.0, 0.0]),  # Will be computed based on patrol
                orientation=0.0,  # Will be computed based on direction
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
        
        # Initialize ground truth objects with various movement patterns
        for i in range(self.num_targets):
            obj = self._create_ground_truth_object(self.time)
            self.ground_truth_objects.append(obj)
        
        # Initialize centralized data aggregator (for weighted state estimation only)
        self.data_aggregator = DataAggregator(
            robots=self.robots,
            tracks_by_robot=self.tracks
        )
        self.fp_objects_by_robot: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {
            r.id: {} for r in self.robots
        }
        # Per-robot counter to mint new FP object ids
        self.fp_next_id_by_robot: Dict[int, int] = {r.id: 0 for r in self.robots} 
        # Initialize trust estimator
        self.trust_estimator = TrustEstimator(negative_bias=3.0, negative_threshold=0.5)
    
    def _enforce_global_object_uniqueness(self):
        """Enforce one track per object globally across all robots"""
        # Collect all tracks with their object IDs
        all_tracks_by_object = {}
        
        for robot_id, tracks in self.tracks.items():
            for track in tracks:
                if track.object_id not in all_tracks_by_object:
                    all_tracks_by_object[track.object_id] = []
                all_tracks_by_object[track.object_id].append((robot_id, track))
        
        # For each object, keep only the track from the robot with highest trust
        for object_id, track_list in all_tracks_by_object.items():
            if len(track_list) > 1:
                # Multiple tracks for same object - keep the one from most trusted robot
                best_robot_id, best_track = max(track_list, key=lambda x: self._get_robot_trust(x[0]))
                
                # Remove this object's tracks from all other robots
                for robot_id, track in track_list:
                    if robot_id != best_robot_id:
                        # Remove track from robot's track list
                        self.tracks[robot_id] = [t for t in self.tracks[robot_id] if t.object_id != object_id]
    
    def _get_robot_trust(self, robot_id: int) -> float:
        """Get robot's current trust value"""
        robot = next((r for r in self.robots if r.id == robot_id), None)
        if robot:
            return robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
        return 0.0
    
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
            # Assign individual center and radius for spread-out circular motion
            # Create multiple possible centers across the world
            potential_centers = [
                (self.world_size[0] * 0.2, self.world_size[1] * 0.2),  # Lower left
                (self.world_size[0] * 0.8, self.world_size[1] * 0.2),  # Lower right  
                (self.world_size[0] * 0.2, self.world_size[1] * 0.8),  # Upper left
                (self.world_size[0] * 0.8, self.world_size[1] * 0.8),  # Upper right
                (self.world_size[0] * 0.5, self.world_size[1] * 0.2),  # Bottom center
                (self.world_size[0] * 0.5, self.world_size[1] * 0.8),  # Top center
                (self.world_size[0] * 0.2, self.world_size[1] * 0.5),  # Left center
                (self.world_size[0] * 0.8, self.world_size[1] * 0.5),  # Right center
            ]
            center_x, center_y = random.choice(potential_centers)
            circular_center = np.array([center_x, center_y, 0.0])
            circular_radius = random.uniform(5, 12)  # Varied radius sizes
            
            # Set initial position on the circle
            angle = random.uniform(0, 2 * np.pi)
            position = circular_center + circular_radius * np.array([
                np.cos(angle), np.sin(angle), 0.0
            ])
            
            # Will be computed dynamically during updates
            velocity = np.array([base_speed, 0.0, 0.0])
        else:  # random_walk or stationary
            velocity = np.array([0.0, 0.0, 0.0])
        
        # Object type
        object_types = ['vehicle', 'person', 'animal']
        object_type = random.choice(object_types)
        
        # Lifespan (-1 for permanent, or random duration)
        if random.random() < 0.7:  # 70% permanent objects
            lifespan = -1
        else:  # 30% temporary objects
            lifespan = random.uniform(10, 30)  # 10-30 seconds
        
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
                # Stationary objects don't move
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
                    # Change direction
                    angle = random.uniform(0, 2*np.pi)
                    obj.velocity = np.array([
                        obj.base_speed * np.cos(angle),
                        obj.base_speed * np.sin(angle),
                        0.0
                    ])
                
                # Move and handle boundaries
                new_pos = obj.position + obj.velocity * self.dt
                if (new_pos[0] <= 0 or new_pos[0] >= self.world_size[0] or 
                    new_pos[1] <= 0 or new_pos[1] >= self.world_size[1]):
                    # Turn around at boundaries
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
                # Circular movement around individual center
                if obj.circular_center is not None:
                    center = obj.circular_center
                    radius = obj.circular_radius
                    angular_velocity = obj.base_speed / radius
                    
                    angle = angular_velocity * (self.time - obj.spawn_time)
                    obj.position = center + radius * np.array([
                        np.cos(angle), np.sin(angle), 0.0
                    ])
                else:
                    # Fallback to old behavior if center not set
                    center = np.array([self.world_size[0]/2, self.world_size[1]/2, 0.0])
                    radius = min(self.world_size) * 0.3
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
            if len(self.ground_truth_objects) < self.num_targets * 2:  # Don't exceed 2x initial
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
            
            # Update orientation to face movement direction
            robot.orientation = np.arctan2(direction_normalized[1], direction_normalized[0])
        else:
            # Reached target, stop and switch direction
            robot.velocity = np.array([0.0, 0.0, 0.0])
            robot.is_returning_home = not robot.is_returning_home
            
            # Add small random pause at waypoints to prevent immediate re-navigation
            if random.random() < 0.3:  # 30% chance to pause for one step
                return
            
            # Set velocity toward new target without recursive call
            new_target = robot.start_position if robot.is_returning_home else robot.goal_position
            new_direction = new_target - robot.position
            new_distance = np.linalg.norm(new_direction)
            
            if new_distance > robot.position_tolerance:
                new_direction_normalized = new_direction / new_distance
                robot.velocity = new_direction_normalized * robot.patrol_speed
                robot.orientation = np.arctan2(new_direction_normalized[1], new_direction_normalized[0])
            else:
                # Still at target after switching, just stop
                robot.velocity = np.array([0.0, 0.0, 0.0])
    
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
        detections = []
        track_id = 0
        
        if robot.is_adversarial:
            # Adversarial robots can report false tracks or ignore existing ones
            return self._generate_adversarial_detections(robot, track_id, noise_std)
        else:
            # Legitimate robots report all true tracks in FOV with minor noise
            return self._generate_legitimate_detections(robot, track_id, noise_std)
    
    def _generate_legitimate_detections(self, robot: RobotState, track_id: int, 
                                      noise_std: float) -> List[Track]:
        """Generate perfect detections for legitimate robots (all true targets in FOV)"""
        detections = []
        
        for gt_obj in self.ground_truth_objects:
            if self.is_in_fov(robot, gt_obj.position):
                # Legitimate robots always detect true targets with minimal noise
                noisy_pos = gt_obj.position + np.random.normal(0, noise_std * 0.5, 3)  # Less noise
                # Use actual object velocity with small noise
                noisy_vel = gt_obj.velocity + np.random.normal(0, 0.2, 3)  # Minimal velocity noise
                
                # High confidence covariance for legitimate detections
                covariance = np.eye(6) * (noise_std * 0.5)**2
                
                object_track_key = f"{robot.id}_gt_obj_{gt_obj.id}"  # Persistent key for same robot-object pair
                
                # Get or initialize track trust from registry using object-based key
                if object_track_key in self.track_trust_registry:
                    trust_alpha, trust_beta = self.track_trust_registry[object_track_key]
                else:
                    # Initialize with neutral Beta(1,1) prior
                    trust_alpha = 1.0
                    trust_beta = 1.0
                    self.track_trust_registry[object_track_key] = (trust_alpha, trust_beta)
                
                track = Track(
                    id=object_track_key,  # Use persistent object-based ID
                    position=noisy_pos,
                    velocity=noisy_vel,
                    covariance=covariance,
                    confidence=random.uniform(0.85, 0.98),  # High confidence
                    timestamp=self.time,
                    source_robot=robot.id,
                    trust_alpha=trust_alpha,
                    trust_beta=trust_beta,
                    object_id=f"gt_obj_{gt_obj.id}"  # Direct object ID from ground truth
                )
                detections.append(track)
                track_id += 1
        
        return detections
    
    def _generate_adversarial_detections(self, robot: RobotState, track_id: int, noise_std: float) -> List[Track]:
        """Generate malicious detections for adversarial robots with persistent FPs."""
        tracks = []

        # Adversarial behavior parameters
        false_negative_rate = 0.1
        false_positive_rate = 0.5 # Fraction of true-in-FOV used to size FP set
        # Use same detection quality as legitimate robots
        position_noise_std = noise_std * 0.5  # Same as legitimate robots
        velocity_noise_std = 0.2              # Same as legitimate robots

        # Collect true objects in FOV
        true_objects_in_fov = [gt for gt in self.ground_truth_objects if self.is_in_fov(robot, gt.position)]

        # --- TRUE objects (with possible FN) ---
        for gt_obj in true_objects_in_fov:
            if random.random() < false_negative_rate:
                continue  # skip some true objects (FN)
            # Use same noise levels as legitimate robots for GT object detections
            noisy_pos = gt_obj.position + np.random.normal(0, position_noise_std, 3)
            noisy_vel = gt_obj.velocity + np.random.normal(0, velocity_noise_std, 3)
            covariance = np.eye(6) * position_noise_std**2

            object_track_key = f"{robot.id}_gt_obj_{gt_obj.id}"  # Persistent key for same robot-object pair
            
            # Track trust init / reuse using object-based key
            trust_alpha, trust_beta = self.track_trust_registry.get(object_track_key, (1.0, 1.0))
            self.track_trust_registry.setdefault(object_track_key, (trust_alpha, trust_beta))

            tracks.append(Track(
                id=object_track_key,  # Use persistent object-based ID
                position=noisy_pos,
                velocity=noisy_vel,
                covariance=covariance,
                confidence=random.uniform(0.85, 0.98),  # Same high confidence as legitimate robots
                timestamp=self.time,
                source_robot=robot.id,
                trust_alpha=trust_alpha,
                trust_beta=trust_beta,
                object_id=f"gt_obj_{gt_obj.id}"  # Direct object ID from ground truth
            ))
            track_id += 1

        # --- PERSISTENT FALSE POSITIVES (FPs) ---
        # Target FP "capacity" based on nearby truth; keep consistent across steps
        target_num_fp = int(len(true_objects_in_fov) * false_positive_rate)

        fp_store = self.fp_objects_by_robot[robot.id]
        # Prune FPs that are far behind / unlikely: keep size near target_num_fp (+/- 1)
        # (Simple policy: if too many FPs, drop random ones)
        if len(fp_store) > target_num_fp + 1:
            to_drop = random.sample(list(fp_store.keys()), len(fp_store) - (target_num_fp + 1))
            for oid in to_drop:
                fp_store.pop(oid, None)

        # Spawn new FP objects if needed
        while len(fp_store) < target_num_fp:
            # Create an FP in robot FOV, not too close to existing GT in FOV
            attempts = 0
            new_pos = None
            while attempts < 20:
                angle = robot.orientation + random.uniform(-robot.fov_angle/2, robot.fov_angle/2)
                distance = random.uniform(3, robot.fov_range)
                candidate = robot.position + np.array([distance*np.cos(angle), distance*np.sin(angle), random.uniform(-5, 5)])

                if all(np.linalg.norm(candidate - gt.position) >= 3.0 for gt in true_objects_in_fov):
                    new_pos = candidate
                    break
                attempts += 1
            if new_pos is None:
                break  # couldn't place; skip

            # Random velocity for FP, modest magnitude
            speed = random.uniform(0.2, 1.5)
            theta = random.uniform(0, 2*np.pi)
            vel = np.array([speed*np.cos(theta), speed*np.sin(theta), random.uniform(-0.2, 0.2)])

            fp_id = f"fp_{robot.id}_{self.fp_next_id_by_robot[robot.id]}"
            self.fp_next_id_by_robot[robot.id] += 1
            fp_store[fp_id] = {'position': new_pos, 'velocity': vel}

        # Update FP dynamics, keep them mostly within FOV/world, and emit detections
        for fp_oid, state in list(fp_store.items()):
            pos = state['position']
            vel = state['velocity']

            # Random small steering to keep them alive / not ballistic
            if random.random() < 0.1:
                steer = random.uniform(-np.pi/6, np.pi/6)
                rot = np.array([[np.cos(steer), -np.sin(steer), 0],
                                [np.sin(steer),  np.cos(steer), 0],
                                [0, 0, 1]])
                vel = rot @ vel
                state['velocity'] = vel

            # Propagate
            pos = pos + vel * self.dt

            # Soft boundary handling (world box)
            if pos[0] <= 0 or pos[0] >= self.world_size[0]: vel[0] *= -1
            if pos[1] <= 0 or pos[1] >= self.world_size[1]: vel[1] *= -1
            pos[0] = np.clip(pos[0], 0, self.world_size[0])
            pos[1] = np.clip(pos[1], 0, self.world_size[1])

            state['position'] = pos
            state['velocity'] = vel

            # Emit FP detection only if still in FOV
            if not self.is_in_fov(robot, pos):
                continue

            # Use persistent FP object ID as track key for continuous trust
            fp_track_key = f"{robot.id}_{fp_oid}"  # Persistent key for same robot-FP pair
            trust_alpha, trust_beta = self.track_trust_registry.get(fp_track_key, (1.0, 1.0))
            self.track_trust_registry.setdefault(fp_track_key, (trust_alpha, trust_beta))

            tracks.append(Track(
                id=fp_track_key,  # Use persistent FP-based ID
                position=pos.copy(),  # Perfect accuracy - no measurement noise for fabricated objects
                velocity=vel.copy(),  # Perfect accuracy - no measurement noise for fabricated objects  
                covariance=np.eye(6) * 1e-6,  # Very low uncertainty - adversary knows exact state
                confidence=random.uniform(0.95, 0.99),  # Very high confidence for fabricated objects
                timestamp=self.time,
                source_robot=robot.id,
                trust_alpha=trust_alpha,
                trust_beta=trust_beta,
                object_id=fp_oid  # Persistent FP object ID maintained by adversary
            ))
            track_id += 1

        return tracks

    
    def merge_overlapping_tracks(self):
        """Merge spatially overlapping tracks, keeping the one with highest mean trust"""
        for robot_id in self.tracks:
            robot_tracks = self.tracks[robot_id]
            if len(robot_tracks) <= 1:
                continue
            
            merged_tracks = []
            remaining_tracks = robot_tracks.copy()
            
            while remaining_tracks:
                # Take first track as reference
                current_track = remaining_tracks.pop(0)
                overlapping_tracks = [current_track]
                
                # Find all tracks that overlap with current track
                i = 0
                while i < len(remaining_tracks):
                    other_track = remaining_tracks[i]
                    distance = np.linalg.norm(current_track.position - other_track.position)
                    
                    if distance <= 1.5:  # Conservative threshold considering detection noise (~3σ)
                        overlapping_tracks.append(remaining_tracks.pop(i))
                    else:
                        i += 1
                
                # If multiple overlapping tracks, keep the one with highest mean trust
                if len(overlapping_tracks) > 1:
                    best_track = None
                    best_trust = -1
                    
                    for track in overlapping_tracks:
                        track_trust = track.trust_alpha / (track.trust_alpha + track.trust_beta)
                        if track_trust > best_trust:
                            best_trust = track_trust
                            best_track = track
                    
                    merged_tracks.append(best_track)
                else:
                    merged_tracks.append(current_track)
            
            self.tracks[robot_id] = merged_tracks
    
    def perform_assignment_based_trust_estimation(self) -> Dict[int, Dict]:
        """Perform trust estimation using assignment-based PSM generation as per paper"""
        trust_updates = {}
        
        # Store PSMs for each robot to be applied later
        robot_psms = {robot.id: [] for robot in self.robots}
        all_track_psms = {}
        
        # OUTER LOOP: For each ego robot, create ego fused tracks
        for ego_robot in self.robots:
            ego_tracks = self.tracks.get(ego_robot.id, [])
            
            # Create ego fused tracks by fusing ego tracks with nearby tracks from other robots
            if not ego_tracks:
                continue
                
            # Create ego robot's fused tracks
            ego_fused_tracks = self.data_aggregator.fuse_tracks_for_ego_robot(ego_robot.id)
            
            if not ego_fused_tracks:
                continue
            
            # INNER LOOP: For each proximal robot, compare its tracks with ego fused tracks
            for proximal_robot in self.robots:
                if proximal_robot.id == ego_robot.id:
                    continue
                    
                # Use raw unfiltered tracks for proximal robot (important for trust evaluation)
                proximal_tracks = self.raw_tracks.get(proximal_robot.id, [])
                if not proximal_tracks:
                    continue
                
                # Generate PSMs based on assignment between ego fused tracks and raw proximal tracks
                self._generate_assignment_psms_paper_style(
                    ego_fused_tracks, proximal_tracks, ego_robot, proximal_robot, 
                    robot_psms, all_track_psms)
        
        # Apply collected PSMs to update robot trust
        for robot in self.robots:
            if robot_psms[robot.id]:
                self.trust_estimator.update_agent_trust(robot, robot_psms[robot.id])
            
            # Update track trust for all tracks that have PSMs (use raw tracks)
            robot_tracks = self.raw_tracks.get(robot.id, [])
            for track in robot_tracks:
                if track.id in all_track_psms:
                    self.trust_estimator.update_track_trust(track, all_track_psms[track.id])
                    # Update the trust registry with the new values
                    self.track_trust_registry[track.id] = (track.trust_alpha, track.trust_beta)
                    
                    # Propagate trust to object trust store if this track has an object_id
                    if track.object_id:
                        self.data_aggregator.update_object_trust(track.object_id, track.trust_alpha, track.trust_beta)
            
            # ALWAYS include ALL robots in trust_updates to maintain continuous trust evolution
            trust_updates[robot.id] = {
                'alpha': robot.trust_alpha,
                'beta': robot.trust_beta,
                'mean_trust': robot.trust_alpha / (robot.trust_alpha + robot.trust_beta),
                'num_agent_psms': len(robot_psms[robot.id]),
                'track_trust_info': {
                    track.id: {
                        'alpha': track.trust_alpha,
                        'beta': track.trust_beta,
                        'mean_trust': track.trust_alpha / (track.trust_alpha + track.trust_beta),
                        'num_psms': len(all_track_psms.get(track.id, []))
                    }
                    for track in robot_tracks
                }
            }
        
        return trust_updates
    
    def _generate_assignment_psms_paper_style(self, ego_fused_tracks: List[Track], proximal_tracks: List[Track],
                                            ego_robot: RobotState, proximal_robot: RobotState,
                                            robot_psms: Dict, all_track_psms: Dict):
        """Generate PSMs using simplified object ID matching"""
        
        if not ego_fused_tracks or not proximal_tracks:
            return
            
        # Filter proximal tracks to those in ego robot's FOV
        fov_proximal_tracks = [
            track for track in proximal_tracks 
            if self.is_in_fov(ego_robot, track.position)
        ]
        
        if not fov_proximal_tracks:
            return
        
        # Build set of ego object IDs for quick lookup
        ego_object_ids = {track.object_id for track in ego_fused_tracks}
        
        # Process each proximal track
        for proximal_track in fov_proximal_tracks:
            if proximal_track.object_id in ego_object_ids:
                # MATCH: Same object ID - positive evidence
                # Find matching ego track
                ego_track = next((t for t in ego_fused_tracks if t.object_id == proximal_track.object_id), None)
                if ego_track:
                    self._generate_positive_psm_paper_style(
                        ego_track, proximal_track, proximal_robot, robot_psms, all_track_psms
                    )
            else:
                # MISMATCH: Different object ID - negative evidence (false positive)
                self._generate_negative_psm_paper_style(
                    proximal_track, proximal_robot, robot_psms, all_track_psms
                )
    
    def _generate_positive_psm_paper_style(self, ego_fused_track: Track, proximal_track: Track,
                                         proximal_robot: RobotState, robot_psms: Dict, all_track_psms: Dict):
        """Generate positive PSM when proximal track matches ego fused track"""
        # Calculate ego fused track trust statistics (used in agent PSM)
        ego_fused_track_expected_trust = self.trust_estimator.get_expected_trust(
            ego_fused_track.trust_alpha, ego_fused_track.trust_beta)
        ego_fused_track_trust_variance = self.trust_estimator.get_trust_variance(
            ego_fused_track.trust_alpha, ego_fused_track.trust_beta)
        
        # Calculate agent trust statistics (used in track PSM)
        agent_expected_trust = self.trust_estimator.get_expected_trust(
            proximal_robot.trust_alpha, proximal_robot.trust_beta)
        
        # Agent PSM: value = E[ego_fused_track_trust], confidence = 1-V[ego_fused_track_trust] (positive)
        agent_value = ego_fused_track_expected_trust
        agent_confidence = 1.0 - ego_fused_track_trust_variance  # Ensure positive confidence
        robot_psms[proximal_robot.id].append((agent_value, agent_confidence))
        
        # Track PSM: value = 1 (match), confidence = E[agent_trust]
        track_value = 1.0
        track_confidence = agent_expected_trust
        
        if proximal_track.id not in all_track_psms:
            all_track_psms[proximal_track.id] = []
        all_track_psms[proximal_track.id].append((track_value, track_confidence))
    
    def _generate_negative_psm_paper_style(self, proximal_track: Track, proximal_robot: RobotState,
                                         robot_psms: Dict, all_track_psms: Dict):
        """Generate negative PSM when proximal track does not match any ego fused track but is in ego's FOV"""
        # Calculate proximal track trust statistics (used in agent PSM)
        proximal_track_expected_trust = self.trust_estimator.get_expected_trust(
            proximal_track.trust_alpha, proximal_track.trust_beta)
        proximal_track_trust_variance = self.trust_estimator.get_trust_variance(
            proximal_track.trust_alpha, proximal_track.trust_beta)
        
        # Calculate agent trust statistics (used in track PSM)
        agent_expected_trust = self.trust_estimator.get_expected_trust(
            proximal_robot.trust_alpha, proximal_robot.trust_beta)
        
        # Agent PSM: value = 1 - E[proximal_track_trust], confidence = 1 - V[proximal_track_trust]
        agent_value = 1.0 - proximal_track_expected_trust  # Corrected: negative evidence
        agent_confidence = 1.0 - proximal_track_trust_variance  # Ensure positive confidence 
        robot_psms[proximal_robot.id].append((agent_value, agent_confidence))
        
        # Track PSM: value = 0 (no match/false positive), confidence = E[agent_trust]
        track_value = 0.0
        track_confidence = agent_expected_trust
        
        if proximal_track.id not in all_track_psms:
            all_track_psms[proximal_track.id] = []
        all_track_psms[proximal_track.id].append((track_value, track_confidence))
    
    def step(self) -> Dict:
        """Execute one simulation step"""
        # Update ground truth objects first
        self._update_ground_truth_objects()
        
        # Update robot positions and navigation
        for robot in self.robots:
            # Update navigation first (sets velocity and orientation)
            self._update_robot_navigation(robot)
            
            # Move robot based on current velocity
            robot.position += robot.velocity * self.dt
            
            # Boundary conditions - bounce off walls if needed
            if robot.position[0] < 0 or robot.position[0] > self.world_size[0]:
                robot.velocity[0] *= -1
                robot.position[0] = np.clip(robot.position[0], 0, self.world_size[0])
            if robot.position[1] < 0 or robot.position[1] > self.world_size[1]:
                robot.velocity[1] *= -1
                robot.position[1] = np.clip(robot.position[1], 0, self.world_size[1])
        
        # Generate detections for each robot
        for robot in self.robots:
            raw_detections = self.generate_detections(robot)
            self.raw_tracks[robot.id] = raw_detections  # Keep raw tracks for PSM generation
            self.tracks[robot.id] = raw_detections.copy()  # Copy for filtering/merging
            
            # Update object trust from generated tracks
            for track in self.tracks[robot.id]:
                if track.object_id:
                    self.data_aggregator.update_object_trust(track.object_id, track.trust_alpha, track.trust_beta)
        
        # Merge overlapping tracks within each robot (keep highest trust)
        self.merge_overlapping_tracks()
        
        # *** PERFORM TRUST ESTIMATION FIRST (before uniqueness enforcement) ***
        # This allows multi-robot consensus to be rewarded before filtering
        self.data_aggregator.tracks_by_robot = self.tracks
        self.data_aggregator.robots = self.robots
        
        # Update object trust from tracks before consensus analysis
        for tracks in self.tracks.values():
            for track in tracks:
                if track.object_id:
                    self.data_aggregator.update_object_trust(track.object_id, track.trust_alpha, track.trust_beta)
        
        # Perform trust estimation using assignment-based PSMs (rewards consensus)
        trust_updates = self.perform_assignment_based_trust_estimation()
        
        # *** THEN enforce global one-track-per-object constraint (for final output) ***
        self._enforce_global_object_uniqueness()
        
        self.time += self.dt
        
        return {
            'time': self.time,
            'robot_states': [(r.id, r.position.tolist(), r.velocity.tolist(), r.is_adversarial) 
                           for r in self.robots],
            'tracks': {robot_id: [(t.id, t.position.tolist(), t.confidence, t.trust_alpha, t.trust_beta) 
                                for t in tracks] 
                      for robot_id, tracks in self.tracks.items()},
            'trust_updates': trust_updates,
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
    
    def visualize_current_state(self, save_path: Optional[str] = None):
        """Visualize current simulation state"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot ground truth objects with different symbols based on movement pattern
        if self.ground_truth_objects:
            for obj in self.ground_truth_objects:
                marker_map = {
                    'stationary': 'x',
                    'linear': 'v',
                    'random_walk': 'd',
                    'circular': 'o'
                }
                color_map = {
                    'vehicle': 'red',
                    'person': 'orange', 
                    'animal': 'brown'
                }
                
                marker = marker_map.get(obj.movement_pattern, 'x')
                color = color_map.get(obj.object_type, 'red')
                
                ax.scatter(obj.position[0], obj.position[1], 
                          c=color, marker=marker, s=100, alpha=0.8,
                          label=f'GT {obj.object_type} ({obj.movement_pattern})' 
                          if obj.id < 4 else "")  # Only label first few to avoid clutter
        
        # Plot robots and their tracks
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.robots)))
        
        for i, robot in enumerate(self.robots):
            # Plot robot position with different markers for legitimate vs adversarial
            trust_val = robot.trust_alpha/(robot.trust_alpha+robot.trust_beta)
            robot_type = "ADV" if robot.is_adversarial else "LEG"
            marker = '^' if robot.is_adversarial else 'o'
            
            ax.scatter(robot.position[0], robot.position[1], 
                      c=[colors[i]], marker=marker, s=200, 
                      label=f'Robot {robot.id} ({robot_type}, Trust: {trust_val:.2f})')
            
            # Plot patrol pattern (start and goal positions)
            if robot.start_position is not None:
                # Start position (home)
                ax.scatter(robot.start_position[0], robot.start_position[1], 
                          c=[colors[i]], marker='s', s=100, alpha=0.7)
            
            if robot.goal_position is not None:
                # Goal position
                ax.scatter(robot.goal_position[0], robot.goal_position[1], 
                          c=[colors[i]], marker='*', s=150, alpha=0.7)
                
                # Draw patrol line
                if robot.start_position is not None:
                    ax.plot([robot.start_position[0], robot.goal_position[0]], 
                           [robot.start_position[1], robot.goal_position[1]], 
                           color=colors[i], linestyle='--', alpha=0.5)
            
            # Plot robot's field of view
            angles = np.linspace(robot.orientation - robot.fov_angle/2,
                               robot.orientation + robot.fov_angle/2, 20)
            fov_x = robot.position[0] + robot.fov_range * np.cos(angles)
            fov_y = robot.position[1] + robot.fov_range * np.sin(angles)
            fov_alpha = 0.05 if robot.is_adversarial else 0.1
            ax.fill(np.concatenate([[robot.position[0]], fov_x, [robot.position[0]]]),
                   np.concatenate([[robot.position[1]], fov_y, [robot.position[1]]]),
                   color=colors[i], alpha=fov_alpha)
            
            # Plot robot's track detections
            if robot.id in self.tracks:
                track_positions = np.array([t.position for t in self.tracks[robot.id]])
                if len(track_positions) > 0:
                    track_marker = 'x' if robot.is_adversarial else '+'
                    ax.scatter(track_positions[:, 0], track_positions[:, 1], 
                              c=[colors[i]], marker=track_marker, s=50, alpha=0.7)
        
        ax.set_xlim(0, self.world_size[0])
        ax.set_ylim(0, self.world_size[1])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Multi-Robot Trust-Based Sensor Fusion - Dynamic Objects (t={self.time:.1f}s)\n'
                    f'Objects: {len(self.ground_truth_objects)} | Robots: Squares=Start, Stars=Goals')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    """Main simulation runner"""
    print("Initializing Neural-Symbolic Trust-Based Sensor Fusion Simulation")
    
    # Create simulation environment with smaller world and more communication
    env = SimulationEnvironment(num_robots=5, num_targets=20, world_size=(50.0, 50.0))
    
    # Visualize initial state
    env.visualize_current_state('initial_state.png')
    
    # Collect training data
    print("Collecting training data...")
    training_data = env.collect_training_data(num_steps=500)
    
    # Save training data
    env.save_data(training_data, 'trust_simulation_data.json')
    
    # Visualize final state
    env.visualize_current_state('final_state.png')
    
    # Print summary statistics
    print("\nSimulation Summary:")
    print(f"Total time steps: {len(training_data)}")
    print(f"Final simulation time: {env.time:.1f} seconds")
    
    print("\nRobot Classifications (Ground Truth):")
    legitimate_robots = [r for r in env.robots if not r.is_adversarial]
    adversarial_robots = [r for r in env.robots if r.is_adversarial]
    
    print(f"Legitimate Robots: {[r.id for r in legitimate_robots]}")
    print(f"Adversarial Robots: {[r.id for r in adversarial_robots]}")
    
    print("\nFinal Trust Levels vs Ground Truth:")
    for robot in env.robots:
        trust_mean = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
        robot_type = "ADVERSARIAL" if robot.is_adversarial else "LEGITIMATE"
        print(f"Robot {robot.id} ({robot_type}): α={robot.trust_alpha:.2f}, β={robot.trust_beta:.2f}, "
              f"Trust={trust_mean:.3f}")


if __name__ == "__main__":
    main()