#!/usr/bin/env python3
"""
Clean robot and track class definitions for multi-robot trust system.

This module provides organized data structures to replace the confusing
multiple track types (local tracks, fused tracks, current updated tracks).

New in this version:
- Integrated RL dual-horizon trust buffers (fast/slow) directly in Robot and Track classes
- RL trust methods (rl_trust_value, update_rl_trust, etc.) for seamless integration
- Maintains backward compatibility with original trust_alpha/trust_beta parameters
"""

import numpy as np
from typing import Dict, List, Optional, Any
import time


class Track:
    """
    Clean track class that describes a robot's observation of an object.
    
    Each track belongs to exactly one robot and represents that robot's
    belief about one object's state and trustworthiness.
    """
    
    def __init__(self, 
                 track_id: str,
                 robot_id: int, 
                 object_id: str,
                 position: np.ndarray,
                 velocity: np.ndarray,
                 trust_alpha: float = 1.0,
                 trust_beta: float = 1.0,
                 timestamp: float = None):
        """
        Initialize a track.
        
        Args:
            track_id: Unique identifier for this track
            robot_id: ID of the robot that owns this track
            object_id: ID of the object being tracked
            position: Current position estimate [x, y]
            velocity: Current velocity estimate [vx, vy]
            trust_alpha: Alpha parameter of Beta trust distribution
            trust_beta: Beta parameter of Beta trust distribution  
            timestamp: When this track was created/updated
        """
        self.track_id = track_id
        self.robot_id = robot_id
        self.object_id = object_id
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        
        # Trust distribution parameters (Beta distribution)
        self.trust_alpha = trust_alpha
        self.trust_beta = trust_beta

        # RL dual-horizon trust buffers (for RL trust system)
        self.rl_fast_alpha = trust_alpha
        self.rl_fast_beta = trust_beta
        self.rl_slow_alpha = trust_alpha
        self.rl_slow_beta = trust_beta
        self.rl_lambda_mix = 0.7  # Mixing parameter (70% fast, 30% slow)
        
        # Metadata
        self.timestamp = timestamp or time.time()
        self.last_updated = self.timestamp
        self.observation_count = 1
    
    @property
    def trust_value(self) -> float:
        """Compute trust value from Beta distribution parameters."""
        return self.trust_alpha / (self.trust_alpha + self.trust_beta)
    
    def update_trust(self, delta_alpha: float, delta_beta: float):
        """Update trust distribution parameters."""
        self.trust_alpha += delta_alpha
        self.trust_beta += delta_beta

    # RL dual-horizon trust methods
    @property
    def rl_effective_alpha(self) -> float:
        """Compute effective alpha from dual-horizon buffers."""
        return self.rl_lambda_mix * self.rl_fast_alpha + (1 - self.rl_lambda_mix) * self.rl_slow_alpha

    @property
    def rl_effective_beta(self) -> float:
        """Compute effective beta from dual-horizon buffers."""
        return self.rl_lambda_mix * self.rl_fast_beta + (1 - self.rl_lambda_mix) * self.rl_slow_beta

    @property
    def rl_trust_value(self) -> float:
        """Compute RL trust value from effective dual-horizon parameters."""
        effective_alpha = self.rl_effective_alpha
        effective_beta = self.rl_effective_beta
        return effective_alpha / (effective_alpha + effective_beta)

    @property
    def rl_confidence(self) -> float:
        """Compute RL trust confidence (strength of belief)."""
        return self.rl_effective_alpha + self.rl_effective_beta

    def update_rl_trust(self, delta_alpha: float, delta_beta: float, fast_ratio: float = 0.8):
        """Update RL dual-horizon trust buffers."""
        # Update fast buffer with higher proportion
        self.rl_fast_alpha += delta_alpha * fast_ratio
        self.rl_fast_beta += delta_beta * fast_ratio

        # Update slow buffer with remaining proportion
        self.rl_slow_alpha += delta_alpha * (1 - fast_ratio)
        self.rl_slow_beta += delta_beta * (1 - fast_ratio)

    def apply_rl_forgetting(self, gamma_forget: float = 0.01):
        """Apply forgetting toward (1,1) for RL trust buffers."""
        # Fast buffer forgetting
        self.rl_fast_alpha = (1 - gamma_forget) * self.rl_fast_alpha + gamma_forget * 1.0
        self.rl_fast_beta = (1 - gamma_forget) * self.rl_fast_beta + gamma_forget * 1.0

        # Slow buffer forgetting
        self.rl_slow_alpha = (1 - gamma_forget) * self.rl_slow_alpha + gamma_forget * 1.0
        self.rl_slow_beta = (1 - gamma_forget) * self.rl_slow_beta + gamma_forget * 1.0

    def apply_rl_strength_cap(self, strength_max: float = 50.0):
        """Apply strength caps to prevent over-confidence in RL buffers."""
        # Cap fast buffer
        fast_strength = self.rl_fast_alpha + self.rl_fast_beta
        if fast_strength > strength_max:
            scale = strength_max / fast_strength
            self.rl_fast_alpha *= scale
            self.rl_fast_beta *= scale

        # Cap slow buffer
        slow_strength = self.rl_slow_alpha + self.rl_slow_beta
        if slow_strength > strength_max:
            scale = strength_max / slow_strength
            self.rl_slow_alpha *= scale
            self.rl_slow_beta *= scale

    def update_state(self, position: np.ndarray, velocity: np.ndarray, timestamp: float):
        """Update track state estimates."""
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.last_updated = timestamp 
        self.timestamp = self.last_updated  # Update timestamp to latest update time
    
    def distance_to(self, other_track: 'Track') -> float:
        """Compute distance to another track."""
        return np.linalg.norm(self.position - other_track.position)
    
    def is_same_object(self, other_track: 'Track', distance_threshold: float = 5.0) -> bool:
        """Check if this track and another track refer to the same object."""
        return (self.object_id == other_track.object_id and 
                self.distance_to(other_track) < distance_threshold)
    
    def was_updated_at_timestep(self, timestep: float, tolerance: float = 1e-6) -> bool:
        """Check if this track was updated at a specific simulation timestep."""
        return abs(self.timestamp - timestep) < tolerance
    
    def was_updated_since(self, since_time: float) -> bool:
        """Check if this track was updated since a given time."""
        return self.last_updated >= since_time
    
    def __repr__(self):
        return (f"Track(id={self.track_id}, robot={self.robot_id}, "
                f"obj={self.object_id}, pos={self.position}, "
                f"trust={self.trust_value:.3f})")


class Robot:
    """
    Clean robot class that keeps all information local to the robot.
    
    Each robot maintains its own tracks, position, velocity, and trust parameters.
    This eliminates the confusion between different track storage systems.
    """
    
    def __init__(self,
                 robot_id: int,
                 position: np.ndarray,
                 velocity: np.ndarray = None,
                 trust_alpha: float = 1.0,
                 trust_beta: float = 1.0,
                 fov_range: float = 50.0,
                 fov_angle: float = np.pi/3,
                 use_spot_fov: bool = False,
                 occ_grid: np.ndarray = None,
                 grid_resolution: float = None,
                 grid_xmin: float = None,
                 grid_ymin: float = None):
        """
        Initialize a robot.

        Args:
            robot_id: Unique identifier for this robot
            position: Current position [x, y]
            velocity: Current velocity [vx, vy]
            trust_alpha: Alpha parameter of robot's trust distribution
            trust_beta: Beta parameter of robot's trust distribution
            fov_range: Field of view range
            fov_angle: Field of view angle in radians
            use_spot_fov: If True, use SPOT dual-camera FoV instead of default single-camera FoV
            occ_grid: Occupancy grid (2D array, 1=occupied, 0=free), optional for line-of-sight checking
            grid_resolution: Resolution of occupancy grid in meters per pixel
            grid_xmin: Minimum X coordinate of grid in world coordinates
            grid_ymin: Minimum Y coordinate of grid in world coordinates
        """
        self.id = robot_id
        self.position = np.array(position)
        self.velocity = np.array(velocity) if velocity is not None else np.zeros(2)
        
        # Calculate initial orientation from velocity
        if velocity is not None and (velocity[0] != 0 or velocity[1] != 0):
            self.orientation = np.arctan2(velocity[1], velocity[0])
        else:
            self.orientation = 0.0  # Default facing east
        
        # Robot trust distribution parameters
        self.trust_alpha = trust_alpha
        self.trust_beta = trust_beta

        # RL dual-horizon trust buffers (for RL trust system)
        self.rl_fast_alpha = trust_alpha
        self.rl_fast_beta = trust_beta
        self.rl_slow_alpha = trust_alpha
        self.rl_slow_beta = trust_beta
        self.rl_lambda_mix = 0.7  # Mixing parameter (70% fast, 30% slow)
        
        # Field of view parameters
        self.fov_range = fov_range
        self.fov_angle = fov_angle
        self.use_spot_fov = use_spot_fov  # Use SPOT dual-camera FoV if True

        # Occupancy grid for line-of-sight checking (optional)
        self.occ_grid = occ_grid
        self.grid_resolution = grid_resolution
        self.grid_xmin = grid_xmin
        self.grid_ymin = grid_ymin
        if occ_grid is not None:
            self.grid_height, self.grid_width = occ_grid.shape

        # Local tracks maintained by this robot
        self.local_tracks: Dict[str, Track] = {}  # object_id -> Track
        
        # Current timestep tracking
        self.current_timestep: float = 0.0
        self.current_timestep_tracks: Dict[str, Track] = {}  # Tracks updated in current timestep
        
        # Metadata
        self.timestamp = time.time()
        self.is_adversarial = False
        self.is_active = True
    
    @property
    def trust_value(self) -> float:
        """Compute robot trust value from Beta distribution parameters."""
        return self.trust_alpha / (self.trust_alpha + self.trust_beta)
    
    def update_trust(self, delta_alpha: float, delta_beta: float):
        """Update robot trust distribution parameters."""
        self.trust_alpha += delta_alpha
        self.trust_beta += delta_beta

    # RL dual-horizon trust methods
    @property
    def rl_effective_alpha(self) -> float:
        """Compute effective alpha from dual-horizon buffers."""
        return self.rl_lambda_mix * self.rl_fast_alpha + (1 - self.rl_lambda_mix) * self.rl_slow_alpha

    @property
    def rl_effective_beta(self) -> float:
        """Compute effective beta from dual-horizon buffers."""
        return self.rl_lambda_mix * self.rl_fast_beta + (1 - self.rl_lambda_mix) * self.rl_slow_beta

    @property
    def rl_trust_value(self) -> float:
        """Compute RL trust value from effective dual-horizon parameters."""
        effective_alpha = self.rl_effective_alpha
        effective_beta = self.rl_effective_beta
        return effective_alpha / (effective_alpha + effective_beta)

    @property
    def rl_confidence(self) -> float:
        """Compute RL trust confidence (strength of belief)."""
        return self.rl_effective_alpha + self.rl_effective_beta

    def update_rl_trust(self, delta_alpha: float, delta_beta: float, fast_ratio: float = 0.8):
        """Update RL dual-horizon trust buffers."""
        # Update fast buffer with higher proportion
        self.rl_fast_alpha += delta_alpha * fast_ratio
        self.rl_fast_beta += delta_beta * fast_ratio

        # Update slow buffer with remaining proportion
        self.rl_slow_alpha += delta_alpha * (1 - fast_ratio)
        self.rl_slow_beta += delta_beta * (1 - fast_ratio)

    def apply_rl_forgetting(self, gamma_forget: float = 0.01):
        """Apply forgetting toward (1,1) for RL trust buffers."""
        # Fast buffer forgetting
        self.rl_fast_alpha = (1 - gamma_forget) * self.rl_fast_alpha + gamma_forget * 1.0
        self.rl_fast_beta = (1 - gamma_forget) * self.rl_fast_beta + gamma_forget * 1.0

        # Slow buffer forgetting
        self.rl_slow_alpha = (1 - gamma_forget) * self.rl_slow_alpha + gamma_forget * 1.0
        self.rl_slow_beta = (1 - gamma_forget) * self.rl_slow_beta + gamma_forget * 1.0

    def apply_rl_strength_cap(self, strength_max: float = 50.0):
        """Apply strength caps to prevent over-confidence in RL buffers."""
        # Cap fast buffer
        fast_strength = self.rl_fast_alpha + self.rl_fast_beta
        if fast_strength > strength_max:
            scale = strength_max / fast_strength
            self.rl_fast_alpha *= scale
            self.rl_fast_beta *= scale

        # Cap slow buffer
        slow_strength = self.rl_slow_alpha + self.rl_slow_beta
        if slow_strength > strength_max:
            scale = strength_max / slow_strength
            self.rl_slow_alpha *= scale
            self.rl_slow_beta *= scale 

    def update_state(self, position: np.ndarray, velocity: np.ndarray = None):
        """Update robot position, velocity, and orientation."""
        self.position = np.array(position)
        if velocity is not None:
            self.velocity = np.array(velocity)
            # Update orientation based on velocity (direction of movement)
            if velocity[0] != 0 or velocity[1] != 0:
                self.orientation = np.arctan2(velocity[1], velocity[0])
        self.timestamp = time.time()
    
    def add_track(self, track: Track):
        """Add a track to this robot's local tracks."""
        if track.robot_id != self.id:
            raise ValueError(f"Track robot_id {track.robot_id} doesn't match robot id {self.id}")
        
        self.local_tracks[track.object_id] = track
        # Mark track as updated in current timestep
        self.current_timestep_tracks[track.object_id] = track
    
    def remove_track(self, object_id: str):
        """Remove a track from this robot's local tracks."""
        if object_id in self.local_tracks:
            del self.local_tracks[object_id]
    
    def get_track(self, object_id: str) -> Optional[Track]:
        """Get a specific track by object ID."""
        return self.local_tracks.get(object_id)
    
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks maintained by this robot."""
        return list(self.local_tracks.values())

    def get_all_current_tracks(self) -> List[Track]:
        """Get all tracks that are currently active (updated in the current timestep)."""
        return list(self.current_timestep_tracks.values())

    def get_track_count(self) -> int:
        """Get number of tracks this robot is maintaining."""
        return len(self.local_tracks)
    
    def create_track_for_object(self, 
                               object_id: str, 
                               position: np.ndarray,
                               velocity: np.ndarray,
                               timestamp: float = None) -> Track:
        """Create a new track for an observed object."""
        track_id = f"robot_{self.id}_obj_{object_id}"
        current_time = timestamp if timestamp is not None else self.current_timestep
        
        track = Track(
            track_id=track_id,
            robot_id=self.id,
            object_id=object_id,
            position=position,
            velocity=velocity,
            timestamp=current_time
        )
        
        self.add_track(track)
        return track
    
    def update_track(self, 
                    object_id: str,
                    position: np.ndarray,
                    velocity: np.ndarray,
                    timestamp: float = None):
        """Update an existing track or create new one if doesn't exist."""
        current_time = timestamp if timestamp is not None else self.current_timestep
        
        if object_id in self.local_tracks:
            self.local_tracks[object_id].update_state(position, velocity, current_time)
            # Mark track as updated in current timestep
            self.current_timestep_tracks[object_id] = self.local_tracks[object_id]
            self.local_tracks[object_id].observation_count += 1
        else:
            self.create_track_for_object(object_id, position, velocity, current_time)
    
    def is_in_fov(self, target_position: np.ndarray) -> bool:
        """
        Check if a position is within this robot's field of view.

        If use_spot_fov is True, uses SPOT dual-camera FoV.
        Otherwise uses default single-camera FoV.
        """
        if self.use_spot_fov:
            # Use SPOT dual-camera FoV for Webots environment
            return self.is_in_spot_dual_camera_fov(target_position, self.fov_angle, self.fov_range)

        # Default single-camera FoV
        # Calculate relative position
        rel_pos = target_position - self.position
        distance = np.linalg.norm(rel_pos[:2])  # 2D distance

        if distance > self.fov_range:
            return False

        # Check angle constraint
        target_angle = np.arctan2(rel_pos[1], rel_pos[0])
        angle_diff = abs(target_angle - self.orientation)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Wrap around

        if angle_diff > self.fov_angle / 2:
            return False

        # Geometric FoV check passed, now check line of sight
        return self.has_line_of_sight(target_position)

    def has_line_of_sight(self, target_position: np.ndarray) -> bool:
        """
        Check if there's a clear line of sight from robot to target using occupancy grid.

        Uses Bresenham's line algorithm to check all cells along the ray from robot to target.

        Args:
            target_position: Target position in world frame [x, y, z]

        Returns:
            True if line of sight is clear (no occupied cells), False otherwise
            Returns True if no occupancy grid is available (no occlusion checking)
        """
        # If no occupancy grid, assume clear line of sight
        if self.occ_grid is None:
            return True

        # Convert positions to grid coordinates
        start_col = int(np.floor((self.position[0] - self.grid_xmin) / self.grid_resolution))
        start_row = int(np.floor((self.position[1] - self.grid_ymin) / self.grid_resolution))
        end_col = int(np.floor((target_position[0] - self.grid_xmin) / self.grid_resolution))
        end_row = int(np.floor((target_position[1] - self.grid_ymin) / self.grid_resolution))

        # Clamp to grid bounds
        start_row = np.clip(start_row, 0, self.grid_height - 1)
        start_col = np.clip(start_col, 0, self.grid_width - 1)
        end_row = np.clip(end_row, 0, self.grid_height - 1)
        end_col = np.clip(end_col, 0, self.grid_width - 1)

        # Bresenham's line algorithm
        dx = abs(end_col - start_col)
        dy = abs(end_row - start_row)
        sx = 1 if end_col > start_col else -1
        sy = 1 if end_row > start_row else -1
        err = dx - dy

        col, row = start_col, start_row

        # Skip the first cell (robot's position) and check cells in between
        first_cell = True

        while True:
            # Reached target (don't check target cell either, it's the object position)
            if col == end_col and row == end_row:
                break

            # Move to next cell before checking
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                col += sx
            if e2 < dx:
                err += dx
                row += sy

            # Skip first iteration (we just moved from start position)
            if first_cell:
                first_cell = False
                continue

            # Check if current cell is occupied (only intermediate cells)
            if col != end_col or row != end_row:  # Don't check target cell
                if 0 <= row < self.grid_height and 0 <= col < self.grid_width:
                    if self.occ_grid[row, col] == 1:
                        return False  # Occupied cell blocks line of sight

        return True  # Clear line of sight

    def is_in_spot_dual_camera_fov(self, target_position: np.ndarray,
                                     fov_angle: float = np.pi/4,
                                     fov_range: float = 20.0) -> bool:
        """
        Check if position is in SPOT robot's dual-camera field of view.

        SPOT has two head cameras (left and right) with specific offsets and orientations.
        An object is visible if it's in EITHER camera's FoV.

        Args:
            target_position: Target position in world frame [x, y, z]
            fov_angle: FoV angle for each camera (default: π/4 = 45°)
            fov_range: Maximum detection range (default: 20m)

        Returns:
            True if target is visible to either camera
        """
        # Camera transforms from Spot.proto (same as filter_detections_by_fov.py)
        # HEAD_SHAPES transformation
        head_trans = np.array([-0.459994, -0.000019, -1.650002])
        head_rot_aa = (0.577350, -0.577354, -0.577346, 2.094389)
        head_R = self._axis_angle_to_rotation_matrix(*head_rot_aa)

        # Right head camera (local to HEAD_SHAPES)
        right_cam_trans_local = np.array([0.044898, 1.677970, -0.922603])
        right_cam_rot_local_aa = (-0.425009, 0.613005, 0.666028, 2.205231)
        right_R_local = self._axis_angle_to_rotation_matrix(*right_cam_rot_local_aa)

        # Left head camera (local to HEAD_SHAPES)
        left_cam_trans_local = np.array([-0.045109, 1.679550, -0.923250])
        left_cam_rot_local_aa = (-0.685445, 0.491870, 0.536869, 1.854317)
        left_R_local = self._axis_angle_to_rotation_matrix(*left_cam_rot_local_aa)

        # Compose transformations
        right_camera_offset = head_trans + head_R @ right_cam_trans_local
        left_camera_offset = head_trans + head_R @ left_cam_trans_local
        right_R_composed = head_R @ right_R_local
        left_R_composed = head_R @ left_R_local

        # Check both cameras
        for camera_offset, camera_R in [(left_camera_offset, left_R_composed),
                                         (right_camera_offset, right_R_composed)]:
            # Get camera world pose
            camera_x, camera_y = self._get_camera_world_pose(self.position[0], self.position[1],
                                                              self.orientation, camera_offset)

            # Get camera pointing direction in world frame
            camera_yaw = self._get_camera_pointing_direction(self.orientation, camera_R)

            # Check if object is in this camera's FoV (geometric check)
            if self._is_in_single_camera_fov(camera_x, camera_y, camera_yaw,
                                             target_position[0], target_position[1],
                                             fov_angle, fov_range):
                # Geometric FoV check passed, now check line of sight
                if self.has_line_of_sight(target_position):
                    return True  # Visible to at least one camera with clear line of sight

        return False  # Not visible to either camera or line of sight blocked

    def _axis_angle_to_rotation_matrix(self, ax: float, ay: float, az: float, angle: float) -> np.ndarray:
        """Convert axis-angle rotation to rotation matrix using Rodrigues formula"""
        norm = np.sqrt(ax*ax + ay*ay + az*az)
        if norm < 1e-10:
            return np.eye(3)
        ax, ay, az = ax/norm, ay/norm, az/norm

        c = np.cos(angle)
        s = np.sin(angle)
        C = 1 - c

        R = np.array([
            [ax*ax*C + c,    ax*ay*C - az*s, ax*az*C + ay*s],
            [ay*ax*C + az*s, ay*ay*C + c,    ay*az*C - ax*s],
            [az*ax*C - ay*s, az*ay*C + ax*s, az*az*C + c]
        ])
        return R

    def _get_camera_world_pose(self, robot_x: float, robot_y: float, robot_yaw: float,
                                camera_offset: np.ndarray) -> tuple:
        """Get camera position in world frame"""
        offset_forward, offset_left, offset_up = camera_offset

        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)

        camera_x = robot_x + (offset_forward * cos_yaw - offset_left * sin_yaw)
        camera_y = robot_y + (offset_forward * sin_yaw + offset_left * cos_yaw)

        return camera_x, camera_y

    def _get_camera_pointing_direction(self, robot_yaw: float, camera_rotation_matrix: np.ndarray) -> float:
        """Get camera pointing direction in world frame"""
        # Camera default direction is +X in mesh coords
        camera_default_dir = np.array([1, 0, 0])
        pointing_dir = camera_rotation_matrix @ camera_default_dir

        # Get yaw in robot frame
        camera_yaw_local = np.arctan2(pointing_dir[1], pointing_dir[0])

        # Convert to world frame
        camera_yaw_world = robot_yaw + camera_yaw_local

        return camera_yaw_world

    def _is_in_single_camera_fov(self, camera_x: float, camera_y: float, camera_yaw: float,
                                   object_x: float, object_y: float,
                                   fov_angle: float, fov_range: float) -> bool:
        """Check if an object is within a single camera's field of view"""
        # Vector from camera to object
        dx = object_x - camera_x
        dy = object_y - camera_y

        # Distance to object
        distance = np.sqrt(dx*dx + dy*dy)

        # Angle to object (from world +X axis)
        angle_to_object = np.arctan2(dy, dx)

        # Angle difference (normalized to [-pi, pi])
        angle_diff = angle_to_object - camera_yaw
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # Check if within FoV
        in_range = distance <= fov_range
        in_angle = abs(angle_diff) <= fov_angle / 2

        return (in_range and in_angle)

    def start_new_timestep(self, timestep: float):
        """Start a new timestep and clear current timestep tracks."""
        self.current_timestep = timestep
        self.current_timestep_tracks.clear()
    
    def get_current_timestep_tracks(self) -> List[Track]:
        """Get tracks that were updated in the current timestep."""
        return list(self.current_timestep_tracks.values())
    
    def get_current_timestep_track_ids(self) -> List[str]:
        """Get object IDs of tracks updated in the current timestep."""
        return list(self.current_timestep_tracks.keys())
    
    def is_track_updated_current_timestep(self, object_id: str) -> bool:
        """Check if a specific track was updated in the current timestep."""
        return object_id in self.current_timestep_tracks
    
    def get_current_timestep_track_count(self) -> int:
        """Get number of tracks updated in current timestep."""
        return len(self.current_timestep_tracks)
    
    def mark_track_as_current(self, object_id: str):
        """Manually mark a track as updated in current timestep."""
        if object_id in self.local_tracks:
            self.current_timestep_tracks[object_id] = self.local_tracks[object_id]
        else:
            raise ValueError(f"Track with object_id {object_id} not found in local tracks")
    
    def update_current_timestep_tracks(self):
        """Check all tracks and mark those updated in current timestep as current."""
        for object_id, track in self.local_tracks.items():
            if track.was_updated_at_timestep(self.current_timestep):
                self.current_timestep_tracks[object_id] = track
    
    def __repr__(self):
        return (f"Robot(id={self.id}, pos={self.position}, "
                f"tracks={len(self.local_tracks)}, "
                f"trust={self.trust_value:.3f})")



# Example usage and testing
if __name__ == "__main__":
    # Create some robots
    robot1 = Robot(robot_id=1, position=[0, 0])
    robot2 = Robot(robot_id=2, position=[10, 10]) 
    robot3 = Robot(robot_id=3, position=[20, 0])
    
    # Create some tracks
    robot1.create_track_for_object("obj_1", position=[5, 5], velocity=[1, 0])
    robot1.create_track_for_object("obj_2", position=[15, 5], velocity=[0, 1])
    
    robot2.create_track_for_object("obj_1", position=[5.2, 4.8], velocity=[1.1, 0.1])
    robot2.create_track_for_object("obj_3", position=[25, 15], velocity=[-1, 0])
    
    robot3.create_track_for_object("obj_2", position=[14.8, 5.1], velocity=[0.1, 0.9])
    
    # Print robot information
    print("Individual Robots:")
    for robot in [robot1, robot2, robot3]:
        print(f"  {robot}")
        for track in robot.get_all_tracks():
            print(f"    {track}")
    print()
    
    # Test track updates
    print("Testing track updates:")
    robot1.update_track("obj_1", position=[6, 6], velocity=[1.2, 0.1])
    print(f"  Updated track: {robot1.get_track('obj_1')}")
    
    # Test trust updates
    print("Testing trust updates:")
    track = robot1.get_track("obj_1")
    original_trust = track.trust_value
    track.update_trust(0.5, 0.2)
    print(f"  Trust changed from {original_trust:.3f} to {track.trust_value:.3f}")
    
    # Test current timestep tracking
    print("\nTesting current timestep tracking:")
    
    # Start new timestep
    current_sim_time = 10.0
    robot1.start_new_timestep(current_sim_time)
    robot2.start_new_timestep(current_sim_time)
    
    print(f"  Started timestep {current_sim_time}")
    print(f"  Robot1 current timestep tracks: {robot1.get_current_timestep_track_count()}")
    print(f"  Robot2 current timestep tracks: {robot2.get_current_timestep_track_count()}")
    
    # Update some tracks in current timestep
    robot1.update_track("obj_1", position=[7, 7], velocity=[1.3, 0.2])
    robot2.create_track_for_object("obj_4", position=[30, 20], velocity=[-0.5, 0.8])
    
    print(f"  After updates - Robot1 current tracks: {robot1.get_current_timestep_track_count()}")
    print(f"  After updates - Robot2 current tracks: {robot2.get_current_timestep_track_count()}")
    
    # Show which tracks were updated
    print(f"  Robot1 updated track IDs: {robot1.get_current_timestep_track_ids()}")
    print(f"  Robot2 updated track IDs: {robot2.get_current_timestep_track_ids()}")
    
    # Check specific track update status
    print(f"  Was obj_1 updated this timestep? {robot1.is_track_updated_current_timestep('obj_1')}")
    print(f"  Was obj_2 updated this timestep? {robot1.is_track_updated_current_timestep('obj_2')}")
    
    # Get the actual tracks updated in current timestep
    current_tracks = robot1.get_current_timestep_tracks()
    print(f"  Current timestep tracks for Robot1:")
    for track in current_tracks:
        print(f"    {track}")
        
    print("\nTimestep tracking functionality added successfully!")