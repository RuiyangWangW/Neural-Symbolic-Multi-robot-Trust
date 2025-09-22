#!/usr/bin/env python3
"""
Clean robot and track class definitions for multi-robot trust system.

This module provides organized data structures to replace the confusing
multiple track types (local tracks, fused tracks, current updated tracks).
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
        # Increment observation count since this represents another observation/update

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
                 fov_angle: float = np.pi/3):
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
        
        # Field of view parameters
        self.fov_range = fov_range
        self.fov_angle = fov_angle
        
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
        """Check if a position is within this robot's field of view."""
        # Calculate relative position
        rel_pos = target_position - self.position
        distance = np.linalg.norm(rel_pos[:2])  # 2D distance

        if distance > self.fov_range:
            return False
        
        # Check angle constraint
        target_angle = np.arctan2(rel_pos[1], rel_pos[0])
        angle_diff = abs(target_angle - self.orientation)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Wrap around

        return angle_diff <= self.fov_angle / 2

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