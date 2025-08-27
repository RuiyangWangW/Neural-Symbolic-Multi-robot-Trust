#!/usr/bin/env python3
"""
Abstract base class for trust algorithms in multi-robot sensor fusion
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


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

    # Persistent fused object identity
    object_id: Optional[str] = None


class TrustAlgorithm(ABC):
    """Abstract base class for trust estimation algorithms"""
    
    @abstractmethod
    def initialize(self, robots: List[RobotState]) -> None:
        """Initialize the trust algorithm with robot states"""
        pass
    
    @abstractmethod
    def update_trust(self, robots: List[RobotState], tracks_by_robot: Dict[int, List[Track]], 
                    robot_object_tracks: Dict[int, Dict[str, Track]], time: float,
                    robot_current_tracks: Optional[Dict[int, Dict[str, Track]]] = None,
                    environment: Optional['SimulationEnvironment'] = None) -> Dict[int, Dict]:
        """
        Update trust values for robots and tracks
        
        Args:
            robots: List of robot states
            tracks_by_robot: Robot ID -> list of tracks
            robot_object_tracks: robot_id -> {object_id: Track}
            time: Current simulation time
            robot_current_tracks: robot_id -> {object_id: Track} - current timestep only
            environment: Simulation environment for proximal range filtering
            
        Returns:
            Dictionary containing trust updates for each robot
        """
        pass
    
    @abstractmethod
    def get_expected_trust(self, alpha: float, beta: float) -> float:
        """Calculate expected value E[trust] = alpha / (alpha + beta)"""
        pass
    
    @abstractmethod
    def get_trust_variance(self, alpha: float, beta: float) -> float:
        """Calculate variance of trust distribution"""
        pass