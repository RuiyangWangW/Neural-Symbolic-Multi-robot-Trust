#!/usr/bin/env python3
"""
Detector Sensor Class for Multi-Robot Trust Simulation

This module defines a realistic detector sensor that models natural sensor limitations
with transient false positive and false negative detections. This is separate from
adversarial manipulation behavior.
"""

import numpy as np
import random
from typing import List, Dict, Tuple


class DetectorSensor:
    """
    Realistic detector sensor with natural FP/FN rates.

    This class models realistic sensor behavior with:
    - Transient false positive detections (sensor artifacts that appear/disappear)
    - False negative detections (missed real objects)
    - Measurement noise on position/velocity

    This represents SENSOR LIMITATIONS, not adversarial behavior.
    """

    def __init__(self,
                 robot_id: int,
                 sensor_fp_rate: float = 0.05,
                 sensor_fn_rate: float = 0.05):
        """
        Initialize a realistic detector sensor.

        Args:
            robot_id: ID of the robot that owns this sensor. Used to namespace transient
                sensor FP object_ids so they're globally unique - every robot has its own
                DetectorSensor instance with its own independent next_natural_fp_id
                counter starting at 0, so without this, two robots' unrelated artifacts
                could collide on the same object_id (e.g. both producing "sensor_fp_0"),
                causing them to be incorrectly fused/co-detected as if they were the same
                object.
            sensor_fp_rate: Natural false positive rate (probability of spurious detections)
            sensor_fn_rate: Natural false negative rate (probability of missing real objects)
        """
        self.robot_id = robot_id
        self.sensor_fp_rate = sensor_fp_rate
        self.sensor_fn_rate = sensor_fn_rate

        # Track transient sensor FP objects (appear and disappear randomly)
        self.natural_fp_objects: List[Dict] = []
        self.next_natural_fp_id = 0

    def generate_detections(self,
                           robot_position: np.ndarray,
                           robot_orientation: float,
                           robot_fov_range: float,
                           robot_fov_angle: float,
                           is_in_fov_func,  # Function to check if position is in FoV
                           ground_truth_objects: List,
                           time: float,
                           noise_std: float,
                           world_size: Tuple[float, float]) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate realistic sensor detections with FP/FN.

        Args:
            robot_position: Robot's current position [x, y, z]
            robot_orientation: Robot's current orientation (radians)
            robot_fov_range: Robot's field of view range
            robot_fov_angle: Robot's field of view angle (radians)
            is_in_fov_func: Function to check if a position is in the robot's FoV
            ground_truth_objects: List of ground truth objects in the environment
            time: Current simulation time
            noise_std: Standard deviation for position/velocity noise
            world_size: Size of the simulation world (for FP generation)

        Returns:
            Tuple of (detected_gt_objects, detected_sensor_fp_objects)
            - detected_gt_objects: List of dicts with GT object detections after sensor FN
            - detected_sensor_fp_objects: List of dicts with transient sensor FP detections
        """
        detected_gt_objects = []
        detected_sensor_fp_objects = []

        # 1. Apply sensor FN to ground truth objects
        for gt_obj in ground_truth_objects:
            if is_in_fov_func(gt_obj.position):
                # Apply false negative rate (sensor limitation)
                if random.random() < self.sensor_fn_rate:
                    continue  # Sensor missed this detection

                # Add measurement noise (realistic sensor imperfection)
                noisy_pos = gt_obj.position + np.random.normal(0, noise_std, 3)
                noisy_vel = gt_obj.velocity + np.random.normal(0, noise_std * 0.1, 3)

                detected_gt_objects.append({
                    'type': 'ground_truth',
                    'gt_object': gt_obj,
                    'object_id': f"gt_obj_{gt_obj.id}",
                    'position': noisy_pos,
                    'velocity': noisy_vel
                })

        # 2. Generate transient sensor FP detections
        # Update existing natural FP objects and potentially create new ones
        self._update_natural_fp_objects(time, robot_position, robot_orientation,
                                       robot_fov_range, robot_fov_angle, world_size)

        # Detect sensor FP objects in FoV
        for fp_obj in self.natural_fp_objects:
            if is_in_fov_func(fp_obj['position']):
                # Add noise to FP position (sensor noise applies to FPs too)
                noisy_pos = fp_obj['position'] + np.random.normal(0, noise_std, 3)
                noisy_vel = fp_obj['velocity'] + np.random.normal(0, noise_std * 0.1, 3)

                detected_sensor_fp_objects.append({
                    'type': 'sensor_fp',
                    'object_id': f"sensor_fp_{self.robot_id}_{fp_obj['id']}",
                    'position': noisy_pos,
                    'velocity': noisy_vel,
                    'spawn_time': fp_obj['spawn_time']
                })

        return detected_gt_objects, detected_sensor_fp_objects

    def _update_natural_fp_objects(self,
                                   time: float,
                                   robot_position: np.ndarray,
                                   robot_orientation: float,
                                   robot_fov_range: float,
                                   robot_fov_angle: float,
                                   world_size: Tuple[float, float]):
        """
        Update transient natural false positive objects (sensor artifacts).

        Natural FPs appear randomly and move with random walk dynamics.
        They represent sensor artifacts (reflections, clutter, etc.), not adversarial manipulations.

        Args:
            time: Current simulation time
            robot_position: Robot's current position
            robot_orientation: Robot's current orientation
            robot_fov_range: Robot's field of view range
            robot_fov_angle: Robot's field of view angle
            world_size: Size of the simulation world
        """
        # Remove old FP objects (they disappear after some time - transient)
        fp_lifetime = 10.0  # seconds
        self.natural_fp_objects = [
            fp for fp in self.natural_fp_objects
            if time - fp['spawn_time'] < fp_lifetime
        ]

        # Potentially create new natural FP objects
        # Expected number of FP objects = sensor_fp_rate * number of real objects in FoV
        # For simplicity, create with small probability each timestep
        if random.random() < self.sensor_fp_rate * 0.1:  # 0.1 is timestep factor
            # Create new natural FP near the robot's FoV (sensor artifact)
            distance = random.uniform(robot_fov_range * 0.3, robot_fov_range * 0.9)
            angle = robot_orientation + random.uniform(-robot_fov_angle/2, robot_fov_angle/2)

            fp_position = robot_position + np.array([
                distance * np.cos(angle),
                distance * np.sin(angle),
                0.0
            ])

            # Ensure within world bounds
            fp_position[0] = np.clip(fp_position[0], 0, world_size[0])
            fp_position[1] = np.clip(fp_position[1], 0, world_size[1])

            # Random velocity (slow movement, like drifting clutter)
            fp_velocity = np.random.normal(0, 0.5, 3)
            fp_velocity[2] = 0.0  # No vertical movement

            fp_obj = {
                'id': self.next_natural_fp_id,
                'position': fp_position,
                'velocity': fp_velocity,
                'spawn_time': time
            }

            self.natural_fp_objects.append(fp_obj)
            self.next_natural_fp_id += 1

        # Update positions of existing natural FP objects (random walk)
        dt = 0.1
        for fp_obj in self.natural_fp_objects:
            # Random walk with occasional direction changes
            if random.random() < 0.05:  # 5% chance to change direction
                fp_obj['velocity'] = np.random.normal(0, 0.5, 3)
                fp_obj['velocity'][2] = 0.0

            # Update position
            fp_obj['position'] = fp_obj['position'] + fp_obj['velocity'] * dt

            # Bounce off boundaries
            if fp_obj['position'][0] <= 0 or fp_obj['position'][0] >= world_size[0]:
                fp_obj['velocity'][0] *= -1
                fp_obj['position'][0] = np.clip(fp_obj['position'][0], 0, world_size[0])
            if fp_obj['position'][1] <= 0 or fp_obj['position'][1] >= world_size[1]:
                fp_obj['velocity'][1] *= -1
                fp_obj['position'][1] = np.clip(fp_obj['position'][1], 0, world_size[1])

    def get_num_active_sensor_fps(self) -> int:
        """Get the number of currently active transient sensor FP objects."""
        return len(self.natural_fp_objects)

    def reset(self):
        """Reset the sensor state (clear all transient FP objects)."""
        self.natural_fp_objects.clear()
        self.next_natural_fp_id = 0
