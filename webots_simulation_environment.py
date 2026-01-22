#!/usr/bin/env python3
"""
Webots Simulation Environment

Loads robot positions and object detections from Webots simulation data.
This environment replays recorded Webots simulation data for trust-based
sensor fusion analysis.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class WebotsSimulationEnvironment:
    """
    Environment that loads data from Webots simulation files.

    The webots_sim folder structure:
    - maps/occ_grid.npy: Occupancy grid (binary: 0=free, 1=occupied)
    - detections/SPOT_X/detections.json: Robot detections per timestep

    Each detection entry contains:
    - t: timestamp
    - robot_pose: {x, y, z, yaw}
    - visible: dict of camera names -> list of detections
      - Each detection: {gid, id, pos: {x, y, z}, yaw}
    """

    def __init__(self, webots_data_path: str = "webots_sim"):
        """
        Initialize Webots simulation environment.

        Args:
            webots_data_path: Path to webots_sim folder
        """
        self.data_path = Path(webots_data_path)

        # Load occupancy grid
        self.occ_grid = np.load(self.data_path / "maps" / "occ_grid.npy")
        self.grid_height, self.grid_width = self.occ_grid.shape

        # Load grid metadata for coordinate transformation
        import json
        meta_file = self.data_path / "maps" / "occ_grid_meta.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                self.grid_meta = json.load(f)

            # Extract key parameters
            self.resolution = self.grid_meta["resolution_m_per_pixel"]
            bbox = self.grid_meta["coverage_world_bbox_m"]
            self.grid_xmin = bbox["xmin"]
            self.grid_xmax = bbox["xmax"]
            self.grid_ymin = bbox["ymin"]
            self.grid_ymax = bbox["ymax"]
        else:
            # Fallback: no metadata available
            print("Warning: No grid metadata found. Using estimated bounds.")
            self.grid_meta = None
            self.resolution = 0.1  # Assume 10cm resolution
            self.grid_xmin = -10.5
            self.grid_xmax = 10.5
            self.grid_ymin = -13.0
            self.grid_ymax = 4.5

        # Load robot detections
        self.robot_data = {}  # robot_id -> list of timestep data
        self._load_robot_detections()

        # Determine number of timesteps (all robots should have same length)
        self.num_timesteps = len(next(iter(self.robot_data.values())))

        # Build ground truth object database
        self.ground_truth_objects = {}  # gid -> {positions_per_timestep, avg_position}
        self._build_ground_truth_objects()

        # Current timestep
        self.current_timestep = 0

    def _load_robot_detections(self):
        """Load detection data for all robots."""
        detections_dir = self.data_path / "detections"

        for robot_dir in sorted(detections_dir.iterdir()):
            if not robot_dir.is_dir() or robot_dir.name.startswith('.'):
                continue

            robot_id = robot_dir.name  # e.g., "SPOT_0"
            detection_file = robot_dir / "detections.json"

            if detection_file.exists():
                with open(detection_file, 'r') as f:
                    self.robot_data[robot_id] = json.load(f)

        print(f"Loaded data for {len(self.robot_data)} robots")
        print(f"Robots: {list(self.robot_data.keys())}")

    def _build_ground_truth_objects(self):
        """
        Build ground truth object database by aggregating detections across all robots.

        For each object (identified by gid), we:
        1. Collect all detections from all robots at all timesteps
        2. Average the positions to get ground truth position per timestep
        3. Store the trajectory
        """
        # Collect all detections per object per timestep
        object_detections_per_timestep = defaultdict(lambda: defaultdict(list))

        for robot_id, timesteps in self.robot_data.items():
            for t_idx, timestep_data in enumerate(timesteps):
                visible = timestep_data.get("visible", {})

                # Use only "union" detections (merged from both cameras)
                # If "union" doesn't exist, fall back to aggregating all cameras
                if "union" in visible:
                    detections_to_process = visible["union"]
                else:
                    # Fallback: aggregate from all cameras
                    detections_to_process = []
                    for camera_name, detections in visible.items():
                        detections_to_process.extend(detections)

                for det in detections_to_process:
                    gid = det["gid"]
                    pos = det["pos"]

                    # Store position (x, y, z)
                    object_detections_per_timestep[gid][t_idx].append([
                        pos["x"], pos["y"], pos["z"]
                    ])

        # Average positions across all detecting robots per timestep
        for gid, timestep_detections in object_detections_per_timestep.items():
            positions_per_timestep = {}

            for t_idx, positions in timestep_detections.items():
                if positions:
                    # Average position across all robots that detected it
                    avg_pos = np.mean(positions, axis=0)
                    positions_per_timestep[t_idx] = avg_pos

            # Calculate overall average position (across all timesteps)
            all_positions = list(positions_per_timestep.values())
            if all_positions:
                avg_position = np.mean(all_positions, axis=0)
            else:
                avg_position = np.array([0, 0, 0])

            self.ground_truth_objects[gid] = {
                "positions_per_timestep": positions_per_timestep,
                "avg_position": avg_position,
                "first_seen": min(positions_per_timestep.keys()) if positions_per_timestep else 0,
                "last_seen": max(positions_per_timestep.keys()) if positions_per_timestep else 0,
                "shape": self._get_object_shape(gid),
            }

        print(f"Identified {len(self.ground_truth_objects)} ground truth objects")

    def _get_object_shape(self, gid: str) -> Dict:
        """
        Get object shape information from gid.

        Args:
            gid: Ground truth object ID (e.g., "DEF:WoodenBox_1", "DEF:GasCanister_2")
                 or false positive object ID (e.g., "fp_obj_1")

        Returns:
            Dict with 'type' and dimensions:
            - 'rectangle': {'type': 'rectangle', 'width': float, 'length': float}
            - 'circle': {'type': 'circle', 'radius': float}
        """
        gid_lower = gid.lower()

        # False positive objects are 0.6x0.6 boxes
        if 'fp_obj' in gid_lower or 'fp' in gid_lower:
            return {'type': 'rectangle', 'width': 0.6, 'length': 0.6}
        # WoodenBox and CardboardBox are 0.6x0.6 squares
        elif 'woodbox' in gid_lower or 'cardbox' in gid_lower:
            return {'type': 'rectangle', 'width': 0.6, 'length': 0.6}
        # GasCanister is a circle with radius 0.175
        elif 'canister' in gid_lower:
            return {'type': 'circle', 'radius': 0.175}
        # Human is a 0.35x0.35 box
        elif 'human' in gid_lower:
            return {'type': 'rectangle', 'width': 0.35, 'length': 0.35}
        # Forklift is a large 1.0x4.0 rectangle
        elif 'forklift' in gid_lower:
            return {'type': 'rectangle', 'width': 1.0, 'length': 4.0}
        else:
            # Default to small square for unknown types
            return {'type': 'rectangle', 'width': 0.3, 'length': 0.3}

    def get_robot_positions(self, timestep: int) -> Dict[str, Tuple[float, float, float]]:
        """
        Get robot positions at a specific timestep.

        Args:
            timestep: Timestep index (0 to num_timesteps-1)

        Returns:
            Dict mapping robot_id -> (x, y, yaw)
        """
        positions = {}

        for robot_id, timesteps in self.robot_data.items():
            if timestep < len(timesteps):
                robot_pose = timesteps[timestep]["robot_pose"]
                positions[robot_id] = (
                    robot_pose["x"],
                    robot_pose["y"],
                    robot_pose["yaw"]
                )

        return positions

    def get_ground_truth_object_positions(self, timestep: int) -> Dict[str, Tuple[float, float, float]]:
        """
        Get ground truth object positions at a specific timestep.

        Args:
            timestep: Timestep index

        Returns:
            Dict mapping gid -> (x, y, z)
        """
        positions = {}

        for gid, obj_data in self.ground_truth_objects.items():
            if timestep in obj_data["positions_per_timestep"]:
                pos = obj_data["positions_per_timestep"][timestep]
                positions[gid] = tuple(pos)

        return positions

    def get_robot_detections(self, robot_id: str, timestep: int) -> List[Dict]:
        """
        Get all detections from a robot at a specific timestep.

        Args:
            robot_id: Robot identifier (e.g., "SPOT_0")
            timestep: Timestep index

        Returns:
            List of detection dicts with {gid, pos: {x, y, z}, yaw}
        """
        if robot_id not in self.robot_data:
            return []

        if timestep >= len(self.robot_data[robot_id]):
            return []

        timestep_data = self.robot_data[robot_id][timestep]
        visible = timestep_data.get("visible", {})

        # Aggregate detections from all cameras
        all_detections = []
        for camera_name, detections in visible.items():
            all_detections.extend(detections)

        return all_detections

    def get_occupancy_grid(self) -> np.ndarray:
        """
        Get the occupancy grid map.

        Returns:
            2D numpy array (0=free, 1=occupied)
        """
        return self.occ_grid

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid indices.

        Args:
            x: World X coordinate (meters)
            y: World Y coordinate (meters)

        Returns:
            (row, col) indices in the occupancy grid
        """
        col = int(np.floor((x - self.grid_xmin) / self.resolution))
        row = int(np.floor((y - self.grid_ymin) / self.resolution))

        # Clamp to grid bounds
        row = np.clip(row, 0, self.grid_height - 1)
        col = np.clip(col, 0, self.grid_width - 1)

        return row, col

    def grid_to_world_center(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert grid indices to world coordinates (cell center).

        Args:
            row: Grid row index
            col: Grid column index

        Returns:
            (x, y) world coordinates of cell center
        """
        x = self.grid_xmin + (col + 0.5) * self.resolution
        y = self.grid_ymin + (row + 0.5) * self.resolution
        return x, y

    def get_grid_extent(self) -> Tuple[float, float, float, float]:
        """
        Get the world extent of the occupancy grid for visualization.

        Returns:
            (xmin, xmax, ymin, ymax) in world coordinates
        """
        return (self.grid_xmin, self.grid_xmax, self.grid_ymin, self.grid_ymax)

    def get_world_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get world bounds. Uses grid metadata if available.

        Returns:
            (min_x, max_x, min_y, max_y)
        """
        if self.grid_meta is not None:
            # Use grid coverage as world bounds
            return (self.grid_xmin, self.grid_xmax, self.grid_ymin, self.grid_ymax)
        else:
            # Fallback: infer from data
            all_x = []
            all_y = []

            # Collect all robot positions
            for timesteps in self.robot_data.values():
                for timestep_data in timesteps:
                    pose = timestep_data["robot_pose"]
                    all_x.append(pose["x"])
                    all_y.append(pose["y"])

            # Collect all object positions
            for obj_data in self.ground_truth_objects.values():
                for pos in obj_data["positions_per_timestep"].values():
                    all_x.append(pos[0])
                    all_y.append(pos[1])

            if all_x and all_y:
                margin = 2.0  # Add margin
                return (
                    min(all_x) - margin,
                    max(all_x) + margin,
                    min(all_y) - margin,
                    max(all_y) + margin
                )
            else:
                return (-10, 10, -10, 10)  # Default bounds

    def reset(self):
        """Reset to timestep 0."""
        self.current_timestep = 0

    def step(self):
        """Advance to next timestep."""
        if self.current_timestep < self.num_timesteps - 1:
            self.current_timestep += 1
        return self.current_timestep

    def get_summary(self) -> str:
        """Get a summary of the loaded data."""
        summary = []
        summary.append(f"Webots Simulation Environment")
        summary.append(f"=" * 60)
        summary.append(f"Occupancy Grid: {self.grid_height} x {self.grid_width}")
        summary.append(f"Number of Robots: {len(self.robot_data)}")
        summary.append(f"Number of Timesteps: {self.num_timesteps}")
        summary.append(f"Ground Truth Objects: {len(self.ground_truth_objects)}")

        world_bounds = self.get_world_bounds()
        summary.append(f"World Bounds: X=[{world_bounds[0]:.1f}, {world_bounds[1]:.1f}], Y=[{world_bounds[2]:.1f}, {world_bounds[3]:.1f}]")

        # Object summary
        summary.append(f"\nGround Truth Objects:")
        for gid, obj_data in sorted(self.ground_truth_objects.items())[:10]:  # Show first 10
            avg_pos = obj_data["avg_position"]
            summary.append(f"  {gid}: avg_pos=({avg_pos[0]:.2f}, {avg_pos[1]:.2f}, {avg_pos[2]:.2f})")

        if len(self.ground_truth_objects) > 10:
            summary.append(f"  ... and {len(self.ground_truth_objects) - 10} more")

        return "\n".join(summary)


def main():
    """Test the Webots simulation environment."""
    env = WebotsSimulationEnvironment("webots_sim")

    print(env.get_summary())
    print()

    # Test timestep 0
    print("=" * 60)
    print("Timestep 0 Data:")
    print("=" * 60)

    robot_positions = env.get_robot_positions(0)
    print(f"\nRobot Positions:")
    for robot_id, (x, y, yaw) in robot_positions.items():
        print(f"  {robot_id}: ({x:.2f}, {y:.2f}, yaw={yaw:.2f})")

    gt_positions = env.get_ground_truth_object_positions(0)
    print(f"\nGround Truth Objects ({len(gt_positions)} visible):")
    for gid, (x, y, z) in list(gt_positions.items())[:5]:
        print(f"  {gid}: ({x:.2f}, {y:.2f}, {z:.2f})")

    # Test robot detections
    print(f"\nSPOT_0 Detections at t=0:")
    detections = env.get_robot_detections("SPOT_0", 0)
    print(f"  Total detections: {len(detections)}")
    for det in detections[:3]:
        pos = det["pos"]
        print(f"    {det['gid']}: ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})")


if __name__ == "__main__":
    main()
