#!/usr/bin/env python3
"""
Create an animated GIF visualization of the trust-based sensor fusion simulation
showing robots, FOVs, detections, ground truth objects, and false positives over time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import json

def convert_raw_data_to_animation_format(raw_data):
    """Convert raw simulation data to format expected by animation

    Handles two formats:
    1. Direct simulation format: frames have 'robot_states', 'ground_truth', etc.
    2. Comparison results format: frames have 'frame_state' with 'robots', 'detections', etc.
    """
    simulation_data = []

    for frame in raw_data:
        frame_data = {
            'time': frame.get('time', 0.0),
            'robots': [],
            'ground_truth_objects': [],
            'detections': {},
            'false_positives': {}
        }

        # Check if this is comparison results format (has frame_state)
        if 'frame_state' in frame:
            # Comparison results format - extract from frame_state
            frame_state = frame['frame_state']
            robot_trust_values = frame.get('robot_trust_values', {})

            # Robots are already in the right format
            for robot in frame_state.get('robots', []):
                robot_id = robot['id']
                # Use trust from robot_trust_values if available, otherwise from robot data
                trust_value = robot_trust_values.get(str(robot_id), robot.get('trust_value', 0.5))

                frame_data['robots'].append({
                    'id': robot_id,
                    'position': robot['position'][:2],
                    'orientation': robot.get('orientation', 0.0),
                    'fov_range': robot.get('fov_range', 50.0),
                    'fov_angle': robot.get('fov_angle', np.pi/3),
                    'is_adversarial': robot.get('is_adversarial', False),
                    'trust': trust_value
                })

                # Initialize detection lists
                frame_data['detections'][robot_id] = []
                frame_data['false_positives'][robot_id] = []

            # Get detections from frame_state
            # Detections is a dict mapping robot_id (string) -> list of detection objects
            detections_dict = frame_state.get('detections', {})

            # Track unique ground truth and FP objects for the ground truth layer
            gt_objects_seen = {}  # object_id -> position
            fp_objects_seen = {}  # object_id -> position

            for robot_id_str, robot_detections in detections_dict.items():
                robot_id = int(robot_id_str)
                for detection in robot_detections:
                    det_data = {
                        'position': detection['position'][:2],
                        'object_id': detection.get('object_id', ''),
                        'trust_mean': detection.get('trust_value', 0.5),
                        'confidence': detection.get('confidence', 1.0)
                    }

                    object_id = detection.get('object_id', '')
                    position = detection['position'][:2]

                    if 'gt_obj_' in object_id:
                        det_data['type'] = 'gt_detection'
                        frame_data['detections'][robot_id].append(det_data)
                        # Track unique GT objects for visualization
                        if object_id not in gt_objects_seen:
                            gt_objects_seen[object_id] = position
                    elif 'fp_' in object_id:
                        det_data['type'] = 'fp_detection'
                        frame_data['false_positives'][robot_id].append(det_data)
                        # Track unique FP objects for visualization
                        if object_id not in fp_objects_seen:
                            fp_objects_seen[object_id] = position

            # Add ground truth objects to frame data for visualization
            for obj_id, position in gt_objects_seen.items():
                frame_data['ground_truth_objects'].append({
                    'id': obj_id,
                    'position': position,
                    'object_type': 'ground_truth',
                    'movement_pattern': 'unknown'  # Not available in comparison format
                })

            # Add FP objects to frame data for visualization
            for obj_id, position in fp_objects_seen.items():
                frame_data['ground_truth_objects'].append({
                    'id': obj_id,
                    'position': position,
                    'object_type': 'false_positive',
                    'movement_pattern': 'unknown'  # Not available in comparison format
                })

        else:
            # Direct simulation format - use old logic
            trust_updates = frame.get('trust_updates', {})

            # Convert robot states
            for robot_state in frame['robot_states']:
                robot_id, position, velocity, is_adversarial, fov_range, fov_angle = robot_state

                # Extract actual trust from trust_updates
                robot_trust_info = trust_updates.get(str(robot_id), {})
                trust_mean = robot_trust_info.get('mean_trust', 0.5)  # Use actual trust or default

                # Calculate orientation from velocity (direction of movement)
                if velocity[0] != 0 or velocity[1] != 0:
                    orientation = np.arctan2(velocity[1], velocity[0])
                else:
                    orientation = 0.0

                frame_data['robots'].append({
                    'id': robot_id,
                    'position': position[:2],  # Only x, y coordinates
                    'orientation': orientation,  # Actual orientation from movement
                    'fov_range': fov_range,  # Correct FOV range from simulation
                    'fov_angle': fov_angle,  # 60 degrees FOV angle
                    'is_adversarial': is_adversarial,
                    'trust': trust_mean  # Actual trust value
                })

                # Initialize detection and false positive lists for each robot
                frame_data['detections'][robot_id] = []
                frame_data['false_positives'][robot_id] = []

            # Convert ground truth objects
            gt_data = frame.get('ground_truth', {})
            gt_objects = gt_data.get('objects', [])
            for gt_obj in gt_objects:
                obj_id, position, _, obj_type, movement_pattern = gt_obj
                frame_data['ground_truth_objects'].append({
                    'id': obj_id,
                    'position': position[:2],  # Only x, y coordinates
                    'object_type': obj_type,
                    'movement_pattern': movement_pattern
                })

            # Convert FP objects (persistent) - these show the actual FP object positions
            shared_fp_objects = frame.get('shared_fp_objects', [])
            for fp_data in shared_fp_objects:
                fp_id, position, velocity, obj_type, movement_pattern = fp_data
                # Add FP objects to the ground truth objects list for visualization (as orange stars)
                frame_data['ground_truth_objects'].append({
                    'id': fp_id,
                    'position': position[:2],  # Only x, y coordinates
                    'object_type': 'false_positive',
                    'movement_pattern': movement_pattern
                })

            # ALSO convert FP detection tracks for debugging comparison
            tracks = frame.get('tracks', {})
            for robot_id_str, robot_tracks in tracks.items():
                robot_id = int(robot_id_str)  # Convert string to int
                for track in robot_tracks:
                    object_id, position, confidence, trust_alpha, trust_beta = track
                    track_data = {
                        'position': position[:2],  # Only x, y coordinates
                        'object_id': object_id,
                        'trust_mean': trust_alpha / (trust_alpha + trust_beta) if (trust_alpha + trust_beta) > 0 else 0.5,
                        'confidence': confidence
                    }

                    # Include both GT detections and FP detections for debugging
                    if 'gt_obj_' in object_id:
                        track_data['type'] = 'gt_detection'
                        frame_data['detections'][robot_id].append(track_data)
                    elif 'fp_' in object_id:
                        track_data['type'] = 'fp_detection'
                        frame_data['false_positives'][robot_id].append(track_data)
        
        simulation_data.append(frame_data)
    
    return simulation_data

def create_simulation_gif(data_file='trust_simulation_data.json', method='paper'):
    """Create animated GIF from existing simulation data

    Args:
        data_file: Path to JSON file (can be comparison results or direct simulation data)
        method: Which method to visualize ('paper', 'supervised', or 'bayesian') if using comparison results
    """
    print("Creating Trust-Based Sensor Fusion Simulation GIF...")
    print(f"Loading data from: {data_file}")

    # Load existing simulation data
    try:
        with open(data_file, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {data_file} not found. Please run the simulation first.")
        return

    # Check if this is comparison results format or direct simulation format
    if isinstance(raw_data, dict) and ('paper_results' in raw_data or 'supervised_results' in raw_data or 'bayesian_results' in raw_data):
        # This is comparison results format - extract the specified method
        print(f"Detected comparison results format. Extracting '{method}' method data...")
        method_key = f"{method}_results"
        if method_key not in raw_data:
            print(f"Error: Method '{method}' not found in data. Available methods: {[k.replace('_results', '') for k in raw_data.keys() if k.endswith('_results')]}")
            return
        frames_data = raw_data[method_key]
    else:
        # This is direct simulation format (array of frames)
        frames_data = raw_data

    # Convert raw data to format expected by animation
    simulation_data = convert_raw_data_to_animation_format(frames_data)
    num_frames = len(simulation_data)

    if num_frames == 0:
        print("Error: No simulation data found.")
        return

    print(f"Loaded {num_frames} frames of simulation data")

    # Determine world size
    if frames_data and isinstance(frames_data[0], dict):
        # Try to get from frame_state first
        if 'frame_state' in frames_data[0]:
            world_size = tuple(frames_data[0]['frame_state'].get('world_size', [100.0, 100.0]))
        else:
            world_size = tuple(frames_data[0].get('world_size', [100.0, 100.0]))
    else:
        world_size = (100.0, 100.0)
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, world_size[0])
    ax.set_ylim(0, world_size[1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=32)
    
    # Colors for robots (consistent throughout animation)
    num_robots = len(simulation_data[0]['robots']) if simulation_data else 4
    
    print("Creating animation...")
    
    def animate_frame(frame):
        """Animation function for each frame"""
        ax.clear()
        ax.set_xlim(0, world_size[0])
        ax.set_ylim(0, world_size[1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=32)

        data = simulation_data[frame]

        # Plot ground truth objects (green circles) and FP objects (red circles)
        for gt_obj in data['ground_truth_objects']:
            if gt_obj.get('object_type') == 'false_positive':
                # FP objects: red circles
                color = 'red'
            else:
                # GT objects: green circles
                color = 'green'

            ax.scatter(gt_obj['position'][0], gt_obj['position'][1],
                      c=color, marker='o', s=200, alpha=0.8,
                      edgecolors='black', linewidth=1.5)

        # Plot robots with FOV
        for i, robot in enumerate(data['robots']):
            pos = robot['position']

            # All robots are custom blue boxes
            marker = 's'  # Square for all robots
            color = '#6093D3'
            robot_size = 200

            ax.scatter(pos[0], pos[1], c=color, marker=marker, s=robot_size,
                      edgecolors='black', linewidth=2, alpha=0.9)

            # Field of View (FOV) visualization
            fov_angles = np.linspace(robot['orientation'] - robot['fov_angle']/2,
                                   robot['orientation'] + robot['fov_angle']/2, 30)

            # FOV triangle
            fov_x = [pos[0]]
            fov_y = [pos[1]]
            for angle in fov_angles:
                fov_x.append(pos[0] + robot['fov_range'] * np.cos(angle))
                fov_y.append(pos[1] + robot['fov_range'] * np.sin(angle))
            fov_x.append(pos[0])
            fov_y.append(pos[1])

            # FOV fill - transparent lightblue
            ax.fill(fov_x, fov_y, color='lightblue', alpha=0.15)
            ax.plot(fov_x, fov_y, color='lightblue', linewidth=1.5, alpha=0.4)
    
    # Create animation (5x faster: reduced interval and increased fps)
    print("Rendering GIF (this may take a moment)...")
    anim = FuncAnimation(fig, animate_frame, frames=num_frames, 
                        interval=40, repeat=True, blit=False)  # 40ms instead of 200ms
    
    # Save as GIF
    writer = PillowWriter(fps=25)  # 25fps instead of 5fps for 5x speed
    gif_path = 'trust_sensor_fusion_simulation.gif'
    anim.save(gif_path, writer=writer, dpi=100)
    
    plt.close(fig)
    
    print(f"✅ GIF saved as: {gif_path}")
    print(f"   Frames: {num_frames}")
    print(f"   Duration: ~{num_frames/25:.1f} seconds (5x faster)")
    print(f"   File size: Check the generated file")
    
    # Create a summary frame as well
    create_summary_plot((0, world_size[0], 0, world_size[1]), simulation_data)

def create_summary_plot(world_bounds, simulation_data):
    """Create a static summary plot showing the final state"""
    print("Creating summary plot...")
    
    final_data = simulation_data[-1]
    x_min, x_max, y_min, y_max = world_bounds
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: Final simulation state
    ax1.set_xlim(x_min - 2, x_max + 2)
    ax1.set_ylim(y_min - 2, y_max + 2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=32)

    num_robots = len(final_data['robots']) if final_data['robots'] else 4
    robot_colors = plt.cm.Set1(np.linspace(0, 1, num_robots))

    # Plot ground truth objects (green circles) and FP objects (red circles)
    for gt_obj in final_data['ground_truth_objects']:
        if gt_obj.get('object_type') == 'false_positive':
            # FP objects: red circles
            color = 'red'
        else:
            # GT objects: green circles
            color = 'green'

        ax1.scatter(gt_obj['position'][0], gt_obj['position'][1],
                   c=color, marker='o', s=200, alpha=0.8,
                   edgecolors='black', linewidth=1.5)

    # Plot robots with FOV
    for i, robot in enumerate(final_data['robots']):
        pos = robot['position']

        # All robots are custom blue boxes
        marker = 's'
        color = '#6093D3'
        robot_size = 200

        ax1.scatter(pos[0], pos[1], c=color, marker=marker, s=robot_size,
                   edgecolors='black', linewidth=2, alpha=0.9)

        # Field of View (FOV) visualization
        fov_angles = np.linspace(robot['orientation'] - robot['fov_angle']/2,
                               robot['orientation'] + robot['fov_angle']/2, 30)

        # FOV triangle
        fov_x = [pos[0]]
        fov_y = [pos[1]]
        for angle in fov_angles:
            fov_x.append(pos[0] + robot['fov_range'] * np.cos(angle))
            fov_y.append(pos[1] + robot['fov_range'] * np.sin(angle))
        fov_x.append(pos[0])
        fov_y.append(pos[1])

        # FOV fill - transparent lightblue
        ax1.fill(fov_x, fov_y, color='lightblue', alpha=0.15)
        ax1.plot(fov_x, fov_y, color='lightblue', linewidth=1.5, alpha=0.4)
    
    # Right plot: Trust evolution over time
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Robot Trust', fontsize=12)
    ax2.set_title('Trust Evolution Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    times = [data['time'] for data in simulation_data]
    
    for i, robot_id in enumerate(range(len(final_data['robots']))):
        robot_trusts = []
        robot_type = None
        
        for data in simulation_data:
            robot_data = next(r for r in data['robots'] if r['id'] == robot_id)
            robot_trusts.append(robot_data['trust'])
            if robot_type is None:
                robot_type = 'ADV' if robot_data['is_adversarial'] else 'LEG'
        
        linestyle = '--' if robot_type == 'ADV' else '-'
        ax2.plot(times, robot_trusts, color=robot_colors[i], linewidth=2, 
                linestyle=linestyle, alpha=0.8, 
                label=f'Robot {robot_id} ({robot_type})')
    
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('simulation_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("✅ Summary plot saved as: simulation_summary.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create animated GIF from trust simulation data')
    parser.add_argument('data_file', nargs='?', default='trust_simulation_data.json',
                        help='Path to JSON file (can be comparison results or direct simulation data)')
    parser.add_argument('--method', default='paper', choices=['paper', 'supervised', 'bayesian'],
                        help='Which method to visualize (for comparison results)')
    args = parser.parse_args()

    create_simulation_gif(args.data_file, method=args.method)