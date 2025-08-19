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
    """Convert raw simulation data to format expected by animation"""
    simulation_data = []
    
    for frame in raw_data:
        frame_data = {
            'time': frame['time'],
            'robots': [],
            'ground_truth_objects': [],
            'detections': {},
            'false_positives': {}
        }
        
        # Get trust updates for this frame
        trust_updates = frame.get('trust_updates', {})
        
        # Convert robot states
        for robot_state in frame['robot_states']:
            robot_id, position, velocity, is_adversarial = robot_state
            
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
                'fov_range': 20.0,  # Correct FOV range from simulation
                'fov_angle': np.pi/3,  # 60 degrees FOV angle
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
        fp_objects = frame.get('fp_objects', {})
        for robot_id_str, fp_list in fp_objects.items():
            robot_id = int(robot_id_str)  # Convert string to int
            for fp_data in fp_list:
                fp_id, position, velocity = fp_data
                fp_obj_data = {
                    'position': position[:2],  # Only x, y coordinates
                    'object_id': fp_id,
                    'trust_mean': 0.3,  # Default low trust for FP objects
                    'confidence': 0.6,   # Default confidence for FP objects
                    'type': 'fp_object'  # Mark as FP object
                }
                frame_data['false_positives'][robot_id].append(fp_obj_data)
        
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

def create_simulation_gif(data_file='trust_simulation_data.json'):
    """Create animated GIF from existing simulation data"""
    print("Creating Trust-Based Sensor Fusion Simulation GIF...")
    print(f"Loading data from: {data_file}")
    
    # Load existing simulation data
    try:
        with open(data_file, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {data_file} not found. Please run the simulation first.")
        return
    
    # Convert raw data to format expected by animation
    simulation_data = convert_raw_data_to_animation_format(raw_data)
    num_frames = len(simulation_data)
    
    if num_frames == 0:
        print("Error: No simulation data found.")
        return
    
    print(f"Loaded {num_frames} frames of simulation data")
    
    # Determine world size - use ground truth objects to set overall bounds
    # but ensure robot movement area is clearly visible
    all_gt_positions = []
    all_robot_positions = []
    
    for frame_data in simulation_data:
        for robot in frame_data['robots']:
            all_robot_positions.append(robot['position'])
        for gt_obj in frame_data['ground_truth_objects']:
            all_gt_positions.append(gt_obj['position'])
    
    if all_gt_positions:
        gt_positions_array = np.array(all_gt_positions)
        x_min, x_max = gt_positions_array[:, 0].min(), gt_positions_array[:, 0].max()
        y_min, y_max = gt_positions_array[:, 1].min(), gt_positions_array[:, 1].max()
        world_size = (x_max - x_min + 10, y_max - y_min + 10)  # Add padding
        
        # Check robot positions to ensure they're visible
        if all_robot_positions:
            robot_positions_array = np.array(all_robot_positions)
            robot_x_min, robot_x_max = robot_positions_array[:, 0].min(), robot_positions_array[:, 0].max()
            robot_y_min, robot_y_max = robot_positions_array[:, 1].min(), robot_positions_array[:, 1].max()
            
            # Expand world bounds if needed to include robot area with good visibility
            x_min = min(x_min, robot_x_min - 5)
            x_max = max(x_max, robot_x_max + 5)
            y_min = min(y_min, robot_y_min - 5)
            y_max = max(y_max, robot_y_max + 5)
            world_size = (x_max - x_min, y_max - y_min)
    else:
        world_size = (50.0, 50.0)  # Default size
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(x_min - 2, x_max + 2)
    ax.set_ylim(y_min - 2, y_max + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    
    # Colors for robots (consistent throughout animation)
    num_robots = len(simulation_data[0]['robots']) if simulation_data else 4
    robot_colors = plt.cm.Set1(np.linspace(0, 1, num_robots))
    
    print("Creating animation...")
    
    def animate_frame(frame):
        """Animation function for each frame"""
        ax.clear()
        ax.set_xlim(x_min - 2, x_max + 2)
        ax.set_ylim(y_min - 2, y_max + 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        
        data = simulation_data[frame]
        
        # Title with time and trust info
        title_parts = [f'Trust-Based Sensor Fusion (t={data["time"]:.1f}s)']
        
        # Calculate average trust for each type
        leg_trusts = [r['trust'] for r in data['robots'] if not r['is_adversarial']]
        adv_trusts = [r['trust'] for r in data['robots'] if r['is_adversarial']]
        
        if leg_trusts:
            title_parts.append(f'LEG Trust: {np.mean(leg_trusts):.2f}')
        if adv_trusts:
            title_parts.append(f'ADV Trust: {np.mean(adv_trusts):.2f}')
            
        ax.set_title(' | '.join(title_parts), fontsize=14, fontweight='bold')
        
        # Plot ground truth objects (all use same star symbol, same blue color)
        for gt_obj in data['ground_truth_objects']:
            ax.scatter(gt_obj['position'][0], gt_obj['position'][1],
                      c='blue', marker='*', s=300, alpha=0.9,
                      edgecolors='black', linewidth=2, 
                      label='Ground Truth' if gt_obj['id'] == 0 else '')
            
            # Add GT object ID label
            ax.text(gt_obj['position'][0], gt_obj['position'][1] + 1.2, f'GT{gt_obj["id"]}', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Plot robots with FOV and detections
        for i, robot in enumerate(data['robots']):
            pos = robot['position']
            
            # All robots use square marker, color based on type
            marker = 's'  # Square for all robots
            if robot['is_adversarial']:
                color = 'red'
                robot_type = 'ADV'
            else:
                color = 'green'
                robot_type = 'LEG'
            
            # Robot size based on trust (higher trust = larger)
            robot_size = 100 + robot['trust'] * 200
            
            ax.scatter(pos[0], pos[1], c=color, marker=marker, s=robot_size,
                      edgecolors='white', linewidth=2, alpha=0.9,
                      label=f'{robot_type} Robot' if (robot_type == 'LEG' and i == 0) or (robot_type == 'ADV' and robot['is_adversarial'] and not any(r['is_adversarial'] and r['id'] < robot['id'] for r in data['robots'])) else '')
            
            # Robot ID text
            ax.text(pos[0], pos[1] - 1.5, f'R{robot["id"]}\\n{robot["trust"]:.2f}', 
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
            
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
            
            # FOV fill (make adversarial FOV more visible)
            fov_alpha = 0.15 if robot['is_adversarial'] else 0.1  # Increased from 0.05 to 0.15
            ax.fill(fov_x, fov_y, color=color, alpha=fov_alpha)
            ax.plot(fov_x, fov_y, color=color, linewidth=2, alpha=0.6)  # Stronger outline
            
            # Plot false positive objects only (same symbol as GT, different color)
            for fp in data['false_positives'][robot['id']]:
                fp_pos = fp['position']
                fp_type = fp.get('type', 'unknown')
                
                # Only show FP objects, not FP detections
                if fp_type == 'fp_object':
                    # FP Objects: Use same star symbol as GT objects but orange color
                    ax.scatter(fp_pos[0], fp_pos[1], 
                              c='orange', marker='*', s=300, alpha=0.9, 
                              edgecolors='black', linewidth=2)
                    
                    # Add FP object ID label (similar to GT objects)
                    ax.text(fp_pos[0], fp_pos[1] + 1.2, f'FP{fp["object_id"].replace("fp_", "")}', 
                           ha='center', va='bottom', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Add consistent legend for all frames
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=12, label='Ground Truth Objects'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', markeredgecolor='black', 
                   markeredgewidth=2, markersize=12, label='False Positive Objects'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Legitimate Robots'), 
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Adversarial Robots'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
        
        # Add info box
        info_text = f"Ground Truth Objects: {len(data['ground_truth_objects'])}\\n"
        
        # Count only FP objects (no longer showing FP detections)
        total_fp_objects = 0
        for fps in data['false_positives'].values():
            for fp in fps:
                if fp.get('type') == 'fp_object':
                    total_fp_objects += 1
        
        info_text += f"False Positive Objects: {total_fp_objects}\\n"
        
        # Add robot trust info
        leg_trusts = [r['trust'] for r in data['robots'] if not r['is_adversarial']]
        adv_trusts = [r['trust'] for r in data['robots'] if r['is_adversarial']]
        if leg_trusts:
            info_text += f"Avg LEG Trust: {np.mean(leg_trusts):.2f}\\n"
        if adv_trusts:
            info_text += f"Avg ADV Trust: {np.mean(adv_trusts):.2f}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
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
    create_summary_plot((x_min, x_max, y_min, y_max), simulation_data)

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
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('Final Simulation State', fontsize=14, fontweight='bold')
    
    num_robots = len(final_data['robots']) if final_data['robots'] else 4
    robot_colors = plt.cm.Set1(np.linspace(0, 1, num_robots))
    
    # Plot everything as in the animation
    for gt_obj in final_data['ground_truth_objects']:
        marker_map = {'stationary': 's', 'linear': '>', 'random_walk': 'd', 'circular': 'o'}
        color_map = {'vehicle': 'red', 'person': 'orange', 'animal': 'brown'}
        
        marker = marker_map.get(gt_obj['movement_pattern'], 'o')
        color = color_map.get(gt_obj['object_type'], 'red')
        
        ax1.scatter(gt_obj['position'][0], gt_obj['position'][1],
                   c=color, marker=marker, s=200, alpha=0.9,
                   edgecolors='black', linewidth=2)
    
    for i, robot in enumerate(final_data['robots']):
        color = robot_colors[i]
        pos = robot['position']
        marker = '^' if robot['is_adversarial'] else 'o'
        robot_size = 100 + robot['trust'] * 200
        
        ax1.scatter(pos[0], pos[1], c=[color], marker=marker, s=robot_size,
                   edgecolors='white', linewidth=2, alpha=0.9)
        
        # FOV
        fov_angles = np.linspace(robot['orientation'] - robot['fov_angle']/2,
                               robot['orientation'] + robot['fov_angle']/2, 30)
        fov_x = [pos[0]] + [pos[0] + robot['fov_range'] * np.cos(a) for a in fov_angles] + [pos[0]]
        fov_y = [pos[1]] + [pos[1] + robot['fov_range'] * np.sin(a) for a in fov_angles] + [pos[1]]
        
        fov_alpha = 0.05 if robot['is_adversarial'] else 0.1
        ax1.fill(fov_x, fov_y, color=color, alpha=fov_alpha)
        
        # Detections
        for detection in final_data['detections'][robot['id']]:
            det_pos = detection['position']
            det_size = 30 + detection['confidence'] * 40 + detection['trust_mean'] * 30
            ax1.scatter(det_pos[0], det_pos[1], c=[color], marker='+', s=det_size, alpha=0.8)
        
        for fp in final_data['false_positives'][robot['id']]:
            fp_pos = fp['position']
            fp_size = 20 + fp['confidence'] * 20
            ax1.scatter(fp_pos[0], fp_pos[1], c='red', marker='x', s=fp_size, alpha=0.7)
    
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
    create_simulation_gif()