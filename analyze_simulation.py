#!/usr/bin/env python3
"""
Analyze the simulation data to understand trust estimation and robot behavior.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_simulation_data(filename='paper_trust_data.json'):
    """Analyze the collected simulation data"""
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} simulation steps")
    
    # Get ground truth information from first step
    first_step = data[0]
    ground_truth = first_step['ground_truth']
    
    print("\nGround Truth:")
    print(f"Objects: {ground_truth['num_objects']} (dynamic)")
    print(f"Legitimate Robots: {ground_truth['legitimate_robots']}")
    print(f"Adversarial Robots: {ground_truth['adversarial_robots']}")
    
    # Analyze object types and movement patterns
    if 'objects' in ground_truth:
        object_types = [obj[3] for obj in ground_truth['objects']]  # obj_type is index 3
        movement_patterns = [obj[4] for obj in ground_truth['objects']]  # movement_pattern is index 4
        print(f"Object types: {set(object_types)}")
        print(f"Movement patterns: {set(movement_patterns)}")
    
    # Analyze track generation patterns
    legitimate_track_counts = []
    adversarial_track_counts = []
    
    print("\nTrack Generation Analysis:")
    for step in data[:10]:  # First 10 steps
        print(f"\nTime {step['time']:.1f}s:")
        for robot_id, tracks in step['tracks'].items():
            robot_id = int(robot_id)
            is_adversarial = robot_id in ground_truth['adversarial_robots']
            robot_type = "ADV" if is_adversarial else "LEG"
            
            print(f"  Robot {robot_id} ({robot_type}): {len(tracks)} tracks")
            
            if is_adversarial:
                adversarial_track_counts.append(len(tracks))
            else:
                legitimate_track_counts.append(len(tracks))
    
    # Analyze trust updates
    print("\nTrust Update Analysis:")
    trust_history = {}
    
    for step in data:
        if 'trust_updates' in step:
            for robot_id, trust_info in step['trust_updates'].items():
                robot_id = int(robot_id)
                if robot_id not in trust_history:
                    trust_history[robot_id] = {
                        'times': [],
                        'trust_means': [],
                        'num_psms': [],
                        'is_adversarial': robot_id in ground_truth['adversarial_robots']
                    }
                
                trust_history[robot_id]['times'].append(step['time'])
                trust_history[robot_id]['trust_means'].append(trust_info['mean_trust'])
                # Handle both old and new format
                num_psms = trust_info.get('num_psms', trust_info.get('num_agent_psms', 0))
                trust_history[robot_id]['num_psms'].append(num_psms)
    
    # Plot trust evolution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for robot_id, history in trust_history.items():
        robot_type = "ADV" if history['is_adversarial'] else "LEG"
        linestyle = '--' if history['is_adversarial'] else '-'
        plt.plot(history['times'], history['trust_means'], 
                label=f'Robot {robot_id} ({robot_type})', linestyle=linestyle)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Trust')
    plt.title('Trust Evolution Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    for robot_id, history in trust_history.items():
        robot_type = "ADV" if history['is_adversarial'] else "LEG"
        linestyle = '--' if history['is_adversarial'] else '-'
        plt.plot(history['times'], history['num_psms'], 
                label=f'Robot {robot_id} ({robot_type})', linestyle=linestyle)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Number of PSMs per Step')
    plt.title('Pseudomeasurement Generation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Track count distributions
    plt.subplot(2, 2, 3)
    if legitimate_track_counts and adversarial_track_counts:
        plt.hist([legitimate_track_counts, adversarial_track_counts], 
                bins=range(max(max(legitimate_track_counts), max(adversarial_track_counts)) + 2),
                alpha=0.7, label=['Legitimate', 'Adversarial'])
        plt.xlabel('Number of Tracks per Robot per Step')
        plt.ylabel('Frequency')
        plt.title('Track Generation Distribution')
        plt.legend()
    
    # Final trust levels comparison
    plt.subplot(2, 2, 4)
    final_trusts_leg = []
    final_trusts_adv = []
    robot_ids_leg = []
    robot_ids_adv = []
    
    for robot_id, history in trust_history.items():
        if history['trust_means']:
            final_trust = history['trust_means'][-1]
            if history['is_adversarial']:
                final_trusts_adv.append(final_trust)
                robot_ids_adv.append(robot_id)
            else:
                final_trusts_leg.append(final_trust)
                robot_ids_leg.append(robot_id)
    
    x_pos = np.arange(len(robot_ids_leg + robot_ids_adv))
    colors = ['blue'] * len(robot_ids_leg) + ['red'] * len(robot_ids_adv)
    
    plt.bar(x_pos, final_trusts_leg + final_trusts_adv, color=colors, alpha=0.7)
    plt.xlabel('Robot ID')
    plt.ylabel('Final Trust Level')
    plt.title('Final Trust Levels (Blue=Legitimate, Red=Adversarial)')
    plt.xticks(x_pos, robot_ids_leg + robot_ids_adv)
    
    plt.tight_layout()
    plt.savefig('simulation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    if legitimate_track_counts:
        print(f"Average tracks per legitimate robot: {np.mean(legitimate_track_counts):.2f}")
    if adversarial_track_counts:
        print(f"Average tracks per adversarial robot: {np.mean(adversarial_track_counts):.2f}")
    
    print(f"\nFinal Trust Levels:")
    for robot_id in sorted(trust_history.keys()):
        history = trust_history[robot_id]
        robot_type = "ADVERSARIAL" if history['is_adversarial'] else "LEGITIMATE"
        if history['trust_means']:
            final_trust = history['trust_means'][-1]
            total_psms = sum(history['num_psms'])
            print(f"Robot {robot_id} ({robot_type}): Final Trust = {final_trust:.3f}, Total PSMs = {total_psms}")

if __name__ == "__main__":
    analyze_simulation_data()