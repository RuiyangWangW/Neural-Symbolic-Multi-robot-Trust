#!/usr/bin/env python3
"""
Analyze track merging to verify implementation
"""

import json
import numpy as np

def analyze_track_merging(filename='paper_trust_data.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"Analyzing track merging from {len(data)} simulation steps")
    
    # Check track counts and trust values over time
    print("\nTrack analysis (first 5 timesteps):")
    for i in range(min(5, len(data))):
        step = data[i]
        if 'tracks' in step:
            tracks = step['tracks']
            print(f"\nTime {step['time']:.1f}s:")
            
            total_tracks = 0
            for robot_id, robot_tracks in tracks.items():
                print(f"  Robot {robot_id}: {len(robot_tracks)} tracks")
                total_tracks += len(robot_tracks)
                
                # Show track trust values for first robot with tracks
                if robot_tracks and robot_id == list(tracks.keys())[0]:
                    print(f"    Track trusts: ", end="")
                    for track_info in robot_tracks[:3]:  # Show first 3 tracks
                        track_id, position, confidence, alpha, beta = track_info
                        trust_mean = alpha / (alpha + beta)
                        print(f"{trust_mean:.3f} ", end="")
                    if len(robot_tracks) > 3:
                        print("...")
                    else:
                        print()
            
            print(f"  Total tracks across all robots: {total_tracks}")
            
            # Compare with ground truth objects
            if 'ground_truth' in step:
                num_objects = step['ground_truth']['num_objects']
                print(f"  Ground truth objects: {num_objects}")
    
    # Analyze track count distribution
    track_counts_per_robot = []
    trust_values = []
    
    for step in data[::50]:  # Sample every 50 steps
        if 'tracks' in step:
            for robot_id, robot_tracks in step['tracks'].items():
                track_counts_per_robot.append(len(robot_tracks))
                
                for track_info in robot_tracks:
                    track_id, position, confidence, alpha, beta = track_info
                    trust_mean = alpha / (alpha + beta)
                    trust_values.append(trust_mean)
    
    if track_counts_per_robot:
        print(f"\nTrack count statistics (sampled):")
        print(f"  Average tracks per robot: {np.mean(track_counts_per_robot):.2f}")
        print(f"  Max tracks per robot: {max(track_counts_per_robot)}")
        print(f"  Min tracks per robot: {min(track_counts_per_robot)}")
    
    if trust_values:
        print(f"\nTrack trust statistics:")
        print(f"  Average track trust: {np.mean(trust_values):.3f}")
        print(f"  Track trust range: {min(trust_values):.3f} - {max(trust_values):.3f}")
    
    # Final timestep analysis
    final_step = data[-1]
    if 'tracks' in final_step:
        print(f"\nFinal timestep track analysis:")
        total_final_tracks = 0
        high_trust_tracks = 0
        
        for robot_id, robot_tracks in final_step['tracks'].items():
            total_final_tracks += len(robot_tracks)
            print(f"  Robot {robot_id}: {len(robot_tracks)} tracks")
            
            if robot_tracks:
                robot_trusts = []
                for track_info in robot_tracks:
                    track_id, position, confidence, alpha, beta = track_info
                    trust_mean = alpha / (alpha + beta)
                    robot_trusts.append(trust_mean)
                    if trust_mean > 0.5:
                        high_trust_tracks += 1
                
                print(f"    Mean track trust: {np.mean(robot_trusts):.3f}")
        
        print(f"  Total tracks: {total_final_tracks}")
        print(f"  High trust tracks (>0.5): {high_trust_tracks}")
        
        if 'ground_truth' in final_step:
            num_objects = final_step['ground_truth']['num_objects']
            print(f"  Ground truth objects: {num_objects}")
            print(f"  Track efficiency: {total_final_tracks}/{num_objects} = {total_final_tracks/num_objects:.2f} tracks per object")
            
            # Analyze ground truth objects vs false positives based on trust
            analyze_track_classification(final_step)

def analyze_track_classification(step_data, high_trust_threshold=0.6, low_trust_threshold=0.4):
    """
    Analyze track classification:
    - Ground truth objects with high track trust (correctly trusted)
    - False positive objects with low track trust (correctly distrusted)
    """
    print(f"\nTrack Classification Analysis:")
    print(f"  High trust threshold: {high_trust_threshold}")
    print(f"  Low trust threshold: {low_trust_threshold}")
    
    if 'tracks' not in step_data:
        print("  No track data available")
        return
    
    # Collect all tracks with their object IDs and trust values
    all_tracks = []
    for robot_id, robot_tracks in step_data['tracks'].items():
        for track_info in robot_tracks:
            track_id, position, confidence, alpha, beta = track_info
            trust_mean = alpha / (alpha + beta)
            
            # Determine if this is a ground truth object or false positive
            # Ground truth objects typically have IDs like "gt_obj_X"
            # False positives typically have IDs like "fp_X_Y" 
            is_ground_truth = 'gt_obj_' in track_id
            is_false_positive = 'fp_' in track_id
            
            all_tracks.append({
                'robot_id': robot_id,
                'track_id': track_id,
                'trust': trust_mean,
                'is_ground_truth': is_ground_truth,
                'is_false_positive': is_false_positive,
                'alpha': alpha,
                'beta': beta
            })
    
    # Count classifications
    gt_high_trust = 0  # Ground truth objects with high trust (good)
    gt_low_trust = 0   # Ground truth objects with low trust (bad)
    gt_medium_trust = 0 # Ground truth objects with medium trust
    fp_high_trust = 0  # False positives with high trust (bad)
    fp_low_trust = 0   # False positives with low trust (good)
    fp_medium_trust = 0 # False positives with medium trust
    other_tracks = 0   # Tracks that don't clearly fall into GT or FP categories
    
    gt_trust_values = []
    fp_trust_values = []
    
    for track in all_tracks:
        if track['is_ground_truth']:
            gt_trust_values.append(track['trust'])
            if track['trust'] >= high_trust_threshold:
                gt_high_trust += 1
            elif track['trust'] <= low_trust_threshold:
                gt_low_trust += 1
            else:
                gt_medium_trust += 1
        elif track['is_false_positive']:
            fp_trust_values.append(track['trust'])
            if track['trust'] >= high_trust_threshold:
                fp_high_trust += 1
            elif track['trust'] <= low_trust_threshold:
                fp_low_trust += 1
            else:
                fp_medium_trust += 1
        else:
            other_tracks += 1
    
    # Display results
    total_gt = gt_high_trust + gt_medium_trust + gt_low_trust
    total_fp = fp_high_trust + fp_medium_trust + fp_low_trust
    
    print(f"\n  📊 Ground Truth Objects:")
    print(f"    High trust (≥{high_trust_threshold}): {gt_high_trust} tracks ✅")
    print(f"    Medium trust ({low_trust_threshold}<t<{high_trust_threshold}): {gt_medium_trust} tracks ⚠️")
    print(f"    Low trust (≤{low_trust_threshold}): {gt_low_trust} tracks ❌")
    print(f"    Total GT tracks: {total_gt}")
    if gt_trust_values:
        print(f"    Average GT trust: {np.mean(gt_trust_values):.3f}")
        print(f"    GT trust range: {min(gt_trust_values):.3f} - {max(gt_trust_values):.3f}")
    
    print(f"\n  📊 False Positive Objects:")
    print(f"    Low trust (≤{low_trust_threshold}): {fp_low_trust} tracks ✅")
    print(f"    Medium trust ({low_trust_threshold}<t<{high_trust_threshold}): {fp_medium_trust} tracks ⚠️")
    print(f"    High trust (≥{high_trust_threshold}): {fp_high_trust} tracks ❌") 
    print(f"    Total FP tracks: {total_fp}")
    if fp_trust_values:
        print(f"    Average FP trust: {np.mean(fp_trust_values):.3f}")
        print(f"    FP trust range: {min(fp_trust_values):.3f} - {max(fp_trust_values):.3f}")
    
    if other_tracks > 0:
        print(f"\n  📊 Other/Unknown tracks: {other_tracks}")
    
    # Summary metrics
    correctly_classified = gt_high_trust + fp_low_trust
    total_classified = total_gt + total_fp
    
    if total_classified > 0:
        accuracy = correctly_classified / total_classified
        print(f"\n  🎯 Classification Accuracy: {correctly_classified}/{total_classified} = {accuracy:.1%}")
        
        if total_gt > 0:
            gt_accuracy = gt_high_trust / total_gt
            print(f"     GT detection rate: {gt_high_trust}/{total_gt} = {gt_accuracy:.1%}")
        
        if total_fp > 0:
            fp_detection_rate = fp_low_trust / total_fp
            print(f"     FP detection rate: {fp_low_trust}/{total_fp} = {fp_detection_rate:.1%}")
    
    return gt_high_trust, total_gt, fp_low_trust, total_fp


def analyze_track_trust_evolution(filename='paper_trust_data.json'):
    """
    Analyze how track trust evolves over time for GT vs FP objects
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"\nTrack Trust Evolution Analysis:")
    
    # Sample data points throughout simulation
    sample_indices = [0, len(data)//4, len(data)//2, 3*len(data)//4, len(data)-1]
    sample_times = []
    gt_trust_over_time = []
    fp_trust_over_time = []
    
    for idx in sample_indices:
        step = data[idx]
        sample_times.append(step['time'])
        
        gt_trusts = []
        fp_trusts = []
        
        if 'tracks' in step:
            for robot_id, robot_tracks in step['tracks'].items():
                for track_info in robot_tracks:
                    track_id, position, confidence, alpha, beta = track_info
                    trust_mean = alpha / (alpha + beta)
                    
                    if 'gt_obj_' in track_id:
                        gt_trusts.append(trust_mean)
                    elif 'fp_' in track_id:
                        fp_trusts.append(trust_mean)
        
        gt_trust_over_time.append(np.mean(gt_trusts) if gt_trusts else 0)
        fp_trust_over_time.append(np.mean(fp_trusts) if fp_trusts else 0)
    
    print(f"  Time Evolution (sampled at {len(sample_indices)} points):")
    print(f"    Time:     ", " ".join(f"{t:6.1f}" for t in sample_times))
    print(f"    GT Trust: ", " ".join(f"{t:6.3f}" for t in gt_trust_over_time))
    print(f"    FP Trust: ", " ".join(f"{t:6.3f}" for t in fp_trust_over_time))


def quick_summary(filename='paper_trust_data.json', high_trust_threshold=0.6, low_trust_threshold=0.4):
    """
    Quick summary of the key metrics you requested
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    final_step = data[-1]
    if 'tracks' not in final_step:
        print("No track data available")
        return
    
    gt_high_trust = 0
    fp_low_trust = 0
    total_gt = 0
    total_fp = 0
    
    for robot_id, robot_tracks in final_step['tracks'].items():
        for track_info in robot_tracks:
            track_id, position, confidence, alpha, beta = track_info
            trust_mean = alpha / (alpha + beta)
            
            if 'gt_obj_' in track_id:
                total_gt += 1
                if trust_mean >= high_trust_threshold:
                    gt_high_trust += 1
            elif 'fp_' in track_id:
                total_fp += 1
                if trust_mean <= low_trust_threshold:
                    fp_low_trust += 1
    
    print(f"\n🎯 QUICK SUMMARY:")
    print(f"  Ground truth objects with high trust (≥{high_trust_threshold}): {gt_high_trust}/{total_gt} = {gt_high_trust/total_gt*100 if total_gt > 0 else 0:.1f}%")
    print(f"  False positive objects with low trust (≤{low_trust_threshold}): {fp_low_trust}/{total_fp} = {fp_low_trust/total_fp*100 if total_fp > 0 else 0:.1f}%")
    print(f"  Total tracks analyzed: {total_gt + total_fp}")
    
    return gt_high_trust, total_gt, fp_low_trust, total_fp


if __name__ == "__main__":
    analyze_track_merging()
    analyze_track_trust_evolution()
    quick_summary()