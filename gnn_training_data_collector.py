#!/usr/bin/env python3
"""
GNN Training Data Collection Script

This script collects training data using the neural symbolic trust algorithm
to train the GNN model.
"""

import time
import numpy as np
from typing import Dict, List
import json

from simulation_environment import SimulationEnvironment
from neural_symbolic_trust_algorithm import NeuralSymbolicTrustAlgorithm
from paper_trust_algorithm import PaperTrustAlgorithm


class GNNDataCollector:
    """Diverse data collector for robust GNN training"""
    
    def __init__(self, seed: int = None):
        """Initialize collector with optional random seed for reproducibility"""
        if seed is not None:
            np.random.seed(seed)
            print(f"🎲 Random seed set to: {seed}")
        
    def collect_diverse_training_data(self, 
                                    num_scenarios: int = 20,
                                    steps_per_scenario: int = 1000,
                                    sample_intervals: int = 50,
                                    target_examples: int = 5000,
                                    output_file: str = "gnn_training_data_diverse.pkl") -> str:
        """
        Collect diverse training data from multiple scenarios with random sampling
        
        Args:
            num_scenarios: Number of different scenarios to simulate
            steps_per_scenario: Number of steps to run each scenario
            sample_intervals: Interval between data samples (for temporal diversity)
            target_examples: Target number of training examples to collect
            output_file: Output file path
            
        Returns:
            Path to saved training data file
        """
        print("🚀 DIVERSE GNN TRAINING DATA COLLECTION")
        print("=" * 60)
        print(f"Scenarios: {num_scenarios}")
        print(f"Steps per scenario: {steps_per_scenario}")
        print(f"Sample intervals: {sample_intervals}")
        print(f"Target examples: {target_examples}")
        
        # Initialize neural symbolic algorithm in learning mode
        neural_algorithm = NeuralSymbolicTrustAlgorithm(learning_mode=True)
        
        start_time = time.time()
        scenario_configs = []
        
        # Generate diverse scenario configurations
        print(f"\n🎯 Generating {num_scenarios} diverse scenarios...")
        scenarios = self._generate_diverse_scenarios(num_scenarios)
        
        for i, scenario in enumerate(scenarios):
            scenario_configs.append(scenario)
            print(f"  Scenario {i+1}: {scenario['num_robots']}R, {scenario['adversarial_ratio']:.2f}adv, "
                  f"FP:{scenario['false_positive_rate']:.2f}, FN:{scenario['false_negative_rate']:.2f}")
        
        print(f"\n📊 Running {num_scenarios} scenarios...")
        total_examples_collected = 0
        
        for scenario_idx, scenario in enumerate(scenarios):
            if total_examples_collected >= target_examples:
                print(f"✅ Reached target of {target_examples} examples, stopping early")
                break
                
            print(f"\n🔄 Scenario {scenario_idx + 1}/{len(scenarios)}")
            print(f"   Config: {scenario['num_robots']}R, {scenario['num_targets']}T, "
                  f"adv:{scenario['adversarial_ratio']:.2f}, "
                  f"FP:{scenario['false_positive_rate']:.3f}, FN:{scenario['false_negative_rate']:.3f}")
            
            # Run this scenario and collect samples
            examples_from_scenario = self._run_scenario_with_sampling(
                neural_algorithm, scenario, steps_per_scenario, sample_intervals, scenario_idx
            )
            
            total_examples_collected = len(neural_algorithm.training_data_collector.training_examples)
            print(f"   📈 Collected {examples_from_scenario} new examples (Total: {total_examples_collected})")
        
        collection_time = time.time() - start_time
        
        # Save training data
        neural_algorithm.save_training_data(output_file)
        
        # Save comprehensive metadata
        metadata = {
            'collection_params': {
                'num_scenarios': num_scenarios,
                'steps_per_scenario': steps_per_scenario,
                'sample_intervals': sample_intervals,
                'target_examples': target_examples,
                'actual_scenarios_run': len([s for s in scenarios if scenarios.index(s) < scenario_idx + 1])
            },
            'scenario_diversity': {
                'num_robots_range': [min(s['num_robots'] for s in scenario_configs), 
                                   max(s['num_robots'] for s in scenario_configs)],
                'adversarial_ratio_range': [min(s['adversarial_ratio'] for s in scenario_configs),
                                          max(s['adversarial_ratio'] for s in scenario_configs)],
                'false_positive_range': [min(s['false_positive_rate'] for s in scenario_configs),
                                       max(s['false_positive_rate'] for s in scenario_configs)],
                'false_negative_range': [min(s['false_negative_rate'] for s in scenario_configs),
                                       max(s['false_negative_rate'] for s in scenario_configs)],
                'world_sizes': list(set(str(s['world_size']) for s in scenario_configs))
            },
            'results': {
                'total_examples': total_examples_collected,
                'collection_time': collection_time,
                'examples_per_second': total_examples_collected / collection_time if collection_time > 0 else 0,
                'scenarios_used': len(scenario_configs)
            },
            'timestamp': str(np.datetime64('now'))
        }
        
        metadata_file = output_file.replace('.pkl', '_metadata.json')
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        metadata_serializable = convert_numpy_types(metadata)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_serializable, f, indent=2)
        
        print(f"\n✅ Diverse training data collection completed!")
        print(f"   📊 Total examples: {total_examples_collected:,}")
        print(f"   🎯 Scenarios run: {len(scenario_configs)}")
        print(f"   ⏱️  Collection time: {collection_time:.1f}s ({collection_time/60:.1f}m)")
        print(f"   📈 Examples per second: {total_examples_collected/collection_time:.1f}")
        print(f"   💾 Training data saved to: {output_file}")
        print(f"   📝 Metadata saved to: {metadata_file}")
        
        # Print diversity summary
        div = metadata['scenario_diversity']
        print(f"\n🌈 Dataset Diversity Summary:")
        print(f"   Robots: {div['num_robots_range'][0]}-{div['num_robots_range'][1]}")
        print(f"   Adversarial ratio: {div['adversarial_ratio_range'][0]:.2f}-{div['adversarial_ratio_range'][1]:.2f}")
        print(f"   False positive rate: {div['false_positive_range'][0]:.3f}-{div['false_positive_range'][1]:.3f}")
        print(f"   False negative rate: {div['false_negative_range'][0]:.3f}-{div['false_negative_range'][1]:.3f}")
        
        if total_examples_collected < target_examples * 0.8:
            print(f"   ⚠️  Collected fewer examples than target. Consider increasing scenarios or steps.")
        
        return output_file
    
    def _generate_diverse_scenarios(self, num_scenarios: int) -> List[Dict]:
        """Generate diverse scenario configurations"""
        scenarios = []
        
        # Define parameter ranges for systematic testing scenarios - Discrete uniform selection
        robot_counts = [4, 5, 6, 7, 8, 10, 12]
        target_multipliers = [2.0, 2.5, 3.0]  # targets = robots * multiplier
        adversarial_ratios = [0.2, 0.3, 0.4, 0.5]  # Exactly 4 values for uniform selection
        world_sizes = [(40, 40), (50, 50), (60, 60), (80, 80), (100, 100)]
        
        # Sensor error rates - Exact values for systematic testing with uniform probability
        false_positive_rates = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]  # 10 values
        false_negative_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]            # 8 values
        
        for i in range(num_scenarios):
            # Select parameters with some correlation but maintain diversity
            num_robots = np.random.choice(robot_counts)
            num_targets = int(num_robots * np.random.choice(target_multipliers))
            
            # Select adversarial ratio from discrete set, ensuring at least 1 adversarial robot
            # Filter valid adversarial ratios for this robot count (must result in >= 1 adversarial robot)
            valid_adversarial_ratios = [ratio for ratio in adversarial_ratios 
                                      if int(num_robots * ratio) >= 1]
            
            if valid_adversarial_ratios:
                adversarial_ratio = np.random.choice(valid_adversarial_ratios)
            else:
                # Fallback: if no valid ratios, use minimum that gives 1 adversarial robot
                # This should rarely happen with robot_counts=[4,5,6,7,8,10,12] and ratios=[0.2,0.3,0.4,0.5]
                adversarial_ratio = 1.0 / num_robots
            
            # Sensor parameters - uniform selection from discrete values
            # Always use independent selection for systematic coverage (no correlation)
            fp_rate = np.random.choice(false_positive_rates)  # Uniform selection from discrete FP rates
            fn_rate = np.random.choice(false_negative_rates)  # Uniform selection from discrete FN rates
            
            # No clamping needed - values are already from the exact discrete sets
            
            scenario = {
                'num_robots': num_robots,
                'num_targets': num_targets,
                'adversarial_ratio': adversarial_ratio,
                'world_size': world_sizes[np.random.randint(len(world_sizes))],
                'false_positive_rate': fp_rate,
                'false_negative_rate': fn_rate,
                # Additional diversity parameters
                'sensor_range': np.random.uniform(20, 40),
                'communication_range': np.random.uniform(30, 60),
                'movement_speed': np.random.uniform(0.5, 2.0),
                'proximal_range': np.random.uniform(30, 70)  # Vary proximal range
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _run_scenario_with_sampling(self, neural_algorithm, scenario, total_steps, sample_interval, scenario_idx=0):
        """Run a scenario with random temporal sampling"""
        examples_before = len(neural_algorithm.training_data_collector.training_examples)
        
        # Create environment with paper trust algorithm for ground truth
        paper_algorithm = PaperTrustAlgorithm()
        env = SimulationEnvironment(
            num_robots=scenario['num_robots'],
            num_targets=scenario['num_targets'],
            world_size=scenario['world_size'],
            adversarial_ratio=scenario['adversarial_ratio'],
            proximal_range=scenario.get('proximal_range', 50.0),  # Default 50.0
            trust_algorithm=paper_algorithm
        )
        
        # Replace the neural algorithm's paper algorithm with the simulation's instance
        # This ensures training data uses the continuously updated trust state
        neural_algorithm.training_data_collector.paper_algorithm = paper_algorithm
        neural_algorithm.training_data_collector.paper_algorithm_initialized = True
        
        # Track steps where we've already collected data to avoid duplicates
        collected_steps = set()
        
        # Set additional parameters if the environment supports them
        self._configure_environment_parameters(env, scenario)
        
        # Calculate sampling points based on sample_interval parameter
        # This creates natural sampling based on the configured interval
        sample_steps = list(range(sample_interval, total_steps, sample_interval))
        print(f"   🎯 Will collect data at {len(sample_steps)} steps: every {sample_interval} steps")
        print(f"   📊 Expected examples: {len(sample_steps)} steps × ~{scenario['num_robots']} robots = ~{len(sample_steps) * scenario['num_robots']} total")
        
        # Run simulation with limited data collection
        for step in range(total_steps):
            # Run simulation step
            env.step()
            
            # Collect data only at specific sample steps, and only once per step
            if step in sample_steps and step not in collected_steps:
                # Temporarily disable the automatic collection in neural algorithm
                original_learning_mode = neural_algorithm.learning_mode
                neural_algorithm.learning_mode = False
                
                # Loop through robots with tracks to collect training examples
                # Each robot with tracks becomes an ego robot for one training example
                if env.robots:
                    collected_for_step = 0
                    
                    for robot in env.robots:
                        # Use current timestep tracks instead of accumulated tracks
                        robot_current_tracks = list(env.robot_current_tracks.get(robot.id, {}).values())
                        if robot_current_tracks:  # Only collect if this robot has current tracks to serve as ego
                            # Convert current timestep tracks to tracks_by_robot format
                            current_tracks_by_robot = {}
                            for robot_id, obj_tracks in env.robot_current_tracks.items():
                                if obj_tracks:  # Only include robots that have current tracks
                                    current_tracks_by_robot[robot_id] = list(obj_tracks.values())
                            
                            # Collect training data using this robot as ego robot with CURRENT tracks
                            neural_algorithm.training_data_collector.collect_from_simulation_step(
                                env.robots, current_tracks_by_robot, env.robot_current_tracks,
                                {}, robot, robot_current_tracks
                            )
                            collected_for_step += 1
                    
                    if collected_for_step > 0:
                        print(f"      Collected {collected_for_step} training examples at step {step}")
                
                # Restore learning mode and mark step as collected
                neural_algorithm.learning_mode = original_learning_mode
                collected_steps.add(step)
        
        examples_after = len(neural_algorithm.training_data_collector.training_examples)
        return examples_after - examples_before
    
    def _configure_environment_parameters(self, env, scenario):
        """Configure environment with additional parameters if supported"""
        try:
            # Try to set sensor parameters if the environment supports them
            if hasattr(env, 'set_sensor_parameters'):
                env.set_sensor_parameters(
                    false_positive_rate=scenario['false_positive_rate'],
                    false_negative_rate=scenario['false_negative_rate']
                )
            
            # Try to set other parameters
            if hasattr(env, 'set_communication_range'):
                env.set_communication_range(scenario['communication_range'])
                
            if hasattr(env, 'set_sensor_range'):
                env.set_sensor_range(scenario['sensor_range'])
                
            if hasattr(env, 'set_movement_speed'):
                env.set_movement_speed(scenario['movement_speed'])
                
        except Exception as e:
            # If environment doesn't support these parameters, that's okay
            print(f"   Note: Some parameters not supported by environment: {e}")

    def collect_training_data(self, *args, **kwargs):
        """Legacy method - redirects to diverse collection"""
        print("ℹ️  Using legacy method - redirecting to diverse collection")
        return self.collect_diverse_training_data(*args, **kwargs)


def main():
    """Main function to collect training data"""
    collector = GNNDataCollector(seed=42)  # Set seed for reproducibility
    
    # Collect diverse training data with smaller parameters for testing
    training_data_file = collector.collect_diverse_training_data(
        num_scenarios=25,            # Fewer scenarios for faster collection
        steps_per_scenario=1000,     # Fewer steps per scenario  
        sample_intervals=20,        # Sampling interval
        target_examples=10000,         # Smaller target for testing
        output_file="gnn_training_data.pkl"
    )
    
    print(f"\n🎉 Training data collection completed!")
    print(f"Collected data file: {training_data_file}")
    print(f"Next step: python train_gnn.py")
    

if __name__ == "__main__":
    main()