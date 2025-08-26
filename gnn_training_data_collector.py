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


class GNNDataCollector:
    """Simple and focused data collector for GNN training"""
    
    def __init__(self):
        pass
    
    def collect_training_data(self, 
                            num_steps: int = 2000,
                            num_robots: int = 6,
                            num_targets: int = 15,
                            adversarial_ratio: float = 0.3,
                            world_size: tuple = (60.0, 60.0),
                            output_file: str = "gnn_training_data.pkl") -> str:
        """
        Collect training data from simulation
        
        Args:
            num_steps: Number of simulation steps
            num_robots: Number of robots in simulation  
            num_targets: Number of targets in simulation
            adversarial_ratio: Ratio of adversarial robots
            world_size: World dimensions
            output_file: Output file path
            
        Returns:
            Path to saved training data file
        """
        print("🚀 GNN TRAINING DATA COLLECTION")
        print("=" * 50)
        print(f"Steps: {num_steps}")
        print(f"Robots: {num_robots}")
        print(f"Targets: {num_targets}")
        print(f"Adversarial ratio: {adversarial_ratio}")
        print(f"World size: {world_size}")
        
        # Initialize neural symbolic algorithm in learning mode
        neural_algorithm = NeuralSymbolicTrustAlgorithm(learning_mode=True)
        
        # Create simulation environment
        env = SimulationEnvironment(
            num_robots=num_robots,
            num_targets=num_targets,
            world_size=world_size,
            adversarial_ratio=adversarial_ratio,
            trust_algorithm=neural_algorithm
        )
        
        print(f"\n📊 Running simulation for {num_steps} steps...")
        start_time = time.time()
        
        # Run simulation to collect training data
        for step in range(num_steps):
            env.step()
            if step % (num_steps // 10) == 0:
                num_examples = len(neural_algorithm.training_data_collector.training_examples)
                progress = (step / num_steps) * 100
                print(f"  Progress: {progress:5.1f}% - Examples: {num_examples:4d}")
        
        collection_time = time.time() - start_time
        total_examples = len(neural_algorithm.training_data_collector.training_examples)
        
        # Save training data
        neural_algorithm.save_training_data(output_file)
        
        # Save metadata
        metadata = {
            'collection_params': {
                'num_steps': num_steps,
                'num_robots': num_robots,
                'num_targets': num_targets,
                'adversarial_ratio': adversarial_ratio,
                'world_size': world_size
            },
            'results': {
                'total_examples': total_examples,
                'collection_time': collection_time,
                'examples_per_second': total_examples / collection_time if collection_time > 0 else 0
            },
            'timestamp': str(np.datetime64('now'))
        }
        
        metadata_file = output_file.replace('.pkl', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Training data collection completed!")
        print(f"   📊 Total examples: {total_examples:,}")
        print(f"   ⏱️  Collection time: {collection_time:.1f}s")
        print(f"   📈 Examples per second: {total_examples/collection_time:.1f}")
        print(f"   💾 Training data saved to: {output_file}")
        print(f"   📝 Metadata saved to: {metadata_file}")
        
        if total_examples < 100:
            print(f"   ⚠️  Warning: Only {total_examples} examples collected. Consider running longer.")
        
        return output_file


def main():
    """Main function to collect training data"""
    collector = GNNDataCollector()
    
    # Collect training data with default parameters
    training_data_file = collector.collect_training_data(
        num_steps=2000,
        num_robots=6,
        num_targets=15,
        adversarial_ratio=0.3,
        world_size=(60.0, 60.0),
        output_file="gnn_training_data.pkl"
    )
    
    print(f"\n🎉 Training data collection completed!")
    print(f"Use 'python train_gnn.py' to train the model")
    

if __name__ == "__main__":
    main()