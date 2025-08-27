#!/usr/bin/env python3
"""
Algorithm Comparison Script

This script compares the trained GNN model with the paper trust algorithm
across different scenarios.
"""

import torch
import numpy as np
import time
import json
import os
from typing import Dict, List

# Set matplotlib backend for headless servers
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for servers
import matplotlib.pyplot as plt

from simulation_environment import SimulationEnvironment
from paper_trust_algorithm import PaperTrustAlgorithm
from neural_symbolic_trust_algorithm import NeuralSymbolicTrustAlgorithm


def show_gpu_status():
    """Display status of all available GPUs (CUDA and MPS)"""
    print("📊 GPU STATUS:")
    
    # Check MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        print("  🍎 Apple Metal Performance Shaders (MPS): ✅ Available")
        print("    Apple Silicon GPU acceleration enabled")
    else:
        print("  🍎 Apple MPS: ❌ Not available")
    
    # Check CUDA
    if torch.cuda.is_available():
        print("  🔥 CUDA GPUs:")
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / 1024**3
            allocated_gb = torch.cuda.memory_allocated(i) / 1024**3
            free_gb = total_gb - allocated_gb
            
            status = "✅ Available" if free_gb > 1.0 else "⚠️  Limited" if free_gb > 0.5 else "❌ Busy"
            
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {allocated_gb:.1f}/{total_gb:.1f} GB used ({free_gb:.1f} GB free) {status}")
    else:
        print("  🔥 CUDA: ❌ Not available")
    
    print()


class AlgorithmComparator:
    """Compares GNN and Paper trust algorithms"""
    
    def _find_best_gpu(self):
        """Find GPU with most free memory"""
        if not torch.cuda.is_available():
            return None
        
        best_gpu = 0
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            free_memory = props.total_memory - allocated
            
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = i
        
        return best_gpu
    
    def __init__(self, device: str = "auto"):
        # Handle different device specifications
        if device == "auto":
            # Auto-select best available device
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
                gpu_id = "mps"
                print(f"🔍 Auto-selected MPS (Apple Silicon GPU)")
            elif torch.cuda.is_available():
                # Find GPU with most free memory
                best_gpu = self._find_best_gpu()
                self.device = torch.device(f'cuda:{best_gpu}')
                gpu_id = best_gpu
                print(f"🔍 Auto-selected CUDA GPU {best_gpu} (most free memory)")
            else:
                self.device = torch.device('cpu')
                gpu_id = None
        elif device == "cpu":
            self.device = torch.device('cpu')
            gpu_id = None
        elif device == "mps":
            self.device = torch.device('mps')
            gpu_id = "mps"
        elif device.startswith("cuda"):
            # Handle cuda, cuda:0, cuda:1, etc.
            self.device = torch.device(device)
            if ":" in device:
                gpu_id = int(device.split(":")[1])
            else:
                gpu_id = 0
        else:
            # Try to parse as GPU ID number
            try:
                gpu_id = int(device)
                self.device = torch.device(f'cuda:{gpu_id}')
            except ValueError:
                print(f"❌ Invalid device '{device}'. Use 'cpu', 'mps', 'cuda', 'cuda:N', or GPU number")
                raise ValueError(f"Invalid device: {device}")
        
        # Verify device availability and set device
        if gpu_id == "mps":
            if not torch.backends.mps.is_available():
                print(f"⚠️  MPS not available, falling back to CPU")
                self.device = torch.device('cpu')
                gpu_id = None
        elif gpu_id is not None and gpu_id != "mps":
            if not torch.cuda.is_available():
                print(f"⚠️  CUDA not available, falling back to CPU")
                self.device = torch.device('cpu')
                gpu_id = None
            elif gpu_id >= torch.cuda.device_count():
                print(f"⚠️  GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs)")
                print(f"   Falling back to CPU to avoid interfering with other GPUs")
                self.device = torch.device('cpu')
                gpu_id = None
            else:
                # Check if GPU has sufficient free memory (at least 1GB recommended for inference)
                torch.cuda.set_device(gpu_id)
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                allocated_memory = torch.cuda.memory_allocated(gpu_id)
                free_memory = total_memory - allocated_memory
                free_memory_gb = free_memory / 1024**3
                
                if free_memory_gb < 1.0:  # Less than 1GB free (inference needs less than training)
                    allocated_gb = allocated_memory / 1024**3
                    total_gb = total_memory / 1024**3
                    print(f"⚠️  GPU {gpu_id} has limited memory: {free_memory_gb:.1f} GB free ({allocated_gb:.1f}/{total_gb:.1f} GB used)")
                    print(f"   Falling back to CPU to avoid memory issues")
                    self.device = torch.device('cpu')
                    gpu_id = None
        
        # Print device information
        if gpu_id == "mps":
            print(f"🍎 Using Apple Silicon GPU (MPS)")
            print(f"   Metal Performance Shaders acceleration enabled")
        elif gpu_id is not None:
            torch.cuda.set_device(gpu_id)  # Set current GPU
            print(f"🚀 Using CUDA GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
            print(f"   Memory Usage: {torch.cuda.memory_allocated(gpu_id)/1024**3:.2f} GB allocated")
        else:
            print(f"🔥 Using CPU")
        
        print(f"🔥 Inference device: {self.device}")
    
    def compare_algorithms(self,
                          gnn_model_file: str = "trained_gnn_model.pth",
                          num_steps: int = 800,
                          output_file: str = "algorithm_comparison_results.json") -> Dict:
        """
        Compare GNN and Paper algorithms across multiple scenarios
        
        Args:
            gnn_model_file: Path to trained GNN model
            num_steps: Number of simulation steps per scenario
            output_file: Output file for results
            
        Returns:
            Comparison results dictionary
        """
        print("🏁 ALGORITHM COMPARISON")
        print("=" * 50)
        
        # Check if trained model exists
        if not os.path.exists(gnn_model_file):
            raise FileNotFoundError(f"Trained GNN model not found: {gnn_model_file}")
        
        print(f"GNN Model: {gnn_model_file}")
        print(f"Simulation steps: {num_steps}")
        
        # Test scenarios with different adversarial conditions
        test_scenarios = [
            {"name": "Low Adversarial", "adversarial_ratio": 0.2, "num_robots": 6, "num_targets": 15},
            {"name": "Medium Adversarial", "adversarial_ratio": 0.35, "num_robots": 6, "num_targets": 15},
            {"name": "High Adversarial", "adversarial_ratio": 0.5, "num_robots": 6, "num_targets": 15},
            {"name": "Dense Environment", "adversarial_ratio": 0.33, "num_robots": 8, "num_targets": 20},
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            print(f"\n📈 Testing Scenario: {scenario['name']}")
            print(f"   Adversarial ratio: {scenario['adversarial_ratio']}")
            print(f"   Robots: {scenario['num_robots']}, Targets: {scenario['num_targets']}")
            
            # Test Paper Algorithm
            print("   🔄 Running Paper Algorithm...")
            paper_results = self._run_algorithm_test(
                algorithm_type="paper",
                model_file=None,
                scenario=scenario,
                num_steps=num_steps
            )
            
            # Test GNN Algorithm
            print("   🔄 Running GNN Algorithm...")
            gnn_results = self._run_algorithm_test(
                algorithm_type="gnn",
                model_file=gnn_model_file,
                scenario=scenario,
                num_steps=num_steps
            )
            
            # Calculate improvements
            separation_improvement = gnn_results['final_separation'] - paper_results['final_separation']
            stability_improvement = paper_results['stability'] - gnn_results['stability']  # Lower is better
            
            scenario_results = {
                'paper': paper_results,
                'gnn': gnn_results,
                'improvements': {
                    'separation_improvement': separation_improvement,
                    'stability_improvement': stability_improvement
                }
            }
            
            results[scenario['name']] = scenario_results
            
            # Print scenario results
            self._print_scenario_results(scenario['name'], scenario_results)
        
        # Overall analysis
        self._print_overall_analysis(results)
        
        # Create visualizations
        self._create_comparison_plots(results)
        
        # Save results
        self._save_results(results, output_file)
        
        return results
    
    def _run_algorithm_test(self, algorithm_type: str, model_file: str, scenario: Dict, num_steps: int) -> Dict:
        """Run a single algorithm test"""
        # Create algorithm instance
        if algorithm_type == "paper":
            algorithm = PaperTrustAlgorithm()
        elif algorithm_type == "gnn":
            algorithm = NeuralSymbolicTrustAlgorithm(model_path=model_file, learning_mode=False)
            algorithm.model = algorithm.model.to(self.device)
            
            # Set model to evaluation mode for inference
            algorithm.model.eval()
            
            # Clear GPU cache if using CUDA or MPS
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'mps':
                torch.mps.empty_cache()
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        
        # Create simulation environment
        env = SimulationEnvironment(
            num_robots=scenario['num_robots'],
            num_targets=scenario['num_targets'],
            world_size=(60.0, 60.0),
            adversarial_ratio=scenario['adversarial_ratio'],
            trust_algorithm=algorithm
        )
        
        # Track metrics during simulation
        separations = []
        inference_times = []
        
        for step in range(num_steps):
            step_start = time.time()
            env.step()
            step_time = time.time() - step_start
            
            if algorithm_type == "gnn":
                inference_times.append(step_time)
            
            # Calculate trust separation every 50 steps
            if step % 50 == 0:
                separation = self._calculate_trust_separation(env.robots)
                separations.append(separation)
        
        # Calculate final metrics
        final_separation = self._calculate_trust_separation(env.robots)
        avg_separation = np.mean(separations) if separations else 0.0
        max_separation = np.max(separations) if separations else 0.0
        stability = np.std(separations) if separations else 0.0
        
        results = {
            'final_separation': final_separation,
            'avg_separation': avg_separation,
            'max_separation': max_separation,
            'stability': stability,
            'separations': separations
        }
        
        if algorithm_type == "gnn":
            results['avg_inference_time'] = np.mean(inference_times) if inference_times else 0.0
        
        return results
    
    def _calculate_trust_separation(self, robots) -> float:
        """Calculate trust separation between legitimate and adversarial robots"""
        legitimate_trusts = []
        adversarial_trusts = []
        
        for robot in robots:
            trust_value = robot.trust_alpha / (robot.trust_alpha + robot.trust_beta)
            if robot.is_adversarial:
                adversarial_trusts.append(trust_value)
            else:
                legitimate_trusts.append(trust_value)
        
        if len(legitimate_trusts) == 0 or len(adversarial_trusts) == 0:
            return 0.0
        
        return np.mean(legitimate_trusts) - np.mean(adversarial_trusts)
    
    def _print_scenario_results(self, scenario_name: str, results: Dict):
        """Print results for a single scenario"""
        paper = results['paper']
        gnn = results['gnn']
        improvements = results['improvements']
        
        print(f"   📊 Results for {scenario_name}:")
        print(f"      Paper Algorithm:")
        print(f"         Final separation: {paper['final_separation']:.3f}")
        print(f"         Avg separation: {paper['avg_separation']:.3f}")
        print(f"         Stability (std): {paper['stability']:.3f}")
        
        print(f"      GNN Algorithm:")
        print(f"         Final separation: {gnn['final_separation']:.3f}")
        print(f"         Avg separation: {gnn['avg_separation']:.3f}")
        print(f"         Stability (std): {gnn['stability']:.3f}")
        print(f"         Avg inference time: {gnn['avg_inference_time']:.4f}s")
        
        print(f"      🔥 GNN vs Paper:")
        print(f"         Separation improvement: {improvements['separation_improvement']:+.3f}")
        print(f"         Stability improvement: {improvements['stability_improvement']:+.3f}")
    
    def _print_overall_analysis(self, results: Dict):
        """Print overall performance analysis"""
        # Calculate average improvements
        avg_separation_improvement = np.mean([r['improvements']['separation_improvement'] for r in results.values()])
        avg_stability_improvement = np.mean([r['improvements']['stability_improvement'] for r in results.values()])
        
        print(f"\n🏆 OVERALL PERFORMANCE ANALYSIS")
        print("=" * 50)
        print(f"📊 Average separation improvement: {avg_separation_improvement:+.3f}")
        print(f"📊 Average stability improvement: {avg_stability_improvement:+.3f}")
        
        # Performance assessment
        if avg_separation_improvement > 0.05:
            print("🎉 GNN shows SIGNIFICANT improvement in trust separation!")
        elif avg_separation_improvement > 0.01:
            print("✅ GNN shows modest improvement in trust separation")
        else:
            print("📝 GNN performance comparable to paper algorithm")
        
        if avg_stability_improvement > 0.01:
            print("🎯 GNN provides better stability (lower variance)")
        else:
            print("📊 Stability performance comparable")
    
    def _create_comparison_plots(self, results: Dict):
        """Create comprehensive comparison visualizations"""
        print(f"\n📊 Creating comparison visualizations...")
        
        scenario_names = list(results.keys())
        
        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Final Separation Comparison
        paper_separations = [results[s]['paper']['final_separation'] for s in scenario_names]
        gnn_separations = [results[s]['gnn']['final_separation'] for s in scenario_names]
        
        x = np.arange(len(scenario_names))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, paper_separations, width, 
                              label='Paper Algorithm', alpha=0.8, color='#3498db')
        bars2 = axes[0, 0].bar(x + width/2, gnn_separations, width, 
                              label='GNN Algorithm', alpha=0.8, color='#e74c3c')
        
        axes[0, 0].set_title('Final Trust Separation Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('Final Trust Separation')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(scenario_names, rotation=15)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Stability Comparison
        paper_stability = [results[s]['paper']['stability'] for s in scenario_names]
        gnn_stability = [results[s]['gnn']['stability'] for s in scenario_names]
        
        bars1 = axes[0, 1].bar(x - width/2, paper_stability, width, 
                              label='Paper Algorithm', alpha=0.8, color='#3498db')
        bars2 = axes[0, 1].bar(x + width/2, gnn_stability, width, 
                              label='GNN Algorithm', alpha=0.8, color='#e74c3c')
        
        axes[0, 1].set_title('Trust Stability Comparison (Lower = Better)', fontweight='bold')
        axes[0, 1].set_ylabel('Stability (Standard Deviation)')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(scenario_names, rotation=15)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Improvement Analysis
        improvements = [results[s]['improvements']['separation_improvement'] for s in scenario_names]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        bars = axes[1, 0].bar(scenario_names, improvements, alpha=0.7, color=colors)
        axes[1, 0].set_title('GNN Separation Improvement vs Paper', fontweight='bold')
        axes[1, 0].set_ylabel('Separation Improvement')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=15)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., 
                           height + (0.005 if height >= 0 else -0.015),
                           f'{height:+.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                           fontsize=10, fontweight='bold')
        
        # 4. GNN Inference Time
        inference_times = [results[s]['gnn']['avg_inference_time'] * 1000 for s in scenario_names]  # Convert to ms
        
        bars = axes[1, 1].bar(scenario_names, inference_times, alpha=0.7, color='#9b59b6')
        axes[1, 1].set_title('GNN Inference Time', fontweight='bold')
        axes[1, 1].set_ylabel('Time per Step (ms)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=15)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(inference_times)*0.02,
                           f'{height:.2f}ms', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = 'algorithm_comparison_results.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # Close figure to free memory
        
        print(f"   📊 Comparison plots saved to: {plot_file}")
    
    def _save_results(self, results: Dict, output_file: str):
        """Save results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for scenario, data in results.items():
            json_results[scenario] = {}
            for algo, metrics in data.items():
                if algo == 'improvements':
                    json_results[scenario][algo] = metrics
                else:
                    json_results[scenario][algo] = {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in metrics.items()
                    }
        
        # Add metadata
        json_results['_metadata'] = {
            'timestamp': str(np.datetime64('now')),
            'device': str(self.device),
            'scenarios_tested': len(results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"   💾 Results saved to: {output_file}")


def main():
    """Main comparison function"""
    import sys
    
    # Check for device argument
    device = "auto"
    if len(sys.argv) > 1:
        device_arg = sys.argv[1]
        
        # Special case: show GPU status
        if device_arg == "status":
            show_gpu_status()
            return
        
        # Valid device options: cpu, mps, cuda, cuda:N, auto, or just GPU number
        valid_devices = ["cpu", "mps", "cuda", "auto"] + [f"cuda:{i}" for i in range(8)] + [str(i) for i in range(8)]
        
        if device_arg in valid_devices or device_arg.startswith("cuda:"):
            device = device_arg
        else:
            print(f"❌ Invalid device '{device_arg}'")
            print(f"Valid options:")
            print(f"  'auto' - Auto-detect best GPU (MPS > CUDA > CPU)")
            print(f"  'cpu'  - Force CPU usage")  
            print(f"  'mps'  - Use Apple Silicon GPU (M1/M2)")
            print(f"  'cuda' - Use CUDA GPU 0")
            print(f"  'cuda:1' - Use specific CUDA GPU (cuda:0, cuda:1, etc.)")
            print(f"  '1' - Use CUDA GPU by number (0, 1, 2, 3)")
            print(f"  'status' - Show GPU status")
            print()
            show_gpu_status()
            return
    
    try:
        comparator = AlgorithmComparator(device=device)
    except Exception as e:
        print(f"❌ Failed to initialize comparator: {e}")
        print()
        show_gpu_status()
        return
    
    # Check if trained model exists
    model_file = "trained_gnn_model.pth"
    if not os.path.exists(model_file):
        print(f"❌ Trained GNN model not found: {model_file}")
        print(f"Run 'python train_gnn.py' first to train the model")
        return
    
    try:
        # Run comparison
        results = comparator.compare_algorithms(
            gnn_model_file=model_file,
            num_steps=800,
            output_file="algorithm_comparison_results.json"
        )
        
        print(f"\n🎉 Algorithm comparison completed successfully!")
        print(f"Check the generated plots and JSON file for detailed results.")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        print(f"Please check your trained model and try again")


if __name__ == "__main__":
    main()