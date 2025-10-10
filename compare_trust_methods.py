#!/usr/bin/env python3
"""
Comparison of Trust Methods: Paper Algorithm vs Supervised Model vs RL Model

This script runs all three methods on the same simulation environment with identical
random seeds and initial conditions to enable fair comparison.
"""

import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import json
import types
from pathlib import Path
from typing import Dict, List, Tuple

# Import required modules
from simulation_environment import SimulationEnvironment
from paper_trust_algorithm import PaperTrustAlgorithm
from supervised_trust_gnn import SupervisedTrustPredictor
from robot_track_classes import Robot, Track

# Import RL trust components
from rl_trust_system import RLTrustSystem
from rl_updater import LearnableUpdater, UpdateDecision


class TrustMethodComparison:
    """Compares paper algorithm vs supervised model vs RL model"""

    def __init__(self,
                 supervised_model_path: str = "trained_gnn_model.pth",
                 rl_model_path: str = "rl_trust_model.pth",
                 num_robots: int = 5,
                 num_targets: int = 10,
                 num_timesteps: int = 500,
                 random_seed: int = 42,
                 world_size: float = 60.0,
                 fov_range: float = 50.0,
                 fov_angle: float = np.pi/3,
                 fixed_step_scale: float = 0.5):
        """
        Initialize comparison with all three trust methods

        Args:
            supervised_model_path: Path to trained supervised trust model
            rl_model_path: Path to trained RL trust model
            num_robots: Number of robots in simulation
            num_targets: Number of ground truth targets
            num_timesteps: Number of simulation steps
            random_seed: Random seed for reproducibility
            world_size: Size of the simulation world
            fov_range: Field of view range for robots
            fov_angle: Field of view angle for robots
        """
        self.supervised_model_path = Path(supervised_model_path) if supervised_model_path else None
        self.rl_model_path = Path(rl_model_path) if rl_model_path else None
        self.num_robots = num_robots
        self.num_targets = num_targets
        self.num_timesteps = num_timesteps
        self.random_seed = random_seed
        self.world_size = world_size
        self.fov_range = fov_range
        self.fov_angle = fov_angle
        self.fixed_step_scale = max(0.0, min(1.0, fixed_step_scale))

        # Initialize trust methods
        self.paper_algorithm = PaperTrustAlgorithm()
        self.supervised_predictor = None
        self.rl_trust_system = None
        self.baseline_trust_system = None

        # Results storage
        self.paper_results = []
        self.supervised_results = []
        self.rl_results = []
        self.baseline_results = []

        # Simulation parameters (can be overridden)
        self.adversarial_ratio = 0.3
        self.false_positive_rate = 0.5
        self.false_negative_rate = 0.0
        self.proximal_range = 50.0

        # Try to load supervised model after proximal_range is set
        try:
            gnn_path_str = str(self.supervised_model_path) if self.supervised_model_path else None
            self.supervised_predictor = SupervisedTrustPredictor(gnn_path_str, proximal_range=self.proximal_range)
            if self.supervised_model_path and self.supervised_model_path.exists():
                print(f"✅ Supervised trust predictor ready ({gnn_path_str})")
            else:
                print("ℹ️ Supervised trust predictor initialized with fresh weights")
        except Exception as e:
            print(f"⚠️ Failed to initialize supervised predictor: {e}")

        # Try to load RL model
        try:
            self._initialize_rl_system()
            if self.rl_model_path and self.rl_model_path.exists():
                model_label = str(self.rl_model_path)
            else:
                model_label = "fresh initialization"
            print(f"✅ RL trust system ready ({model_label})")
        except Exception as e:
            print(f"⚠️ Failed to load RL model: {e}")

        # Initialize baseline fixed-step system
        try:
            self._initialize_baseline_system()
            print(f"✅ Initialized fixed-step baseline with scale={self.fixed_step_scale:.2f}")
        except Exception as e:
            print(f"⚠️ Failed to initialize fixed-step baseline: {e}")

    def _initialize_rl_system(self):
        """Initialize the RL trust update system with ego-centric architecture"""
        device = 'cpu'  # Use CPU for comparison consistency

        rl_model_path = self.rl_model_path
        rl_checkpoint_missing = False
        if rl_model_path and not rl_model_path.exists():
            legacy_path = Path('rl_trust_model.pth')
            if legacy_path.exists():
                print(f"ℹ️ RL model '{rl_model_path}' not found. Falling back to '{legacy_path}'.")
                self.rl_model_path = legacy_path
            else:
                print(f"⚠️ RL model '{rl_model_path}' not found and no legacy model available. Using fresh weights.")
                self.rl_model_path = None
                rl_checkpoint_missing = True
        elif self.rl_model_path is None:
            rl_checkpoint_missing = True

        evidence_path = str(self.supervised_model_path) if self.supervised_model_path and self.supervised_model_path.exists() else None
        updater_path = str(self.rl_model_path) if self.rl_model_path and self.rl_model_path.exists() else None

        if self.supervised_model_path and not self.supervised_model_path.exists():
            print(f"ℹ️ Supervised evidence model '{self.supervised_model_path}' not found. Initializing with fresh weights.")
        if rl_checkpoint_missing:
            print("ℹ️ RL updater model not found. Initializing with fresh weights.")

        # Initialize RLTrustSystem with updated ego-centric parameters
        self.rl_trust_system = RLTrustSystem(
            evidence_model_path=evidence_path,
            updater_model_path=updater_path,
            device=device,
            step_size=1.0,
        )

    def _initialize_baseline_system(self):
        """Set up an RL trust system with fixed step scales."""
        device = 'cpu'
        self.baseline_trust_system = RLTrustSystem(
            evidence_model_path=str(self.supervised_model_path) if self.supervised_model_path and self.supervised_model_path.exists() else None,
            updater_model_path=None,
            device=device,
            step_size=1.0,
        )

        updater = self.baseline_trust_system.updater
        updater.baseline_scale = self.fixed_step_scale

        def fixed_get_step_scales(self_updater, ego_robots, ego_tracks, participating_robots,
                                  participating_tracks, robot_scores, track_scores, track_observers):
            scale = float(self_updater.baseline_scale)
            robot_steps = {robot.id: (scale, scale) for robot in participating_robots}
            track_steps = {track.track_id: (scale, scale) for track in participating_tracks}
            return UpdateDecision(robot_steps, track_steps)

        updater.get_step_scales = types.MethodType(fixed_get_step_scales, updater)

    def create_identical_environments(self, count: int = 3) -> List[SimulationEnvironment]:
        """Create identical simulation environments for fair comparison."""
        envs = []
        for _ in range(count):
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            env = SimulationEnvironment(
                num_robots=self.num_robots,
                num_targets=self.num_targets,
                world_size=(self.world_size, self.world_size),
                adversarial_ratio=self.adversarial_ratio,
                proximal_range=self.proximal_range,
                fov_range=self.fov_range,
                fov_angle=self.fov_angle,
                false_positive_rate=self.false_positive_rate,
                false_negative_rate=self.false_negative_rate
            )
            envs.append(env)
        return envs

    def run_paper_algorithm_simulation(self, env: SimulationEnvironment) -> List[Dict]:
        """Run simulation using paper trust algorithm"""
        print("🚀 Running paper algorithm simulation...")

        results = []

        for step in range(self.num_timesteps):
            # CRITICAL: Reset random seed for each step to ensure identical randomness
            step_seed = self.random_seed + step
            np.random.seed(step_seed)
            random.seed(step_seed)

            # Step the simulation
            frame_data = env.step()

            # Update current timestep tracks for all robots
            for robot in env.robots:
                robot.update_current_timestep_tracks()

            # Apply paper trust algorithm
            try:
                trust_updates = self.paper_algorithm.update_trust(env.robots, env)
            except Exception as e:
                print(f"⚠️ Paper algorithm error at step {step}: {e}")
                trust_updates = {}

            # Collect results
            step_result = {
                'step': step,
                'time': env.time,
                'robot_trust_values': {
                    robot.id: robot.trust_value for robot in env.robots
                },
                'robot_alpha_beta': {
                    robot.id: {'alpha': robot.trust_alpha, 'beta': robot.trust_beta}
                    for robot in env.robots
                },
                'track_trust_values': {},
                'trust_updates': trust_updates,
                'adversarial_robots': [r.id for r in env.robots if r.is_adversarial],
                'legitimate_robots': [r.id for r in env.robots if not r.is_adversarial]
            }

            # Collect track trust values
            for robot in env.robots:
                step_result['track_trust_values'][robot.id] = {
                    track.track_id: {
                        'trust_value': track.trust_value,
                        'alpha': track.trust_alpha,
                        'beta': track.trust_beta,
                        'object_id': track.object_id
                    }
                    for track in robot.get_all_tracks()
                }

            results.append(step_result)

            if step % 100 == 0:
                print(f"  Paper algorithm step {step}/{self.num_timesteps}")

        print("✅ Paper algorithm simulation completed")
        return results

    def run_baseline_simulation(self, env: SimulationEnvironment) -> List[Dict]:
        """Run simulation using fixed step-scale policy"""
        if self.baseline_trust_system is None:
            print("❌ Baseline system not available, skipping")
            return []

        print(f"🚀 Running fixed step-scale simulation (scale={self.fixed_step_scale:.2f})...")

        return self._run_trust_simulation(env, self.baseline_trust_system, label="Baseline")

    def run_rl_model_simulation(self, env: SimulationEnvironment) -> List[Dict]:
        """Run simulation using RL trust model"""
        if self.rl_trust_system is None:
            print("❌ RL model not available, skipping")
            return []

        print("🚀 Running RL model simulation...")

        return self._run_trust_simulation(env, self.rl_trust_system, label="RL")

    def _run_trust_simulation(self, env: SimulationEnvironment, trust_system: RLTrustSystem, label: str) -> List[Dict]:
        """Shared simulation loop for RL-style trust systems."""

        results = []

        for step in range(self.num_timesteps):
            # CRITICAL: Reset random seed for each step to ensure identical randomness
            step_seed = self.random_seed + step
            np.random.seed(step_seed)
            random.seed(step_seed)

            # Step the simulation
            frame_data = env.step()

            # Update current timestep tracks for all robots
            for robot in env.robots:
                robot.update_current_timestep_tracks()

            # Apply RL trust updates
            try:
                trust_system.update_trust(env.robots)
            except Exception as e:
                print(f"⚠️ {label} trust system error at step {step}: {e}")

            # Collect results AFTER RL updates are applied
            step_result = {
                'step': step,
                'time': env.time,
                'robot_trust_values': {
                    robot.id: robot.trust_value for robot in env.robots
                },
                'robot_alpha_beta': {
                    robot.id: {'alpha': robot.trust_alpha, 'beta': robot.trust_beta}
                    for robot in env.robots
                },
                'track_trust_values': {},
                'adversarial_robots': [r.id for r in env.robots if r.is_adversarial],
                'legitimate_robots': [r.id for r in env.robots if not r.is_adversarial]
            }

            # Collect track trust values
            for robot in env.robots:
                step_result['track_trust_values'][robot.id] = {
                    track.track_id: {
                        'trust_value': track.trust_value,
                        'alpha': track.trust_alpha,
                        'beta': track.trust_beta,
                        'object_id': track.object_id
                    }
                    for track in robot.get_all_tracks()
                }

            results.append(step_result)

            if step % 100 == 0:
                print(f"  {label} model step {step}/{self.num_timesteps}")

        print(f"✅ {label} simulation completed")
        return results

    def run_comparison(self) -> Dict:
        """Run complete comparison between both methods"""
        print("🔄 Starting trust method comparison...")
        print(f"Configuration: {self.num_robots} robots, {self.num_targets} targets, {self.num_timesteps} steps")
        print(f"Random seed: {self.random_seed}")

        # Create identical environments for each method
        paper_env, baseline_env, rl_env = self.create_identical_environments(3)

        # Run both simulations
        print("\n" + "="*50)
        self.paper_results = self.run_paper_algorithm_simulation(paper_env)

        print("\n" + "="*50)
        self.baseline_results = self.run_baseline_simulation(baseline_env)

        print("\n" + "="*50)
        self.rl_results = self.run_rl_model_simulation(rl_env)

        # Generate comparison results
        comparison_results = {
            'configuration': {
                'num_robots': self.num_robots,
                'num_targets': self.num_targets,
                'num_timesteps': self.num_timesteps,
                'random_seed': self.random_seed,
                'rl_model_path': str(self.rl_model_path) if self.rl_model_path else None,
                'fixed_step_scale': self.fixed_step_scale
            },
            'paper_results': self.paper_results,
            'baseline_results': self.baseline_results,
            'rl_results': self.rl_results,
            'comparison_metrics': self._compute_comparison_metrics()
        }

        return comparison_results

    def _compute_comparison_metrics(self) -> Dict:
        """Compute comparison metrics across all trust methods"""
        if not self.paper_results or not self.baseline_results or not self.rl_results:
            print("⚠️ Incomplete results for comparison")
            return {}

        print("📊 Computing comparison metrics...")

        metrics = {
            'trust_convergence': {},
            'final_trust_values': {},
            'method_differences': {}
        }

        method_results = {
            'paper': self.paper_results,
            'baseline': self.baseline_results,
            'rl': self.rl_results
        }

        robot_ids = list(self.paper_results[0]['robot_trust_values'].keys())
        adversarial_ids = set(self.paper_results[0]['adversarial_robots'])
        legitimate_ids = set(self.paper_results[0]['legitimate_robots'])

        for robot_id in robot_ids:
            robot_type = 'adversarial' if robot_id in adversarial_ids else 'legitimate'
            method_stats = {}

            for method, results in method_results.items():
                series = [step['robot_trust_values'][robot_id] for step in results]
                values = np.array(series)
                method_stats[method] = {
                    'final': float(values[-1]),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }

            # Correlations between methods
            correlations = {}
            paper_series = np.array([step['robot_trust_values'][robot_id] for step in self.paper_results])
            baseline_series = np.array([step['robot_trust_values'][robot_id] for step in self.baseline_results])
            rl_series = np.array([step['robot_trust_values'][robot_id] for step in self.rl_results])

            def safe_corr(a, b):
                if len(a) < 2:
                    return 0.0
                try:
                    corr = np.corrcoef(a, b)[0, 1]
                    if np.isnan(corr):
                        return 0.0
                    return float(corr)
                except Exception:
                    return 0.0

            correlations['paper_vs_baseline'] = safe_corr(paper_series, baseline_series)
            correlations['paper_vs_rl'] = safe_corr(paper_series, rl_series)
            correlations['baseline_vs_rl'] = safe_corr(baseline_series, rl_series)

            metrics['trust_convergence'][robot_id] = {
                'robot_type': robot_type,
                'methods': method_stats,
                'correlations': correlations
            }

        # Aggregate metrics by robot type for final trust values
        for robot_type, id_set in [('legitimate', legitimate_ids), ('adversarial', adversarial_ids)]:
            if not id_set:
                continue
            stats = {}
            for method in method_results.keys():
                finals = [metrics['trust_convergence'][rid]['methods'][method]['final']
                          for rid in id_set]
                stats[method] = {
                    'mean': float(np.mean(finals)),
                    'std': float(np.std(finals))
                }
            metrics['final_trust_values'][robot_type] = stats

        # Method-wise difference summaries (RL vs others)
        diff_stats = {}
        for method in ['paper', 'baseline']:
            diffs = []
            for robot_id in robot_ids:
                rl_final = metrics['trust_convergence'][robot_id]['methods']['rl']['final']
                other_final = metrics['trust_convergence'][robot_id]['methods'][method]['final']
                diffs.append(rl_final - other_final)
            diff_stats[f'rl_minus_{method}'] = {
                'mean': float(np.mean(diffs)),
                'std': float(np.std(diffs))
            }

        metrics['method_differences'] = diff_stats

        print("✅ Comparison metrics computed")
        return metrics

    def save_results(self, filename: str = "trust_comparison_results.json"):
        """Save comparison results to file"""
        results = {
            'configuration': {
                'num_robots': self.num_robots,
                'num_targets': self.num_targets,
                'num_timesteps': self.num_timesteps,
                'random_seed': self.random_seed,
                'rl_model_path': str(self.rl_model_path) if self.rl_model_path else None,
                'fixed_step_scale': self.fixed_step_scale
            },
            'paper_results': self.paper_results,
            'baseline_results': self.baseline_results,
            'rl_results': self.rl_results,
            'comparison_metrics': self._compute_comparison_metrics()
        }

        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        results = convert_numpy_types(results)

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"📁 Results saved to {filename}")
        return filename

    def visualize_comparison(self, save_path: str = "trust_comparison.png"):
        """Create visualization comparing paper, baseline, and RL methods"""
        if not (self.paper_results and self.baseline_results and self.rl_results):
            print("⚠️ Incomplete results available for visualization")
            return

        print("📈 Creating comparison visualization...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Trust Method Comparison: Paper vs Fixed Step vs RL', fontsize=16, fontweight='bold')

        timesteps = [step['step'] for step in self.paper_results]
        adversarial_robots = self.paper_results[0]['adversarial_robots']
        legitimate_robots = self.paper_results[0]['legitimate_robots']

        method_styles = {
            'Paper': {'results': self.paper_results, 'linestyle': '-', 'alpha': 0.7},
            'Baseline': {'results': self.baseline_results, 'linestyle': '-.', 'alpha': 0.8},
            'RL': {'results': self.rl_results, 'linestyle': '--', 'alpha': 0.9}
        }

        # Plot 1: Legitimate robots
        ax1 = axes[0, 0]
        for robot_id in legitimate_robots:
            for label, style in method_styles.items():
                trust_values = [step['robot_trust_values'][robot_id] for step in style['results']]
                ax1.plot(timesteps, trust_values, linestyle=style['linestyle'], alpha=style['alpha'],
                         label=f'{label} R{robot_id}')
        ax1.set_title('Legitimate Robots Trust Evolution')
        ax1.set_xlabel('Simulation Step')
        ax1.set_ylabel('Trust Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Adversarial robots
        ax2 = axes[0, 1]
        if adversarial_robots:
            for robot_id in adversarial_robots:
                for label, style in method_styles.items():
                    trust_values = [step['robot_trust_values'][robot_id] for step in style['results']]
                    ax2.plot(timesteps, trust_values, linestyle=style['linestyle'], alpha=style['alpha'],
                             label=f'{label} R{robot_id}')
        else:
            ax2.text(0.5, 0.5, 'No Adversarial Robots\nin this Scenario',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes, fontsize=12, style='italic')
        ax2.set_title('Adversarial Robots Trust Evolution')
        ax2.set_xlabel('Simulation Step')
        ax2.set_ylabel('Trust Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Average trust over time
        ax3 = axes[1, 0]
        for label, style in method_styles.items():
            leg_avg = np.mean([[step['robot_trust_values'][rid] for rid in legitimate_robots]
                               for step in style['results']], axis=1)
            ax3.plot(timesteps, leg_avg, label=f'{label} - Legitimate', linewidth=2,
                     linestyle=style['linestyle'])
            if adversarial_robots:
                adv_avg = np.mean([[step['robot_trust_values'][rid] for rid in adversarial_robots]
                                   for step in style['results']], axis=1)
                ax3.plot(timesteps, adv_avg, label=f'{label} - Adversarial', linewidth=2,
                         linestyle=style['linestyle'])
        ax3.set_title('Average Trust by Robot Type')
        ax3.set_xlabel('Simulation Step')
        ax3.set_ylabel('Average Trust Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Final trust comparison
        ax4 = axes[1, 1]
        method_order = ['Paper', 'Baseline', 'RL']
        categories = []
        means = []
        stds = []
        colors = []

        color_map = {
            'Paper': 'tab:blue',
            'Baseline': 'tab:orange',
            'RL': 'tab:green'
        }

        for label in method_order:
            results = method_styles[label]['results']
            final_leg = [results[-1]['robot_trust_values'][rid] for rid in legitimate_robots]
            categories.append(f'{label}\nLegitimate')
            means.append(np.mean(final_leg))
            stds.append(np.std(final_leg))
            colors.append(color_map[label])

        if adversarial_robots:
            for label in method_order:
                results = method_styles[label]['results']
                final_adv = [results[-1]['robot_trust_values'][rid] for rid in adversarial_robots]
                categories.append(f'{label}\nAdversarial')
                means.append(np.mean(final_adv))
                stds.append(np.std(final_adv))
                colors.append(color_map[label])

        x_pos = np.arange(len(categories))
        bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax4.set_title('Final Trust Values Comparison')
        ax4.set_ylabel('Final Trust Value')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories, rotation=15)
        ax4.grid(True, alpha=0.3, axis='y')

        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Visualization saved to {save_path}")

    def print_summary(self):
        """Print comparison summary"""
        if not self.rl_results:
            print("⚠️ No RL results available for summary")
            return

        print("\n" + "="*60)
        print("🎯 TRUST METHOD COMPARISON SUMMARY")
        print("="*60)

        metrics = self._compute_comparison_metrics()

        print(f"Configuration: {self.num_robots} robots, {self.num_timesteps} steps, seed {self.random_seed}")
        rl_label = str(self.rl_model_path) if self.rl_model_path else "fresh initialization"
        print(f"RL Model: {rl_label}")
        print(f"Fixed Step Scale: {self.fixed_step_scale:.2f}")

        # Final trust values by robot type
        if 'final_trust_values' in metrics:
            print("\n📊 Final Trust Values:")
            for robot_type, method_vals in metrics['final_trust_values'].items():
                print(f"  {robot_type.title()} Robots:")
                for method, stats in method_vals.items():
                    print(f"    {method.capitalize():<9}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        # Method differences summary
        if 'method_differences' in metrics:
            print("\n🔍 RL Improvements over Other Methods:")
            for key, stats in metrics['method_differences'].items():
                print(f"  {key}: {stats['mean']:+.3f} ± {stats['std']:.3f}")

        # Individual robot results
        print("\n🤖 Individual Robot Results:")
        legitimate_robots = self.paper_results[0]['legitimate_robots']
        adversarial_robots = self.paper_results[0]['adversarial_robots']

        print("  Legitimate Robots:")
        for robot_id in legitimate_robots:
            if robot_id in metrics['trust_convergence']:
                conv = metrics['trust_convergence'][robot_id]
                methods = conv['methods']
                print(f"    Robot {robot_id}: "
                      f"Paper={methods['paper']['final']:.3f}, "
                      f"Baseline={methods['baseline']['final']:.3f}, "
                      f"RL={methods['rl']['final']:.3f}, "
                      f"Corr(RL,B)= {conv['correlations']['baseline_vs_rl']:.3f}")

        print("  Adversarial Robots:")
        for robot_id in adversarial_robots:
            if robot_id in metrics['trust_convergence']:
                conv = metrics['trust_convergence'][robot_id]
                methods = conv['methods']
                print(f"    Robot {robot_id}: "
                      f"Paper={methods['paper']['final']:.3f}, "
                      f"Baseline={methods['baseline']['final']:.3f}, "
                      f"RL={methods['rl']['final']:.3f}, "
                      f"Corr(RL,B)= {conv['correlations']['baseline_vs_rl']:.3f}")

        print("\n✨ Comparison completed successfully!")


def main():
    """Main function to run three specific scenarios"""
    print("🚀 Starting Trust Method Comparison - Three Scenarios")

    # =============================================================================
    # GLOBAL PARAMETERS - All configuration in one place
    # =============================================================================

    # Simulation Parameters
    NUM_ROBOTS = 10
    NUM_TARGETS = 20
    NUM_TIMESTEPS = 100
    RANDOM_SEED = 8

    # Environment Parameters
    WORLD_SIZE = 100.0
    ADVERSARIAL_RATIO = 0.3
    PROXIMAL_RANGE = 50.0
    FOV_RANGE = 50.0
    FOV_ANGLE = np.pi/3

    # Model Parameters
    SUPERVISED_MODEL_PATH = "supervised_trust_model.pth"
    RL_MODEL_PATH = "rl_trust_model.pth"

    # Scenario-specific parameters (only FP/FN rates vary)
    scenarios = [
        {
            "name": "Scenario_1_Low_FP_Low_FN",
            "false_positive_rate": 0.3,
            "false_negative_rate": 0.1
        },
        {
            "name": "Scenario_2_High_FP_Low_FN",
            "false_positive_rate": 0.8,
            "false_negative_rate": 0.1
        },
        {
            "name": "Scenario_3_Low_FP_High_FN",
            "false_positive_rate": 0.3,
            "false_negative_rate": 0.5
        }
    ]

    # =============================================================================
    # SCENARIO EXECUTION
    # =============================================================================

    all_results = {}
    summary_stats = []

    for i, scenario in enumerate(scenarios):
        print(f"\n{'='*60}")
        print(f"Running {scenario['name']} ({i+1}/3)")
        print('='*60)
        print(f"Configuration:")
        print(f"  - Robots: {NUM_ROBOTS}")
        print(f"  - Targets: {NUM_TARGETS}")
        print(f"  - Timesteps: {NUM_TIMESTEPS}")
        print(f"  - Random seed: {RANDOM_SEED}")
        print(f"  - Adversarial ratio: {ADVERSARIAL_RATIO}")
        print(f"  - World size: {WORLD_SIZE}x{WORLD_SIZE}")
        print(f"  - Proximal range: {PROXIMAL_RANGE}")
        print(f"  - FOV range: {FOV_RANGE}")
        print(f"  - FOV angle: π/3")
        print(f"  - False positive rate: {scenario['false_positive_rate']}")
        print(f"  - False negative rate: {scenario['false_negative_rate']}")

        # Create comparison instance with centralized parameters
        comparison = TrustMethodComparison(
            supervised_model_path=SUPERVISED_MODEL_PATH,
            rl_model_path=RL_MODEL_PATH,
            num_robots=NUM_ROBOTS,
            num_targets=NUM_TARGETS,
            num_timesteps=NUM_TIMESTEPS,
            random_seed=RANDOM_SEED,
            world_size=WORLD_SIZE,
            fov_range=FOV_RANGE,
            fov_angle=FOV_ANGLE
        )

        # Set global and scenario-specific parameters
        comparison.adversarial_ratio = ADVERSARIAL_RATIO
        comparison.false_positive_rate = scenario['false_positive_rate']
        comparison.false_negative_rate = scenario['false_negative_rate']
        comparison.proximal_range = PROXIMAL_RANGE

        try:
            # Run comparison
            results = comparison.run_comparison()

            # Store results
            all_results[scenario['name']] = results

            # Save individual results
            results_file = f"trust_comparison_{scenario['name']}.json"
            comparison.save_results(results_file)

            # Create visualization
            viz_file = f"trust_comparison_{scenario['name']}.png"
            comparison.visualize_comparison(viz_file)

            # Extract summary statistics
            metrics = results['comparison_metrics']
            if 'final_trust_values' in metrics:
                legit_stats = metrics['final_trust_values'].get('legitimate', {})
                adv_stats = metrics['final_trust_values'].get('adversarial', {})
                diff_stats = metrics.get('method_differences', {})

                scenario_summary = {
                    'scenario': scenario['name'],
                    'false_positive_rate': scenario['false_positive_rate'],
                    'false_negative_rate': scenario['false_negative_rate'],
                    'fixed_step_scale': comparison.fixed_step_scale,
                    'legitimate_paper': legit_stats.get('paper', {}).get('mean', 0),
                    'legitimate_baseline': legit_stats.get('baseline', {}).get('mean', 0),
                    'legitimate_rl': legit_stats.get('rl', {}).get('mean', 0),
                    'adversarial_paper': adv_stats.get('paper', {}).get('mean', 0),
                    'adversarial_baseline': adv_stats.get('baseline', {}).get('mean', 0),
                    'adversarial_rl': adv_stats.get('rl', {}).get('mean', 0),
                    'rl_minus_paper': diff_stats.get('rl_minus_paper', {}).get('mean', 0),
                    'rl_minus_baseline': diff_stats.get('rl_minus_baseline', {}).get('mean', 0)
                }
                summary_stats.append(scenario_summary)

            # Print summary
            comparison.print_summary()

            print(f"✅ {scenario['name']} completed successfully!")
            print(f"📁 Results saved to {results_file}")
            print(f"📊 Visualization saved to {viz_file}")

        except Exception as e:
            print(f"❌ Error in {scenario['name']}: {e}")
            continue

    # Print overall comparison
    print(f"\n{'='*70}")
    print("🎯 THREE SCENARIO COMPARISON SUMMARY")
    print('='*70)

    if summary_stats:
        print("\n📊 Performance Summary:")
        header = "Scenario Name               | FP Rate | FN Rate | Leg (P/B/R)        | Adv (P/B/R)        | RL-P Δ  | RL-B Δ"
        print(header)
        print("-" * len(header))

        for stat in summary_stats:
            print(f"{stat['scenario']:<26} | "
                  f"{stat['false_positive_rate']:<7} | "
                  f"{stat['false_negative_rate']:<7} | "
                  f"{stat['legitimate_paper']:.3f}/{stat['legitimate_baseline']:.3f}/{stat['legitimate_rl']:.3f} | "
                  f"{stat['adversarial_paper']:.3f}/{stat['adversarial_baseline']:.3f}/{stat['adversarial_rl']:.3f} | "
                  f"{stat['rl_minus_paper']:+.3f} | {stat['rl_minus_baseline']:+.3f}")

        avg_rl_minus_paper = np.mean([s['rl_minus_paper'] for s in summary_stats])
        avg_rl_minus_baseline = np.mean([s['rl_minus_baseline'] for s in summary_stats])

        print("-" * len(header))
        print(f"{'AVERAGE':<26} | {'':>7} | {'':>7} | {'':>18} | {'':>18} | {avg_rl_minus_paper:+.3f} | {avg_rl_minus_baseline:+.3f}")

        print(f"\n🔍 Key Findings:")
        print(f"   • Average RL vs Paper final trust difference: {avg_rl_minus_paper:+.3f}")
        print(f"   • Average RL vs Baseline final trust difference: {avg_rl_minus_baseline:+.3f}")

        print(f"\n📈 Scenario Analysis:")
        for stat in summary_stats:
            print(f"   • {stat['scenario']}:")
            print(f"     - FP/FN rates: {stat['false_positive_rate']:.1f}/{stat['false_negative_rate']:.1f}")
            print(f"     - RL vs Paper: {stat['rl_minus_paper']:+.3f}")
            print(f"     - RL vs Baseline: {stat['rl_minus_baseline']:+.3f}")

    # Save comprehensive results using centralized parameters
    comprehensive_results = {
        'scenarios': all_results,
        'summary_statistics': summary_stats,
        'configuration': {
            'num_robots': NUM_ROBOTS,
            'num_targets': NUM_TARGETS,
            'adversarial_ratio': ADVERSARIAL_RATIO,
            'world_size': WORLD_SIZE,
            'proximal_range': PROXIMAL_RANGE,
            'fov_range': FOV_RANGE,
            'fov_angle': 'π/3',
            'num_timesteps': NUM_TIMESTEPS,
            'random_seed': RANDOM_SEED,
            'rl_model_path': RL_MODEL_PATH,
            'fixed_step_scale': summary_stats[0]['fixed_step_scale'] if summary_stats else 0.5
        }
    }

    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    comprehensive_results = convert_numpy_types(comprehensive_results)

    with open("three_scenario_comparison.json", 'w') as f:
        json.dump(comprehensive_results, f, indent=2)

    print(f"\n✅ Three scenario comparison completed successfully!")
    print(f"📁 Comprehensive results saved to three_scenario_comparison.json")


if __name__ == "__main__":
    main()
