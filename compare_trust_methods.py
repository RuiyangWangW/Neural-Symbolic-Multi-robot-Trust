#!/usr/bin/env python3
"""
Comparison of Supervised Trust Model vs Paper Trust Algorithm

This script runs both methods on the same simulation environment with identical
random seeds and initial conditions to enable fair comparison.
"""

import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple

# Import required modules
from simulation_environment import SimulationEnvironment
from paper_trust_algorithm import PaperTrustAlgorithm
from supervised_trust_gnn import SupervisedTrustPredictor
from robot_track_classes import Robot, Track


class TrustMethodComparison:
    """Compares supervised trust model against paper trust algorithm"""

    def __init__(self,
                 model_path: str = "supervised_trust_model.pth",
                 num_robots: int = 5,
                 num_targets: int = 10,
                 num_timesteps: int = 500,
                 random_seed: int = 42,
                 world_size: float = 60.0,
                 fov_range: float = 50.0,
                 fov_angle: float = np.pi/3):
        """
        Initialize comparison with both trust methods

        Args:
            model_path: Path to trained supervised trust model
            num_robots: Number of robots in simulation
            num_targets: Number of ground truth targets
            num_timesteps: Number of simulation steps
            random_seed: Random seed for reproducibility
            world_size: Size of the simulation world
            fov_range: Field of view range for robots
            fov_angle: Field of view angle for robots
        """
        self.model_path = model_path
        self.num_robots = num_robots
        self.num_targets = num_targets
        self.num_timesteps = num_timesteps
        self.random_seed = random_seed
        self.world_size = world_size
        self.fov_range = fov_range
        self.fov_angle = fov_angle

        # Initialize trust methods
        self.paper_algorithm = PaperTrustAlgorithm()
        self.supervised_predictor = None

        # Results storage
        self.paper_results = []
        self.supervised_results = []

        # Simulation parameters (can be overridden)
        self.adversarial_ratio = 0.3
        self.false_positive_rate = 0.5
        self.false_negative_rate = 0.0
        self.proximal_range = 50.0

        # Try to load supervised model after proximal_range is set
        try:
            self.supervised_predictor = SupervisedTrustPredictor(model_path, proximal_range=self.proximal_range)
            print(f"✅ Loaded supervised trust model from {model_path}")
        except Exception as e:
            print(f"⚠️ Failed to load supervised model: {e}")
            print("Will only run paper algorithm comparison")

    def create_identical_environments(self) -> Tuple[SimulationEnvironment, SimulationEnvironment]:
        """Create two identical simulation environments for fair comparison"""

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        # Create first environment
        env1 = SimulationEnvironment(
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

        # Store initial state for reproduction
        initial_robot_states = []
        for robot in env1.robots:
            initial_robot_states.append({
                'id': robot.id,
                'position': robot.position.copy(),
                'velocity': robot.velocity.copy(),
                'orientation': robot.orientation,
                'is_adversarial': robot.is_adversarial,
                'start_position': robot.start_position.copy(),
                'goal_position': robot.goal_position.copy(),
                'patrol_speed': robot.patrol_speed,
                'trust_alpha': robot.trust_alpha,
                'trust_beta': robot.trust_beta
            })

        initial_gt_objects = []
        for obj in env1.ground_truth_objects:
            initial_gt_objects.append({
                'id': obj.id,
                'position': obj.position.copy(),
                'velocity': obj.velocity.copy(),
                'object_type': obj.object_type,
                'movement_pattern': obj.movement_pattern,
                'spawn_time': obj.spawn_time,
                'lifespan': obj.lifespan,
                'base_speed': obj.base_speed,
                'turn_probability': obj.turn_probability,
                'direction_change_time': obj.direction_change_time,
                'circular_center': obj.circular_center.copy() if obj.circular_center is not None else None,
                'circular_radius': obj.circular_radius
            })

        # Reset random seed and create second identical environment
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        env2 = SimulationEnvironment(
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

        return env1, env2

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

    def run_supervised_model_simulation(self, env: SimulationEnvironment) -> List[Dict]:
        """Run simulation using supervised trust model"""
        if self.supervised_predictor is None:
            print("❌ Supervised model not available, skipping")
            return []

        print("🚀 Running supervised model simulation...")

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

            # Apply supervised model predictions for each robot's ego graph
            for ego_robot in env.robots:
                try:
                    # Get predictions from supervised model
                    result = self.supervised_predictor.predict_from_robots_tracks(
                        ego_robot, env.robots, threshold=0.5
                    )

                    # Update alpha/beta values based on model predictions
                    self._update_trust_from_predictions(result['predictions'], result['graph_data'])

                except Exception as e:
                    print(f"⚠️ Supervised model error for robot {ego_robot.id} at step {step}: {e}")

            # Collect results (same structure as paper algorithm)
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
                print(f"  Supervised model step {step}/{self.num_timesteps}")

        print("✅ Supervised model simulation completed")
        return results

    def _update_trust_from_predictions(self, predictions: Dict, graph_data):
        """
        Update alpha/beta values based on supervised model predictions using correct node mappings

        Args:
            predictions: Model predictions containing trust probabilities
            graph_data: Graph data with node mappings (agent_nodes, track_nodes)
        """

        # Update agent (robot) trust values using correct mappings
        if 'agent' in predictions and hasattr(graph_data, 'agent_nodes'):
            agent_probs = predictions['agent']['probabilities']

            # Use the exact node mapping from graph_data
            for robot_id, node_idx in graph_data.agent_nodes.items():
                if node_idx < len(agent_probs):
                    p = agent_probs[node_idx][0]  # Extract probability from array

                    # Convert probability to alpha/beta update
                    if p >= 0.5:
                        # High trust prediction - increase alpha
                        delta_alpha = p
                        delta_beta = 0.0
                    else:
                        # Low trust prediction - increase beta
                        delta_alpha = 0.0
                        delta_beta = (1 - p)

                    # Find the robot object and apply update
                    for robot in graph_data._proximal_robots:
                        if robot.id == robot_id:
                            # Apply update with small step size to prevent rapid changes
                            step_size = 0.1
                            robot.update_trust(delta_alpha * step_size, delta_beta * step_size)
                            break

        # Update track trust values using correct mappings
        if 'track' in predictions and hasattr(graph_data, 'track_nodes'):
            track_probs = predictions['track']['probabilities']

            # Use the exact node mapping from graph_data
            for track_id, node_idx in graph_data.track_nodes.items():
                if node_idx < len(track_probs):
                    p = track_probs[node_idx][0]  # Extract probability from array

                    # Convert probability to alpha/beta update
                    if p >= 0.5:
                        # High trust prediction - increase alpha
                        delta_alpha = p
                        delta_beta = 0.0
                    else:
                        # Low trust prediction - increase beta
                        delta_alpha = 0.0
                        delta_beta = (1 - p)

                    # Find the track object and apply update
                    for robot in graph_data._proximal_robots:
                        for track in robot.get_all_tracks():
                            if track.track_id == track_id:
                                # Apply update with small step size
                                step_size = 0.1
                                track.update_trust(delta_alpha * step_size, delta_beta * step_size)
                                break


    def run_comparison(self) -> Dict:
        """Run complete comparison between both methods"""
        print("🔄 Starting trust method comparison...")
        print(f"Configuration: {self.num_robots} robots, {self.num_targets} targets, {self.num_timesteps} steps")
        print(f"Random seed: {self.random_seed}")

        # Create identical environments
        paper_env, supervised_env = self.create_identical_environments()

        # Run both simulations
        print("\n" + "="*50)
        self.paper_results = self.run_paper_algorithm_simulation(paper_env)

        print("\n" + "="*50)
        self.supervised_results = self.run_supervised_model_simulation(supervised_env)

        # Generate comparison results
        comparison_results = {
            'configuration': {
                'num_robots': self.num_robots,
                'num_targets': self.num_targets,
                'num_timesteps': self.num_timesteps,
                'random_seed': self.random_seed,
                'model_path': self.model_path
            },
            'paper_results': self.paper_results,
            'supervised_results': self.supervised_results,
            'comparison_metrics': self._compute_comparison_metrics()
        }

        return comparison_results

    def _compute_comparison_metrics(self) -> Dict:
        """Compute comparison metrics between the two methods"""
        if not self.supervised_results:
            print("⚠️ No supervised results available for comparison")
            return {}

        print("📊 Computing comparison metrics...")

        metrics = {
            'trust_convergence': {},
            'final_trust_values': {},
            'trust_evolution_stats': {},
            'method_differences': {}
        }

        # Extract trust evolution for both methods
        paper_trust_evolution = {}
        supervised_trust_evolution = {}

        # Initialize trust evolution tracking
        for robot_id in self.paper_results[0]['robot_trust_values'].keys():
            paper_trust_evolution[robot_id] = []
            supervised_trust_evolution[robot_id] = []

        # Collect trust values over time
        for step_data in self.paper_results:
            for robot_id, trust_val in step_data['robot_trust_values'].items():
                paper_trust_evolution[robot_id].append(trust_val)

        for step_data in self.supervised_results:
            for robot_id, trust_val in step_data['robot_trust_values'].items():
                supervised_trust_evolution[robot_id].append(trust_val)

        # Compute metrics for each robot
        for robot_id in paper_trust_evolution.keys():
            paper_values = np.array(paper_trust_evolution[robot_id])
            supervised_values = np.array(supervised_trust_evolution[robot_id])

            # Get robot type (adversarial or legitimate)
            robot_type = "adversarial" if robot_id in self.paper_results[0]['adversarial_robots'] else "legitimate"

            metrics['trust_convergence'][robot_id] = {
                'robot_type': robot_type,
                'paper_final': float(paper_values[-1]),
                'supervised_final': float(supervised_values[-1]),
                'paper_mean': float(np.mean(paper_values)),
                'supervised_mean': float(np.mean(supervised_values)),
                'paper_std': float(np.std(paper_values)),
                'supervised_std': float(np.std(supervised_values)),
                'correlation': float(np.corrcoef(paper_values, supervised_values)[0, 1]) if len(paper_values) > 1 else 0.0
            }

        # Aggregate metrics by robot type
        legitimate_robots = [rid for rid in paper_trust_evolution.keys()
                           if rid in self.paper_results[0]['legitimate_robots']]
        adversarial_robots = [rid for rid in paper_trust_evolution.keys()
                            if rid in self.paper_results[0]['adversarial_robots']]

        for robot_type, robot_ids in [('legitimate', legitimate_robots), ('adversarial', adversarial_robots)]:
            if robot_ids:
                paper_finals = [metrics['trust_convergence'][rid]['paper_final'] for rid in robot_ids]
                supervised_finals = [metrics['trust_convergence'][rid]['supervised_final'] for rid in robot_ids]

                metrics['final_trust_values'][robot_type] = {
                    'paper_mean': float(np.mean(paper_finals)),
                    'supervised_mean': float(np.mean(supervised_finals)),
                    'paper_std': float(np.std(paper_finals)),
                    'supervised_std': float(np.std(supervised_finals)),
                    'difference': float(np.mean(supervised_finals) - np.mean(paper_finals))
                }

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
                'model_path': self.model_path
            },
            'paper_results': self.paper_results,
            'supervised_results': self.supervised_results,
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
        """Create visualization comparing both methods"""
        if not self.supervised_results:
            print("⚠️ No supervised results available for visualization")
            return

        print("📈 Creating comparison visualization...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Trust Method Comparison: Paper Algorithm vs Supervised Model', fontsize=16, fontweight='bold')

        # Extract data for plotting
        timesteps = [step['step'] for step in self.paper_results]

        # Get robot categories
        adversarial_robots = self.paper_results[0]['adversarial_robots']
        legitimate_robots = self.paper_results[0]['legitimate_robots']

        # Plot 1: Trust evolution for legitimate robots
        ax1 = axes[0, 0]
        for robot_id in legitimate_robots:
            paper_trust = [step['robot_trust_values'][robot_id] for step in self.paper_results]
            supervised_trust = [step['robot_trust_values'][robot_id] for step in self.supervised_results]

            ax1.plot(timesteps, paper_trust, label=f'Paper R{robot_id}', linestyle='-', alpha=0.7)
            ax1.plot(timesteps, supervised_trust, label=f'Supervised R{robot_id}', linestyle='--', alpha=0.7)

        ax1.set_title('Legitimate Robots Trust Evolution')
        ax1.set_xlabel('Simulation Step')
        ax1.set_ylabel('Trust Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Trust evolution for adversarial robots
        ax2 = axes[0, 1]
        for robot_id in adversarial_robots:
            paper_trust = [step['robot_trust_values'][robot_id] for step in self.paper_results]
            supervised_trust = [step['robot_trust_values'][robot_id] for step in self.supervised_results]

            ax2.plot(timesteps, paper_trust, label=f'Paper R{robot_id}', linestyle='-', alpha=0.7)
            ax2.plot(timesteps, supervised_trust, label=f'Supervised R{robot_id}', linestyle='--', alpha=0.7)

        ax2.set_title('Adversarial Robots Trust Evolution')
        ax2.set_xlabel('Simulation Step')
        ax2.set_ylabel('Trust Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Average trust by robot type
        ax3 = axes[1, 0]

        # Calculate average trust over time for each method and robot type
        paper_leg_avg = np.mean([[step['robot_trust_values'][rid] for rid in legitimate_robots]
                                for step in self.paper_results], axis=1)
        supervised_leg_avg = np.mean([[step['robot_trust_values'][rid] for rid in legitimate_robots]
                                     for step in self.supervised_results], axis=1)

        paper_adv_avg = np.mean([[step['robot_trust_values'][rid] for rid in adversarial_robots]
                                for step in self.paper_results], axis=1)
        supervised_adv_avg = np.mean([[step['robot_trust_values'][rid] for rid in adversarial_robots]
                                     for step in self.supervised_results], axis=1)

        ax3.plot(timesteps, paper_leg_avg, label='Paper - Legitimate', color='green', linestyle='-', linewidth=2)
        ax3.plot(timesteps, supervised_leg_avg, label='Supervised - Legitimate', color='green', linestyle='--', linewidth=2)
        ax3.plot(timesteps, paper_adv_avg, label='Paper - Adversarial', color='red', linestyle='-', linewidth=2)
        ax3.plot(timesteps, supervised_adv_avg, label='Supervised - Adversarial', color='red', linestyle='--', linewidth=2)

        ax3.set_title('Average Trust by Robot Type')
        ax3.set_xlabel('Simulation Step')
        ax3.set_ylabel('Average Trust Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Final trust comparison
        ax4 = axes[1, 1]

        final_paper_leg = [self.paper_results[-1]['robot_trust_values'][rid] for rid in legitimate_robots]
        final_supervised_leg = [self.supervised_results[-1]['robot_trust_values'][rid] for rid in legitimate_robots]
        final_paper_adv = [self.paper_results[-1]['robot_trust_values'][rid] for rid in adversarial_robots]
        final_supervised_adv = [self.supervised_results[-1]['robot_trust_values'][rid] for rid in adversarial_robots]

        x_pos = np.arange(4)
        means = [np.mean(final_paper_leg), np.mean(final_supervised_leg),
                np.mean(final_paper_adv), np.mean(final_supervised_adv)]
        stds = [np.std(final_paper_leg), np.std(final_supervised_leg),
               np.std(final_paper_adv), np.std(final_supervised_adv)]
        labels = ['Paper\nLegitimate', 'Supervised\nLegitimate', 'Paper\nAdversarial', 'Supervised\nAdversarial']
        colors = ['lightgreen', 'darkgreen', 'lightcoral', 'darkred']

        bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax4.set_title('Final Trust Values Comparison')
        ax4.set_ylabel('Final Trust Value')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels)
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
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
        if not self.supervised_results:
            print("⚠️ No supervised results available for summary")
            return

        print("\n" + "="*60)
        print("🎯 TRUST METHOD COMPARISON SUMMARY")
        print("="*60)

        metrics = self._compute_comparison_metrics()

        print(f"Configuration: {self.num_robots} robots, {self.num_timesteps} steps, seed {self.random_seed}")
        print(f"Model: {self.model_path}")

        # Final trust values by robot type
        if 'final_trust_values' in metrics:
            print("\n📊 Final Trust Values:")
            for robot_type, values in metrics['final_trust_values'].items():
                print(f"  {robot_type.title()} Robots:")
                print(f"    Paper Algorithm:    {values['paper_mean']:.3f} ± {values['paper_std']:.3f}")
                print(f"    Supervised Model:   {values['supervised_mean']:.3f} ± {values['supervised_std']:.3f}")
                print(f"    Difference:         {values['difference']:+.3f}")

        # Individual robot results
        print("\n🤖 Individual Robot Results:")
        legitimate_robots = self.paper_results[0]['legitimate_robots']
        adversarial_robots = self.paper_results[0]['adversarial_robots']

        print("  Legitimate Robots:")
        for robot_id in legitimate_robots:
            if robot_id in metrics['trust_convergence']:
                conv = metrics['trust_convergence'][robot_id]
                print(f"    Robot {robot_id}: Paper={conv['paper_final']:.3f}, "
                      f"Supervised={conv['supervised_final']:.3f}, "
                      f"Corr={conv['correlation']:.3f}")

        print("  Adversarial Robots:")
        for robot_id in adversarial_robots:
            if robot_id in metrics['trust_convergence']:
                conv = metrics['trust_convergence'][robot_id]
                print(f"    Robot {robot_id}: Paper={conv['paper_final']:.3f}, "
                      f"Supervised={conv['supervised_final']:.3f}, "
                      f"Corr={conv['correlation']:.3f}")

        print("\n✨ Comparison completed successfully!")


def main():
    """Main function to run three specific scenarios"""
    print("🚀 Starting Trust Method Comparison - Three Scenarios")

    # =============================================================================
    # GLOBAL PARAMETERS - All configuration in one place
    # =============================================================================

    # Simulation Parameters
    NUM_ROBOTS = 5
    NUM_TARGETS = 10
    NUM_TIMESTEPS = 150
    RANDOM_SEED = 42

    # Environment Parameters
    WORLD_SIZE = 100.0
    ADVERSARIAL_RATIO = 0.2
    PROXIMAL_RANGE = 50.0
    FOV_RANGE = 50.0
    FOV_ANGLE = np.pi/3

    # Model Parameters
    MODEL_PATH = "supervised_trust_model.pth"

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
            model_path=MODEL_PATH,
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
                scenario_summary = {
                    'scenario': scenario['name'],
                    'false_positive_rate': scenario['false_positive_rate'],
                    'false_negative_rate': scenario['false_negative_rate'],
                    'legitimate_paper': metrics['final_trust_values'].get('legitimate', {}).get('paper_mean', 0),
                    'legitimate_supervised': metrics['final_trust_values'].get('legitimate', {}).get('supervised_mean', 0),
                    'adversarial_paper': metrics['final_trust_values'].get('adversarial', {}).get('paper_mean', 0),
                    'adversarial_supervised': metrics['final_trust_values'].get('adversarial', {}).get('supervised_mean', 0),
                    'legitimate_improvement': metrics['final_trust_values'].get('legitimate', {}).get('difference', 0),
                    'adversarial_improvement': metrics['final_trust_values'].get('adversarial', {}).get('difference', 0)
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
        print("Scenario Name               | FP Rate | FN Rate | Leg (P→S)    | Adv (P→S)    | Leg Δ   | Adv Δ")
        print("-" * 85)

        for stat in summary_stats:
            print(f"{stat['scenario']:<26} | "
                  f"{stat['false_positive_rate']:<7} | "
                  f"{stat['false_negative_rate']:<7} | "
                  f"{stat['legitimate_paper']:.3f}→{stat['legitimate_supervised']:.3f} | "
                  f"{stat['adversarial_paper']:.3f}→{stat['adversarial_supervised']:.3f} | "
                  f"{stat['legitimate_improvement']:+.3f} | {stat['adversarial_improvement']:+.3f}")

        # Calculate overall averages
        avg_leg_improvement = np.mean([s['legitimate_improvement'] for s in summary_stats])
        avg_adv_improvement = np.mean([s['adversarial_improvement'] for s in summary_stats])

        print("-" * 85)
        print(f"{'AVERAGE':<26} | {'':>7} | {'':>7} | {'':>12} | {'':>12} | {avg_leg_improvement:+.3f} | {avg_adv_improvement:+.3f}")

        print(f"\n🔍 Key Findings:")
        print(f"   • Average legitimate robot trust improvement: {avg_leg_improvement:+.3f}")
        print(f"   • Average adversarial robot trust change: {avg_adv_improvement:+.3f}")

        # Analysis by scenario type
        print(f"\n📈 Scenario Analysis:")
        for stat in summary_stats:
            print(f"   • {stat['scenario']}:")
            print(f"     - FP/FN rates: {stat['false_positive_rate']:.1f}/{stat['false_negative_rate']:.1f}")
            print(f"     - Legitimate trust: {stat['legitimate_improvement']:+.3f}")
            print(f"     - Adversarial detection: {stat['adversarial_improvement']:+.3f}")

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
            'model_path': MODEL_PATH
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