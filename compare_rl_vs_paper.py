#!/usr/bin/env python3
"""
Simple RL-GNN vs Paper Algorithm Comparison

Both algorithms receive robot states and track observations from the SAME simulation.
No complicated environment wrapping - just one simulation feeding both methods.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

from train_gnn_rl import PPOTrustGNN, PPOTrainer, RLTrustEnvironment
from neural_symbolic_trust_algorithm import PPOTrustGNN as NeuralPPOTrustGNN, NeuralSymbolicTrustAlgorithm
from paper_trust_algorithm import PaperTrustAlgorithm
from simulation_environment import SimulationEnvironment


class SimpleTrustComparison:
    """Simple comparison using one simulation feeding both algorithms"""
    
    def __init__(self, model_path='ppo_trust_gnn.pth'):
        self.model_path = model_path
        
        # Load RL model with correct architecture (7 track features: 5 predicates + alpha + beta)
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        self.rl_model = PPOTrustGNN(agent_features=6, track_features=7, hidden_dim=64)
        self.rl_trainer = PPOTrainer(self.rl_model, device=torch.device('cpu'))
        self.rl_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ Loaded RL model: Episode {checkpoint['episode']}, Best reward: {checkpoint['best_reward']:.2f}")
        
        # Paper algorithm
        self.paper_algo = PaperTrustAlgorithm()
        
        # RL algorithm wrapper (not actually used, but kept for compatibility)
        self.rl_algo = NeuralSymbolicTrustAlgorithm(learning_mode=False)
        
        # Set model to evaluation mode
        self.rl_model.eval()
        
    def run_comparison(self, scenario_config):
        """Run both algorithms on the same simulation"""
        print(f"\n🔬 Testing scenario: {scenario_config['name']}")
        print(f"   Robots: {scenario_config['num_robots']}, Objects: {scenario_config['num_objects']}")
        print(f"   Adversarial ratio: {scenario_config['adversarial_ratio']:.2f}")
        
        # Create ONE simulation environment with scenario parameters
        world_size = scenario_config.get('world_size', (50.0, 50.0))
        proximal_range = scenario_config.get('proximal_range', 50.0)
        
        sim_env = SimulationEnvironment(
            num_robots=scenario_config['num_robots'],
            num_targets=scenario_config['num_objects'],
            world_size=world_size,
            adversarial_ratio=scenario_config['adversarial_ratio'],
            proximal_range=proximal_range
        )
        
        # Print ground truth
        adversarial_ids = [r.id for r in sim_env.robots if r.is_adversarial]
        legitimate_ids = [r.id for r in sim_env.robots if not r.is_adversarial]
        print(f"   Ground Truth: Adversarial {adversarial_ids}, Legitimate {legitimate_ids}")
        
        # Run simulation and collect data for both algorithms
        max_steps = scenario_config.get('max_steps', 100)
        
        # Storage for results
        rl_trust_evolution = []
        paper_trust_evolution = []
        
        # Initialize trust values for both algorithms
        rl_robot_trusts = {r.id: {'alpha': 1.0, 'beta': 1.0} for r in sim_env.robots}
        paper_robot_trusts = {r.id: {'alpha': 1.0, 'beta': 1.0} for r in sim_env.robots}
        
        # Paper algorithm is ready to use without initialization
        
        # Run simulation steps
        for step in range(max_steps):
            # Step the simulation
            step_data = sim_env.step()
            
            # Extract current robot states and track observations
            robot_states = {r.id: r for r in sim_env.robots}
            robot_tracks = {r.id: r.get_all_tracks() for r in sim_env.robots}  # Dict[robot_id, List[Track]]
            
            # === RL ALGORITHM ===
            # Convert simulation data to RL format and get RL decisions
            rl_state = self._convert_to_rl_state(robot_states, robot_tracks, rl_robot_trusts, sim_env)
            if rl_state is not None:
                try:
                    # Get RL actions (trust updates)
                    actions, _, _ = self.rl_trainer.select_action(rl_state, deterministic=True)
                    # Apply RL trust updates using the proper RL method
                    self._apply_rl_actions(actions, rl_robot_trusts, robot_states, rl_state)
                except Exception as e:
                    print(f"RL algorithm error: {e}")
                    # Fallback to small increment
                    for robot_id in robot_states.keys():
                        if robot_id in rl_robot_trusts:
                            rl_robot_trusts[robot_id]['alpha'] += 0.001
                            rl_robot_trusts[robot_id]['beta'] += 0.001
            
            # === PAPER ALGORITHM ===
            # Apply paper algorithm trust updates
            self._apply_paper_algorithm(paper_robot_trusts, robot_states, robot_tracks, sim_env)
            
            # Record trust evolution for both algorithms
            rl_trust_evolution.append({
                'step': step,
                'robot_trusts': self._extract_trust_values(rl_robot_trusts, robot_states, 'RL-GNN')
            })
            
            paper_trust_evolution.append({
                'step': step, 
                'robot_trusts': self._extract_trust_values(paper_robot_trusts, robot_states, 'Paper')
            })
        
        # Generate results
        results = {
            'config': scenario_config,
            'ground_truth': {
                'adversarial_ids': adversarial_ids,
                'legitimate_ids': legitimate_ids
            },
            'rl_gnn': {
                'trust_evolution': rl_trust_evolution,
                'final_trusts': rl_trust_evolution[-1]['robot_trusts'] if rl_trust_evolution else {}
            },
            'paper': {
                'trust_evolution': paper_trust_evolution,
                'final_trusts': paper_trust_evolution[-1]['robot_trusts'] if paper_trust_evolution else {}
            }
        }
        
        return results
    
    def _convert_to_rl_state(self, robot_states, robot_tracks, rl_robot_trusts, sim_env):
        """Convert simulation data to RL state format - use the same method as train_gnn_rl.py"""
        try:
            # Create a temporary RL environment to use its graph construction method
            temp_rl_env = RLTrustEnvironment(
                num_robots=len(robot_states),
                num_targets=10, # doesn't matter for state generation
                adversarial_ratio=0.3
            )
            
            # Replace its simulation with our shared environment
            temp_rl_env.sim_env = sim_env
            temp_rl_env.episode_ego_robot_id = list(robot_states.keys())[0]  # Use first robot as ego
            
            # Set robot trust values in simulation - CRITICAL: Update the actual simulation robots
            for robot_id, trust_data in rl_robot_trusts.items():
                robot = robot_states[robot_id]
                robot.trust_alpha = trust_data['alpha']
                robot.trust_beta = trust_data['beta']
                # Also update in sim_env.robots list
                for sim_robot in sim_env.robots:
                    if sim_robot.id == robot_id:
                        sim_robot.trust_alpha = trust_data['alpha']
                        sim_robot.trust_beta = trust_data['beta']
            
            # Update track trust values as well
            for robot_id, tracks in robot_tracks.items():
                for track in tracks:
                    if not hasattr(track, 'trust_alpha'):
                        track.trust_alpha = 2.0
                        track.trust_beta = 1.0
            
            # Generate proper RL state using the same method as train_gnn_rl.py
            rl_state = temp_rl_env._get_current_state()
            
            # CRITICAL: Attach current robots to state for action application
            if rl_state is not None:
                rl_state._current_robots = list(robot_states.values())
            
            return rl_state
            
        except Exception as e:
            print(f"RL state conversion error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _apply_rl_actions(self, actions, rl_robot_trusts, robot_states, rl_state):
        """Apply RL trust update actions using the exact same method as train_gnn_rl.py"""
        try:
            # Use the same logic as RLTrustEnvironment._apply_trust_updates method
            if not actions or not rl_state:
                return
                
            # Update robot trust distributions directly in simulation environment
            if 'agent' in actions and hasattr(rl_state, 'agent_nodes') and hasattr(rl_state, '_current_robots'):
                agent_actions = actions['agent']
                current_robots = rl_state._current_robots
                
                for robot_id, node_idx in rl_state.agent_nodes.items():
                    if node_idx < agent_actions['value'].shape[0]:
                        # Find robot in current robots list
                        robot = None
                        for r in current_robots:
                            if r.id == robot_id:
                                robot = r
                                break
                        
                        if robot is not None and hasattr(robot, 'trust_alpha'):
                            # Store original trust for logging
                            original_alpha = robot.trust_alpha
                            original_beta = robot.trust_beta
                            
                            # Convert GNN action to trust parameter updates
                            psm_value = agent_actions['value'][node_idx].item()
                            psm_confidence = agent_actions['confidence'][node_idx].item()
                            
                            # Update robot trust using PSM approach (exact same as train_gnn_rl.py)
                            delta_alpha = psm_confidence * psm_value * 0.05  # Gradual changes
                            delta_beta = psm_confidence * (1.0 - psm_value) * 0.05
                            
                            # Apply updates directly to robot
                            robot.trust_alpha += delta_alpha
                            robot.trust_beta += delta_beta
                            
                            # Update our tracking dict as well
                            rl_robot_trusts[robot_id]['alpha'] = robot.trust_alpha
                            rl_robot_trusts[robot_id]['beta'] = robot.trust_beta
                            
                            # Update robot_states dict
                            robot_states[robot_id].trust_alpha = robot.trust_alpha
                            robot_states[robot_id].trust_beta = robot.trust_beta
                            
                            # Debug logging (same format as train_gnn_rl.py)
                            print(f"  [TRUST UPDATE] Robot {robot_id}: α {original_alpha:.3f}→{robot.trust_alpha:.3f} (Δ{delta_alpha:+.3f}), β {original_beta:.3f}→{robot.trust_beta:.3f} (Δ{delta_beta:+.3f})")
            
            # Update track trust distributions with proper propagation (if needed)
            # This matches the track trust update logic from train_gnn_rl.py
            if 'track' in actions and hasattr(rl_state, 'track_nodes'):
                track_actions = actions['track']
                # Track trust updates could be implemented here if needed
                pass
            
        except Exception as e:
            print(f"RL action application error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            # Fallback to small increment
            for robot_id in robot_states.keys():
                if robot_id in rl_robot_trusts:
                    rl_robot_trusts[robot_id]['alpha'] += 0.001
                    rl_robot_trusts[robot_id]['beta'] += 0.001
    
    def _apply_paper_algorithm(self, paper_robot_trusts, robot_states, robot_tracks, sim_env):
        """Apply paper algorithm trust updates using the actual algorithm"""
        try:
            # Set robot trust values before calling paper algorithm
            for robot_id, trust_data in paper_robot_trusts.items():
                robot = robot_states[robot_id]
                robot.trust_alpha = trust_data['alpha']
                robot.trust_beta = trust_data['beta']
            
            # Set track trust values  
            for robot_id, tracks in robot_tracks.items():
                for track in tracks:
                    if not hasattr(track, 'trust_alpha'):
                        track.trust_alpha = 2.0
                        track.trust_beta = 1.0
            
            # Call the actual paper trust algorithm (simplified signature)
            trust_updates = self.paper_algo.update_trust(
                list(robot_states.values()), 
                environment=sim_env
            )
            
            # Extract updated trust values back to our tracking
            for robot_id, robot in robot_states.items():
                paper_robot_trusts[robot_id]['alpha'] = getattr(robot, 'trust_alpha', paper_robot_trusts[robot_id]['alpha'])
                paper_robot_trusts[robot_id]['beta'] = getattr(robot, 'trust_beta', paper_robot_trusts[robot_id]['beta'])
                
        except Exception as e:
            import traceback
            print(f"Paper algorithm error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Fallback to simple increment if paper algorithm fails
            for robot_id in robot_states.keys():
                if robot_id in paper_robot_trusts:
                    paper_robot_trusts[robot_id]['alpha'] += 0.005
                    paper_robot_trusts[robot_id]['beta'] += 0.005
    
    def _extract_trust_values(self, robot_trusts, robot_states, algorithm_name):
        """Extract trust values with ground truth labels"""
        extracted = {}
        for robot_id, trust_data in robot_trusts.items():
            robot = robot_states[robot_id]
            trust_mean = trust_data['alpha'] / (trust_data['alpha'] + trust_data['beta'])
            
            is_adversarial = robot.is_adversarial
            ground_truth_label = 'Adversarial' if is_adversarial else 'Legitimate'
            
            extracted[robot_id] = {
                'trust': trust_mean,
                'alpha': trust_data['alpha'],
                'beta': trust_data['beta'],
                'is_adversarial': is_adversarial,
                'ground_truth_label': ground_truth_label,
                'robot_type': f'Robot {robot_id} ({ground_truth_label})'
            }
        
        return extracted
    
    def _generate_diverse_scenarios(self):
        """Generate diverse test scenarios"""
        return [
            {
                'name': 'Dense Small Scale',
                'num_robots': 4,
                'num_objects': 12,
                'adversarial_ratio': 0.25,
                'max_steps': 80,
                'world_size': (40.0, 40.0),
                'proximal_range': 40.0
            },
            {
                'name': 'Sparse Medium Scale',
                'num_robots': 6,
                'num_objects': 18,
                'adversarial_ratio': 0.33,
                'max_steps': 100,
                'world_size': (60.0, 60.0), 
                'proximal_range': 45.0
            },
            {
                'name': 'High Adversarial',
                'num_robots': 5,
                'num_objects': 15,
                'adversarial_ratio': 0.6,
                'max_steps': 120,
                'world_size': (50.0, 50.0),
                'proximal_range': 35.0
            },
            {
                'name': 'Large Scale Low Adversarial',
                'num_robots': 8,
                'num_objects': 30,
                'adversarial_ratio': 0.125,
                'max_steps': 150,
                'world_size': (80.0, 80.0),
                'proximal_range': 60.0
            }
        ]
    
    def run_all_scenarios(self, use_diverse=False):
        """Run comparison across multiple scenarios"""
        if use_diverse:
            print("🌈 Using diverse scenario generation...")
            scenarios = self._generate_diverse_scenarios()
        else:
            scenarios = [
                {
                    'name': 'Simulation Environment Match',
                    'num_robots': 5,
                    'num_objects': 20,
                    'adversarial_ratio': 0.3,
                    'max_steps': 100
                },
                {
                    'name': 'Small Scale Dense',
                    'num_robots': 3, 
                    'num_objects': 15,
                    'adversarial_ratio': 0.4,
                    'max_steps': 50
                },
                {
                    'name': 'High Adversarial Dense',
                    'num_robots': 4,
                    'num_objects': 18,
                    'adversarial_ratio': 0.5,
                    'max_steps': 75
                },
                {
                    'name': 'Large Scale',
                    'num_robots': 8,
                    'num_objects': 25,
                    'adversarial_ratio': 0.25,
                    'max_steps': 80
                }
            ]
        
        all_results = []
        
        for scenario in scenarios:
            result = self.run_comparison(scenario)
            all_results.append(result)
        
        # Generate plots
        self._generate_plots(all_results)
        
        return all_results
    
    def _generate_plots(self, results):
        """Generate comparison plots"""
        print(f"\n📊 Generating comparison plots...")
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Create plots for each scenario
        for scenario_idx, scenario_result in enumerate(results):
            config = scenario_result['config']
            scenario_name = config['name'].replace(' ', '_')
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # RL-GNN plot
            self._plot_trust_evolution(ax1, scenario_result['rl_gnn']['trust_evolution'], 
                                     scenario_result['rl_gnn']['final_trusts'],
                                     f"RL-GNN Method - {config['name']}", colors)
            
            # Paper algorithm plot
            self._plot_trust_evolution(ax2, scenario_result['paper']['trust_evolution'],
                                     scenario_result['paper']['final_trusts'], 
                                     f"Paper Algorithm - {config['name']}", colors)
            
            # Add scenario info
            info_text = f"Robots: {config['num_robots']}, Objects: {config['num_objects']}, " + \
                       f"Adversarial: {config['adversarial_ratio']:.0%}, Steps: {config['max_steps']}"
            fig.suptitle(f"Trust Evolution Comparison - {config['name']}\n{info_text}", 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            filename = f'trust_evolution_scenario_{scenario_idx+1}_{scenario_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved {filename}")
            plt.close()
    
    def _plot_trust_evolution(self, ax, evolution_data, final_trusts, title, colors):
        """Plot trust evolution for a single algorithm"""
        if not evolution_data:
            ax.text(0.5, 0.5, 'No evolution data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        steps = [data['step'] for data in evolution_data]
        robot_ids = list(final_trusts.keys()) if final_trusts else []
        
        # Plot each robot's trust evolution
        for i, robot_id in enumerate(robot_ids):
            robot_trusts = []
            for data in evolution_data:
                if robot_id in data['robot_trusts']:
                    robot_trusts.append(data['robot_trusts'][robot_id]['trust'])
                else:
                    robot_trusts.append(0.5)
            
            # Get robot label with adversarial status
            if final_trusts and robot_id in final_trusts:
                robot_data = final_trusts[robot_id]
                is_adversarial = robot_data['is_adversarial']
                ground_truth_label = robot_data['ground_truth_label']
                label = f'Robot {robot_id} ({ground_truth_label})'
                
                # Visual styling based on adversarial status
                if is_adversarial:
                    linestyle = '--'
                    linewidth = 3
                    marker = 'x'
                    markersize = 8
                else:
                    linestyle = '-'
                    linewidth = 2
                    marker = 'o'
                    markersize = 6
            else:
                label = f'Robot {robot_id}'
                linestyle = ':'
                linewidth = 2
                marker = 's'
                markersize = 5
            
            color = colors[i % len(colors)]
            ax.plot(steps, robot_trusts, color=color, linestyle=linestyle,
                   linewidth=linewidth, marker=marker, markersize=markersize,
                   label=label, alpha=0.8)
        
        ax.set_xlabel('Simulation Steps')
        ax.set_ylabel('Trust Value')
        ax.set_title(title, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add reference lines
        ax.axhline(y=0.45, color='red', linestyle=':', alpha=0.7, linewidth=2)
        ax.axhline(y=0.55, color='green', linestyle=':', alpha=0.7, linewidth=2)
        
        # Add shaded regions  
        ax.axhspan(0.0, 0.45, alpha=0.1, color='red')
        ax.axhspan(0.55, 1.0, alpha=0.1, color='green')
        ax.axhspan(0.45, 0.55, alpha=0.1, color='yellow')
        
        # Add zone labels
        ax.text(0.02, 0.22, 'Adversarial\nZone', transform=ax.transAxes, 
                color='darkred', fontsize=9, fontweight='bold')
        ax.text(0.02, 0.77, 'Legitimate\nZone', transform=ax.transAxes, 
                color='darkgreen', fontsize=9, fontweight='bold')
        ax.text(0.02, 0.50, 'Uncertain\nZone', transform=ax.transAxes, 
                color='darkorange', fontsize=8, fontweight='bold')


def main():
    """Run the simple comparison"""
    import sys
    
    print("🔬 SIMPLE RL-GNN vs PAPER ALGORITHM COMPARISON")
    print("=" * 50)
    
    # Check for diverse scenarios flag
    use_diverse = '--diverse' in sys.argv or '--diverse-scenarios' in sys.argv
    if use_diverse:
        print("🌈 Using diverse scenario generation for comparison")
    
    comparator = SimpleTrustComparison()
    results = comparator.run_all_scenarios(use_diverse=use_diverse)
    
    print(f"\n🎉 Comparison completed!")
    print(f"   Generated {len(results)} scenario comparison plots")
    
    if use_diverse:
        print("   Used diverse scenarios with systematic parameter variations")


if __name__ == "__main__":
    main()