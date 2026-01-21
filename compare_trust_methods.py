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
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import required modules
from simulation_environment import SimulationEnvironment
from paper_trust_algorithm import PaperTrustAlgorithm
from supervised_trust_algorithm import SupervisedTrustAlgorithm
from bayesian_ego_graph_trust import BayesianEgoGraphTrust
from robot_track_classes import Robot, Track


class TrustMethodComparison:
    """Runs paper trust algorithm simulation"""

    def __init__(self,
                 supervised_model_path: str = "supervised_trust_model.pth",
                 robot_density: float = 0.0005,
                 target_density_multiplier: Optional[float] = 2.0,
                 target_density: Optional[float] = None,
                 num_timesteps: int = 500,
                 random_seed: int = 42,
                 world_size: float = 100.0,
                 fov_range: float = 50.0,
                 fov_angle: float = np.pi/3,
                 fixed_step_scale: float = 0.5,
                 allow_fp_codetection: bool = True,
                 ):
        """
        Initialize comparison with all three trust methods

        Args:
            supervised_model_path: Path to trained supervised GNN trust model
            robot_density: Robots per unit area in the fixed world
            target_density_multiplier: Multiplier applied to robot density to derive target density
            target_density: Optional explicit target density (overrides multiplier if provided)
            num_timesteps: Number of simulation steps
            random_seed: Random seed for reproducibility
            world_size: Side length of the (square) simulation world
            fov_range: Field of view range for robots
            fov_angle: Field of view angle for robots
            fixed_step_scale: Step scale for baseline fixed-step method (default: 0.5)
            allow_fp_codetection: Whether to allow FP codetection (default: True)
        """
        self.supervised_model_path = Path(supervised_model_path) if supervised_model_path else None
        self.world_size = float(world_size)
        self.world_area = self.world_size * self.world_size
        self.robot_density = float(robot_density)

        resolved_multiplier = target_density_multiplier
        if target_density is not None and resolved_multiplier is None:
            if self.robot_density <= 0:
                raise ValueError("Robot density must be positive to derive target density multiplier.")
            resolved_multiplier = float(target_density) / self.robot_density

        if resolved_multiplier is None:
            raise ValueError("Either target_density_multiplier or target_density must be provided.")

        self.target_density_multiplier = float(resolved_multiplier)
        self.target_density = round(self.robot_density * self.target_density_multiplier, 8)
        self.num_robots = max(1, int(round(self.robot_density * self.world_area)))
        self.num_targets = max(1, int(round(self.target_density * self.world_area)))
        self.num_timesteps = num_timesteps
        self.random_seed = random_seed
        self.fov_range = fov_range
        self.fov_angle = fov_angle
        self.fixed_step_scale = max(0.0, min(1.0, fixed_step_scale))

        # Initialize trust methods
        self.paper_algorithm = PaperTrustAlgorithm()
        self.supervised_algorithm = None
        self.bayesian_algorithm = None

        # Results storage
        self.paper_results = []
        self.supervised_results = []
        self.bayesian_results = []
        self.baseline_results = []

        # Simulation parameters (can be overridden)
        self.adversarial_ratio = 0.3
        self.false_positive_rate = 0.5
        self.false_negative_rate = 0.0
        self.proximal_range = 50.0
        self.allow_fp_codetection = allow_fp_codetection  # Can be set to True for FP codetection experiments

        # Try to load supervised model after proximal_range is set
        try:
            gnn_path_str = str(self.supervised_model_path) if self.supervised_model_path else None
            self.supervised_algorithm = SupervisedTrustAlgorithm(
                model_path=gnn_path_str,
                proximal_range=self.proximal_range
            )
            if self.supervised_model_path and self.supervised_model_path.exists():
                print(f"✅ Supervised trust algorithm ready ({gnn_path_str})")
            else:
                print("ℹ️ Supervised trust algorithm initialized with fresh weights")
        except Exception as e:
            print(f"⚠️ Failed to initialize supervised algorithm: {e}")

        # Initialize Bayesian ego graph trust algorithm
        try:
            self.bayesian_algorithm = BayesianEgoGraphTrust(
                proximity_radius=self.proximal_range
            )
            print("✅ Bayesian ego graph trust algorithm ready")
        except Exception as e:
            print(f"⚠️ Failed to initialize Bayesian algorithm: {e}")

    def create_identical_environments(self, count: int = 4) -> List[SimulationEnvironment]:
        """Create identical simulation environments."""
        envs = []
        for _ in range(count):
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            env = SimulationEnvironment(
                world_size=(self.world_size, self.world_size),
                robot_density=self.robot_density,
                target_density=self.target_density,
                adversarial_ratio=self.adversarial_ratio,
                proximal_range=self.proximal_range,
                fov_range=self.fov_range,
                fov_angle=self.fov_angle,
                false_positive_rate=self.false_positive_rate,
                false_negative_rate=self.false_negative_rate,
                allow_fp_codetection=self.allow_fp_codetection
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

            # Add frame state for visualization
            def classify_track_object(object_id: str) -> str:
                if object_id.startswith("gt_"):
                    return "ground_truth"
                if object_id.startswith("fp_"):
                    return "false_positive"
                return "unknown"

            robot_states = []
            robot_detections = {}
            for robot in env.robots:
                robot_states.append({
                    'id': robot.id,
                    'position': robot.position[:2].tolist() if hasattr(robot.position, '__iter__') else [float(robot.position)],
                    'orientation': float(getattr(robot, 'orientation', 0.0)),
                    'trust_value': float(robot.trust_value),
                    'is_adversarial': bool(robot.is_adversarial),
                    'fov_range': float(robot.fov_range),
                    'fov_angle': float(robot.fov_angle),
                })

                detections = []
                for track in robot.get_current_timestep_tracks():
                    detections.append({
                        'track_id': track.track_id,
                        'object_id': track.object_id,
                        'position': track.position[:2].tolist() if hasattr(track.position, '__iter__') else [float(track.position)],
                        'type': classify_track_object(track.object_id),
                        'trust_value': float(track.trust_value),
                    })
                robot_detections[str(robot.id)] = detections

            step_result['frame_state'] = {
                'world_size': list(env.world_size),
                'robots': robot_states,
                'detections': robot_detections,
            }

            results.append(step_result)

            if step % 100 == 0:
                print(f"  Paper algorithm step {step}/{self.num_timesteps}")

        print("✅ Paper algorithm simulation completed")
        return results

    def run_supervised_model_simulation(self, env: SimulationEnvironment) -> List[Dict]:
        """Run simulation using supervised trust algorithm"""
        if self.supervised_algorithm is None:
            print("❌ Supervised algorithm not available, skipping")
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

            # Apply supervised trust algorithm
            # This internally loops over all robots as ego and updates their trust
            try:
                trust_updates = self.supervised_algorithm.update_trust(env.robots, env)
            except Exception as e:
                print(f"⚠️ Supervised algorithm error at step {step}: {e}")
                trust_updates = {}

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

            # Add frame state for visualization
            def classify_track_object(object_id: str) -> str:
                if object_id.startswith("gt_"):
                    return "ground_truth"
                if object_id.startswith("fp_"):
                    return "false_positive"
                return "unknown"

            robot_states = []
            robot_detections = {}
            for robot in env.robots:
                robot_states.append({
                    'id': robot.id,
                    'position': robot.position[:2].tolist() if hasattr(robot.position, '__iter__') else [float(robot.position)],
                    'orientation': float(getattr(robot, 'orientation', 0.0)),
                    'trust_value': float(robot.trust_value),
                    'is_adversarial': bool(robot.is_adversarial),
                    'fov_range': float(robot.fov_range),
                    'fov_angle': float(robot.fov_angle),
                })

                detections = []
                for track in robot.get_current_timestep_tracks():
                    detections.append({
                        'track_id': track.track_id,
                        'object_id': track.object_id,
                        'position': track.position[:2].tolist() if hasattr(track.position, '__iter__') else [float(track.position)],
                        'type': classify_track_object(track.object_id),
                        'trust_value': float(track.trust_value),
                    })
                robot_detections[str(robot.id)] = detections

            step_result['frame_state'] = {
                'world_size': list(env.world_size),
                'robots': robot_states,
                'detections': robot_detections,
            }

            results.append(step_result)

            if step % 100 == 0:
                print(f"  Supervised model step {step}/{self.num_timesteps}")

        print("✅ Supervised model simulation completed")
        return results

    def run_bayesian_simulation(self, env: SimulationEnvironment) -> List[Dict]:
        """Run simulation using Bayesian ego graph trust algorithm"""
        if self.bayesian_algorithm is None:
            print("❌ Bayesian algorithm not available, skipping")
            return []

        print("🚀 Running Bayesian ego graph simulation...")

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

            # Apply Bayesian trust algorithm
            try:
                trust_updates = self.bayesian_algorithm.update_trust(env.robots, env)
            except Exception as e:
                print(f"⚠️ Bayesian algorithm error at step {step}: {e}")
                trust_updates = {}

            # Collect results (same structure as other algorithms)
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

            # Add frame state for visualization
            def classify_track_object(object_id: str) -> str:
                if object_id.startswith("gt_"):
                    return "ground_truth"
                if object_id.startswith("fp_"):
                    return "false_positive"
                return "unknown"

            robot_states = []
            robot_detections = {}
            for robot in env.robots:
                robot_states.append({
                    'id': robot.id,
                    'position': robot.position[:2].tolist() if hasattr(robot.position, '__iter__') else [float(robot.position)],
                    'orientation': float(getattr(robot, 'orientation', 0.0)),
                    'trust_value': float(robot.trust_value),
                    'is_adversarial': bool(robot.is_adversarial),
                    'fov_range': float(robot.fov_range),
                    'fov_angle': float(robot.fov_angle),
                })

                detections = []
                for track in robot.get_current_timestep_tracks():
                    detections.append({
                        'track_id': track.track_id,
                        'object_id': track.object_id,
                        'position': track.position[:2].tolist() if hasattr(track.position, '__iter__') else [float(track.position)],
                        'type': classify_track_object(track.object_id),
                        'trust_value': float(track.trust_value),
                    })
                robot_detections[str(robot.id)] = detections

            step_result['frame_state'] = {
                'world_size': list(env.world_size),
                'robots': robot_states,
                'detections': robot_detections,
            }

            results.append(step_result)

            if step % 100 == 0:
                print(f"  Bayesian model step {step}/{self.num_timesteps}")

        print("✅ Bayesian model simulation completed")
        return results

    def run_baseline_simulation(self, env: SimulationEnvironment) -> List[Dict]:
        """Run simulation with no trust updates (baseline that believes all robots equally)"""
        print("🚀 Running baseline (no trust) simulation...")

        results = []

        for step in range(self.num_timesteps):
            # CRITICAL: Reset random seed for each step to ensure identical randomness
            step_seed = self.random_seed + step
            np.random.seed(step_seed)
            random.seed(step_seed)

            # Step the simulation WITHOUT any trust updates
            frame_data = env.step()

            # Update current timestep tracks for all robots
            for robot in env.robots:
                robot.update_current_timestep_tracks()

            # NO TRUST UPDATES - robots and tracks keep initial trust (0.5)

            # Collect results (same structure as other algorithms)
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

            # Store per-frame robot state and detections
            robot_states = []
            robot_detections = {}

            for robot in env.robots:
                robot_states.append({
                    'id': robot.id,
                    'position': robot.position[:2].tolist() if hasattr(robot.position, '__iter__') else [float(robot.position)],
                    'orientation': float(getattr(robot, 'orientation', 0.0)),
                    'trust_value': float(robot.trust_value),
                    'is_adversarial': bool(robot.is_adversarial),
                    'fov_range': float(robot.fov_range),
                    'fov_angle': float(robot.fov_angle),
                    'velocity': robot.velocity[:2].tolist() if hasattr(robot.velocity, '__iter__') else [float(robot.velocity)],
                })

                detections = []
                for track in robot.get_current_timestep_tracks():
                    object_type = "ground_truth" if track.object_id.startswith("gt_") else "false_positive" if track.object_id.startswith("fp_") else "unknown"
                    detections.append({
                        'track_id': track.track_id,
                        'object_id': track.object_id,
                        'position': track.position[:2].tolist() if hasattr(track.position, '__iter__') else [float(track.position)],
                        'type': object_type,
                        'trust_value': float(track.trust_value),
                    })
                robot_detections[str(robot.id)] = detections

            step_result['frame_state'] = {
                'world_size': list(env.world_size),
                'robots': robot_states,
                'detections': robot_detections,
            }

            results.append(step_result)

            if step % 100 == 0:
                print(f"  Baseline step {step}/{self.num_timesteps}")

        print("✅ Baseline simulation completed")
        return results

    def _run_trust_simulation(self, env: SimulationEnvironment, trust_system, label: str) -> List[Dict]:
        """Shared simulation loop for trust systems (baseline or mass-based)."""

        results = []

        def classify_track_object(object_id: str) -> str:
            if object_id.startswith("gt_"):
                return "ground_truth"
            if object_id.startswith("fp_"):
                return "false_positive"
            return "unknown"

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

            # Store per-frame robot state and detections for downstream visualisations
            robot_states = []
            robot_detections = {}
            ego_graph_snapshots = {}

            for robot in env.robots:
                robot_states.append({
                    'id': robot.id,
                    'position': robot.position[:2].tolist() if hasattr(robot.position, '__iter__') else [float(robot.position)],
                    'orientation': float(getattr(robot, 'orientation', 0.0)),
                    'trust_value': float(robot.trust_value),
                    'is_adversarial': bool(robot.is_adversarial),
                    'fov_range': float(robot.fov_range),
                    'fov_angle': float(robot.fov_angle),
                    'velocity': robot.velocity[:2].tolist() if hasattr(robot.velocity, '__iter__') else [float(robot.velocity)],
                })

                detections = []
                for track in robot.get_current_timestep_tracks():
                    detections.append({
                        'track_id': track.track_id,
                        'object_id': track.object_id,
                        'position': track.position[:2].tolist() if hasattr(track.position, '__iter__') else [float(track.position)],
                        'type': classify_track_object(track.object_id),
                        'trust_value': float(track.trust_value),
                    })
                robot_detections[str(robot.id)] = detections

                # Capture ego-graph snapshot if GNN evidence is available
                if trust_system.evidence_extractor.available:
                    try:
                        ego_result = trust_system.evidence_extractor.predict_from_robots_tracks(robot, env.robots)

                        # Check if ego_result is None (no cross-validation or no meaningful tracks)
                        if ego_result is None:
                            # Skip this ego-graph - no cross-validation constraints met
                            continue

                        predictions = ego_result['predictions']
                        graph_data = ego_result['graph_data']
                        meaningful_track_indices = ego_result.get('meaningful_track_indices', [])

                        agent_scores_array = predictions.get('agent', {}).get('trust_scores', [])
                        track_scores_array = predictions.get('track', {}).get('trust_scores', [])
                        agent_nodes = getattr(graph_data, 'agent_nodes', {})
                        track_nodes = getattr(graph_data, 'track_nodes', {})

                        # Only include ego robot (index 0) in agent scores
                        agent_scores = {}
                        for agent_id, idx in agent_nodes.items():
                            if idx == 0:  # Only ego robot
                                if idx < len(agent_scores_array):
                                    agent_scores[str(agent_id)] = float(np.clip(np.asarray(agent_scores_array[idx]).item(), 0.0, 1.0))
                                else:
                                    agent_scores[str(agent_id)] = 0.5

                        # Only include meaningful tracks in track scores
                        track_scores = {}
                        for track_id, idx in track_nodes.items():
                            if idx in meaningful_track_indices:  # Only meaningful tracks
                                if idx < len(track_scores_array):
                                    track_scores[track_id] = float(np.clip(np.asarray(track_scores_array[idx]).item(), 0.0, 1.0))
                                else:
                                    track_scores[track_id] = 0.5

                        edge_index = {}
                        for edge_type, edge_tensor in graph_data.edge_index_dict.items():
                            key = "|".join(edge_type)
                            edge_index[key] = edge_tensor.detach().cpu().numpy().tolist()

                        ego_graph_snapshots[str(robot.id)] = {
                            'agent_nodes': {str(rid): idx for rid, idx in agent_nodes.items()},
                            'track_nodes': track_nodes,
                            'agent_scores': agent_scores,
                            'track_scores': track_scores,
                            'edges': edge_index,
                            'meaningful_track_indices': meaningful_track_indices,
                        }
                    except Exception as graph_err:
                        print(f"⚠️ Failed to capture ego graph for robot {robot.id} at step {step}: {graph_err}")

            step_result['frame_state'] = {
                'world_size': list(env.world_size),
                'robots': robot_states,
                'detections': robot_detections,
            }
            if ego_graph_snapshots:
                step_result['ego_graphs'] = ego_graph_snapshots

            results.append(step_result)

            if step % 100 == 0:
                print(f"  {label} model step {step}/{self.num_timesteps}")

        print(f"✅ {label} simulation completed")
        return results

    def run_comparison(self) -> Dict:
        """Run complete comparison between paper algorithm, supervised model, Bayesian method, and baseline"""
        print("🔄 Starting trust method comparison...")
        print(
            f"Configuration: world={self.world_size}m, robots={self.num_robots} "
            f"(density {self.robot_density:.6f}), targets={self.num_targets} "
            f"(multiplier {self.target_density_multiplier:.3f}, density {self.target_density:.6f}), "
            f"steps={self.num_timesteps}"
        )
        print(f"Random seed: {self.random_seed}")

        # Create identical environments for all four methods
        paper_env, supervised_env, bayesian_env, baseline_env = self.create_identical_environments(4)

        # Run all four simulations
        print("\n" + "="*50)
        self.paper_results = self.run_paper_algorithm_simulation(paper_env)

        print("\n" + "="*50)
        self.supervised_results = self.run_supervised_model_simulation(supervised_env)

        print("\n" + "="*50)
        self.bayesian_results = self.run_bayesian_simulation(bayesian_env)

        print("\n" + "="*50)
        self.baseline_results = self.run_baseline_simulation(baseline_env)

        # Generate comparison results
        comparison_results = {
            'configuration': {
                'world_size': self.world_size,
                'robot_density': self.robot_density,
                'target_density': self.target_density,
                'target_density_multiplier': self.target_density_multiplier,
                'derived_num_robots': self.num_robots,
                'derived_num_targets': self.num_targets,
                'num_timesteps': self.num_timesteps,
                'random_seed': self.random_seed
            },
            'paper_results': self.paper_results,
            'supervised_results': self.supervised_results,
            'bayesian_results': self.bayesian_results,
            'baseline_results': self.baseline_results,
            'comparison_metrics': self._compute_comparison_metrics()
        }

        return comparison_results

    def _compute_comparison_metrics(self) -> Dict:
        """Compute comparison metrics between paper algorithm, supervised model, and Bayesian method"""
        if not self.paper_results or not self.supervised_results or not self.bayesian_results:
            print("⚠️ Incomplete results for comparison metrics")
            return {}

        print("📊 Computing comparison metrics...")

        metrics = {
            'trust_convergence': {},
            'final_trust_values': {},
            'method_differences': {}
        }

        method_results = {
            'paper': self.paper_results,
            'supervised': self.supervised_results,
            'bayesian': self.bayesian_results
        }

        robot_ids = list(self.paper_results[0]['robot_trust_values'].keys())
        adversarial_ids = set(self.paper_results[0]['adversarial_robots'])
        legitimate_ids = set(self.paper_results[0]['legitimate_robots'])

        # Compute per-robot metrics for each method
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

            metrics['trust_convergence'][robot_id] = {
                'robot_type': robot_type,
                **method_stats
            }

        # Aggregate metrics by robot type and method
        for robot_type, id_set in [('legitimate', legitimate_ids), ('adversarial', adversarial_ids)]:
            if not id_set:
                continue

            metrics['final_trust_values'][robot_type] = {}

            for method in ['paper', 'supervised', 'bayesian']:
                finals = [metrics['trust_convergence'][rid][method]['final'] for rid in id_set]
                metrics['final_trust_values'][robot_type][method] = {
                    'mean': float(np.mean(finals)),
                    'std': float(np.std(finals))
                }

        # Compute method differences
        all_paper_finals = []
        all_supervised_finals = []
        all_bayesian_finals = []
        for robot_id in robot_ids:
            all_paper_finals.append(metrics['trust_convergence'][robot_id]['paper']['final'])
            all_supervised_finals.append(metrics['trust_convergence'][robot_id]['supervised']['final'])
            all_bayesian_finals.append(metrics['trust_convergence'][robot_id]['bayesian']['final'])

        diff_supervised_paper = np.array(all_supervised_finals) - np.array(all_paper_finals)
        diff_bayesian_paper = np.array(all_bayesian_finals) - np.array(all_paper_finals)
        diff_bayesian_supervised = np.array(all_bayesian_finals) - np.array(all_supervised_finals)

        metrics['method_differences']['supervised_minus_paper'] = {
            'mean': float(np.mean(diff_supervised_paper)),
            'std': float(np.std(diff_supervised_paper))
        }
        metrics['method_differences']['bayesian_minus_paper'] = {
            'mean': float(np.mean(diff_bayesian_paper)),
            'std': float(np.std(diff_bayesian_paper))
        }
        metrics['method_differences']['bayesian_minus_supervised'] = {
            'mean': float(np.mean(diff_bayesian_supervised)),
            'std': float(np.std(diff_bayesian_supervised))
        }

        print("✅ Comparison metrics computed")
        return metrics

    def save_results(self, filename: str = "trust_comparison_results.json"):
        """Save comparison results to file"""
        results = {
            'configuration': {
                'robot_density': self.robot_density,
                'target_density_multiplier': self.target_density_multiplier,
                'target_density': self.target_density,
                'num_robots': self.num_robots,
                'num_targets': self.num_targets,
                'num_timesteps': self.num_timesteps,
                'random_seed': self.random_seed
            },
            'paper_results': self.paper_results,
            'supervised_results': self.supervised_results,
            'bayesian_results': self.bayesian_results,
            'metrics': self._compute_comparison_metrics()
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

    def visualize_results(self, save_path: str = "trust_comparison_results.png"):
        """Create visualization comparing paper algorithm, supervised model, and Bayesian method"""
        if not self.paper_results or not self.supervised_results or not self.bayesian_results:
            print("⚠️ Incomplete results for visualization")
            return

        print("📈 Creating comparison visualization...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Trust Method Comparison: Paper vs Supervised vs Bayesian', fontsize=16, fontweight='bold')

        timesteps = [step['step'] for step in self.paper_results]
        adversarial_robots = self.paper_results[0]['adversarial_robots']
        legitimate_robots = self.paper_results[0]['legitimate_robots']

        # Define colors for each method
        method_colors = {
            'paper': '#1f77b4',      # Blue
            'supervised': '#ff7f0e', # Orange
            'bayesian': '#2ca02c'    # Green
        }

        # Plot 1: Legitimate robots comparison (show first 2 robots for clarity)
        ax1 = axes[0, 0]
        for robot_id in legitimate_robots[:2]:  # Limit to 2 robots for clarity with 3 methods
            paper_trust = [step['robot_trust_values'][robot_id] for step in self.paper_results]
            supervised_trust = [step['robot_trust_values'][robot_id] for step in self.supervised_results]
            bayesian_trust = [step['robot_trust_values'][robot_id] for step in self.bayesian_results]

            ax1.plot(timesteps, paper_trust, label=f'Paper R{robot_id}',
                    color=method_colors['paper'], linestyle='-', alpha=0.7, linewidth=1.5)
            ax1.plot(timesteps, supervised_trust, label=f'Supervised R{robot_id}',
                    color=method_colors['supervised'], linestyle='--', alpha=0.7, linewidth=1.5)
            ax1.plot(timesteps, bayesian_trust, label=f'Bayesian R{robot_id}',
                    color=method_colors['bayesian'], linestyle=':', alpha=0.7, linewidth=1.5)
        ax1.set_title('Legitimate Robots Trust Evolution')
        ax1.set_xlabel('Simulation Step')
        ax1.set_ylabel('Trust Value')
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Adversarial robots comparison
        ax2 = axes[0, 1]
        if adversarial_robots:
            for robot_id in adversarial_robots[:2]:  # Limit to 2 robots for clarity with 3 methods
                paper_trust = [step['robot_trust_values'][robot_id] for step in self.paper_results]
                supervised_trust = [step['robot_trust_values'][robot_id] for step in self.supervised_results]
                bayesian_trust = [step['robot_trust_values'][robot_id] for step in self.bayesian_results]

                ax2.plot(timesteps, paper_trust, label=f'Paper R{robot_id}',
                        color=method_colors['paper'], linestyle='-', alpha=0.7, linewidth=1.5)
                ax2.plot(timesteps, supervised_trust, label=f'Supervised R{robot_id}',
                        color=method_colors['supervised'], linestyle='--', alpha=0.7, linewidth=1.5)
                ax2.plot(timesteps, bayesian_trust, label=f'Bayesian R{robot_id}',
                        color=method_colors['bayesian'], linestyle=':', alpha=0.7, linewidth=1.5)
            ax2.legend(fontsize=7, ncol=2)
        else:
            ax2.text(0.5, 0.5, 'No Adversarial Robots\nin this Scenario',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes, fontsize=12, style='italic')
        ax2.set_title('Adversarial Robots Trust Evolution')
        ax2.set_xlabel('Simulation Step')
        ax2.set_ylabel('Trust Value')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Average trust by robot type and method
        ax3 = axes[1, 0]
        paper_leg_avg = np.mean([[step['robot_trust_values'][rid] for rid in legitimate_robots]
                                 for step in self.paper_results], axis=1)
        supervised_leg_avg = np.mean([[step['robot_trust_values'][rid] for rid in legitimate_robots]
                                      for step in self.supervised_results], axis=1)
        bayesian_leg_avg = np.mean([[step['robot_trust_values'][rid] for rid in legitimate_robots]
                                    for step in self.bayesian_results], axis=1)

        ax3.plot(timesteps, paper_leg_avg, label='Paper - Legitimate',
                color=method_colors['paper'], linestyle='-', linewidth=2.5)
        ax3.plot(timesteps, supervised_leg_avg, label='Supervised - Legitimate',
                color=method_colors['supervised'], linestyle='-', linewidth=2.5)
        ax3.plot(timesteps, bayesian_leg_avg, label='Bayesian - Legitimate',
                color=method_colors['bayesian'], linestyle='-', linewidth=2.5)

        if adversarial_robots:
            paper_adv_avg = np.mean([[step['robot_trust_values'][rid] for rid in adversarial_robots]
                                     for step in self.paper_results], axis=1)
            supervised_adv_avg = np.mean([[step['robot_trust_values'][rid] for rid in adversarial_robots]
                                          for step in self.supervised_results], axis=1)
            bayesian_adv_avg = np.mean([[step['robot_trust_values'][rid] for rid in adversarial_robots]
                                        for step in self.bayesian_results], axis=1)

            ax3.plot(timesteps, paper_adv_avg, label='Paper - Adversarial',
                    color=method_colors['paper'], linestyle='--', linewidth=2.5, alpha=0.7)
            ax3.plot(timesteps, supervised_adv_avg, label='Supervised - Adversarial',
                    color=method_colors['supervised'], linestyle='--', linewidth=2.5, alpha=0.7)
            ax3.plot(timesteps, bayesian_adv_avg, label='Bayesian - Adversarial',
                    color=method_colors['bayesian'], linestyle='--', linewidth=2.5, alpha=0.7)

        ax3.set_title('Average Trust by Robot Type and Method')
        ax3.set_xlabel('Simulation Step')
        ax3.set_ylabel('Average Trust Value')
        ax3.legend(fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Final trust comparison
        ax4 = axes[1, 1]
        final_paper_leg = [self.paper_results[-1]['robot_trust_values'][rid] for rid in legitimate_robots]
        final_supervised_leg = [self.supervised_results[-1]['robot_trust_values'][rid] for rid in legitimate_robots]
        final_bayesian_leg = [self.bayesian_results[-1]['robot_trust_values'][rid] for rid in legitimate_robots]
        final_paper_adv = [self.paper_results[-1]['robot_trust_values'][rid] for rid in adversarial_robots] if adversarial_robots else []
        final_supervised_adv = [self.supervised_results[-1]['robot_trust_values'][rid] for rid in adversarial_robots] if adversarial_robots else []
        final_bayesian_adv = [self.bayesian_results[-1]['robot_trust_values'][rid] for rid in adversarial_robots] if adversarial_robots else []

        categories = []
        means = []
        stds = []
        colors = []

        # Legitimate robots
        categories.extend(['Paper\nLegitimate', 'Supervised\nLegitimate', 'Bayesian\nLegitimate'])
        means.extend([np.mean(final_paper_leg), np.mean(final_supervised_leg), np.mean(final_bayesian_leg)])
        stds.extend([np.std(final_paper_leg), np.std(final_supervised_leg), np.std(final_bayesian_leg)])
        colors.extend([method_colors['paper'], method_colors['supervised'], method_colors['bayesian']])

        # Adversarial robots
        if adversarial_robots:
            categories.extend(['Paper\nAdversarial', 'Supervised\nAdversarial', 'Bayesian\nAdversarial'])
            means.extend([np.mean(final_paper_adv), np.mean(final_supervised_adv), np.mean(final_bayesian_adv)])
            stds.extend([np.std(final_paper_adv), np.std(final_supervised_adv), np.std(final_bayesian_adv)])
            # Use darker shades for adversarial
            colors.extend(['#0d47a1', '#e65100', '#1b5e20'])

        x_pos = np.arange(len(categories))
        bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax4.set_title('Final Trust Values Comparison')
        ax4.set_ylabel('Final Trust Value')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories, fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Visualization saved to {save_path}")

    def print_summary(self):
        """Print comparison summary"""
        if not self.paper_results or not self.supervised_results or not self.bayesian_results:
            print("⚠️ Incomplete results for summary")
            return

        print("\n" + "="*70)
        print("🎯 TRUST METHOD COMPARISON SUMMARY")
        print("="*70)

        metrics = self._compute_comparison_metrics()

        print(
            f"Configuration: world={self.world_size}m, robots≈{self.num_robots} "
            f"(density {self.robot_density:.6f}), targets≈{self.num_targets} "
            f"(multiplier {self.target_density_multiplier:.3f}, density {self.target_density:.6f}), "
            f"steps={self.num_timesteps}, seed {self.random_seed}"
        )

        # Final trust values by robot type and method
        if 'final_trust_values' in metrics:
            print("\n📊 Final Trust Values Comparison:")
            for robot_type in ['legitimate', 'adversarial']:
                if robot_type in metrics['final_trust_values']:
                    print(f"\n  {robot_type.title()} Robots:")
                    for method in ['paper', 'supervised', 'bayesian']:
                        if method in metrics['final_trust_values'][robot_type]:
                            stats = metrics['final_trust_values'][robot_type][method]
                            print(f"    {method.title()}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        # Method differences
        if 'method_differences' in metrics:
            diff_sp = metrics['method_differences']['supervised_minus_paper']
            diff_bp = metrics['method_differences']['bayesian_minus_paper']
            diff_bs = metrics['method_differences']['bayesian_minus_supervised']
            print(f"\n📈 Method Differences:")
            print(f"    Supervised vs Paper: {diff_sp['mean']:+.3f} ± {diff_sp['std']:.3f}")
            print(f"    Bayesian vs Paper: {diff_bp['mean']:+.3f} ± {diff_bp['std']:.3f}")
            print(f"    Bayesian vs Supervised: {diff_bs['mean']:+.3f} ± {diff_bs['std']:.3f}")

        print("\n✨ Comparison completed successfully!")


def main():
    """Main function to trust methods on three specific scenarios"""
    print("🚀 Starting Trust Method Simulation - Three Scenarios")

    # =============================================================================
    # GLOBAL PARAMETERS - All configuration in one place
    # =============================================================================

    # Simulation Parameters
    ROBOT_DENSITY = 0.0010  # ≈10 robots in 100x100 world
    TARGET_DENSITY_MULTIPLIER = 2.0  # Targets are twice robot density
    NUM_TIMESTEPS = 100
    RANDOM_SEED = 8

    # Environment Parameters
    WORLD_SIZE = 100.0
    ADVERSARIAL_RATIO = 0.3
    PROXIMAL_RANGE = 50.0
    FOV_RANGE = 50.0
    FOV_ANGLE = np.pi/3

    WORLD_AREA = WORLD_SIZE * WORLD_SIZE
    TARGET_DENSITY = round(ROBOT_DENSITY * TARGET_DENSITY_MULTIPLIER, 8)
    NUM_ROBOTS = max(1, int(round(ROBOT_DENSITY * WORLD_AREA)))
    NUM_TARGETS = max(1, int(round(TARGET_DENSITY * WORLD_AREA)))

    # Model Parameters
    SUPERVISED_MODEL_PATH = "supervised_trust_model.pth"

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
            "false_negative_rate": 0.3
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
        print(f"  - Target multiplier: {TARGET_DENSITY_MULTIPLIER}")
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
            robot_density=ROBOT_DENSITY,
            target_density_multiplier=TARGET_DENSITY_MULTIPLIER,
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
            comparison.visualize_results(viz_file)

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
                    'target_density_multiplier': comparison.target_density_multiplier,
                    'legitimate_paper': legit_stats.get('paper', {}).get('mean', 0),
                    'legitimate_supervised': legit_stats.get('supervised', {}).get('mean', 0),
                    'adversarial_paper': adv_stats.get('paper', {}).get('mean', 0),
                    'adversarial_supervised': adv_stats.get('supervised', {}).get('mean', 0),
                    'supervised_minus_paper': diff_stats.get('supervised_minus_paper', {}).get('mean', 0),
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
        header = "Scenario Name               | FP Rate | FN Rate | Leg (P/S)      | Adv (P/S)      | S-P Δ"
        print(header)
        print("-" * len(header))

        for stat in summary_stats:
            print(f"{stat['scenario']:<26} | "
                  f"{stat['false_positive_rate']:<7.1f} | "
                  f"{stat['false_negative_rate']:<7.1f} | "
                  f"{stat['legitimate_paper']:.3f}/{stat['legitimate_supervised']:.3f} | "
                  f"{stat['adversarial_paper']:.3f}/{stat['adversarial_supervised']:.3f} | "
                  f"{stat['supervised_minus_paper']:+.3f}")

        avg_leg_paper = np.mean([s['legitimate_paper'] for s in summary_stats])
        avg_leg_supervised = np.mean([s['legitimate_supervised'] for s in summary_stats])
        avg_adv_paper = np.mean([s['adversarial_paper'] for s in summary_stats])
        avg_adv_supervised = np.mean([s['adversarial_supervised'] for s in summary_stats])
        avg_diff = np.mean([s['supervised_minus_paper'] for s in summary_stats])

        print("-" * len(header))
        print(f"{'AVERAGE':<26} | {'':>7} | {'':>7} | "
              f"{avg_leg_paper:.3f}/{avg_leg_supervised:.3f} | "
              f"{avg_adv_paper:.3f}/{avg_adv_supervised:.3f} | "
              f"{avg_diff:+.3f}")

        print(f"\n🔍 Key Findings:")
        print(f"   • Paper Algorithm:")
        print(f"     - Legitimate: {avg_leg_paper:.3f}")
        print(f"     - Adversarial: {avg_adv_paper:.3f}")
        print(f"     - Gap: {avg_leg_paper - avg_adv_paper:+.3f}")
        print(f"   • Supervised Model:")
        print(f"     - Legitimate: {avg_leg_supervised:.3f}")
        print(f"     - Adversarial: {avg_adv_supervised:.3f}")
        print(f"     - Gap: {avg_leg_supervised - avg_adv_supervised:+.3f}")
        print(f"   • Average Supervised vs Paper Difference: {avg_diff:+.3f}")

        print(f"\n📈 Scenario Analysis:")
        for stat in summary_stats:
            print(f"   • {stat['scenario']}:")
            print(f"     - FP/FN rates: {stat['false_positive_rate']:.1f}/{stat['false_negative_rate']:.1f}")
            print(f"     - Paper:      Leg={stat['legitimate_paper']:.3f}, Adv={stat['adversarial_paper']:.3f}")
            print(f"     - Supervised: Leg={stat['legitimate_supervised']:.3f}, Adv={stat['adversarial_supervised']:.3f}")
            print(f"     - Difference: {stat['supervised_minus_paper']:+.3f}")

    # Save comprehensive results using centralized parameters
    comprehensive_results = {
        'scenarios': all_results,
        'summary_statistics': summary_stats,
        'configuration': {
            'world_size': WORLD_SIZE,
            'robot_density': ROBOT_DENSITY,
            'target_density': TARGET_DENSITY,
            'target_density_multiplier': TARGET_DENSITY_MULTIPLIER,
            'derived_num_robots': int(round(ROBOT_DENSITY * WORLD_SIZE * WORLD_SIZE)),
            'derived_num_targets': int(round(TARGET_DENSITY * WORLD_SIZE * WORLD_SIZE)),
            'adversarial_ratio': ADVERSARIAL_RATIO,
            'proximal_range': PROXIMAL_RANGE,
            'fov_range': FOV_RANGE,
            'fov_angle': 'π/3',
            'num_timesteps': NUM_TIMESTEPS,
            'random_seed': RANDOM_SEED,
            'trust_update_method': 'paper_algorithm'
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
