#!/usr/bin/env python3
"""
Webots Trust Method Comparison: 10 Scenarios with Random Adversarial Assignments

Compares Paper, Supervised, Bayesian, and Baseline trust methods across scenarios with
randomly selected adversarial robots, running on real Webots replay data via
WebotsTrustEnvironment (synchronized with the main simulation pipeline's LegitimateRobot/
AdversarialRobot architecture and reported_tracks/all_tracks track model).

Features:
- FP co-detection enabled by default
- Adversarial robots run the real optimized ("aggressive", delta_plus=delta_minus=3.0)
  MILP policy - not a random FP/FN rate - for both persistent FP injection and GT
  suppression decisions
- Reports robot precision/recall and object precision/recall
"""

import argparse
import numpy as np
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple
import itertools

from webots_trust_environment import WebotsTrustEnvironment
from paper_trust_algorithm import PaperTrustAlgorithm
from supervised_trust_algorithm import SupervisedTrustAlgorithm
from bayesian_ego_graph_trust import BayesianEgoGraphTrust

# Method display configuration (matching in_sample_benchmark.py)
METHOD_ORDER = ["baseline", "bayesian", "paper", "supervised"]
METHOD_DISPLAY_NAMES = {
    "baseline": "Baseline (No Trust)",
    "bayesian": "Naïve Bayesian",
    "paper": "PSM Aggregation",
    "supervised": "NeST-Bayes",
}


def classification_metrics(labels: List[int], scores: List[float], threshold: float) -> Dict[str, float]:
    if not labels:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    preds = [1 if score >= threshold else 0 for score in scores]
    labels_arr = np.array(labels, dtype=np.int32)
    preds_arr = np.array(preds, dtype=np.int32)

    correct = np.sum(labels_arr == preds_arr)
    accuracy = correct / len(labels_arr)

    tp = np.sum((preds_arr == 1) & (labels_arr == 1))
    fp = np.sum((preds_arr == 1) & (labels_arr == 0))
    fn = np.sum((preds_arr == 0) & (labels_arr == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }



class WebotsTrustComparison:
    """Runs comparison of trust methods on Webots simulation data"""

    def __init__(self,
                 webots_data_path: str = "webots_sim_filtered_corrected",
                 supervised_model_path: str = "supervised_trust_model.pth",
                 num_scenarios: int = 100,
                 num_adversarial: int = 2,
                 num_timesteps: int = 100,
                 random_seed: int = 42):
        """
        Initialize comparison

        Args:
            webots_data_path: Path to filtered Webots data
            supervised_model_path: Path to trained supervised GNN model
            num_scenarios: Number of different scenarios to test
            num_adversarial: Number of robots to make adversarial
            num_timesteps: Number of timesteps to simulate
            random_seed: Base random seed
        """
        self.webots_data_path = webots_data_path
        self.supervised_model_path = Path(supervised_model_path) if supervised_model_path else None
        self.num_scenarios = num_scenarios
        self.num_adversarial = num_adversarial
        self.num_timesteps = num_timesteps
        self.random_seed = random_seed

        # Results storage
        self.results = []

    # Adversarial policy aggressiveness levels (delta_plus=delta_minus), matching
    # optimized_policy_benchmark.py's conservative/moderate/aggressive sweep.
    POLICY_DELTAS = {
        'conservative': 1.0,
        'moderate': 2.0,
        'aggressive': 3.0,
    }

    def generate_scenarios(self, robot_names: List[str]) -> List[Dict]:
        """
        Generate num_scenarios test scenarios with random adversarial assignments.

        Each scenario has:
        - num_adversarial randomly selected adversarial robots
        - Random persistent FP injection rate (0.1 to 0.3)
        - A randomly selected adversarial policy aggressiveness (conservative/moderate/
          aggressive -> delta_plus=delta_minus in {1.0, 2.0, 3.0}). Adversarial robots run
          the real optimized MILP policy (report/suppress decisions driven by
          delta_plus/delta_minus), not a random FP/FN rate.

        Args:
            robot_names: List of available robot names

        Returns:
            List of scenario configurations
        """
        scenarios = []
        random.seed(self.random_seed)

        policy_names = list(self.POLICY_DELTAS.keys())

        for i in range(self.num_scenarios):
            # Randomly select adversarial robots
            adversarial_robots = random.sample(robot_names, self.num_adversarial)

            # Random persistent FP injection rate
            fp_injection_rate = random.uniform(0.1, 0.3)

            # Randomly select adversarial policy aggressiveness for this scenario
            policy = random.choice(policy_names)
            delta = self.POLICY_DELTAS[policy]

            scenario = {
                'id': i,
                'adversarial_robots': adversarial_robots,
                'fp_injection_rate': fp_injection_rate,
                'policy': policy,
                'delta_plus': delta,
                'delta_minus': delta,
                'seed': self.random_seed + i  # Unique seed per scenario
            }

            scenarios.append(scenario)

        return scenarios

    def run_all_scenarios(self, output_dir: str = "benchmark_results", output_name: str = "webots"):
        """Run all scenarios and compare trust methods.

        Args:
            output_dir: Directory to save results (default: benchmark_results/, matching
                the other benchmark scripts' default so analyze_benchmark_results.py's
                default search path picks this up automatically)
            output_name: Base filename, produces <output_dir>/<output_name>_detailed.json
        """
        # First, load base environment to get robot names (more efficient than full trust environment)
        from webots_simulation_environment import WebotsSimulationEnvironment
        temp_env = WebotsSimulationEnvironment(webots_data_path=self.webots_data_path)
        robot_names = list(temp_env.robot_data.keys())
        self.num_robots_total = len(robot_names)

        # Generate scenarios
        scenarios = self.generate_scenarios(robot_names)

        print("=" * 80)
        print("WEBOTS TRUST COMPARISON")
        print("=" * 80)
        print(f"Dataset: {self.webots_data_path}")
        print(f"Number of scenarios: {len(scenarios)}")
        print(f"Adversarial robots per scenario: {self.num_adversarial}")
        print(f"Timesteps per scenario: {self.num_timesteps}")
        print(f"Random seed: {self.random_seed}")
        print(f"\nRunning {len(scenarios)} test scenarios...")
        print("=" * 80)

        # Run each scenario
        for scenario in scenarios:
            result = self.run_single_scenario(scenario)
            self.results.append(result)

        # Save results
        self.save_results(output_dir=output_dir, name=output_name)

        # Print summary statistics
        self.print_summary_statistics()

    def run_single_scenario(self, scenario: Dict) -> Dict:
        """
        Run a single scenario with all four trust methods.

        Args:
            scenario: Scenario configuration

        Returns:
            Dictionary with results from all methods
        """
        # Initialize trust algorithms
        paper_algo = PaperTrustAlgorithm()
        supervised_algo = SupervisedTrustAlgorithm(model_path=str(self.supervised_model_path))
        bayesian_algo = BayesianEgoGraphTrust()

        # Run simulation with each method (matching METHOD_ORDER)
        methods = {
            'baseline': None,  # No trust baseline
            'bayesian': bayesian_algo,
            'paper': paper_algo,
            'supervised': supervised_algo,
        }

        method_results = {}

        for method_name, algorithm in methods.items():

            # Reset environment (with FP co-detection enabled, per-scenario optimized policy
            # aggressiveness - conservative/moderate/aggressive, see generate_scenarios)
            env = WebotsTrustEnvironment(
                webots_data_path=self.webots_data_path,
                adversarial_robot_ids=scenario['adversarial_robots'],
                adversarial_fp_injection_rate=scenario['fp_injection_rate'],
                allow_fp_codetection=True,  # Enable FP co-detection by default
                delta_plus=scenario['delta_plus'],
                delta_minus=scenario['delta_minus'],
                random_seed=scenario['seed']
            )

            # Run timesteps
            trust_history = []

            for t in range(min(self.num_timesteps, env.num_timesteps)):
                env.step(t)

                # Update trust using this algorithm (pass list of Robot objects and environment)
                # For baseline, skip trust update (all robots keep default trust of 0.5)
                if algorithm is not None:
                    algorithm.update_trust(list(env.robots.values()), environment=env)

                # Record trust scores
                trust_scores = env.get_robot_trust_scores()
                trust_history.append(trust_scores.copy())

            # Calculate metrics
            metrics = self.calculate_metrics(env, trust_history, scenario['adversarial_robots'])
            method_results[method_name] = metrics

        result = {
            'scenario': scenario,
            'methods': method_results
        }

        return result

    def compute_robot_metrics(self, env: WebotsTrustEnvironment,
                             adversarial_robots: List[str],
                             threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute robot-level precision/recall metrics.

        Args:
            env: Environment with final robot trust values
            adversarial_robots: List of adversarial robot names
            threshold: Classification threshold for trust

        Returns:
            Dictionary with robot precision, recall, F1, accuracy
        """
        adversarial_set = set(adversarial_robots)
        labels = []
        scores = []
        legit_trust = []
        adv_trust = []

        for robot_name, robot in env.robots.items():
            is_legit = robot_name not in adversarial_set
            labels.append(1 if is_legit else 0)
            scores.append(robot.trust_value)

            if is_legit:
                legit_trust.append(robot.trust_value)
            else:
                adv_trust.append(robot.trust_value)

        metrics = classification_metrics(labels, scores, threshold)
        metrics['mean_legitimate_trust'] = float(np.mean(legit_trust)) if legit_trust else 0.0
        metrics['mean_adversarial_trust'] = float(np.mean(adv_trust)) if adv_trust else 0.0

        return metrics

    def compute_object_metrics(self, env: WebotsTrustEnvironment,
                               threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute object-level precision/recall using each robot's real (unmanipulated)
        track trust values - no adversarial lie substitution.

        Args:
            env: Environment with robot tracks
            threshold: Classification threshold

        Returns:
            Dictionary with object precision, recall, F1, accuracy
        """
        # Filter to believed-legitimate robots (trust >= threshold)
        legitimate_robots = {name for name, robot in env.robots.items()
                           if robot.trust_value >= threshold}

        # FIXED DENOMINATOR: every object ANY robot ever actually detected (union of
        # all_tracks across all robots, independent of trust), NOT every object that
        # exists in the environment definition. An object no robot ever sensed the whole
        # episode is a sensing-coverage gap outside every method's control (occlusion,
        # range, FoV) - it should not count against any method's recall, since no method
        # could possibly have recalled an object nobody ever observed. This denominator is
        # still identical across all four methods within a scenario: it only depends on
        # what robots detected (same sensors, same replay data, same is_in_fov/DetectorSensor
        # sampling per method run - not on which trust algorithm is scoring afterward).
        all_objects = set()
        for robot_name, robot in env.robots.items():
            for track in robot.get_all_tracks():
                obj_id = track.object_id
                if obj_id.startswith('DEF:') or obj_id.startswith('fp_obj_'):
                    all_objects.add(obj_id)

        # Aggregate tracks by object_id from legitimate robots only
        object_weighted_trusts = {}  # object_id -> list of (robot_trust, track_trust)

        for robot_name, robot in env.robots.items():
            if robot_name not in legitimate_robots:
                continue  # Skip robots with trust < threshold

            robot_trust = robot.trust_value

            for track in robot.get_all_tracks():
                obj_id = track.object_id
                track_trust = track.trust_value

                if obj_id.startswith('DEF:') or obj_id.startswith('fp_obj_'):
                    if obj_id not in object_weighted_trusts:
                        object_weighted_trusts[obj_id] = []
                    object_weighted_trusts[obj_id].append((robot_trust, track_trust))

        # Calculate weighted average trust per object and evaluate
        labels = []
        scores = []
        gt_trust = []
        fp_trust = []

        # Evaluate ALL objects, including those not seen by trusted robots
        for obj_id in all_objects:
            if obj_id in object_weighted_trusts:
                weighted_values = object_weighted_trusts[obj_id]
                # Calculate weighted average: sum(robot_trust * track_trust) / sum(robot_trust)
                sum_weighted = sum(robot_trust * track_trust for robot_trust, track_trust in weighted_values)
                sum_weights = sum(robot_trust for robot_trust, _ in weighted_values)
                avg_trust = float(sum_weighted / sum_weights) if sum_weights > 0 else 0.0
            else:
                # Object only seen by untrusted robots -> score = 0.0
                avg_trust = 0.0

            # Webots gids use 'DEF:' for ground truth; persistent adversarial FP objects
            # use robot_types.py's 'fp_obj_{id}' convention
            if obj_id.startswith('DEF:'):  # Ground truth object
                labels.append(1)
                gt_trust.append(avg_trust)
            elif obj_id.startswith('fp_obj_'):  # False positive object
                labels.append(0)
                fp_trust.append(avg_trust)
            else:
                continue

            scores.append(avg_trust)

        metrics = classification_metrics(labels, scores, threshold)
        metrics['mean_true_object_trust'] = float(np.mean(gt_trust)) if gt_trust else 0.0
        metrics['mean_false_object_trust'] = float(np.mean(fp_trust)) if fp_trust else 0.0

        return metrics

    def calculate_metrics(self, env: WebotsTrustEnvironment,
                         trust_history: List[Dict[str, float]],
                         adversarial_robots: List[str]) -> Dict:
        """
        Calculate comprehensive performance metrics for a method.

        Returns the same {"robots": {...}, "objects": {...}} shape as
        comprehensive_trust_benchmark.py's evaluate_methods, so results saved by this
        script are directly readable by analyze_benchmark_results.py.

        Args:
            env: Environment
            trust_history: List of trust scores over time (unused now that
                compute_robot_metrics computes mean_legitimate_trust/mean_adversarial_trust
                directly from final robot state; kept as a parameter for call-site
                compatibility)
            adversarial_robots: List of adversarial robot names

        Returns:
            Dictionary with 'robots' and 'objects' metric sub-dicts
        """
        robot_metrics = self.compute_robot_metrics(env, adversarial_robots, threshold=0.5)
        object_metrics = self.compute_object_metrics(env, threshold=0.5)

        return {
            'robots': robot_metrics,
            'objects': object_metrics,
        }

    def print_summary_statistics(self):
        """Print summary statistics matching in_sample_benchmark format"""
        # Collect metrics for each method
        method_robot_precisions = {method: [] for method in METHOD_ORDER}
        method_robot_recalls = {method: [] for method in METHOD_ORDER}
        method_robot_accuracies = {method: [] for method in METHOD_ORDER}
        method_object_precisions = {method: [] for method in METHOD_ORDER}
        method_object_recalls = {method: [] for method in METHOD_ORDER}
        method_object_accuracies = {method: [] for method in METHOD_ORDER}

        for result in self.results:
            methods_data = result['methods']
            for method in METHOD_ORDER:
                if method in methods_data:
                    robot_prec = methods_data[method]["robots"].get("precision", 0.0)
                    robot_rec = methods_data[method]["robots"].get("recall", 0.0)
                    robot_acc = methods_data[method]["robots"].get("accuracy", 0.0)
                    object_prec = methods_data[method]["objects"].get("precision", 0.0)
                    object_rec = methods_data[method]["objects"].get("recall", 0.0)
                    object_acc = methods_data[method]["objects"].get("accuracy", 0.0)
                    method_robot_precisions[method].append(robot_prec)
                    method_robot_recalls[method].append(robot_rec)
                    method_robot_accuracies[method].append(robot_acc)
                    method_object_precisions[method].append(object_prec)
                    method_object_recalls[method].append(object_rec)
                    method_object_accuracies[method].append(object_acc)

        print("\n" + "=" * 80)
        print("WEBOTS TRUST COMPARISON RESULTS")
        print("=" * 80)
        print("\nROBOT PRECISION (Adversarial Rejection):")
        print("-" * 80)
        print(f"{'Method':<25} {'Mean':<15} {'Std Dev':<15} {'N':<10}")
        print("-" * 80)
        for method in METHOD_ORDER:
            precs = method_robot_precisions[method]
            if precs:
                mean_prec = np.mean(precs)
                std_prec = np.std(precs)
                print(f"{METHOD_DISPLAY_NAMES[method]:<25} {mean_prec:.4f}          {std_prec:.4f}          {len(precs)}")

        print("\nROBOT RECALL (Legitimate Robot Identification):")
        print("-" * 80)
        print(f"{'Method':<25} {'Mean':<15} {'Std Dev':<15} {'N':<10}")
        print("-" * 80)
        for method in METHOD_ORDER:
            recs = method_robot_recalls[method]
            if recs:
                mean_rec = np.mean(recs)
                std_rec = np.std(recs)
                print(f"{METHOD_DISPLAY_NAMES[method]:<25} {mean_rec:.4f}          {std_rec:.4f}          {len(recs)}")

        print("\nROBOT ACCURACY:")
        print("-" * 80)
        print(f"{'Method':<25} {'Mean':<15} {'Std Dev':<15} {'N':<10}")
        print("-" * 80)
        for method in METHOD_ORDER:
            accs = method_robot_accuracies[method]
            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                print(f"{METHOD_DISPLAY_NAMES[method]:<25} {mean_acc:.4f}          {std_acc:.4f}          {len(accs)}")

        print("\nOBJECT PRECISION (FP Rejection):")
        print("-" * 80)
        print(f"{'Method':<25} {'Mean':<15} {'Std Dev':<15} {'N':<10}")
        print("-" * 80)
        for method in METHOD_ORDER:
            precs = method_object_precisions[method]
            if precs:
                mean_prec = np.mean(precs)
                std_prec = np.std(precs)
                print(f"{METHOD_DISPLAY_NAMES[method]:<25} {mean_prec:.4f}          {std_prec:.4f}          {len(precs)}")

        print("\nOBJECT RECALL (GT Object Identification):")
        print("-" * 80)
        print(f"{'Method':<25} {'Mean':<15} {'Std Dev':<15} {'N':<10}")
        print("-" * 80)
        for method in METHOD_ORDER:
            recs = method_object_recalls[method]
            if recs:
                mean_rec = np.mean(recs)
                std_rec = np.std(recs)
                print(f"{METHOD_DISPLAY_NAMES[method]:<25} {mean_rec:.4f}          {std_rec:.4f}          {len(recs)}")

        print("\nOBJECT ACCURACY:")
        print("-" * 80)
        print(f"{'Method':<25} {'Mean':<15} {'Std Dev':<15} {'N':<10}")
        print("-" * 80)
        for method in METHOD_ORDER:
            accs = method_object_accuracies[method]
            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                print(f"{METHOD_DISPLAY_NAMES[method]:<25} {mean_acc:.4f}          {std_acc:.4f}          {len(accs)}")


    def save_results(self, output_dir: str = "benchmark_results", name: str = "webots"):
        """Save results in the same {"metadata": ..., "scenarios": [...]} _detailed.json
        format used by unified_benchmark.py / optimized_policy_benchmark.py /
        deceptive_policy_benchmark.py, so analyze_benchmark_results.py can read this
        script's output directly (it discovers files via *_detailed.json glob and reads
        scenario["parameters"]/scenario["evaluation"][method]["robots"/"objects"]).

        Args:
            output_dir: Directory to save results (default: benchmark_results/, matching
                the other benchmark scripts' default)
            name: Base filename, produces <output_dir>/<name>_detailed.json
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        scenarios = []
        for i, result in enumerate(self.results):
            scenario = result['scenario']
            # extract_metrics_dataframe requires: name, robot_density, adversarial_ratio,
            # random_seed (no fallback for these). Webots has no robot_density (fixed real
            # robot count from the recording, not sampled) - report num robots instead so
            # the field is at least present and informative.
            parameters = {
                "name": f"webots_{scenario['id']:03d}",
                "benchmark_type": "webots",
                # Webots has no sampled robot_density (fixed real robot count from the
                # recording) - report the actual robot count instead so the field is at
                # least present and informative for extract_metrics_dataframe.
                "robot_density": self.num_robots_total,
                "adversarial_ratio": self.num_adversarial / max(1, self.num_robots_total),
                "adversarial_fp_injection_rate": scenario['fp_injection_rate'],
                "adversarial_fn_suppression_rate": 0.0,  # Policy-driven (MILP), not a rate
                "policy": scenario['policy'],  # conservative / moderate / aggressive
                "delta_plus": scenario['delta_plus'],
                "delta_minus": scenario['delta_minus'],
                "adversarial_robots": scenario['adversarial_robots'],
                "random_seed": scenario['seed'],
            }
            scenarios.append({
                "scenario_index": i,
                "parameters": parameters,
                "evaluation": result['methods'],
            })

        detailed_results = {
            "metadata": {
                "benchmark_type": "webots",
                "adversarial_mode": "optimized",
                "description": "Webots replay data - optimized policy, per-scenario random "
                               "aggressiveness (conservative/moderate/aggressive, "
                               "delta_plus=delta_minus in {1.0, 2.0, 3.0})",
                "num_scenarios": len(self.results),
                "base_seed": self.random_seed,
                "robot_modes": {
                    "legitimate": "webots_replay",
                    "adversarial": "optimized"
                },
                "adversarial_lie": False,
                "parameter_ranges": {
                    "num_adversarial": self.num_adversarial,
                    "num_timesteps": self.num_timesteps,
                    "policies": list(self.POLICY_DELTAS.keys()),
                    "delta_values": list(self.POLICY_DELTAS.values()),
                },
            },
            "scenarios": scenarios,
        }

        detailed_path = output_path / f"{name}_detailed.json"
        with open(detailed_path, 'w') as f:
            json.dump(convert_to_serializable(detailed_results), f, indent=2)

        print(f"\n✓ Results saved to: {detailed_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Webots trust method comparison - optimized aggressive adversarial policy"
    )
    parser.add_argument("--webots-data-path", type=str, default="webots_sim_filtered_corrected",
                        help="Path to filtered Webots data")
    parser.add_argument("--supervised-model", type=str, default="supervised_trust_model.pth",
                        help="Path to trained supervised GNN model")
    parser.add_argument("--num-scenarios", type=int, default=100,
                        help="Number of scenarios to test (default: 100)")
    parser.add_argument("--num-adversarial", type=int, default=2,
                        help="Number of adversarial robots per scenario (default: 2)")
    parser.add_argument("--num-timesteps", type=int, default=100,
                        help="Number of timesteps to simulate per scenario (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save results (default: benchmark_results/, "
                             "matching the other benchmark scripts so "
                             "analyze_benchmark_results.py's default search picks it up)")
    parser.add_argument("--output-name", type=str, default="webots",
                        help="Base filename, produces <output-dir>/<output-name>_detailed.json")
    args = parser.parse_args()

    comparison = WebotsTrustComparison(
        webots_data_path=args.webots_data_path,
        supervised_model_path=args.supervised_model,
        num_scenarios=args.num_scenarios,
        num_adversarial=args.num_adversarial,
        num_timesteps=args.num_timesteps,
        random_seed=args.seed
    )

    comparison.run_all_scenarios(output_dir=args.output_dir, output_name=args.output_name)


if __name__ == "__main__":
    main()
