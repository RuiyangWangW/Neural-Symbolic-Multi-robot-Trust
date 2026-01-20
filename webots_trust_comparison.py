#!/usr/bin/env python3
"""
Webots Trust Method Comparison: 10 Scenarios with Random Adversarial Assignments

Compares Paper, Supervised, Bayesian, and Ego-Graph trust methods
across 10 scenarios with randomly selected adversarial robots.

Features:
- FP co-detection enabled by default
- Adversarial robots lie about track trust values
- Reports robot precision/recall and object precision/recall
"""

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
                 num_scenarios: int = 10,
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

    def generate_scenarios(self, robot_names: List[str]) -> List[Dict]:
        """
        Generate 10 different test scenarios with random adversarial assignments.

        Each scenario has:
        - 2 randomly selected adversarial robots
        - Random FP rate (0.1 to 0.3)
        - Random FN rate (0.0 to 0.3)

        Args:
            robot_names: List of available robot names

        Returns:
            List of scenario configurations
        """
        scenarios = []
        random.seed(self.random_seed)

        for i in range(self.num_scenarios):
            # Randomly select adversarial robots
            adversarial_robots = random.sample(robot_names, self.num_adversarial)

            # Random FP and FN rates
            fp_rate = random.uniform(0.1, 0.3)
            fn_rate = random.uniform(0.0, 0.3)

            scenario = {
                'id': i,
                'adversarial_robots': adversarial_robots,
                'fp_rate': fp_rate,
                'fn_rate': fn_rate,
                'seed': self.random_seed + i  # Unique seed per scenario
            }

            scenarios.append(scenario)

        return scenarios

    def run_all_scenarios(self):
        """Run all scenarios and compare trust methods"""
        # First, load base environment to get robot names (more efficient than full trust environment)
        from webots_simulation_environment import WebotsSimulationEnvironment
        temp_env = WebotsSimulationEnvironment(webots_data_path=self.webots_data_path)
        robot_names = list(temp_env.robot_data.keys())

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
        self.save_results()

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

            # Reset environment (with FP co-detection enabled)
            env = WebotsTrustEnvironment(
                webots_data_path=self.webots_data_path,
                adversarial_robot_ids=scenario['adversarial_robots'],
                false_positive_rate=scenario['fp_rate'],
                false_negative_rate=scenario['fn_rate'],
                allow_fp_codetection=True,  # Enable FP co-detection by default
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

    def compute_object_metrics_with_adversarial_lies(self, env: WebotsTrustEnvironment,
                                                     adversarial_robots: List[str],
                                                     threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute object-level precision/recall with ADVERSARIAL TRACK LIES.

        Adversarial robots lie about track trust values:
        - GT objects: lie low (random uniform 0.0 to 0.2)
        - FP objects: lie high (random uniform 0.8 to 1.0)

        Args:
            env: Environment with robot tracks
            adversarial_robots: List of adversarial robot names
            threshold: Classification threshold

        Returns:
            Dictionary with object precision, recall, F1, accuracy
        """
        adversarial_set = set(adversarial_robots)

        # Filter to believed-legitimate robots (trust >= threshold)
        legitimate_robots = {name for name, robot in env.robots.items()
                           if robot.trust_value >= threshold}

        # Collect ALL objects (from all robots)
        # Webots uses 'DEF:' for ground truth and 'FP:' for false positives
        all_objects = set()
        for robot_name, robot in env.robots.items():
            for track in robot.local_tracks.values():
                if hasattr(track, 'object_id'):
                    obj_id = track.object_id
                    if obj_id.startswith('DEF:') or obj_id.startswith('FP:'):
                        all_objects.add(obj_id)

        # Aggregate tracks by object_id from legitimate robots only
        object_weighted_trusts = {}  # object_id -> list of (robot_trust, track_trust)

        for robot_name, robot in env.robots.items():
            if robot_name not in legitimate_robots:
                continue  # Skip robots with trust < threshold

            robot_trust = robot.trust_value
            is_adversarial = robot_name in adversarial_set

            for track in robot.local_tracks.values():
                if not hasattr(track, 'object_id'):
                    print("Warning: Track missing object_id attribute, skipping.")
                    continue

                obj_id = track.object_id
                track_trust = track.trust_value

                # ADVERSARIAL TRACK LIES: Override track trust if robot is adversarial
                # Webots uses 'DEF:' for ground truth and 'FP:' for false positives
                if is_adversarial:
                    if obj_id.startswith('DEF:'):  # Ground truth object
                        track_trust = random.uniform(0.0, 0.2)
                    elif obj_id.startswith('FP:'):  # False positive object
                        track_trust = random.uniform(0.8, 1.0)

                if obj_id.startswith('DEF:') or obj_id.startswith('FP:'):
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

            # Webots uses 'DEF:' for ground truth and 'FP:' for false positives
            if obj_id.startswith('DEF:'):  # Ground truth object
                labels.append(1)
                gt_trust.append(avg_trust)
            elif obj_id.startswith('FP:'):  # False positive object
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

        Metrics:
        - Trust separation metrics (legacy)
        - Robot precision/recall
        - Object precision/recall (with adversarial lies)

        Args:
            env: Environment
            trust_history: List of trust scores over time
            adversarial_robots: List of adversarial robot names

        Returns:
            Dictionary of metrics
        """
        adversarial_set = set(adversarial_robots)
        legitimate_robots = [name for name in env.robots.keys() if name not in adversarial_set]

        # Final trust scores
        final_trusts = trust_history[-1] if trust_history else {}

        leg_trust_final = np.mean([final_trusts[r] for r in legitimate_robots])
        adv_trust_final = np.mean([final_trusts[r] for r in adversarial_robots])
        trust_separation = leg_trust_final - adv_trust_final

        # Trust over time
        leg_trust_over_time = []
        adv_trust_over_time = []

        for trust_scores in trust_history:
            leg_mean = np.mean([trust_scores[r] for r in legitimate_robots])
            adv_mean = np.mean([trust_scores[r] for r in adversarial_robots])

            leg_trust_over_time.append(leg_mean)
            adv_trust_over_time.append(adv_mean)

        # Compute robot and object metrics
        robot_metrics = self.compute_robot_metrics(env, adversarial_robots, threshold=0.5)
        object_metrics = self.compute_object_metrics_with_adversarial_lies(env, adversarial_robots, threshold=0.5)

        metrics = {
            # Trust separation metrics (legacy)
            'final_legitimate_trust': float(leg_trust_final),
            'final_adversarial_trust': float(adv_trust_final),
            'trust_separation': float(trust_separation),
            'legitimate_trust_history': leg_trust_over_time,
            'adversarial_trust_history': adv_trust_over_time,

            # Robot classification metrics (nested)
            'robot_classification': {
                'precision': robot_metrics['precision'],
                'recall': robot_metrics['recall'],
                'f1': robot_metrics['f1'],
                'accuracy': robot_metrics['accuracy']
            },

            # Object classification metrics (nested)
            'object_classification': {
                'precision': object_metrics['precision'],
                'recall': object_metrics['recall'],
                'f1': object_metrics['f1'],
                'accuracy': object_metrics['accuracy'],
                'mean_true_object_trust': object_metrics['mean_true_object_trust'],
                'mean_false_object_trust': object_metrics['mean_false_object_trust']
            }
        }

        return metrics

    def format_evaluation_results(self) -> List[Dict]:
        """Format results to match comprehensive_trust_benchmark structure"""
        formatted_results = []

        for result in self.results:
            scenario = result['scenario']
            methods_data = result['methods']

            # Format metrics for each method
            formatted_methods = {}
            for method_name, metrics in methods_data.items():
                formatted_methods[method_name] = {
                    'robots': {
                        'precision': metrics['robot_classification']['precision'],
                        'recall': metrics['robot_classification']['recall'],
                        'f1': metrics['robot_classification']['f1'],
                        'accuracy': metrics['robot_classification']['accuracy']
                    },
                    'objects': {
                        'precision': metrics['object_classification']['precision'],
                        'recall': metrics['object_classification']['recall'],
                        'f1': metrics['object_classification']['f1'],
                        'accuracy': metrics['object_classification']['accuracy']
                    }
                }

            formatted_results.append({
                'name': f"webots_{scenario['id']:03d}",
                'metrics': formatted_methods
            })

        return formatted_results

    def print_summary_statistics(self):
        """Print summary statistics matching in_sample_benchmark format"""
        summary = self.format_evaluation_results()

        # Collect metrics for each method
        method_robot_precisions = {method: [] for method in METHOD_ORDER}
        method_robot_recalls = {method: [] for method in METHOD_ORDER}
        method_robot_accuracies = {method: [] for method in METHOD_ORDER}
        method_object_precisions = {method: [] for method in METHOD_ORDER}
        method_object_recalls = {method: [] for method in METHOD_ORDER}
        method_object_accuracies = {method: [] for method in METHOD_ORDER}

        for entry in summary:
            for method in METHOD_ORDER:
                if method in entry["metrics"]:
                    robot_prec = entry["metrics"][method]["robots"].get("precision", 0.0)
                    robot_rec = entry["metrics"][method]["robots"].get("recall", 0.0)
                    robot_acc = entry["metrics"][method]["robots"].get("accuracy", 0.0)
                    object_prec = entry["metrics"][method]["objects"].get("precision", 0.0)
                    object_rec = entry["metrics"][method]["objects"].get("recall", 0.0)
                    object_acc = entry["metrics"][method]["objects"].get("accuracy", 0.0)
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


    def save_results(self, output_path: str = "webots_trust_comparison_results.json"):
        """Save results to JSON file"""
        output_file = Path(output_path)

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

        serializable_results = convert_to_serializable(self.results)

        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")


def main():
    """Run comparison with default settings"""
    comparison = WebotsTrustComparison(
        webots_data_path="webots_sim_filtered_corrected",
        supervised_model_path="supervised_trust_model.pth",
        num_scenarios=10,
        num_adversarial=2,
        num_timesteps=100,
        random_seed=42
    )

    comparison.run_all_scenarios()


if __name__ == "__main__":
    main()
