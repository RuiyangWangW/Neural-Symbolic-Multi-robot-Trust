#!/usr/bin/env python3
"""
Comprehensive benchmarking script comparing paper algorithm and supervised GNN model
across in-distribution and out-of-distribution scenarios.

The script reuses `TrustMethodComparison` to run simulations, saves per-scenario
results (including full time-series data) for reproducibility, and computes
classification metrics for robots and tracks. Aggregated metrics and plots are
stored under the chosen output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from compare_trust_methods import TrustMethodComparison


def make_scenario(
    name: str,
    description: str,
    *,
    robot_density: float,
    target_density_multiplier: float,
    adversarial_ratio: float,
    false_positive_rate: float,
    false_negative_rate: float,
    num_timesteps: int,
    random_seed: int,
) -> Dict:
    target_density = round(float(robot_density) * float(target_density_multiplier), 8)
    return {
        "name": name,
        "description": description,
        "world_size": WORLD_SIZE_METERS,
        "robot_density": float(robot_density),
        "target_density_multiplier": float(target_density_multiplier),
        "target_density": target_density,
        "adversarial_ratio": float(adversarial_ratio),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "num_timesteps": int(num_timesteps),
        "random_seed": int(random_seed),
    }


BASE_ROBOT_DENSITY = 0.0005  # ≈5 robots in 100x100 world
BASE_TARGET_DENSITY_MULTIPLIER = 4.0  # ≈20 targets in 100x100 world
BASE_TARGET_DENSITY = round(BASE_ROBOT_DENSITY * BASE_TARGET_DENSITY_MULTIPLIER, 8)
WORLD_SIZE_METERS = 100.0

IN_SAMPLE_SCENARIOS: List[Dict] = []
robot_density_values = [0.00045, 0.00048, 0.0005, 0.00052, 0.00055, 0.00046, 0.00049, 0.00053, 0.00057, 0.00051]
target_density_values = [0.0018, 0.0019, 0.0020, 0.0021, 0.0022, 0.00185, 0.00195, 0.00205, 0.00215, 0.00225]

for idx, (rd, td) in enumerate(zip(robot_density_values, target_density_values)):
    target_multiplier = round(td / rd, 6) if rd > 0 else BASE_TARGET_DENSITY_MULTIPLIER
    fp = 0.45 + (idx % 3) * 0.02
    fn = 0.02 if idx % 4 == 0 else 0.0
    IN_SAMPLE_SCENARIOS.append(
        make_scenario(
            name=f"in_sample_{idx+1:02d}",
            description=f"In-distribution scenario variant {idx+1}.",
            robot_density=rd,
            target_density_multiplier=target_multiplier,
            adversarial_ratio=0.5,
            false_positive_rate=fp,
            false_negative_rate=fn,
            num_timesteps=100 + (idx % 3) * 10,
            random_seed=1000 + idx * 37,
        )
    )


OOD_SCENARIOS: List[Dict] = []

# Lower robot/target density to mimic larger operational areas
low_density_pairs = [
    (0.00020, 0.0010),
    (0.00024, 0.0012),
    (0.00028, 0.0014),
    (0.00032, 0.0015),
    (0.00035, 0.0016),
]
for idx, (rd, td) in enumerate(low_density_pairs):
    target_multiplier = round(td / rd, 6) if rd > 0 else BASE_TARGET_DENSITY_MULTIPLIER
    OOD_SCENARIOS.append(
        make_scenario(
            name=f"ood_low_density_{idx+1:02d}",
            description="OOD: sparse coverage reminiscent of larger operational areas.",
            robot_density=rd,
            target_density_multiplier=target_multiplier,
            adversarial_ratio=0.5,
            false_positive_rate=0.5,
            false_negative_rate=0.05,
            num_timesteps=120,
            random_seed=2000 + idx * 41,
        )
    )

# Higher density (more robots/targets in fixed world)
high_density_pairs = [
    (0.00070, 0.0026),
    (0.00080, 0.0028),
    (0.00090, 0.0030),
    (0.00100, 0.0032),
    (0.00110, 0.0034),
]
for idx, (rd, td) in enumerate(high_density_pairs):
    target_multiplier = round(td / rd, 6) if rd > 0 else BASE_TARGET_DENSITY_MULTIPLIER
    OOD_SCENARIOS.append(
        make_scenario(
            name=f"ood_high_density_{idx+1:02d}",
            description="OOD: crowded environment with many robots and targets.",
            robot_density=rd,
            target_density_multiplier=target_multiplier,
            adversarial_ratio=0.5,
            false_positive_rate=0.55,
            false_negative_rate=0.05,
            num_timesteps=130,
            random_seed=2500 + idx * 53,
        )
    )

# Elevated false-positive rates
high_false_positive_settings = [0.7, 0.75, 0.8, 0.85, 0.9]
for idx, fp_rate in enumerate(high_false_positive_settings):
    OOD_SCENARIOS.append(
        make_scenario(
            name=f"ood_high_false_positive_{idx+1:02d}",
            description="OOD: sensors prone to frequent false positives.",
            robot_density=BASE_ROBOT_DENSITY,
            target_density_multiplier=BASE_TARGET_DENSITY_MULTIPLIER,
            adversarial_ratio=0.5,
            false_positive_rate=fp_rate,
            false_negative_rate=0.1,
            num_timesteps=130,
            random_seed=3000 + idx * 61,
        )
    )

# Elevated false-negative rates
high_false_negative_settings = [0.3, 0.35, 0.4, 0.45, 0.5]
for idx, fn_rate in enumerate(high_false_negative_settings):
    OOD_SCENARIOS.append(
        make_scenario(
            name=f"ood_high_false_negative_{idx+1:02d}",
            description="OOD: many true detections are missed.",
            robot_density=BASE_ROBOT_DENSITY,
            target_density_multiplier=BASE_TARGET_DENSITY_MULTIPLIER,
            adversarial_ratio=0.5,
            false_positive_rate=0.55,
            false_negative_rate=fn_rate,
            num_timesteps=130,
            random_seed=3500 + idx * 67,
        )
    )

# High adversarial presence
high_adversarial_settings = [0.6, 0.65, 0.7, 0.8, 0.9]
for idx, adv_ratio in enumerate(high_adversarial_settings):
    rd = 0.0006
    td = 0.0022
    target_multiplier = round(td / rd, 6) if rd > 0 else BASE_TARGET_DENSITY_MULTIPLIER
    OOD_SCENARIOS.append(
        make_scenario(
            name=f"ood_high_adversarial_{idx+1:02d}",
            description="OOD: high proportion of adversarial robots.",
            robot_density=rd,
            target_density_multiplier=target_multiplier,
            adversarial_ratio=adv_ratio,
            false_positive_rate=0.6,
            false_negative_rate=0.1,
            num_timesteps=130,
            random_seed=4000 + idx * 73,
        )
    )

DEFAULT_SCENARIOS: List[Dict] = IN_SAMPLE_SCENARIOS + OOD_SCENARIOS

METHOD_ORDER = ["baseline", "bayesian", "paper", "supervised"]
METHOD_DISPLAY_NAMES = {
    "baseline": "Baseline (No Trust)",
    "paper": "PSM Aggregation",
    "supervised": "Supervised GNN",
    "bayesian": "Bayesian Ego Graph",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark trust methods across multiple scenarios.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comprehensive_benchmark"),
        help="Directory to store per-scenario results, metrics, and plots.",
    )
    parser.add_argument(
        "--supervised-model",
        type=Path,
        default=Path("supervised_trust_model.pth"),
        help="Path to supervised GNN model checkpoint.",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        help="Optional JSON file describing scenarios. Defaults to hard-coded set if omitted.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Trust threshold used to convert trust values into binary labels.",
    )
    return parser.parse_args()


def load_scenarios(args: argparse.Namespace) -> List[Dict]:
    if args.scenarios and args.scenarios.exists():
        with args.scenarios.open() as fh:
            data = json.load(fh)
        return data
    return DEFAULT_SCENARIOS


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


def compute_robot_metrics(final_step: Dict, threshold: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    robot_trust = final_step["robot_trust_values"]
    adversarial = set(final_step.get("adversarial_robots", []))
    labels = []
    scores = []
    legit_trust = []
    adversarial_trust = []

    for robot_id_str, trust in robot_trust.items():
        robot_id = int(robot_id_str)
        is_legit = robot_id not in adversarial
        labels.append(1 if is_legit else 0)
        scores.append(float(trust))
        if is_legit:
            legit_trust.append(float(trust))
        else:
            adversarial_trust.append(float(trust))

    metrics = classification_metrics(labels, scores, threshold)
    stats = {
        "mean_legitimate_trust": float(np.mean(legit_trust)) if legit_trust else 0.0,
        "mean_adversarial_trust": float(np.mean(adversarial_trust)) if adversarial_trust else 0.0,
    }
    return metrics, stats


def compute_object_metrics(final_step: Dict, threshold: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute object-level accuracy using only believed-legitimate robots (trust >= threshold).

    Algorithm:
    1. Filter to robots with trust >= threshold (believed legitimate)
    2. Aggregate tracks by object_id using weighted average from filtered robots only
    3. Objects with aggregated trust > threshold are "believed to be true"
    4. Accuracy = (GT objects believed true) / (GT + FP objects believed true)

    This measures: "Of all objects we trust (based on trusted robots), what fraction are actually real?"
    """
    track_data = final_step.get("track_trust_values", {})
    robot_trust_data = final_step.get("robot_trust_values", {})

    # Filter to believed-legitimate robots (trust >= threshold)
    legitimate_robots = {robot_id for robot_id, trust in robot_trust_data.items()
                        if float(trust) >= threshold}

    # Aggregate tracks by object_id from legitimate robots only
    object_weighted_trusts = {}  # object_id -> list of (robot_trust, track_trust) tuples

    for robot_id, robot_tracks in track_data.items():
        if robot_id not in legitimate_robots:
            continue  # Skip robots with trust <= threshold

        robot_trust = float(robot_trust_data.get(robot_id, 0.5))

        for _, track_info in robot_tracks.items():
            object_id = track_info.get("object_id", "")
            track_trust = float(track_info.get("trust_value", 0.5))

            if object_id.startswith("gt_") or object_id.startswith("fp_"):
                if object_id not in object_weighted_trusts:
                    object_weighted_trusts[object_id] = []
                object_weighted_trusts[object_id].append((robot_trust, track_trust))

    # Calculate weighted average trust per object and evaluate
    labels = []
    scores = []
    gt_trust = []
    fp_trust = []

    for object_id, weighted_values in object_weighted_trusts.items():
        # Calculate weighted average: sum(robot_trust * track_trust) / sum(robot_trust)
        sum_weighted = sum(robot_trust * track_trust for robot_trust, track_trust in weighted_values)
        sum_weights = sum(robot_trust for robot_trust, track_trust in weighted_values)

        # Avoid division by zero
        avg_trust = float(sum_weighted / sum_weights) if sum_weights > 0 else 0.5

        if object_id.startswith("gt_"):
            labels.append(1)
            gt_trust.append(avg_trust)
        elif object_id.startswith("fp_"):
            labels.append(0)
            fp_trust.append(avg_trust)
        else:
            continue

        scores.append(avg_trust)

    metrics = classification_metrics(labels, scores, threshold)
    stats = {
        "mean_true_object_trust": float(np.mean(gt_trust)) if gt_trust else 0.0,
        "mean_false_object_trust": float(np.mean(fp_trust)) if fp_trust else 0.0,
    }
    return metrics, stats


def evaluate_methods(results: Dict, threshold: float) -> Dict[str, Dict[str, Dict[str, float]]]:
    evaluation: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method_key in METHOD_ORDER:
        # Map method key to result key (e.g., "paper" -> "paper_results")
        result_key = f"{method_key}_results"
        method_results = results.get(result_key, [])
        if not method_results:
            continue
        final_step = method_results[-1]
        robot_metrics, robot_stats = compute_robot_metrics(final_step, threshold)
        object_metrics, object_stats = compute_object_metrics(final_step, threshold)
        evaluation[method_key] = {
            "robots": {**robot_metrics, **robot_stats},
            "objects": {**object_metrics, **object_stats},
        }
    return evaluation


def plot_accuracy(summary: List[Dict], output_dir: Path) -> None:
    scenarios = [entry["name"] for entry in summary]
    x = np.arange(len(scenarios))
    width = 0.18  # Adjusted for 4 methods

    # Define consistent colors for each method
    method_colors = {
        'baseline': '#d62728',   # Red
        'bayesian': '#2ca02c',   # Green
        'paper': '#1f77b4',      # Blue
        'supervised': '#ff7f0e', # Orange
    }

    robot_fig, (ax_robot, ax_object) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for idx, method in enumerate(METHOD_ORDER):
        robot_acc = [entry["metrics"].get(method, {}).get("robots", {}).get("accuracy", 0.0) for entry in summary]
        object_acc = [entry["metrics"].get(method, {}).get("objects", {}).get("accuracy", 0.0) for entry in summary]
        offsets = x + (idx - 1.5) * width
        color = method_colors.get(method, f'C{idx}')
        ax_robot.bar(offsets, robot_acc, width=width, label=METHOD_DISPLAY_NAMES[method],
                    color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax_object.bar(offsets, object_acc, width=width, label=METHOD_DISPLAY_NAMES[method],
                    color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

    for ax, title in zip((ax_robot, ax_object), ("Robot Accuracy", "Object Accuracy")):
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    ax_object.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=10)
    robot_fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    robot_fig.savefig(output_dir / "trust_method_accuracy_summary.png", dpi=200, bbox_inches="tight")
    plt.close(robot_fig)


def run_scenario(scenario: Dict, args: argparse.Namespace, output_dir: Path) -> Tuple[Dict, Dict]:
    scenario_dir = output_dir / scenario["name"]
    scenario_dir.mkdir(parents=True, exist_ok=True)

    comparison = TrustMethodComparison(
        supervised_model_path=str(args.supervised_model),
        robot_density=scenario["robot_density"],
        target_density_multiplier=scenario["target_density_multiplier"],
        num_timesteps=scenario["num_timesteps"],
        random_seed=scenario["random_seed"],
        world_size=scenario["world_size"],
        fov_range=50.0,
        fov_angle=np.pi / 3,
    )
    comparison.adversarial_ratio = scenario["adversarial_ratio"]
    comparison.false_positive_rate = scenario["false_positive_rate"]
    comparison.false_negative_rate = scenario["false_negative_rate"]
    comparison.proximal_range = 50.0

    print(f"\n=== Running scenario: {scenario['name']} ===")
    results = comparison.run_comparison()

    detailed_path = scenario_dir / f"{scenario['name']}_results.json"
    comparison.save_results(str(detailed_path))

    evaluation = evaluate_methods(results, args.threshold)
    return results, evaluation


def main() -> None:
    args = parse_args()
    scenarios = load_scenarios(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    aggregated_results = {
        "benchmark_parameters": {
            "threshold": args.threshold,
            "supervised_model": str(args.supervised_model),
            "methods": ["paper_algorithm", "supervised_gnn"],
        },
        "scenarios": [],
    }

    for scenario in scenarios:
        results, evaluation = run_scenario(scenario, args, args.output_dir)
        summary.append(
            {
                "name": scenario["name"],
                "description": scenario["description"],
                "robot_density": scenario["robot_density"],
                "target_density_multiplier": scenario["target_density_multiplier"],
                "target_density": scenario["target_density"],
                "metrics": evaluation,
            }
        )
        scenario_record = dict(scenario)
        area = scenario_record["world_size"] ** 2
        scenario_record["derived_num_robots"] = int(round(scenario_record["robot_density"] * area))
        scenario_record["derived_num_targets"] = int(round(scenario_record["target_density"] * area))

        aggregated_results["scenarios"].append(
            {
                "name": scenario["name"],
                "description": scenario["description"],
                "parameters": scenario_record,
                "metrics": evaluation,
            }
        )

    metrics_path = args.output_dir / "comprehensive_metrics.json"
    metrics_path.write_text(json.dumps(aggregated_results, indent=2))
    print(f"\n✅ Aggregated metrics saved to {metrics_path}")

    plot_accuracy(summary, args.output_dir)
    print(f"📈 Accuracy summary plot saved to {args.output_dir / 'trust_method_accuracy_summary.png'}")


if __name__ == "__main__":
    main()
