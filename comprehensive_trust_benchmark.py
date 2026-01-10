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


# ============================================================================
# SCENARIO DEFINITIONS - Matching the 6 benchmark files
# ============================================================================
# This comprehensive benchmark includes:
#   - 10 In-sample scenarios (matching training distribution)
#   - 40 OOD scenarios across 4 dimensions:
#     * 10 higher FP rate (0.4-0.9)
#     * 10 higher FN rate (0.4-0.8)
#     * 10 higher adversarial ratio (0.4-0.5)
#     * 10 even higher adversarial ratio (0.5-0.8)
# Total: 50 scenarios
# ============================================================================

# Base parameters matching the 6 benchmark files
BASE_ROBOT_DENSITY = 0.0005  # ≈5 robots in 100x100 world
BASE_TARGET_DENSITY_MULTIPLIER = 2.0  # ≈10 targets (matches benchmark files)
BASE_TARGET_DENSITY = round(BASE_ROBOT_DENSITY * BASE_TARGET_DENSITY_MULTIPLIER, 8)
WORLD_SIZE_METERS = 100.0

# In-sample parameter ranges (matching in_sample_benchmark.py and generate_supervised_data.py)
ROBOT_DENSITY_RANGE = (0.0005, 0.0020)
TARGET_DENSITY_MULTIPLIER = 2.0
ADVERSARIAL_RATIO_RANGE = (0.1, 0.3)
FALSE_POSITIVE_RATE_RANGE = (0.1, 0.3)
FALSE_NEGATIVE_RATE_RANGE = (0.0, 0.3)

IN_SAMPLE_SCENARIOS: List[Dict] = []
# Sample 10 in-distribution scenarios with varied parameters
robot_density_values = [0.0006, 0.0008, 0.0010, 0.0012, 0.0015, 0.0018, 0.0007, 0.0009, 0.0011, 0.0013]
adversarial_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.15, 0.2, 0.25, 0.1, 0.3]
fp_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.2, 0.15, 0.3, 0.25, 0.1]
fn_values = [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.0, 0.3, 0.15, 0.25]

for idx, (rd, adv, fp, fn) in enumerate(zip(robot_density_values, adversarial_values, fp_values, fn_values)):
    IN_SAMPLE_SCENARIOS.append(
        make_scenario(
            name=f"in_sample_{idx+1:02d}",
            description=f"In-distribution scenario variant {idx+1}.",
            robot_density=rd,
            target_density_multiplier=TARGET_DENSITY_MULTIPLIER,
            adversarial_ratio=adv,
            false_positive_rate=fp,
            false_negative_rate=fn,
            num_timesteps=100,
            random_seed=1000 + idx * 37,
        )
    )


OOD_SCENARIOS: List[Dict] = []

# 1. Higher False Positive Rate (matching higher_false_positive_rate_benchmark.py)
# FP range: 0.4-0.9, keep other params in-sample
robot_density_values_fp = [0.0008, 0.0010, 0.0012, 0.0015, 0.0018, 0.0007, 0.0011, 0.0014, 0.0016, 0.0009]
fp_values_ood = [0.45, 0.5, 0.6, 0.7, 0.8, 0.55, 0.65, 0.75, 0.85, 0.9]

for idx, (rd, fp) in enumerate(zip(robot_density_values_fp, fp_values_ood)):
    # Sample adversarial and FN from in-sample ranges
    adv = 0.1 + (idx % 3) * 0.1  # 0.1, 0.2, 0.3
    fn = 0.0 if idx % 3 == 0 else (0.1 if idx % 3 == 1 else 0.2)
    OOD_SCENARIOS.append(
        make_scenario(
            name=f"ood_high_fp_{idx+1:02d}",
            description="OOD: Higher false positive rate (0.4-0.9).",
            robot_density=rd,
            target_density_multiplier=TARGET_DENSITY_MULTIPLIER,
            adversarial_ratio=adv,
            false_positive_rate=fp,
            false_negative_rate=fn,
            num_timesteps=100,
            random_seed=2000 + idx * 41,
        )
    )

# 2. Higher False Negative Rate (matching higher_false_negative_rate_benchmark.py)
# FN range: 0.4-0.8, keep other params in-sample
robot_density_values_fn = [0.0009, 0.0011, 0.0013, 0.0016, 0.0019, 0.0008, 0.0012, 0.0015, 0.0017, 0.0010]
fn_values_ood = [0.4, 0.45, 0.5, 0.6, 0.7, 0.5, 0.55, 0.65, 0.75, 0.8]

for idx, (rd, fn) in enumerate(zip(robot_density_values_fn, fn_values_ood)):
    # Sample adversarial and FP from in-sample ranges
    adv = 0.15 + (idx % 3) * 0.05  # 0.15, 0.2, 0.25
    fp = 0.15 + (idx % 4) * 0.05  # 0.15, 0.2, 0.25, 0.3
    OOD_SCENARIOS.append(
        make_scenario(
            name=f"ood_high_fn_{idx+1:02d}",
            description="OOD: Higher false negative rate (0.4-0.8).",
            robot_density=rd,
            target_density_multiplier=TARGET_DENSITY_MULTIPLIER,
            adversarial_ratio=adv,
            false_positive_rate=fp,
            false_negative_rate=fn,
            num_timesteps=100,
            random_seed=2500 + idx * 53,
        )
    )

# 3. Higher Adversarial Ratio (matching higher_adversarial_ratio_benchmark.py)
# Adversarial range: 0.4-0.5, keep other params in-sample
robot_density_values_adv = [0.0007, 0.0010, 0.0013, 0.0016, 0.0019, 0.0008, 0.0011, 0.0014, 0.0017, 0.0012]
adv_values_ood1 = [0.40, 0.42, 0.44, 0.46, 0.48, 0.41, 0.43, 0.45, 0.47, 0.50]

for idx, (rd, adv) in enumerate(zip(robot_density_values_adv, adv_values_ood1)):
    # Sample FP and FN from in-sample ranges
    fp = 0.15 + (idx % 3) * 0.05  # 0.15, 0.2, 0.25
    fn = 0.0 if idx % 4 == 0 else (0.1 if idx % 4 == 1 else (0.2 if idx % 4 == 2 else 0.3))
    OOD_SCENARIOS.append(
        make_scenario(
            name=f"ood_higher_adv_{idx+1:02d}",
            description="OOD: Higher adversarial ratio (0.4-0.5).",
            robot_density=rd,
            target_density_multiplier=TARGET_DENSITY_MULTIPLIER,
            adversarial_ratio=adv,
            false_positive_rate=fp,
            false_negative_rate=fn,
            num_timesteps=100,
            random_seed=3000 + idx * 61,
        )
    )

# 4. Even Higher Adversarial Ratio (matching even_higher_adversarial_ratio_benchmark.py)
# Adversarial range: 0.5-0.8, keep other params in-sample
robot_density_values_adv2 = [0.0006, 0.0009, 0.0012, 0.0015, 0.0018, 0.0007, 0.0010, 0.0013, 0.0016, 0.0011]
adv_values_ood2 = [0.50, 0.55, 0.60, 0.65, 0.70, 0.58, 0.62, 0.68, 0.75, 0.80]

for idx, (rd, adv) in enumerate(zip(robot_density_values_adv2, adv_values_ood2)):
    # Sample FP and FN from in-sample ranges
    fp = 0.2 + (idx % 3) * 0.05  # 0.2, 0.25, 0.3
    fn = 0.1 if idx % 3 == 0 else (0.15 if idx % 3 == 1 else 0.2)
    OOD_SCENARIOS.append(
        make_scenario(
            name=f"ood_even_higher_adv_{idx+1:02d}",
            description="OOD: Even higher adversarial ratio (0.5-0.8).",
            robot_density=rd,
            target_density_multiplier=TARGET_DENSITY_MULTIPLIER,
            adversarial_ratio=adv,
            false_positive_rate=fp,
            false_negative_rate=fn,
            num_timesteps=100,
            random_seed=3500 + idx * 67,
        )
    )

DEFAULT_SCENARIOS: List[Dict] = IN_SAMPLE_SCENARIOS + OOD_SCENARIOS

METHOD_ORDER = ["baseline", "bayesian", "paper", "supervised"]
METHOD_DISPLAY_NAMES = {
    "baseline": "Baseline (No Trust)",
    "bayesian": "Naïve Bayesian",
    "paper": "PSM Aggregation",
    "supervised": "NeST-Bayes",
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
        fov_range=80.0,
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
