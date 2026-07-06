#!/usr/bin/env python3
"""
Deceptive Policy Benchmark System for Multi-Robot Trust Methods

This benchmark tests trust methods against DECEPTIVE adversarial mode. 'deceptive'
runs the exact same MILP-based per-object report/ignore policy as 'optimized' mode
(see AdversarialRobot.generate_detections in robot_types.py - 'optimized' and
'deceptive' share the same code path) - it does NOT manipulate simulated track
trust/confidence values, since that would leak into every trust method's live
agent-trust computation (e.g. paper_trust_algorithm.py reads a robot's own reported
track trust as PSM evidence), making 'deceptive' diverge from 'optimized' far beyond
the intended scope.

Instead, 'deceptive' is evaluated with adversarial_lie=True (see evaluate_methods in
comprehensive_trust_benchmark.py / compute_object_metrics_with_adversarial_lies):
purely at object-metrics time, each adversarial robot's track trust is substituted
with a fresh lie before aggregation - GT objects lied low (0.0-0.2), FP-like objects
lied high (0.8-1.0) - without affecting the simulated robots' actual trust state.

Test scenarios:
- In-sample (training distribution)
- Higher adversarial ratio (0.4-0.5)
- Higher FP injection (0.4-0.5) - more persistent false hypotheses

Robot modes:
- Legitimate: realistic (natural sensor noise)
- Adversarial: deceptive (MILP-based report/ignore policy, same as optimized;
  lies only applied at object-metrics evaluation time)
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_trust_methods import TrustMethodComparison
from comprehensive_trust_benchmark import (
    evaluate_methods,
    print_summary_statistics,
)


# Benchmark configurations
@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark scenario type."""
    name: str
    robot_density_range: Tuple[float, float]
    target_density_multiplier: float
    adversarial_ratio_range: Tuple[float, float]
    adversarial_fp_injection_rate_range: Tuple[float, float]
    sensor_fp_rate: float = 0.05  # Sensor FP rate (transient)
    sensor_fn_rate: float = 0.05  # Sensor FN rate (transient)
    description: str = ""


# Define all benchmark types
BENCHMARK_CONFIGS = {
    "in_sample": BenchmarkConfig(
        name="in_sample",
        robot_density_range=(0.0005, 0.0020),
        target_density_multiplier=2.0,
        adversarial_ratio_range=(0.1, 0.3),
        adversarial_fp_injection_rate_range=(0.1, 0.3),
        sensor_fp_rate=0.05,
        sensor_fn_rate=0.05,
        description="In-sample (training distribution) - Deceptive policy"
    ),
    "higher_adv_ratio": BenchmarkConfig(
        name="higher_adv_ratio",
        robot_density_range=(0.0005, 0.0020),
        target_density_multiplier=2.0,
        adversarial_ratio_range=(0.4, 0.5),  # HIGHER
        adversarial_fp_injection_rate_range=(0.1, 0.3),
        sensor_fp_rate=0.05,
        sensor_fn_rate=0.05,
        description="Higher adversarial ratio (0.4-0.5) - Deceptive policy"
    ),
    "higher_fp_injection": BenchmarkConfig(
        name="higher_fp_injection",
        robot_density_range=(0.0005, 0.0020),
        target_density_multiplier=2.0,
        adversarial_ratio_range=(0.1, 0.3),
        adversarial_fp_injection_rate_range=(0.4, 0.5),  # HIGHER (more persistent FP hypotheses)
        sensor_fp_rate=0.05,
        sensor_fn_rate=0.05,
        description="Higher adversarial FP injection rate (0.4-0.5) - Deceptive policy"
    ),
}


# Simulation constants
WORLD_SIZE = 100.0
NUM_TIMESTEPS = 100
PROXIMAL_RANGE = 80.0
FOV_RANGE = 50.0
FOV_ANGLE = np.pi / 3

# Robot modes
LEGITIMATE_MODE = 'realistic'  # Natural sensor noise
ADVERSARIAL_MODE = 'deceptive'  # Policy-based strategic attacks + trust manipulation

METHOD_ORDER = ["baseline", "bayesian", "paper", "supervised"]
METHOD_DISPLAY_NAMES = {
    "baseline": "Baseline (No Trust)",
    "bayesian": "Naïve Bayesian",
    "paper": "PSM Aggregation",
    "supervised": "NeST-Bayes",
}


def sample_scenario_parameters(
    scenario_idx: int,
    base_seed: int,
    config: BenchmarkConfig
) -> Dict:
    """Sample parameters from the specified benchmark configuration.

    Args:
        scenario_idx: Index of the scenario
        base_seed: Base seed for this run (allows reproducibility)
        config: Benchmark configuration defining parameter ranges

    Returns:
        Dictionary with scenario parameters including a unique random_seed
    """
    # Create scenario-specific seed from base seed
    scenario_seed = base_seed + scenario_idx * 1000

    random.seed(scenario_seed)
    np.random.seed(scenario_seed)

    # Sample robot density (increment: 0.0001)
    robot_density = round(
        random.uniform(*config.robot_density_range), 4
    )

    # Sample adversarial ratio (increment: 0.05)
    min_adv, max_adv = config.adversarial_ratio_range
    adv_steps = int((max_adv - min_adv) / 0.05)
    adv_step = random.randint(0, adv_steps)
    adversarial_ratio = round(min_adv + (adv_step * 0.05), 2)

    # Sample adversarial FP injection rate (increment: 0.05)
    min_fp, max_fp = config.adversarial_fp_injection_rate_range
    fp_steps = int((max_fp - min_fp) / 0.05)
    fp_step = random.randint(0, fp_steps)
    adversarial_fp_injection_rate = round(min_fp + (fp_step * 0.05), 2)

    # Note: eta_f and eta_r are legacy parameters, no longer used
    # Deceptive mode now uses objective-driven policy

    target_density = round(
        robot_density * config.target_density_multiplier, 8
    )

    # Generate simulation random seed (will be same for all methods)
    simulation_seed = random.randint(1000, 999999)

    return {
        "name": f"{config.name}_{scenario_idx:03d}",
        "benchmark_type": config.name,
        "robot_density": robot_density,
        "target_density": target_density,
        "target_density_multiplier": config.target_density_multiplier,
        "adversarial_ratio": adversarial_ratio,
        "adversarial_fp_injection_rate": adversarial_fp_injection_rate,
        "adversarial_fn_suppression_rate": 0.0,  # Not used in deceptive mode (policy-based instead)
        "sensor_fp_rate": config.sensor_fp_rate,
        "sensor_fn_rate": config.sensor_fn_rate,
        "random_seed": simulation_seed,
        "legitimate_mode": LEGITIMATE_MODE,
        "adversarial_mode": ADVERSARIAL_MODE,
    }


def run_scenario(
    scenario: Dict,
    supervised_model_path: str,
    threshold: float,
    adversarial_lie: bool = True  # Default TRUE for deceptive mode
) -> Dict:
    """Run a single scenario and return evaluation metrics.

    Args:
        scenario: Scenario parameters
        supervised_model_path: Path to supervised model checkpoint
        threshold: Trust threshold for binary classification
        adversarial_lie: If True, use adversarial track lies evaluation
                        (should be True for deceptive mode to match the trust manipulation)

    Returns:
        Dictionary with evaluation metrics for all methods
    """
    comparison = TrustMethodComparison(
        supervised_model_path=supervised_model_path,
        robot_density=scenario["robot_density"],
        target_density_multiplier=scenario["target_density_multiplier"],
        num_timesteps=NUM_TIMESTEPS,
        random_seed=scenario["random_seed"],
        world_size=WORLD_SIZE,
        fov_range=FOV_RANGE,
        fov_angle=FOV_ANGLE,
        proximal_range=PROXIMAL_RANGE,
        allow_fp_codetection=True,  # Allow adversarial robots to co-detect FPs
        legitimate_mode=scenario.get("legitimate_mode", LEGITIMATE_MODE),
        adversarial_mode=scenario.get("adversarial_mode", ADVERSARIAL_MODE),
    )
    comparison.adversarial_ratio = scenario["adversarial_ratio"]
    comparison.adversarial_fp_injection_rate = scenario["adversarial_fp_injection_rate"]
    comparison.adversarial_fn_suppression_rate = scenario["adversarial_fn_suppression_rate"]
    comparison.sensor_fp_rate = scenario["sensor_fp_rate"]
    comparison.sensor_fn_rate = scenario["sensor_fn_rate"]

    # Note: eta_f and eta_r are no longer used (legacy parameters)
    # Deceptive mode now uses objective-driven policy

    results = comparison.run_comparison()
    evaluation = evaluate_methods(results, threshold=threshold, adversarial_lie=adversarial_lie, object_threshold=threshold)
    return evaluation


def run_benchmark(
    config: BenchmarkConfig,
    num_scenarios: int,
    supervised_model_path: str,
    threshold: float,
    base_seed: int,
    adversarial_lie: bool = True  # Default TRUE for deceptive mode
) -> List[Dict]:
    """Run a complete benchmark with the specified configuration.

    Args:
        config: Benchmark configuration
        num_scenarios: Number of scenarios to run
        supervised_model_path: Path to supervised model checkpoint
        threshold: Trust threshold for binary classification
        base_seed: Base random seed for reproducibility
        adversarial_lie: If True, use adversarial track lies evaluation

    Returns:
        List of evaluation results for all scenarios
    """
    print(f"\n{'=' * 80}")
    print(f"Running {config.name.upper()} Benchmark (Deceptive Policy)")
    print(f"{'=' * 80}")
    print(f"Description: {config.description}")
    print(f"Robot modes: Legitimate={LEGITIMATE_MODE}, Adversarial={ADVERSARIAL_MODE}")
    print(f"Adversarial trust manipulation: GT objects (0.0-0.2), FP objects (0.8-1.0)")
    print(f"Parameter ranges:")
    print(f"  Robot density: {config.robot_density_range}")
    print(f"  Adversarial ratio: {config.adversarial_ratio_range}")
    print(f"  Adversarial FP injection rate: {config.adversarial_fp_injection_rate_range} (persistent)")
    print(f"  Sensor FP/FN rates: {config.sensor_fp_rate}/{config.sensor_fn_rate} (transient)")
    print(f"Running {num_scenarios} scenarios...")
    print(f"{'=' * 80}\n")

    all_results = []
    for i in range(num_scenarios):
        scenario = sample_scenario_parameters(i, base_seed, config)
        evaluation = run_scenario(
            scenario, supervised_model_path, threshold, adversarial_lie
        )
        all_results.append({
            "scenario": scenario,
            "evaluation": evaluation
        })

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_scenarios} scenarios")

    return all_results


def save_results(
    results: List[Dict],
    output_dir: Path,
    config: BenchmarkConfig,
    adversarial_lie: bool = True,
    base_seed: int = None
):
    """Save benchmark results to JSON files.

    Saves two files:
    1. Evaluation-only results (backward compatible)
    2. Detailed results with scenario parameters (for analysis)

    Args:
        results: List of benchmark results
        output_dir: Directory to save results
        config: Benchmark configuration
        adversarial_lie: Whether adversarial lies were used
        base_seed: Base random seed used for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine base filename
    if adversarial_lie:
        base_filename = f"deceptive_{config.name}"  # Deceptive mode always uses lies
    else:
        base_filename = f"deceptive_no_lie_{config.name}"

    # File 1: Evaluation-only results (backward compatible)
    eval_path = output_dir / f"{base_filename}_results.json"
    evaluation_results = [r["evaluation"] for r in results]
    with open(eval_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    # File 2: Detailed results with scenario parameters (for analysis)
    detailed_path = output_dir / f"{base_filename}_detailed.json"
    detailed_results = {
        "metadata": {
            "benchmark_type": config.name,
            "adversarial_mode": ADVERSARIAL_MODE,
            "description": config.description,
            "num_scenarios": len(results),
            "base_seed": base_seed,
            "robot_modes": {
                "legitimate": LEGITIMATE_MODE,
                "adversarial": ADVERSARIAL_MODE
            },
            "adversarial_lie": adversarial_lie,
            "trust_manipulation": {
                "gt_objects": "0.0-0.2 (lie low)",
                "fp_objects": "0.8-1.0 (lie high)"
            },
            "parameter_ranges": {
                "robot_density": config.robot_density_range,
                "adversarial_ratio": config.adversarial_ratio_range,
                "adversarial_fp_injection_rate": config.adversarial_fp_injection_rate_range,
                "sensor_fp_rate": config.sensor_fp_rate,
                "sensor_fn_rate": config.sensor_fn_rate,
            },
            "simulation_constants": {
                "world_size": WORLD_SIZE,
                "num_timesteps": NUM_TIMESTEPS,
                "proximal_range": PROXIMAL_RANGE,
                "fov_range": FOV_RANGE,
                "fov_angle": FOV_ANGLE,
            }
        },
        "scenarios": [
            {
                "scenario_index": i,
                "parameters": r["scenario"],
                "evaluation": r["evaluation"]
            }
            for i, r in enumerate(results)
        ]
    }
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\n✅ Results saved:")
    print(f"   Evaluation only: {eval_path}")
    print(f"   Detailed (with parameters): {detailed_path}")


def print_benchmark_summary(results: List[Dict], config: BenchmarkConfig):
    """Print summary statistics for the benchmark.

    Args:
        results: List of benchmark results
        config: Benchmark configuration
    """
    # Convert to format expected by print_summary_statistics
    summary = []
    for i, result in enumerate(results):
        summary.append({
            "name": result["scenario"]["name"],
            "metrics": result["evaluation"]
        })

    print(f"\n{'=' * 80}")
    print(f"{config.name.upper()} BENCHMARK SUMMARY (Deceptive Policy)")
    print(f"{'=' * 80}")
    print_summary_statistics(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Deceptive policy benchmark for multi-robot trust methods"
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=list(BENCHMARK_CONFIGS.keys()) + ["all"],
        default="all",
        help="Which benchmark to run (default: all)"
    )

    # Scenario parameters
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=100,
        help="Number of test scenarios to run (default: 100)"
    )

    # Model and output
    parser.add_argument(
        "--supervised-model",
        type=Path,
        default=Path("supervised_trust_model.pth"),
        help="Path to supervised model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory to save results"
    )

    # Trust parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Trust threshold for binary classification (default: 0.3, consistent across all benchmarks)"
    )

    # Adversarial behavior
    parser.add_argument(
        "--no-adversarial-lie",
        action="store_true",
        help="Disable adversarial track lies evaluation (default: enabled for deceptive mode)"
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for reproducibility (default: random)"
    )

    args = parser.parse_args()

    # Generate base seed
    if args.seed is not None:
        base_seed = args.seed
    else:
        base_seed = random.randint(1, 999999)

    # Deceptive mode should use adversarial lies by default
    adversarial_lie = not args.no_adversarial_lie

    print(f"\n{'=' * 80}")
    print("DECEPTIVE POLICY MULTI-ROBOT TRUST BENCHMARK")
    print(f"{'=' * 80}")
    print(f"Base seed: {base_seed} (use --seed {base_seed} to reproduce)")
    print(f"Robot modes:")
    print(f"  Legitimate: {LEGITIMATE_MODE} (natural sensor noise)")
    print(f"  Adversarial: {ADVERSARIAL_MODE} (policy-based attacks + trust manipulation)")
    print(f"Adversarial trust manipulation:")
    print(f"  GT objects: lie low (0.0-0.2) - make them seem less credible")
    print(f"  FP objects: lie high (0.8-1.0) - make them seem more credible")
    if adversarial_lie:
        print(f"  Evaluation mode: ENABLED (matches trust manipulation)")
    else:
        print(f"  Evaluation mode: DISABLED (use --no-adversarial-lie to disable)")
    print(f"{'=' * 80}")

    # Determine which benchmarks to run
    if args.benchmark == "all":
        benchmarks_to_run = list(BENCHMARK_CONFIGS.keys())
    else:
        benchmarks_to_run = [args.benchmark]

    # Run each benchmark
    all_benchmark_results = {}
    for benchmark_name in benchmarks_to_run:
        config = BENCHMARK_CONFIGS[benchmark_name]

        # Run benchmark
        results = run_benchmark(
            config=config,
            num_scenarios=args.num_scenarios,
            supervised_model_path=str(args.supervised_model),
            threshold=args.threshold,
            base_seed=base_seed,
            adversarial_lie=adversarial_lie
        )

        # Save results
        save_results(results, args.output_dir, config, adversarial_lie, base_seed)

        # Print summary
        print_benchmark_summary(results, config)

        # Store for later
        all_benchmark_results[benchmark_name] = results

    # Final summary
    print(f"\n{'=' * 80}")
    print("ALL BENCHMARKS COMPLETED (Deceptive Policy)")
    print(f"{'=' * 80}")
    print(f"Benchmarks run: {', '.join(benchmarks_to_run)}")
    print(f"Scenarios per benchmark: {args.num_scenarios}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
