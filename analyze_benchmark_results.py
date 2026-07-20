#!/usr/bin/env python3
"""
Benchmark Results Analysis Script

This script helps analyze the detailed results from unified_benchmark.py
and generate visualizations and statistics.

Features:
- Analyze single or multiple benchmark result files
- Compare performance across different benchmark configurations
- Generate comprehensive visualizations and statistics
- Export data to CSV for further analysis
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob


def load_detailed_results(filepath: Path) -> Dict:
    """Load detailed benchmark results from JSON file.

    Args:
        filepath: Path to detailed results JSON file

    Returns:
        Dictionary with metadata and scenarios
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_metrics_dataframe(results: Dict, method: str = "supervised") -> pd.DataFrame:
    """Extract metrics into a pandas DataFrame for analysis.

    Args:
        results: Detailed results dictionary
        method: Which method to analyze (supervised, paper, bayesian, baseline)

    Returns:
        DataFrame with scenario parameters and metrics
    """
    rows = []

    for scenario in results["scenarios"]:
        params = scenario["parameters"]
        metrics = scenario["evaluation"][method]

        # Handle different metric structures (some have robot_metrics/object_metrics, some have robots/objects)
        robot_metrics = metrics.get("robot_metrics", metrics.get("robots", {}))
        object_metrics = metrics.get("object_metrics", metrics.get("objects", {}))

        row = {
            # Scenario parameters
            "scenario_index": scenario["scenario_index"],
            "scenario_name": params["name"],
            "benchmark_type": params.get("benchmark_type", "unknown"),
            "robot_density": params["robot_density"],
            "adversarial_ratio": params["adversarial_ratio"],
            "adversarial_fp_injection_rate": params.get("adversarial_fp_injection_rate", params.get("false_positive_rate", 0.0)),
            "adversarial_fn_suppression_rate": params.get("adversarial_fn_suppression_rate", params.get("false_negative_rate", 0.0)),
            "sensor_fp_rate": params.get("sensor_fp_rate", 0.05),
            "sensor_fn_rate": params.get("sensor_fn_rate", 0.05),
            "legitimate_mode": params.get("legitimate_mode", "optimal"),
            "adversarial_mode": params.get("adversarial_mode", "normal"),
            "random_seed": params["random_seed"],

            # Robot classification metrics
            "robot_accuracy": robot_metrics.get("accuracy", 0.0),
            "robot_precision": robot_metrics.get("precision", 0.0),
            "robot_recall": robot_metrics.get("recall", 0.0),
            "robot_f1": robot_metrics.get("f1", 0.0),

            # Object detection metrics
            "object_accuracy": object_metrics.get("accuracy", 0.0),
            "object_precision": object_metrics.get("precision", 0.0),
            "object_recall": object_metrics.get("recall", 0.0),
            "object_f1": object_metrics.get("f1", 0.0),

            # Statistics (handle different naming conventions)
            "mean_legit_trust": robot_metrics.get("mean_legitimate_trust", 0.0),
            "mean_adv_trust": robot_metrics.get("mean_adversarial_trust", 0.0),
            "mean_true_obj_trust": object_metrics.get("mean_true_object_trust", 0.0),
            "mean_false_obj_trust": object_metrics.get("mean_false_object_trust", 0.0),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def print_summary_statistics(df: pd.DataFrame, method: str):
    """Print summary statistics for a method.

    Args:
        df: DataFrame with metrics
        method: Method name
    """
    print(f"\n{'=' * 80}")
    print(f"SUMMARY STATISTICS - {method.upper()}")
    print(f"{'=' * 80}\n")

    # Robot classification - concise version
    print("ROBOT CLASSIFICATION:")
    print(f"  Accuracy:  {df['robot_accuracy'].mean():.4f} ± {df['robot_accuracy'].std():.4f}")
    print(f"  Precision: {df['robot_precision'].mean():.4f} ± {df['robot_precision'].std():.4f} (Adv. identified / Total flagged as adv.)")
    print(f"  Recall:    {df['robot_recall'].mean():.4f} ± {df['robot_recall'].std():.4f} (Adv. identified / Total actual adv.)")

    # Object detection - concise version
    print("\nOBJECT DETECTION:")
    print(f"  Accuracy:  {df['object_accuracy'].mean():.4f} ± {df['object_accuracy'].std():.4f}")
    print(f"  Precision: {df['object_precision'].mean():.4f} ± {df['object_precision'].std():.4f} (True obj. / Total accepted)")
    print(f"  Recall:    {df['object_recall'].mean():.4f} ± {df['object_recall'].std():.4f} (True obj. / Total actual true)")


def plot_metrics_vs_parameters(df: pd.DataFrame, output_dir: Path, method: str, prefix: str = ""):
    """Plot metrics vs scenario parameters.

    Args:
        df: DataFrame with metrics
        output_dir: Directory to save plots
        method: Method name
        prefix: Prefix for output filenames (e.g., benchmark name)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parameters = ["adversarial_ratio", "adversarial_fp_injection_rate", "adversarial_fn_suppression_rate"]
    param_labels = {
        "adversarial_ratio": "Adversarial Ratio",
        "adversarial_fp_injection_rate": "Adversarial FP Injection Rate",
        "adversarial_fn_suppression_rate": "Adversarial FN Suppression Rate"
    }

    metrics = {
        "Robot F1": "robot_f1",
        "Object F1": "object_f1",
        "Robot Accuracy": "robot_accuracy",
        "Object Accuracy": "object_accuracy"
    }

    for param in parameters:
        # Skip if parameter has no variation
        if df[param].nunique() <= 1:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{method.upper()} - Metrics vs {param_labels[param]}")

        for idx, (metric_name, metric_col) in enumerate(metrics.items()):
            ax = axes[idx // 2, idx % 2]

            # Scatter plot
            ax.scatter(df[param], df[metric_col], alpha=0.5)

            # Trend line
            z = np.polyfit(df[param], df[metric_col], 1)
            p = np.poly1d(z)
            ax.plot(df[param], p(df[param]), "r--", alpha=0.8, label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}")

            ax.set_xlabel(param_labels[param])
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"{prefix}{method}_{param}_analysis.png" if prefix else f"{method}_{param}_analysis.png"
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()

        print(f"✓ Saved: {output_dir / filename}")


def compare_methods(results: Dict, output_dir: Path):
    """Compare all methods side by side.

    Args:
        results: Detailed results dictionary
        output_dir: Directory to save plots
    """
    # Discover methods from the file (ablation benchmark -> variant names; otherwise the
    # standard baseline/bayesian/paper/supervised set).
    methods = discover_methods(results)

    # Extract metrics for each method
    method_dfs = {}
    for method in methods:
        method_dfs[method] = extract_metrics_dataframe(results, method)

    # Compare robot F1 scores
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = np.arange(len(methods))
    robot_f1_means = [method_dfs[m]["robot_f1"].mean() for m in methods]
    robot_f1_stds = [method_dfs[m]["robot_f1"].std() for m in methods]

    ax.bar(positions, robot_f1_means, yerr=robot_f1_stds, capsize=5)
    ax.set_xticks(positions)
    ax.set_xticklabels([m.title() for m in methods])
    ax.set_ylabel("Robot F1 Score")
    ax.set_title("Robot Classification F1 Score Comparison")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison_robot_f1.png", dpi=150)
    plt.close()

    print(f"✓ Saved: {output_dir / 'method_comparison_robot_f1.png'}")

    # Compare object F1 scores
    fig, ax = plt.subplots(figsize=(10, 6))

    object_f1_means = [method_dfs[m]["object_f1"].mean() for m in methods]
    object_f1_stds = [method_dfs[m]["object_f1"].std() for m in methods]

    ax.bar(positions, object_f1_means, yerr=object_f1_stds, capsize=5)
    ax.set_xticks(positions)
    ax.set_xticklabels([m.title() for m in methods])
    ax.set_ylabel("Object F1 Score")
    ax.set_title("Object Detection F1 Score Comparison")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison_object_f1.png", dpi=150)
    plt.close()

    print(f"✓ Saved: {output_dir / 'method_comparison_object_f1.png'}")


def export_to_csv(results: Dict, output_dir: Path, prefix: str = ""):
    """Export results to CSV files for further analysis.

    Args:
        results: Detailed results dictionary
        output_dir: Directory to save CSV files
        prefix: Prefix for output filenames (e.g., benchmark name)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["baseline", "bayesian", "paper", "supervised"]

    for method in methods:
        df = extract_metrics_dataframe(results, method)
        filename = f"{prefix}{method}_metrics.csv" if prefix else f"{method}_metrics.csv"
        csv_path = output_dir / filename
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")


def discover_methods(results: Dict) -> List[str]:
    """
    Determine which methods a benchmark file contains for `--method all`.

    Prefers an explicit metadata["methods"] list (used by the ablation benchmark, whose
    "methods" are ablation variant names like no_gat/homogeneous rather than the fixed
    baseline/bayesian/paper/supervised). Falls back to the keys actually present in the
    first scenario's evaluation, then to the four standard methods.
    """
    meta_methods = results.get("metadata", {}).get("methods")
    if meta_methods:
        return list(meta_methods)
    scenarios = results.get("scenarios", [])
    if scenarios and "evaluation" in scenarios[0]:
        keys = list(scenarios[0]["evaluation"].keys())
        if keys:
            return keys
    return ["baseline", "bayesian", "paper", "supervised"]


def find_benchmark_files(benchmark_dir: Path) -> Dict[str, List[Path]]:
    """Find all benchmark result files and group them by configuration.

    Args:
        benchmark_dir: Directory containing benchmark results

    Returns:
        Dictionary mapping benchmark names to their detailed result files
    """
    detailed_files = list(benchmark_dir.glob("*_detailed.json"))

    # Group by benchmark configuration
    benchmarks = {}
    for file in detailed_files:
        # Extract benchmark name (remove _detailed.json suffix)
        name = file.stem.replace("_detailed", "")
        benchmarks[name] = file

    return benchmarks


def load_all_benchmarks(benchmark_files: Dict[str, Path]) -> Dict[str, Dict]:
    """Load all benchmark result files.

    Args:
        benchmark_files: Dictionary mapping benchmark names to file paths

    Returns:
        Dictionary mapping benchmark names to their loaded results
    """
    all_results = {}
    for name, filepath in benchmark_files.items():
        print(f"  Loading: {name}...")
        all_results[name] = load_detailed_results(filepath)

    return all_results


def compare_benchmarks_across_methods(all_results: Dict[str, Dict], output_dir: Path, method: str = "supervised"):
    """Compare a specific method's performance across different benchmark configurations.

    Args:
        all_results: Dictionary of all loaded benchmark results
        output_dir: Directory to save plots
        method: Which method to compare across benchmarks
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract dataframes for each benchmark
    benchmark_dfs = {}
    for benchmark_name, results in all_results.items():
        benchmark_dfs[benchmark_name] = extract_metrics_dataframe(results, method)

    # Prepare data for comparison
    benchmark_names = []
    robot_f1_means = []
    robot_f1_stds = []
    object_f1_means = []
    object_f1_stds = []
    robot_acc_means = []
    robot_acc_stds = []
    object_acc_means = []
    object_acc_stds = []

    for name, df in benchmark_dfs.items():
        benchmark_names.append(name.replace("_", "\n"))
        robot_f1_means.append(df["robot_f1"].mean())
        robot_f1_stds.append(df["robot_f1"].std())
        object_f1_means.append(df["object_f1"].mean())
        object_f1_stds.append(df["object_f1"].std())
        robot_acc_means.append(df["robot_accuracy"].mean())
        robot_acc_stds.append(df["robot_accuracy"].std())
        object_acc_means.append(df["object_accuracy"].mean())
        object_acc_stds.append(df["object_accuracy"].std())

    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{method.upper()} - Performance Across Benchmark Configurations", fontsize=16)

    positions = np.arange(len(benchmark_names))
    width = 0.6

    # Robot F1
    ax = axes[0, 0]
    bars = ax.bar(positions, robot_f1_means, width, yerr=robot_f1_stds, capsize=5, alpha=0.7)
    ax.set_ylabel("Robot F1 Score", fontsize=12)
    ax.set_title("Robot Classification F1 Score")
    ax.set_xticks(positions)
    ax.set_xticklabels(benchmark_names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    # Color bars by performance
    for i, bar in enumerate(bars):
        if robot_f1_means[i] > 0.8:
            bar.set_color('green')
        elif robot_f1_means[i] > 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    # Object F1
    ax = axes[0, 1]
    bars = ax.bar(positions, object_f1_means, width, yerr=object_f1_stds, capsize=5, alpha=0.7)
    ax.set_ylabel("Object F1 Score", fontsize=12)
    ax.set_title("Object Detection F1 Score")
    ax.set_xticks(positions)
    ax.set_xticklabels(benchmark_names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    for i, bar in enumerate(bars):
        if object_f1_means[i] > 0.8:
            bar.set_color('green')
        elif object_f1_means[i] > 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    # Robot Accuracy
    ax = axes[1, 0]
    bars = ax.bar(positions, robot_acc_means, width, yerr=robot_acc_stds, capsize=5, alpha=0.7)
    ax.set_ylabel("Robot Accuracy", fontsize=12)
    ax.set_title("Robot Classification Accuracy")
    ax.set_xticks(positions)
    ax.set_xticklabels(benchmark_names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    for i, bar in enumerate(bars):
        if robot_acc_means[i] > 0.8:
            bar.set_color('green')
        elif robot_acc_means[i] > 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    # Object Accuracy
    ax = axes[1, 1]
    bars = ax.bar(positions, object_acc_means, width, yerr=object_acc_stds, capsize=5, alpha=0.7)
    ax.set_ylabel("Object Accuracy", fontsize=12)
    ax.set_title("Object Detection Accuracy")
    ax.set_xticks(positions)
    ax.set_xticklabels(benchmark_names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    for i, bar in enumerate(bars):
        if object_acc_means[i] > 0.8:
            bar.set_color('green')
        elif object_acc_means[i] > 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.tight_layout()
    plt.savefig(output_dir / f"{method}_cross_benchmark_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_dir / f'{method}_cross_benchmark_comparison.png'}")


def create_summary_table(all_results: Dict[str, Dict], output_dir: Path):
    """Create comprehensive summary tables comparing all methods across all benchmarks.

    Creates separate tables for:
    - Normal mode benchmarks (realistic + normal)
    - Optimized mode benchmarks (realistic + optimized)
    - Deceptive mode benchmarks (realistic + deceptive)
    - Combined table with all benchmarks

    Args:
        all_results: Dictionary of all loaded benchmark results
        output_dir: Directory to save the tables
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["baseline", "bayesian", "paper", "supervised"]

    # Group benchmarks by type
    normal_benchmarks = {}
    optimized_benchmarks = {}
    deceptive_benchmarks = {}

    for benchmark_name, results in all_results.items():
        if benchmark_name.startswith("deceptive_"):
            deceptive_benchmarks[benchmark_name] = results
        elif benchmark_name.startswith("optimized_"):
            optimized_benchmarks[benchmark_name] = results
        else:
            normal_benchmarks[benchmark_name] = results

    def create_table_for_group(benchmarks_dict: Dict[str, Dict], group_name: str) -> pd.DataFrame:
        """Create summary table for a specific group of benchmarks."""
        summary_rows = []

        for benchmark_name, results in benchmarks_dict.items():
            for method in methods:
                df = extract_metrics_dataframe(results, method)

                summary_rows.append({
                    "Benchmark": benchmark_name,
                    "Method": method,
                    "Robot Prec (mean±std)": f"{df['robot_precision'].mean():.3f}±{df['robot_precision'].std():.3f}",
                    "Robot Rec (mean±std)": f"{df['robot_recall'].mean():.3f}±{df['robot_recall'].std():.3f}",
                    "Robot Acc (mean±std)": f"{df['robot_accuracy'].mean():.3f}±{df['robot_accuracy'].std():.3f}",
                    "Object Prec (mean±std)": f"{df['object_precision'].mean():.3f}±{df['object_precision'].std():.3f}",
                    "Object Rec (mean±std)": f"{df['object_recall'].mean():.3f}±{df['object_recall'].std():.3f}",
                    "Object Acc (mean±std)": f"{df['object_accuracy'].mean():.3f}±{df['object_accuracy'].std():.3f}",
                })

        return pd.DataFrame(summary_rows)

    # Create and save tables for each group
    if normal_benchmarks:
        print(f"\n{'=' * 120}")
        print("NORMAL MODE BENCHMARKS (realistic + normal adversarial)")
        print(f"{'=' * 120}\n")
        normal_df = create_table_for_group(normal_benchmarks, "Normal")
        csv_path = output_dir / "summary_normal_mode.csv"
        normal_df.to_csv(csv_path, index=False)
        print(normal_df.to_string(index=False))
        print(f"\n✓ Saved: {csv_path}\n")

    if optimized_benchmarks:
        print(f"\n{'=' * 120}")
        print("OPTIMIZED POLICY BENCHMARKS (realistic + optimized adversarial)")
        print(f"{'=' * 120}\n")
        optimized_df = create_table_for_group(optimized_benchmarks, "Optimized")
        csv_path = output_dir / "summary_optimized_mode.csv"
        optimized_df.to_csv(csv_path, index=False)
        print(optimized_df.to_string(index=False))
        print(f"\n✓ Saved: {csv_path}\n")

    if deceptive_benchmarks:
        print(f"\n{'=' * 120}")
        print("DECEPTIVE POLICY BENCHMARKS (realistic + deceptive adversarial)")
        print(f"{'=' * 120}\n")
        deceptive_df = create_table_for_group(deceptive_benchmarks, "Deceptive")
        csv_path = output_dir / "summary_deceptive_mode.csv"
        deceptive_df.to_csv(csv_path, index=False)
        print(deceptive_df.to_string(index=False))
        print(f"\n✓ Saved: {csv_path}\n")

    # Create combined table
    print(f"\n{'=' * 120}")
    print("COMPREHENSIVE SUMMARY - ALL BENCHMARKS & METHODS")
    print(f"{'=' * 120}\n")

    all_rows = []
    if normal_benchmarks:
        all_rows.extend(create_table_for_group(normal_benchmarks, "Normal").to_dict('records'))
    if optimized_benchmarks:
        all_rows.extend(create_table_for_group(optimized_benchmarks, "Optimized").to_dict('records'))
    if deceptive_benchmarks:
        all_rows.extend(create_table_for_group(deceptive_benchmarks, "Deceptive").to_dict('records'))

    summary_df = pd.DataFrame(all_rows)
    csv_path = output_dir / "comprehensive_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\n✓ Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results - defaults to analyzing all files in benchmark_results/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all benchmarks (default behavior)
  python analyze_benchmark_results.py

  # Analyze all benchmarks with all methods and export CSV
  python analyze_benchmark_results.py --method all --export-csv

  # Analyze a single benchmark file
  python analyze_benchmark_results.py benchmark_results/in_sample_detailed.json --single-file

  # Analyze all benchmarks in a custom directory
  python analyze_benchmark_results.py custom_results/ --method all
        """
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        type=Path,
        default=Path("benchmark_results"),
        help="Path to detailed results JSON file or directory (default: benchmark_results/)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results"),
        help="Directory to save analysis outputs (default: analysis_results)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="supervised",
        choices=["baseline", "bayesian", "paper", "supervised", "all"],
        help="Which method to analyze (default: supervised)"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export metrics to CSV files"
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Force single-file analysis mode (for analyzing one specific benchmark file)"
    )

    args = parser.parse_args()

    # Determine if we're processing multiple files or a single file
    # Default behavior: multi-file if directory, unless --single-file is specified
    if args.results_file.is_dir() and not args.single_file:
        # Multi-file analysis mode
        print(f"{'=' * 80}")
        print("MULTI-BENCHMARK ANALYSIS MODE")
        print(f"{'=' * 80}\n")

        # Find all benchmark files
        benchmark_dir = args.results_file if args.results_file.is_dir() else args.results_file.parent
        print(f"Searching for benchmark files in: {benchmark_dir}")
        benchmark_files = find_benchmark_files(benchmark_dir)
        print(f"Found {len(benchmark_files)} benchmark configurations:\n")
        for name in sorted(benchmark_files.keys()):
            print(f"  - {name}")
        print()

        # Load all benchmarks
        print("Loading all benchmark results...")
        all_results = load_all_benchmarks(benchmark_files)
        print(f"✓ Loaded {len(all_results)} benchmarks\n")

        # Determine methods to analyze
        if args.method == "all":
            # Union of methods across all loaded benchmarks (handles ablation variant names).
            methods_to_analyze = []
            for res in all_results.values():
                for m in discover_methods(res):
                    if m not in methods_to_analyze:
                        methods_to_analyze.append(m)
            if not methods_to_analyze:
                methods_to_analyze = ["baseline", "bayesian", "paper", "supervised"]
        else:
            methods_to_analyze = [args.method]

        # Analyze each benchmark individually
        for benchmark_name, results in sorted(all_results.items()):
            print(f"\n{'=' * 80}")
            print(f"ANALYZING BENCHMARK: {benchmark_name.upper()}")
            print(f"{'=' * 80}")

            # Print metadata
            print(f"\nBenchmark type: {results['metadata']['benchmark_type']}")
            print(f"Description: {results['metadata']['description']}")
            print(f"Number of scenarios: {results['metadata']['num_scenarios']}")
            print(f"Robot modes: {results['metadata']['robot_modes']}")
            print(f"Adversarial lie: {results['metadata']['adversarial_lie']}")

            # Create subdirectory for this benchmark
            benchmark_output_dir = args.output_dir / benchmark_name

            # Analyze each method
            for method in methods_to_analyze:
                print(f"\n  Method: {method.upper()}")
                df = extract_metrics_dataframe(results, method)
                print_summary_statistics(df, method)
                plot_metrics_vs_parameters(df, benchmark_output_dir, method)

            # Compare methods within this benchmark
            if args.method == "all":
                print(f"\n  Comparing all methods for {benchmark_name}...")
                compare_methods(results, benchmark_output_dir)

            # Export CSV for this benchmark
            if args.export_csv:
                export_to_csv(results, benchmark_output_dir / "csv", prefix=f"{benchmark_name}_")

        # Cross-benchmark comparisons
        print(f"\n{'=' * 80}")
        print("CROSS-BENCHMARK COMPARISONS")
        print(f"{'=' * 80}\n")

        for method in methods_to_analyze:
            print(f"Creating cross-benchmark comparison for {method.upper()}...")
            compare_benchmarks_across_methods(all_results, args.output_dir, method)

        # Create comprehensive summary table
        print("\nCreating comprehensive summary table...")
        create_summary_table(all_results, args.output_dir)

    else:
        # Single-file analysis mode
        print(f"{'=' * 80}")
        print("SINGLE-BENCHMARK ANALYSIS MODE")
        print(f"{'=' * 80}\n")

        # Load results
        print(f"Loading results from: {args.results_file}")
        results = load_detailed_results(args.results_file)

        # Print metadata (use .get so files without every optional key - e.g. the ablation
        # benchmark - still print cleanly)
        meta = results.get('metadata', {})
        print(f"\n{'=' * 80}")
        print("BENCHMARK METADATA")
        print(f"{'=' * 80}")
        print(f"Benchmark type: {meta.get('benchmark_type', 'unknown')}")
        print(f"Description: {meta.get('description', '')}")
        print(f"Number of scenarios: {meta.get('num_scenarios', 'unknown')}")
        print(f"Base seed: {meta.get('base_seed', 'unknown')}")
        print(f"Robot modes: {meta.get('robot_modes', 'n/a')}")
        print(f"Adversarial lie: {meta.get('adversarial_lie', 'n/a')}")

        # Analyze specific method or all methods (for ablation files, "all" = variant names)
        if args.method == "all":
            methods_to_analyze = discover_methods(results)
        else:
            methods_to_analyze = [args.method]

        # Analysis for each method
        for method in methods_to_analyze:
            print(f"\n{'=' * 80}")
            print(f"ANALYZING: {method.upper()}")
            print(f"{'=' * 80}")

            df = extract_metrics_dataframe(results, method)
            print_summary_statistics(df, method)
            plot_metrics_vs_parameters(df, args.output_dir, method)

        # Compare all methods
        if args.method == "all":
            print(f"\n{'=' * 80}")
            print("COMPARING ALL METHODS")
            print(f"{'=' * 80}")
            compare_methods(results, args.output_dir)

        # Export to CSV if requested
        if args.export_csv:
            print(f"\n{'=' * 80}")
            print("EXPORTING TO CSV")
            print(f"{'=' * 80}")
            export_to_csv(results, args.output_dir / "csv")

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
