#!/usr/bin/env python3
"""
Benchmark all supervised-trust ablation variants on the AGGRESSIVE OPTIMIZED policy.

Evaluates each ablation variant with the exact same aggressive-optimized adversarial policy
(delta_plus = delta_minus = 3.0, matching the training distribution) and the same scenario
seeds, so every row is directly comparable. Reuses optimized_policy_benchmark.py's "aggressive"
config and TrustMethodComparison's environment/evaluation machinery - only the supervised
method's model/temporal-aggregation is swapped per row.

The 5 ablation-study models (each row = the SUPERVISED method's robot/object metrics):

  full          models_ablation/supervised_model_full.pth      (reference)
  no_gat        models_ablation/supervised_model_no_gat.pth
  homogeneous   models_ablation/supervised_model_homogeneous.pth
  triplet_init  models_ablation/supervised_model_triplet_init.pth
  no_beta       FULL model, NO retraining - inference-side temporal_mode='mean_scores'
                (trust = running mean of validated per-step evidence scores)

Usage:
  python benchmark_ablation_models.py --models-dir models_ablation --num-scenarios 50
  python benchmark_ablation_models.py --variants full no_gat no_beta
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from compare_trust_methods import TrustMethodComparison
from comprehensive_trust_benchmark import evaluate_methods
from supervised_trust_algorithm import SupervisedTrustAlgorithm
from optimized_policy_benchmark import (
    BENCHMARK_CONFIGS,
    sample_scenario_parameters,
    NUM_TIMESTEPS,
    WORLD_SIZE,
    FOV_RANGE,
    FOV_ANGLE,
    PROXIMAL_RANGE,
    LEGITIMATE_MODE,
    ADVERSARIAL_MODE,
)

# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
# The 5 ablation-study models. Each variant maps to (checkpoint_key, temporal_mode):
#   - Architecture variants load supervised_model_<key>.pth and use Beta accumulation.
#   - 'no_beta' reuses the FULL model with inference-side temporal_mode='mean_scores'
#     (running mean of validated evidence scores) - no retraining.
ARCH_VARIANTS = ['full', 'no_gat', 'homogeneous', 'triplet_init']
ALL_VARIANTS = ARCH_VARIANTS + ['no_beta']

# Human-readable titles for each method (baseline + the 5 ablation variants).
VARIANT_DISPLAY_NAMES = {
    'baseline': 'Baseline (No Trust)',
    'full': 'Full (reference)',
    'no_gat': 'No-GAT (0 message-passing layers)',
    'homogeneous': 'Homogeneous GAT (single relation)',
    'triplet_init': 'No-Triplet-Init (learned type embeddings)',
    'no_beta': 'No-Beta (mean of per-step scores)',
}


def variant_spec(variant: str, models_dir: Path):
    """Return (model_checkpoint_key, temporal_mode) for a variant name."""
    if variant in ARCH_VARIANTS:
        return variant, 'beta'
    if variant == 'no_beta':
        return 'full', 'mean_scores'
    raise ValueError(f"Unknown variant '{variant}'")


def run_scenario(scenario: Dict, variants: List[str], models_dir: Path,
                 threshold: float) -> Dict[str, Dict]:
    """Run a single scenario and return evaluation metrics keyed by method.

    Mirrors unified_benchmark.py / optimized_policy_benchmark.py's run_scenario, except the
    "methods" are the baseline (no trust) plus the ablation variants. Each method runs on its
    OWN fresh SimulationEnvironment built with the SAME scenario seed (via
    create_identical_environments), so every method sees an identical world and starts trust
    accumulation fresh - exactly the per-method reset the standard benchmarks already do.

    Returns {"baseline": {robots, objects}, <variant>: {robots, objects}, ...}.
    """
    # One TrustMethodComparison holds the shared scenario config and the environment factory.
    comparison = TrustMethodComparison(
        supervised_model_path=str(models_dir / "supervised_model_full.pth"),
        robot_density=scenario["robot_density"],
        target_density_multiplier=scenario["target_density_multiplier"],
        num_timesteps=NUM_TIMESTEPS,
        random_seed=scenario["random_seed"],
        world_size=WORLD_SIZE,
        fov_range=FOV_RANGE,
        fov_angle=FOV_ANGLE,
        proximal_range=PROXIMAL_RANGE,
        allow_fp_codetection=True,
        legitimate_mode=scenario.get("legitimate_mode", LEGITIMATE_MODE),
        adversarial_mode=scenario.get("adversarial_mode", ADVERSARIAL_MODE),
    )
    comparison.adversarial_ratio = scenario["adversarial_ratio"]
    comparison.adversarial_fp_injection_rate = scenario["adversarial_fp_injection_rate"]
    comparison.adversarial_fn_suppression_rate = scenario["adversarial_fn_suppression_rate"]
    comparison.sensor_fp_rate = scenario["sensor_fp_rate"]
    comparison.sensor_fn_rate = scenario["sensor_fn_rate"]
    comparison.delta_plus = scenario["delta_plus"]
    comparison.delta_minus = scenario["delta_minus"]

    # Fresh identical-seed environment per method (baseline + each variant).
    envs = comparison.create_identical_environments(1 + len(variants))
    baseline_env, variant_envs = envs[0], envs[1:]

    # --- Baseline (no trust): also the shared GT/FP object-metric denominator ---
    baseline_results = comparison.run_baseline_simulation(baseline_env)
    shared_gt_ids = baseline_results[-1]['all_gt_object_ids'] if baseline_results else []
    shared_fp_ids = baseline_results[-1]['all_fp_object_ids'] if baseline_results else []

    evaluation = {}
    baseline_eval = evaluate_methods(
        {'baseline_results': baseline_results, 'supervised_results': [],
         'paper_results': [], 'bayesian_results': []},
        threshold=threshold, adversarial_lie=False, object_threshold=threshold)
    evaluation['baseline'] = baseline_eval.get('baseline', {})

    # --- Each ablation variant on its own fresh, same-seed environment ---
    for variant, env in zip(variants, variant_envs):
        ckpt_key, temporal_mode = variant_spec(variant, models_dir)
        model_path = models_dir / f"supervised_model_{ckpt_key}.pth"

        # The variant's supervised algorithm (correct ablation auto-detected from checkpoint
        # tag, correct temporal_mode). finalize_temporal for the No-Beta variant.
        comparison.supervised_algorithm = SupervisedTrustAlgorithm(
            model_path=str(model_path), proximal_range=PROXIMAL_RANGE, temporal_mode=temporal_mode)
        sup_results = comparison.run_supervised_model_simulation(env)
        if temporal_mode != 'beta' and sup_results:
            comparison.supervised_algorithm.finalize_temporal(env.robots)
            last = sup_results[-1]
            last['robot_trust_values'] = {r.id: r.trust_value for r in env.robots}
            last['robot_alpha_beta'] = {r.id: {'alpha': r.trust_alpha, 'beta': r.trust_beta}
                                        for r in env.robots}
            for robot in env.robots:
                last['track_trust_values'][robot.id] = {
                    t.track_id: {'trust_value': t.trust_value, 'alpha': t.trust_alpha,
                                 'beta': t.trust_beta, 'object_id': t.object_id}
                    for t in robot.get_all_tracks()}

        # Shared baseline denominator for object metrics.
        for step_result in sup_results:
            step_result['all_gt_object_ids'] = shared_gt_ids
            step_result['all_fp_object_ids'] = shared_fp_ids

        var_eval = evaluate_methods(
            {'supervised_results': sup_results, 'baseline_results': baseline_results,
             'paper_results': [], 'bayesian_results': []},
            threshold=threshold, adversarial_lie=False, object_threshold=threshold)
        evaluation[variant] = var_eval.get('supervised', {})

    return evaluation


def main():
    parser = argparse.ArgumentParser(description="Benchmark supervised ablation variants (aggressive optimized)")
    parser.add_argument('--models-dir', type=str, default='models_ablation',
                        help='Directory holding supervised_model_<variant>.pth checkpoints')
    parser.add_argument('--variants', nargs='+',
                        default=ALL_VARIANTS, choices=ALL_VARIANTS,
                        help='Which of the 5 ablation models to benchmark (default: all)')
    parser.add_argument('--num-scenarios', type=int, default=50,
                        help='Number of aggressive scenarios (default: 50)')
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Directory to save results (default: benchmark_results/, matching '
                             'the other benchmarks so analyze_benchmark_results.py picks it up)')
    parser.add_argument('--output-name', type=str, default='ablation',
                        help='Base filename, produces <output-dir>/<output-name>_detailed.json')
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    config = BENCHMARK_CONFIGS['aggressive']  # delta_plus = delta_minus = 3.0

    # Resolve which variants actually have a usable checkpoint.
    usable_variants = []
    for variant in args.variants:
        ckpt_key, _ = variant_spec(variant, models_dir)
        if (models_dir / f"supervised_model_{ckpt_key}.pth").exists():
            usable_variants.append(variant)
        else:
            print(f"⚠️ SKIP {variant}: checkpoint not found: "
                  f"{models_dir / f'supervised_model_{ckpt_key}.pth'}")

    # Same scenario set (same seeds) evaluated by EVERY variant -> paired comparison.
    scenarios = [sample_scenario_parameters(i, args.base_seed, config)
                 for i in range(args.num_scenarios)]

    print("=" * 80)
    # Methods = baseline (no trust) + the ablation variants, all keyed in each scenario's
    # evaluation dict (matching how the other benchmarks key baseline/bayesian/paper/supervised).
    methods = ['baseline'] + usable_variants

    print("ABLATION BENCHMARK - aggressive optimized policy (delta_plus=delta_minus=3.0)")
    print("=" * 80)
    print(f"Models dir:   {models_dir}")
    print(f"Methods:      {methods}")
    print(f"Scenarios:    {args.num_scenarios} (base_seed={args.base_seed})")
    print("=" * 80)

    # Build the _detailed.json structure: one entry per scenario, every method appearing
    # as its own key inside that scenario's `evaluation` dict (matching how
    # analyze_benchmark_results.py reads scenario["evaluation"][method]).
    scenario_entries = []
    for i, scenario in enumerate(scenarios):
        try:
            # run_scenario builds a fresh identical-seed environment per method (baseline +
            # each variant), so trust starts fresh for every one - matches the standard
            # benchmarks' per-method reset via create_identical_environments.
            evaluation = run_scenario(scenario, usable_variants, models_dir, args.threshold)
        except Exception as e:
            print(f"  scenario {i}: ERROR {e}")
            evaluation = {}

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{args.num_scenarios} scenarios")

        scenario_entries.append({
            "scenario_index": i,
            # Store the full scenario dict as parameters, exactly like unified_benchmark.py /
            # optimized_policy_benchmark.py ("parameters": r["scenario"]) - it already has
            # name/benchmark_type/robot_density/adversarial_ratio/random_seed/etc.
            "parameters": scenario,
            "evaluation": evaluation,
        })

    detailed_results = {
        "metadata": {
            "benchmark_type": "ablation",
            "description": "Supervised model architecture ablation - aggressive optimized "
                           "policy (delta_plus=delta_minus=3.0). Each 'method' is an ablation "
                           "variant of the supervised model.",
            "adversarial_mode": ADVERSARIAL_MODE,
            "num_scenarios": args.num_scenarios,
            "base_seed": args.base_seed,
            "threshold": args.threshold,
            "models_dir": str(models_dir),
            "methods": methods,  # baseline + ablation variant names, keyed in each evaluation
            "method_display_names": {m: VARIANT_DISPLAY_NAMES[m] for m in methods},
            "parameter_ranges": {
                "delta_plus": config.delta_plus,
                "delta_minus": config.delta_minus,
            },
        },
        "scenarios": scenario_entries,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = output_dir / f"{args.output_name}_detailed.json"
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=float)

    # Print a quick comparison table (aggregate across scenarios).
    def agg(variant, level, metric):
        vals = [e["evaluation"][variant][level][metric]
                for e in scenario_entries
                if variant in e["evaluation"] and level in e["evaluation"][variant]
                and metric in e["evaluation"][variant][level]]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

    print("\n" + "=" * 120)
    print("ABLATION RESULTS (aggressive optimized) - baseline + one row per ablation model")
    print("=" * 120)
    print(f"{'method':<20}{'robot_prec':<16}{'robot_rec':<16}{'robot_acc':<16}"
          f"{'obj_prec':<16}{'obj_rec':<16}{'obj_acc':<16}")
    print("-" * 120)
    for method in methods:
        rp = agg(method, 'robots', 'precision'); rr = agg(method, 'robots', 'recall'); ra = agg(method, 'robots', 'accuracy')
        op = agg(method, 'objects', 'precision'); orr = agg(method, 'objects', 'recall'); oa = agg(method, 'objects', 'accuracy')
        print(f"{method:<20}{rp[0]:.3f}±{rp[1]:.3f}   {rr[0]:.3f}±{rr[1]:.3f}   {ra[0]:.3f}±{ra[1]:.3f}   "
              f"{op[0]:.3f}±{op[1]:.3f}   {orr[0]:.3f}±{orr[1]:.3f}   {oa[0]:.3f}±{oa[1]:.3f}")

    print(f"\n✓ Saved: {detailed_path}")
    print(f"  Analyze with: python analyze_benchmark_results.py {detailed_path} --single-file --method all")


if __name__ == '__main__':
    main()
