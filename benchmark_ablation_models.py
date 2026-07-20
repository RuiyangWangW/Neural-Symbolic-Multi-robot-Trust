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

# Human-readable titles for each variant (used in the metadata/table).
VARIANT_DISPLAY_NAMES = {
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


class AblationComparison(TrustMethodComparison):
    """
    TrustMethodComparison variant that lets the supervised method use a chosen temporal_mode
    and, for the temporal ablation, finalizes the per-step scores at the end of the episode
    and rewrites the last step's trust values (which is what evaluate_methods reads).
    """

    def __init__(self, *args, supervised_temporal_mode: str = 'beta', **kwargs):
        self._supervised_temporal_mode = supervised_temporal_mode
        super().__init__(*args, **kwargs)
        # Rebuild the supervised algorithm with the requested temporal_mode (the base class
        # already built one with temporal_mode='beta'; replace it). The ablation architecture
        # is auto-detected from the checkpoint's stored tag.
        gnn_path_str = str(self.supervised_model_path) if self.supervised_model_path else None
        try:
            self.supervised_algorithm = SupervisedTrustAlgorithm(
                model_path=gnn_path_str,
                proximal_range=self.proximal_range,
                temporal_mode=self._supervised_temporal_mode,
            )
        except Exception as e:
            print(f"⚠️ Failed to init supervised ablation algorithm: {e}")

    def run_supervised_model_simulation(self, env) -> List[Dict]:
        results = super().run_supervised_model_simulation(env)
        # Temporal ablation: fold the recorded per-step scores into a single final trust,
        # then overwrite the LAST step's trust fields (evaluate_methods uses the final step).
        if self._supervised_temporal_mode != 'beta' and results:
            self.supervised_algorithm.finalize_temporal(env.robots)
            last = results[-1]
            last['robot_trust_values'] = {r.id: r.trust_value for r in env.robots}
            last['robot_alpha_beta'] = {
                r.id: {'alpha': r.trust_alpha, 'beta': r.trust_beta} for r in env.robots
            }
            for robot in env.robots:
                last['track_trust_values'][robot.id] = {
                    track.track_id: {
                        'trust_value': track.trust_value,
                        'alpha': track.trust_alpha,
                        'beta': track.trust_beta,
                        'object_id': track.object_id,
                    }
                    for track in robot.get_all_tracks()
                }
        return results

    def run_comparison(self) -> dict:
        """
        Ablation-only comparison: run ONLY the supervised method (the ablation variant under
        test) plus the baseline. The other baselines (paper, bayesian) are skipped entirely
        - we only compare ablation models against each other, so computing paper/bayesian
        every scenario would be wasted work. Baseline is still run because its ever-observed
        GT/FP object set is the shared object-metric denominator (same as run_comparison in
        the parent), and evaluate_methods needs it to score the supervised object metrics
        on the correct detectable-object set. paper_results/bayesian_results are left empty,
        which evaluate_methods gracefully skips.
        """
        supervised_env, baseline_env = self.create_identical_environments(2)

        self.supervised_results = self.run_supervised_model_simulation(supervised_env)
        self.baseline_results = self.run_baseline_simulation(baseline_env)
        self.paper_results = []
        self.bayesian_results = []

        # Propagate baseline's ever-observed GT/FP object sets into supervised results, so the
        # object-level denominator is the shared detectable-object set (matches the parent's
        # run_comparison behavior).
        if self.baseline_results:
            final_gt_ids = self.baseline_results[-1]['all_gt_object_ids']
            final_fp_ids = self.baseline_results[-1]['all_fp_object_ids']
            for step_result in self.supervised_results:
                step_result['all_gt_object_ids'] = final_gt_ids
                step_result['all_fp_object_ids'] = final_fp_ids

        return {
            'supervised_results': self.supervised_results,
            'baseline_results': self.baseline_results,
            'paper_results': self.paper_results,
            'bayesian_results': self.bayesian_results,
        }


def run_variant_scenario(scenario: Dict, model_path: str, temporal_mode: str,
                         threshold: float) -> Dict:
    """Run one scenario for one variant; return the supervised method's evaluation."""
    comparison = AblationComparison(
        supervised_model_path=model_path,
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
        supervised_temporal_mode=temporal_mode,
    )
    comparison.adversarial_ratio = scenario["adversarial_ratio"]
    comparison.adversarial_fp_injection_rate = scenario["adversarial_fp_injection_rate"]
    comparison.adversarial_fn_suppression_rate = scenario["adversarial_fn_suppression_rate"]
    comparison.sensor_fp_rate = scenario["sensor_fp_rate"]
    comparison.sensor_fn_rate = scenario["sensor_fn_rate"]
    comparison.delta_plus = scenario["delta_plus"]
    comparison.delta_minus = scenario["delta_minus"]

    results = comparison.run_comparison()
    evaluation = evaluate_methods(results, threshold=threshold,
                                  adversarial_lie=False, object_threshold=threshold)
    return evaluation.get('supervised', {})


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
    print("ABLATION BENCHMARK - aggressive optimized policy (delta_plus=delta_minus=3.0)")
    print("=" * 80)
    print(f"Models dir:   {models_dir}")
    print(f"Variants:     {usable_variants}")
    print(f"Scenarios:    {args.num_scenarios} (base_seed={args.base_seed})")
    print("=" * 80)

    # Build the _detailed.json structure: one entry per scenario, every variant appearing
    # as its own "method" inside that scenario's `evaluation` dict (matching how
    # analyze_benchmark_results.py reads scenario["evaluation"][method]).
    scenario_entries = []
    for i, scenario in enumerate(scenarios):
        evaluation = {}
        for variant in usable_variants:
            ckpt_key, temporal_mode = variant_spec(variant, models_dir)
            model_path = models_dir / f"supervised_model_{ckpt_key}.pth"
            try:
                supervised_eval = run_variant_scenario(
                    scenario, str(model_path), temporal_mode, args.threshold)
                # supervised_eval is {"robots": {...}, "objects": {...}} for this variant.
                evaluation[variant] = supervised_eval
            except Exception as e:
                print(f"  scenario {i} variant {variant}: ERROR {e}")

        scenario_entries.append({
            "scenario_index": i,
            "parameters": {
                "name": f"ablation_{i:03d}",
                "benchmark_type": "ablation",
                "robot_density": scenario["robot_density"],
                "adversarial_ratio": scenario["adversarial_ratio"],
                "adversarial_fp_injection_rate": scenario["adversarial_fp_injection_rate"],
                "adversarial_fn_suppression_rate": scenario["adversarial_fn_suppression_rate"],
                "sensor_fp_rate": scenario["sensor_fp_rate"],
                "sensor_fn_rate": scenario["sensor_fn_rate"],
                "delta_plus": scenario["delta_plus"],
                "delta_minus": scenario["delta_minus"],
                "legitimate_mode": scenario.get("legitimate_mode", LEGITIMATE_MODE),
                "adversarial_mode": scenario.get("adversarial_mode", ADVERSARIAL_MODE),
                "random_seed": scenario["random_seed"],
            },
            "evaluation": evaluation,
        })
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{args.num_scenarios} scenarios done")

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
            "methods": usable_variants,  # ablation variant names, keyed in each evaluation
            "method_display_names": {v: VARIANT_DISPLAY_NAMES[v] for v in usable_variants},
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
    print("ABLATION RESULTS (aggressive optimized) - each row is one ablation variant")
    print("=" * 120)
    print(f"{'variant':<20}{'robot_prec':<16}{'robot_rec':<16}{'robot_acc':<16}"
          f"{'obj_prec':<16}{'obj_rec':<16}{'obj_acc':<16}")
    print("-" * 120)
    for variant in usable_variants:
        rp = agg(variant, 'robots', 'precision'); rr = agg(variant, 'robots', 'recall'); ra = agg(variant, 'robots', 'accuracy')
        op = agg(variant, 'objects', 'precision'); orr = agg(variant, 'objects', 'recall'); oa = agg(variant, 'objects', 'accuracy')
        print(f"{variant:<20}{rp[0]:.3f}±{rp[1]:.3f}   {rr[0]:.3f}±{rr[1]:.3f}   {ra[0]:.3f}±{ra[1]:.3f}   "
              f"{op[0]:.3f}±{op[1]:.3f}   {orr[0]:.3f}±{orr[1]:.3f}   {oa[0]:.3f}±{oa[1]:.3f}")

    print(f"\n✓ Saved: {detailed_path}")
    print(f"  Analyze with: python analyze_benchmark_results.py {detailed_path} --single-file --method all")


if __name__ == '__main__':
    main()
