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
                        help='Number of aggressive scenarios per variant (default: 50)')
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='ablation_benchmark_results.json')
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    config = BENCHMARK_CONFIGS['aggressive']  # delta_plus = delta_minus = 3.0

    # Same scenario set (same seeds) across all variants -> paired comparison.
    scenarios = [sample_scenario_parameters(i, args.base_seed, config)
                 for i in range(args.num_scenarios)]

    print("=" * 80)
    print("ABLATION BENCHMARK - aggressive optimized policy (delta_plus=delta_minus=3.0)")
    print("=" * 80)
    print(f"Models dir:   {models_dir}")
    print(f"Variants:     {args.variants}")
    print(f"Scenarios:    {args.num_scenarios} (base_seed={args.base_seed})")
    print("=" * 80)

    all_results = {}  # variant -> list of supervised evaluations
    for variant in args.variants:
        ckpt_key, temporal_mode = variant_spec(variant, models_dir)
        model_path = models_dir / f"supervised_model_{ckpt_key}.pth"
        if not model_path.exists():
            print(f"⚠️ SKIP {variant}: checkpoint not found: {model_path}")
            continue

        print(f"\n{'='*80}\n▶ Variant: {variant}  (checkpoint={ckpt_key}, temporal={temporal_mode})\n{'='*80}")
        variant_evals = []
        for i, scenario in enumerate(scenarios):
            try:
                supervised_eval = run_variant_scenario(
                    scenario, str(model_path), temporal_mode, args.threshold)
                variant_evals.append(supervised_eval)
            except Exception as e:
                print(f"  scenario {i}: ERROR {e}")
            if (i + 1) % 10 == 0:
                print(f"  ...{i+1}/{args.num_scenarios} scenarios done")
        all_results[variant] = variant_evals

    # Aggregate + print a comparison table
    def agg(evals, level, metric):
        vals = [e[level][metric] for e in evals if level in e and metric in e[level]]
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (0.0, 0.0, 0)

    print("\n" + "=" * 120)
    print("ABLATION RESULTS (supervised method, aggressive optimized)")
    print("=" * 120)
    header = f"{'variant':<24}{'robot_prec':<16}{'robot_rec':<16}{'robot_acc':<16}{'obj_prec':<16}{'obj_rec':<16}{'obj_acc':<16}"
    print(header)
    print("-" * 120)
    table = {}
    for variant in args.variants:
        evals = all_results.get(variant, [])
        if not evals:
            continue
        rp = agg(evals, 'robots', 'precision'); rr = agg(evals, 'robots', 'recall'); ra = agg(evals, 'robots', 'accuracy')
        op = agg(evals, 'objects', 'precision'); orr = agg(evals, 'objects', 'recall'); oa = agg(evals, 'objects', 'accuracy')
        table[variant] = {
            'n': rp[2],
            'robot_precision': rp[:2], 'robot_recall': rr[:2], 'robot_accuracy': ra[:2],
            'object_precision': op[:2], 'object_recall': orr[:2], 'object_accuracy': oa[:2],
        }
        print(f"{variant:<24}{rp[0]:.3f}±{rp[1]:.3f}   {rr[0]:.3f}±{rr[1]:.3f}   {ra[0]:.3f}±{ra[1]:.3f}   "
              f"{op[0]:.3f}±{op[1]:.3f}   {orr[0]:.3f}±{orr[1]:.3f}   {oa[0]:.3f}±{oa[1]:.3f}")

    out = {
        'metadata': {
            'benchmark': 'aggressive_optimized',
            'delta_plus': config.delta_plus,
            'delta_minus': config.delta_minus,
            'num_scenarios': args.num_scenarios,
            'base_seed': args.base_seed,
            'threshold': args.threshold,
            'models_dir': str(models_dir),
        },
        'summary': table,
        'per_scenario': all_results,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n✓ Saved: {args.output}")


if __name__ == '__main__':
    main()
