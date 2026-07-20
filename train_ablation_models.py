#!/usr/bin/env python3
"""
Train all ablation variants of the supervised trust model.

Trains the following architecture variants (each a separate model), ALL on the same
existing training dataset (default: supervised_trust_dataset.pkl) with IDENTICAL
hyperparameters, so the only thing that differs between rows is the ablated component:

  full          - reference architecture (ablation=None)
  no_gat        - skip the HeteroGAT stack entirely (triplet-init h^(0) -> heads)
  homogeneous   - fully type-blind: one node type + one untyped edge type (merges agent/track
                  and all 6 relations); triplet features also neutralized so no type/relation
                  info at init either - only edge presence/count survives
  triplet_init  - replace the triplet encoder with a learned nn.Embedding(2, 128) per type

These are 4 of the 5 ablation-study models. The 5th (No-Beta) needs NO training - it is the
FULL model with inference-side temporal_mode='mean_scores' (see benchmark_ablation_models.py /
SupervisedTrustAlgorithm). So it is intentionally NOT a training row here.

Each variant is trained by invoking train_supervised_trust.py as a subprocess with
--ablation <variant>, writing:
  <output-dir>/supervised_model_<variant>.pth   (checkpoint, carries its ablation tag)
  <output-dir>/supervised_model_<variant>_training.log

Usage:
  python train_ablation_models.py                         # train all variants, default settings
  python train_ablation_models.py --variants full no_gat  # subset
  python train_ablation_models.py --epochs 300            # override hyperparameters (applied to all)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# variant name -> --ablation value passed to train_supervised_trust.py
# 'full' maps to no --ablation flag (the reference model).
VARIANTS = {
    'full': None,
    'no_gat': 'no_gat',
    'homogeneous': 'homogeneous',
    'triplet_init': 'triplet_init',
}


def main():
    parser = argparse.ArgumentParser(description="Train all supervised-trust ablation variants")
    parser.add_argument('--data', type=str, default='supervised_trust_dataset.pkl',
                        help='Path to the (shared) training dataset .pkl')
    parser.add_argument('--output-dir', type=str, default='models_ablation',
                        help='Directory for the per-variant checkpoints/logs (default: models_ablation/)')
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()),
                        help='Which variants to train (default: all)')
    # Hyperparameters - defaults match train_supervised_trust.py's defaults so the ablation
    # rows are directly comparable to the existing full model. Override to apply to ALL variants.
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--agent-loss-weight', type=float, default=5.0)
    parser.add_argument('--track-loss-weight', type=float, default=1.0)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the commands that would run without executing them')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ABLATION TRAINING")
    print("=" * 80)
    print(f"Dataset:     {args.data}")
    print(f"Output dir:  {output_dir}")
    print(f"Variants:    {args.variants}")
    print(f"Epochs:      {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}  Patience: {args.patience}")
    print("=" * 80)

    results = {}
    for variant in args.variants:
        ablation = VARIANTS[variant]
        output_path = output_dir / f"supervised_model_{variant}.pth"

        cmd = [
            sys.executable, 'train_supervised_trust.py',
            '--data', args.data,
            '--output', str(output_path),
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--lr', str(args.lr),
            '--patience', str(args.patience),
            '--agent-loss-weight', str(args.agent_loss_weight),
            '--track-loss-weight', str(args.track_loss_weight),
            '--train-ratio', str(args.train_ratio),
            '--split-seed', str(args.split_seed),
            '--device', args.device,
        ]
        if ablation is not None:
            cmd += ['--ablation', ablation]

        print(f"\n{'='*80}\n▶ Training variant: {variant} (ablation={ablation})\n  -> {output_path}\n{'='*80}")
        print("  " + " ".join(cmd))

        if args.dry_run:
            results[variant] = 'dry-run'
            continue

        t0 = time.time()
        proc = subprocess.run(cmd)
        dt = time.time() - t0
        status = 'OK' if proc.returncode == 0 else f'FAILED (exit {proc.returncode})'
        results[variant] = status
        print(f"  {variant}: {status}  ({dt/60:.1f} min)")

    print("\n" + "=" * 80)
    print("ABLATION TRAINING SUMMARY")
    print("=" * 80)
    for variant in args.variants:
        ckpt = output_dir / f"supervised_model_{variant}.pth"
        exists = "✓" if ckpt.exists() else "✗"
        print(f"  {variant:22s} {results.get(variant, '-'):20s} checkpoint {exists} {ckpt}")


if __name__ == '__main__':
    main()
