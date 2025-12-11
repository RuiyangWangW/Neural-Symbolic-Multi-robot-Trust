#!/usr/bin/env python3
"""
Supervised Trust GNN Training with Weights & Biases Integration

This script enables hyperparameter tuning with wandb sweeps.
"""

import torch
import argparse
import pickle
import wandb
from torch.utils.data import DataLoader

from supervised_trust_gnn import SupervisedTrustGNN
from train_supervised_trust import SupervisedTrustTrainer, SupervisedTrustDataset, collate_batch_pyg
from generate_supervised_data import SupervisedDataSample


def train_with_config(config=None):
    """
    Train model with given config (called by wandb sweep or directly)

    Args:
        config: Config dict from wandb sweep or parsed args
    """
    # Initialize wandb
    with wandb.init(config=config) as run:
        config = wandb.config

        print("="*80)
        print("WANDB HYPERPARAMETER TUNING RUN")
        print("="*80)
        print(f"Run ID: {run.id}")
        print(f"Run name: {run.name}")
        print()
        print("Configuration:")
        for key, value in dict(config).items():
            print(f"  {key}: {value}")
        print()

        # Load dataset
        print(f"Loading dataset from {config.data}...")
        with open(config.data, 'rb') as f:
            data = pickle.load(f)

        print(f"Loaded {len(data['samples'])} samples")

        # Split dataset
        train_size = int(0.8 * len(data['samples']))
        train_samples = data['samples'][:train_size]
        val_samples = data['samples'][train_size:]

        print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

        # Create datasets and loaders
        train_dataset = SupervisedTrustDataset(train_samples)
        val_dataset = SupervisedTrustDataset(val_samples)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_batch_pyg,
            num_workers=0,
            pin_memory=(config.device != 'cpu')
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_batch_pyg,
            num_workers=0,
            pin_memory=(config.device != 'cpu')
        )

        # Create model
        print(f"\nCreating model (hidden_dim={config.hidden_dim})...")
        model = SupervisedTrustGNN(hidden_dim=config.hidden_dim)

        # Create trainer
        trainer = SupervisedTrustTrainer(
            model,
            device=config.device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            agent_loss_weight=config.agent_loss_weight,
            track_loss_weight=config.track_loss_weight
        )

        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Loss weights: Agent={config.agent_loss_weight:.2f}, Track={config.track_loss_weight:.2f}")
        print()

        # Training loop
        print("Starting training...")
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        patience_counter = 0

        for epoch in range(config.epochs):
            # Train
            train_loss, train_metrics = trainer.train_epoch(train_loader)

            # Validate
            val_loss, val_metrics = trainer.validate_epoch(val_loader)

            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_agent_accuracy': train_metrics.get('agent_accuracy', 0),
                'val_agent_accuracy': val_metrics.get('agent_accuracy', 0),
                'train_track_accuracy': train_metrics.get('track_accuracy', 0),
                'val_track_accuracy': val_metrics.get('track_accuracy', 0),
                'train_agent_loss': train_metrics.get('agent_loss', 0),
                'val_agent_loss': val_metrics.get('agent_loss', 0),
                'train_track_loss': train_metrics.get('track_loss', 0),
                'val_track_loss': val_metrics.get('track_loss', 0),
            })

            # Print progress
            if epoch % 5 == 0 or epoch < 5:
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                      f"Agent Acc={val_metrics.get('agent_accuracy', 0):.3f}, "
                      f"Track Acc={val_metrics.get('track_accuracy', 0):.3f}")

            # Early stopping
            val_accuracy = val_metrics.get('agent_accuracy', 0)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Log final results
        wandb.summary['best_val_loss'] = best_val_loss
        wandb.summary['best_val_accuracy'] = best_val_accuracy

        print()
        print("="*80)
        print("RUN COMPLETE")
        print("="*80)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best agent accuracy: {best_val_accuracy:.3f}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Train supervised trust model with wandb')

    # Data
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset file')

    # Wandb
    parser.add_argument('--wandb-project', type=str, default='robot-trust-tuning',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Wandb entity (username or team)')
    parser.add_argument('--sweep', action='store_true',
                       help='Run as part of a wandb sweep (config from sweep)')

    # Training (defaults for non-sweep runs)
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension')

    # Hyperparameters
    parser.add_argument('--agent-loss-weight', type=float, default=3.3,
                       help='Agent loss weight')
    parser.add_argument('--track-loss-weight', type=float, default=1.0,
                       help='Track loss weight')
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                       help='Weight decay')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/mps/auto)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    print(f"Using device: {args.device}")

    # Setup wandb
    if not args.sweep:
        # Single run (not part of sweep)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )
        train_with_config(vars(args))
    else:
        # Part of sweep - config will come from sweep agent
        train_with_config()


if __name__ == '__main__':
    main()
