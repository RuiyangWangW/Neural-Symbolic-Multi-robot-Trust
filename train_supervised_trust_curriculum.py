#!/usr/bin/env python3
"""
Supervised Trust GNN Training Script with Curriculum Learning

This script extends the standard training with curriculum learning support,
gradually introducing harder samples during training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from typing import List, Dict, Tuple
import argparse
import datetime

# Import from existing training script
from train_supervised_trust import (
    SupervisedTrustDataset,
    SupervisedTrustTrainer,
    collate_batch_pyg,
    collate_batch_individual,
    load_dataset,
    plot_training_results
)
from supervised_trust_gnn import SupervisedTrustGNN
from generate_supervised_data import SupervisedDataSample
from curriculum_learning import CurriculumDataLoader


def train_with_curriculum(trainer: SupervisedTrustTrainer,
                         curriculum_loader: CurriculumDataLoader,
                         val_loader: DataLoader,
                         epochs: int,
                         save_path: str,
                         patience: int,
                         log_print,
                         collate_fn,
                         num_workers: int = 0,
                         pin_memory: bool = False) -> Dict:
    """
    Train model with curriculum learning.

    Args:
        trainer: SupervisedTrustTrainer instance
        curriculum_loader: CurriculumDataLoader with sorted dataset
        val_loader: Validation data loader
        epochs: Number of training epochs
        save_path: Path to save best model
        patience: Early stopping patience
        log_print: Logging function
        collate_fn: Collate function for DataLoader
        num_workers: Number of DataLoader workers
        pin_memory: Whether to pin memory for DataLoader

    Returns:
        Training history dictionary
    """
    log_print(f"🚀 Starting curriculum learning training for {epochs} epochs...")
    log_print(f"📊 Device: {trainer.device}")
    log_print(f"⏰ Early stopping patience: {patience} epochs")
    log_print(f"📚 Curriculum strategy: {curriculum_loader.strategy}")
    log_print("")

    best_val_loss = float('inf')
    patience_counter = 0

    # Track curriculum progress
    curriculum_stats = curriculum_loader.get_statistics()
    log_print(f"📈 Curriculum Statistics:")
    log_print(f"   Total samples: {curriculum_stats['total_samples']}")
    log_print(f"   Easiest difficulty: {curriculum_stats['easiest_difficulty']:.4f}")
    log_print(f"   Hardest difficulty: {curriculum_stats['hardest_difficulty']:.4f}")
    log_print("")

    for epoch in range(epochs):
        # Get training data for this epoch based on curriculum
        epoch_data = curriculum_loader.get_epoch_data(epoch)
        num_epoch_samples = len(epoch_data)

        # Create DataLoader for this epoch
        epoch_dataset = SupervisedTrustDataset(epoch_data)
        train_loader = DataLoader(
            epoch_dataset,
            batch_size=curriculum_loader.batch_size,
            shuffle=True,  # Shuffle within current difficulty level
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        # Log curriculum progress
        pct_used = 100.0 * num_epoch_samples / curriculum_stats['total_samples']
        log_print(f"📚 Epoch {epoch+1}/{epochs} - Using {num_epoch_samples} samples ({pct_used:.1f}% of dataset)")

        # Training
        train_loss, train_metrics = trainer.train_epoch(train_loader)
        trainer.train_losses.append(train_loss)
        trainer.train_metrics.append(train_metrics)

        # Validation (always on full validation set)
        val_loss, val_metrics = trainer.validate_epoch(val_loader)
        trainer.val_losses.append(val_loss)
        trainer.val_metrics.append(val_metrics)

        # Learning rate scheduling
        trainer.scheduler.step(val_loss)

        # Progress logging
        log_print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Print separate losses for monitoring
        train_agent_loss = train_metrics.get('agent_loss', 0.0)
        train_track_loss = train_metrics.get('track_loss', 0.0)
        val_agent_loss = val_metrics.get('agent_loss', 0.0)
        val_track_loss = val_metrics.get('track_loss', 0.0)
        agent_samples = train_metrics.get('agent_samples', 0)
        track_samples = train_metrics.get('track_samples', 0)

        log_print(f"             | Agent Loss: {train_agent_loss:.4f} (n={agent_samples}) | Track Loss: {train_track_loss:.4f} (n={track_samples})")
        log_print(f"             | Val Agent: {val_agent_loss:.4f} | Val Track: {val_track_loss:.4f}")

        # Print detailed metrics every 10 epochs or on last epoch
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Print overall metrics for agents
            if 'agent_accuracy' in train_metrics:
                log_print(f"  Agent Overall: Train Acc={train_metrics['agent_accuracy']:.3f}, Val Acc={val_metrics.get('agent_accuracy', 0):.3f}")
                log_print(f"                 Train F1={train_metrics['agent_f1']:.3f}, Val F1={val_metrics.get('agent_f1', 0):.3f}")

            # Print overall metrics for tracks if available
            if 'track_accuracy' in train_metrics:
                log_print(f"  Track Overall: Train Acc={train_metrics['track_accuracy']:.3f}, Val Acc={val_metrics.get('track_accuracy', 0):.3f}")
                log_print(f"                 Train F1={train_metrics['track_f1']:.3f}, Val F1={val_metrics.get('track_f1', 0):.3f}")

            log_print("-" * 50)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'curriculum_stats': curriculum_stats
            }, save_path)

            log_print(f"✅ Saved best model (val_loss: {best_val_loss:.4f}) to {save_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            log_print(f"⚠️ Early stopping triggered after {patience} epochs without improvement")
            break

    log_print(f"🎉 Training completed! Best validation loss: {best_val_loss:.4f}")

    return {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'train_metrics': trainer.train_metrics,
        'val_metrics': trainer.val_metrics
    }


def main():
    """Main training function with curriculum learning"""
    parser = argparse.ArgumentParser(description='Train supervised trust model with curriculum learning')
    parser.add_argument('--data', type=str, default='supervised_trust_dataset.pkl',
                       help='Path to dataset file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Number of samples per DataLoader batch')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda/mps/auto)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of DataLoader workers (default: 0)')
    parser.add_argument('--patience', type=int, default=100,
                       help='Early stopping patience - epochs without improvement')
    parser.add_argument('--output', type=str, default='supervised_trust_model_curriculum.pth',
                       help='Output model path')
    parser.add_argument('--log', type=str, default=None,
                       help='Log file path (default: output_path.replace(.pth, _training.log))')
    parser.add_argument('--no-pyg-batch', action='store_true',
                       help='Disable PyTorch Geometric batching')
    parser.add_argument('--agent-loss-weight', type=float, default=10.0,
                       help='Weight for agent loss (default: 10.0)')
    parser.add_argument('--track-loss-weight', type=float, default=1.0,
                       help='Weight for track loss (default: 1.0)')

    # Curriculum learning parameters
    parser.add_argument('--curriculum-strategy', type=str, default='linear',
                       choices=['linear', 'root', 'exponential', 'step'],
                       help='Curriculum learning strategy (default: linear)')
    parser.add_argument('--no-curriculum', action='store_true',
                       help='Disable curriculum learning (use standard training)')

    args = parser.parse_args()

    # Set up logging
    log_path = args.log
    if log_path is None:
        log_path = args.output.replace('.pth', '_training.log')

    # Create log file and tee output
    log_file = open(log_path, 'w', buffering=1)  # Line buffering

    def log_print(*print_args, **kwargs):
        """Print to both console and log file"""
        message = ' '.join(str(arg) for arg in print_args)
        print(message, **kwargs)
        log_file.write(message + '\n')
        log_file.flush()

    # Log header
    log_print("=" * 80)
    log_print("SUPERVISED TRUST MODEL TRAINING WITH CURRICULUM LEARNING")
    log_print("=" * 80)
    log_print(f"Start time: {datetime.datetime.now()}")
    log_print(f"Dataset: {args.data}")
    log_print(f"Output model: {args.output}")
    log_print(f"Log file: {log_path}")
    log_print("")

    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            log_print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            log_print(f"🚀 Apple Silicon MPS detected")
        else:
            device = 'cpu'
            log_print(f"⚠️  No GPU acceleration available, using CPU")
    else:
        device = args.device

    log_print(f"🖥️  Using device: {device}")

    # Load dataset
    if not os.path.exists(args.data):
        log_print(f"❌ Dataset file not found: {args.data}")
        log_print("Please run generate_supervised_data.py first to create the dataset")
        log_file.close()
        return

    dataset = load_dataset(args.data, log_print=log_print)

    # Split dataset into train and validation
    np.random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    log_print(f"📊 Dataset split: {len(train_data)} train, {len(val_data)} validation")

    # Create validation loader (normal, no curriculum)
    val_dataset = SupervisedTrustDataset(val_data)

    # Optimize DataLoader settings based on device
    pin_memory = 'cuda' in device
    num_workers = args.num_workers if args.num_workers >= 0 else 0

    # Select collate function based on batching mode
    if args.no_pyg_batch:
        collate_fn = collate_batch_individual
        batching_mode = "Individual Processing"
    else:
        collate_fn = collate_batch_pyg
        batching_mode = "PyG Batching (GPU-optimized)"

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_fn,
                           num_workers=num_workers, pin_memory=pin_memory)

    # Create supervised model
    model = SupervisedTrustGNN(hidden_dim=128)

    log_print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log_print(f"📦 Batching mode: {batching_mode}")
    log_print(f"👷 DataLoader workers: {num_workers}")
    log_print(f"📌 Pin memory: {pin_memory}")

    # Create trainer with loss weighting
    log_print(f"⚖️  Loss weights: Agent={args.agent_loss_weight}, Track={args.track_loss_weight}")
    trainer = SupervisedTrustTrainer(
        model,
        device=device,
        learning_rate=args.lr,
        agent_loss_weight=args.agent_loss_weight,
        track_loss_weight=args.track_loss_weight
    )

    if args.no_curriculum:
        # Standard training without curriculum
        log_print("⚠️  Curriculum learning disabled - using standard training")
        train_dataset = SupervisedTrustDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=True, collate_fn=collate_fn,
                                 num_workers=num_workers, pin_memory=pin_memory)

        history = trainer.train(train_loader, val_loader, epochs=args.epochs,
                               save_path=args.output, patience=args.patience,
                               log_print=log_print)
    else:
        # Curriculum learning
        log_print(f"📚 Initializing curriculum learning with strategy: {args.curriculum_strategy}")
        curriculum_loader = CurriculumDataLoader(
            train_data,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            strategy=args.curriculum_strategy,
            shuffle_within_curriculum=True
        )

        # Train with curriculum
        history = train_with_curriculum(
            trainer=trainer,
            curriculum_loader=curriculum_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            save_path=args.output,
            patience=args.patience,
            log_print=log_print,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    # Plot results
    plot_path = args.output.replace('.pth', '_training_results.png')
    plot_training_results(history, save_path=plot_path, log_print=log_print)

    log_print("")
    log_print("=" * 80)
    log_print(f"End time: {datetime.datetime.now()}")
    log_print("✅ Training completed successfully!")
    log_print("=" * 80)

    log_file.close()


if __name__ == "__main__":
    main()
