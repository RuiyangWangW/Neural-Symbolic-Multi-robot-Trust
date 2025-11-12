#!/usr/bin/env python3
"""
Supervised Trust GNN Training Script

This script trains a supervised learning model for trust prediction using
ground truth labels generated from simulation data.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from supervised_trust_gnn import SupervisedTrustGNN
from generate_supervised_data import SupervisedDataSample
from torch_geometric.data import HeteroData, Batch as HeteroBatch


class SupervisedTrustDataset(Dataset):
    """
    PyTorch Dataset for supervised trust learning
    """

    def __init__(self, data_samples: List[SupervisedDataSample]):
        """
        Initialize dataset

        Args:
            data_samples: List of supervised data samples
        """
        self.samples = data_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'x_dict': sample.x_dict,
            'edge_index_dict': sample.edge_index_dict,
            'agent_labels': sample.agent_labels,
            'track_labels': sample.track_labels,
            'timestep': sample.timestep,
            'episode': sample.episode,
            'ego_robot_id': sample.ego_robot_id
        }


def collate_batch(batch: List[Dict]) -> Dict:
    """
    Optimized collate function using PyTorch Geometric's native batching

    Args:
        batch: List of sample dictionaries

    Returns:
        Dictionary with PyG batched data
    """
    # Pre-allocate lists for better performance
    hetero_data_list = []
    all_agent_labels = []
    all_track_labels = []

    # Process batch more efficiently
    for sample in batch:
        # Create HeteroData object with pre-defined node and edge types
        hetero_data = HeteroData()

        # Add node features directly without iteration
        hetero_data['agent'].x = sample['x_dict']['agent']
        hetero_data['track'].x = sample['x_dict']['track']

        # Add edge indices directly - use correct edge types from supervised_trust_gnn.py
        for edge_type, edge_index in sample['edge_index_dict'].items():
            hetero_data[edge_type].edge_index = edge_index

        hetero_data_list.append(hetero_data)
        all_agent_labels.append(sample['agent_labels'])
        all_track_labels.append(sample['track_labels'])

    # Use PyG's optimized native batching
    try:
        batched_hetero = HeteroBatch.from_data_list(hetero_data_list)

        # Concatenate labels efficiently
        batched_agent_labels = torch.cat(all_agent_labels, dim=0)
        batched_track_labels = torch.cat(all_track_labels, dim=0)

        return {
            'hetero_batch': batched_hetero,
            'agent_labels': batched_agent_labels,
            'track_labels': batched_track_labels,
            'batch_size': len(batch),
            'use_pyg_batch': True
        }

    except Exception as e:
        # Fallback to individual processing if batching fails
        print(f"PyG batching failed: {e}, using individual processing")
        return {
            'samples': batch,
            'batch_size': len(batch),
            'use_individual': True
        }

class SupervisedTrustTrainer:
    """
    Trainer for supervised trust prediction model
    """

    def __init__(self,
                 model: SupervisedTrustGNN,
                 device: str = 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                ):
        """
        Initialize trainer

        Args:
            model: Supervised trust GNN model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        # Optimizer and loss function
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=10
        )

        # Binary cross-entropy loss for classification with mean reduction
        self.criterion = nn.BCELoss(reduction='mean')

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

        # MPS-specific optimizations
        if device == 'mps':
            # Enable MPS optimizations
            try:
                # Try to enable CPU fallback if available (older PyTorch versions)
                if hasattr(torch.backends.mps, 'enable_cpu_fallback'):
                    torch.backends.mps.enable_cpu_fallback(True)
            except:
                pass  # Fallback method not available in this PyTorch version
            print("🍎 MPS optimizations enabled")

    def _transfer_to_device(self, sample: Dict, non_blocking: bool = True) -> Tuple[Dict, Dict, torch.Tensor, torch.Tensor]:
        """
        Efficiently transfer sample data to device with MPS error handling

        Args:
            sample: Data sample dictionary
            non_blocking: Whether to use non-blocking transfer (for MPS/CUDA)

        Returns:
            Tuple of (x_dict, edge_index_dict, agent_labels, track_labels)
        """
        try:
            # For MPS, disable non_blocking transfers to avoid placeholder tensor issues
            use_non_blocking = non_blocking and (self.device.type != 'mps')

            # Transfer feature dictionaries with validation
            x_dict = {}
            for k, v in sample['x_dict'].items():
                if v.numel() > 0:  # Check for non-empty tensors
                    x_dict[k] = v.to(self.device, non_blocking=use_non_blocking)
                else:
                    # Handle empty tensors for MPS compatibility
                    x_dict[k] = v.to(self.device, non_blocking=False)

            # Transfer edge indices with validation
            edge_index_dict = {}
            for k, v in sample['edge_index_dict'].items():
                if v.numel() > 0:  # Check for non-empty tensors
                    edge_index_dict[k] = v.to(self.device, non_blocking=use_non_blocking)
                else:
                    # Handle empty edge indices
                    edge_index_dict[k] = v.to(self.device, non_blocking=False)

            # Transfer labels with validation
            agent_labels = sample['agent_labels'].to(self.device, non_blocking=use_non_blocking) if sample['agent_labels'].numel() > 0 else sample['agent_labels'].to(self.device, non_blocking=False)
            track_labels = sample['track_labels'].to(self.device, non_blocking=use_non_blocking) if sample['track_labels'].numel() > 0 else sample['track_labels'].to(self.device, non_blocking=False)

            return x_dict, edge_index_dict, agent_labels, track_labels

        except Exception as e:
            # Fallback to synchronous transfer for problematic samples
            x_dict = {k: v.to(self.device, non_blocking=False) for k, v in sample['x_dict'].items()}
            edge_index_dict = {k: v.to(self.device, non_blocking=False) for k, v in sample['edge_index_dict'].items()}
            agent_labels = sample['agent_labels'].to(self.device, non_blocking=False)
            track_labels = sample['track_labels'].to(self.device, non_blocking=False)

            if self.device.type == 'mps':
                print(f"MPS transfer fallback used for sample due to: {e}")

            return x_dict, edge_index_dict, agent_labels, track_labels

    def _compute_loss(self, predictions: Dict, agent_labels: torch.Tensor, track_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for predictions
        """
        loss = 0.0
        loss_components = 0

        if 'agent' in predictions and agent_labels.shape[0] > 0:
            agent_loss = self.criterion(predictions['agent'], agent_labels)
            loss += agent_loss 
            loss_components += 1

        if 'track' in predictions and track_labels.shape[0] > 0:
            track_loss = self.criterion(predictions['track'], track_labels)
            loss += track_loss 
            loss_components += 1

        if loss_components > 0:
            return loss * 100
        return None

    def _compute_metrics(self, predictions: Dict, labels: Dict) -> Dict:
        """
        Compute evaluation metrics

        Args:
            predictions: Model predictions dict
            labels: Ground truth labels dict

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        for node_type in predictions:
            if node_type in labels and labels[node_type].shape[0] > 0:
                y_true = labels[node_type].detach().cpu().numpy().flatten()
                y_prob = predictions[node_type].detach().cpu().numpy().flatten()
                y_pred = (y_prob > 0.5).astype(float)

                # Compute metrics only if we have both classes
                if len(np.unique(y_true)) > 1:
                    metrics[f'{node_type}_accuracy'] = accuracy_score(y_true, y_pred)
                    metrics[f'{node_type}_precision'] = precision_score(y_true, y_pred, zero_division=0)
                    metrics[f'{node_type}_recall'] = recall_score(y_true, y_pred, zero_division=0)
                    metrics[f'{node_type}_f1'] = f1_score(y_true, y_pred, zero_division=0)
                    metrics[f'{node_type}_auc'] = roc_auc_score(y_true, y_prob)
                else:
                    # Only one class present
                    metrics[f'{node_type}_accuracy'] = accuracy_score(y_true, y_pred)
                    metrics[f'{node_type}_precision'] = 0.0
                    metrics[f'{node_type}_recall'] = 0.0
                    metrics[f'{node_type}_f1'] = 0.0
                    metrics[f'{node_type}_auc'] = 0.0

        return metrics

    def _compute_detailed_metrics(self, predictions: Dict, labels: Dict, agent_metadata: List[Dict], track_metadata: List[Dict]) -> Dict:
        """
        Compute detailed metrics using metadata for better insights

        Args:
            predictions: Model predictions dict
            labels: Ground truth labels dict
            agent_metadata: Agent metadata list
            track_metadata: Track metadata list

        Returns:
            Dictionary of detailed metrics
        """
        detailed_metrics = {}

        # Agent-specific metrics
        if 'agent' in predictions and labels['agent'].shape[0] > 0:
            y_true = labels['agent'].detach().cpu().numpy().flatten()
            y_prob = predictions['agent'].detach().cpu().numpy().flatten()
            y_pred = (y_prob > 0.5).astype(float)

            # Overall agent metrics
            detailed_metrics.update(self._compute_metrics(predictions, labels))

            # Adversarial vs honest robot metrics
            adversarial_indices = [i for i, meta in enumerate(agent_metadata) if meta['is_adversarial']]
            honest_indices = [i for i, meta in enumerate(agent_metadata) if not meta['is_adversarial']]

            if adversarial_indices:
                adv_true = y_true[adversarial_indices]
                adv_pred = y_pred[adversarial_indices]
                if len(np.unique(adv_true)) > 1:
                    detailed_metrics['adversarial_agent_accuracy'] = accuracy_score(adv_true, adv_pred)
                    detailed_metrics['adversarial_agent_f1'] = f1_score(adv_true, adv_pred, zero_division=0)
                else:
                    detailed_metrics['adversarial_agent_accuracy'] = accuracy_score(adv_true, adv_pred)
                    detailed_metrics['adversarial_agent_f1'] = 0.0

            if honest_indices:
                honest_true = y_true[honest_indices]
                honest_pred = y_pred[honest_indices]
                if len(np.unique(honest_true)) > 1:
                    detailed_metrics['honest_agent_accuracy'] = accuracy_score(honest_true, honest_pred)
                    detailed_metrics['honest_agent_f1'] = f1_score(honest_true, honest_pred, zero_division=0)
                else:
                    detailed_metrics['honest_agent_accuracy'] = accuracy_score(honest_true, honest_pred)
                    detailed_metrics['honest_agent_f1'] = 0.0

        # Track-specific metrics
        if 'track' in predictions and labels['track'].shape[0] > 0:
            y_true = labels['track'].detach().cpu().numpy().flatten()
            y_prob = predictions['track'].detach().cpu().numpy().flatten()
            y_pred = (y_prob > 0.5).astype(float)

            # Ground truth vs false positive track metrics
            gt_indices = [i for i, meta in enumerate(track_metadata) if meta['is_ground_truth']]
            fp_indices = [i for i, meta in enumerate(track_metadata) if not meta['is_ground_truth']]

            if gt_indices:
                gt_true = y_true[gt_indices]
                gt_pred = y_pred[gt_indices]
                if len(np.unique(gt_true)) > 1:
                    detailed_metrics['ground_truth_track_accuracy'] = accuracy_score(gt_true, gt_pred)
                    detailed_metrics['ground_truth_track_f1'] = f1_score(gt_true, gt_pred, zero_division=0)
                else:
                    detailed_metrics['ground_truth_track_accuracy'] = accuracy_score(gt_true, gt_pred)
                    detailed_metrics['ground_truth_track_f1'] = 0.0

            if fp_indices:
                fp_true = y_true[fp_indices]
                fp_pred = y_pred[fp_indices]
                if len(np.unique(fp_true)) > 1:
                    detailed_metrics['false_positive_track_accuracy'] = accuracy_score(fp_true, fp_pred)
                    detailed_metrics['false_positive_track_f1'] = f1_score(fp_true, fp_pred, zero_division=0)
                else:
                    detailed_metrics['false_positive_track_accuracy'] = accuracy_score(fp_true, fp_pred)
                    detailed_metrics['false_positive_track_f1'] = 0.0

        return detailed_metrics

    def _process_sample(self, sample: Dict) -> Tuple[float, Dict]:
        """
        Process a single sample

        Args:
            sample: Single data sample

        Returns:
            Tuple of (loss, metrics)
        """
        # Move data to device
        x_dict, edge_index_dict, agent_labels, track_labels = self._transfer_to_device(sample)

        # Skip samples with empty data
        if agent_labels.numel() == 0 and track_labels.numel() == 0:
            return 0.0, {}

        # Forward pass
        predictions = self.model(x_dict, edge_index_dict)

        # Compute loss
        loss = self._compute_loss(predictions, agent_labels, track_labels)
        if loss is None:
            return 0.0, {}

        # Compute metrics
        labels_dict = {'agent': agent_labels, 'track': track_labels}
        metrics = self._compute_metrics(predictions, labels_dict)

        return loss, metrics



    def _process_pyg_batch(self, batch_data: Dict) -> Tuple[float, Dict]:
        """
        Process a PyTorch Geometric batched HeteroData

        Args:
            batch_data: PyG batched data

        Returns:
            Tuple of (loss, metrics)
        """
        # Move data to device
        hetero_batch = batch_data['hetero_batch'].to(self.device)
        agent_labels = batch_data['agent_labels'].to(self.device)
        track_labels = batch_data['track_labels'].to(self.device)

        # Forward pass - PyG handles the batching automatically
        predictions = self.model(hetero_batch.x_dict, hetero_batch.edge_index_dict)

        # Compute loss - PyG concatenated everything, so labels should align
        loss = self._compute_loss(predictions, agent_labels, track_labels)
        if loss is None:
            return 0.0, {}

        # Compute metrics
        labels_dict = {'agent': agent_labels, 'track': track_labels}
        metrics = self._compute_metrics(predictions, labels_dict)

        return loss, metrics

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """
        Train for one epoch with support for both batched and individual processing

        Args:
            dataloader: Training data loader

        Returns:
            Tuple of (average_loss, average_metrics)
        """
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        all_metrics = []

        for batch_data in dataloader:
            try:
                # Zero gradients
                self.optimizer.zero_grad()

                if batch_data.get('use_pyg_batch', False):
                    # Process as a PyG batch
                    loss, metrics = self._process_pyg_batch(batch_data)
                    current_batch_size = batch_data['batch_size']

                    if loss > 0:
                        # Backward pass
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                        total_loss += loss.item()
                        num_samples += current_batch_size

                        if metrics:
                            all_metrics.append(metrics)

                else:
                    # Process samples individually (fallback or single sample)
                    samples = batch_data['samples']

                    for sample in samples:
                        try:
                            # Process single sample
                            loss, metrics = self._process_sample(sample)

                            if loss > 0:
                                # Backward pass
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                self.optimizer.step()

                                total_loss += loss.item()
                                num_samples += 1

                                if metrics:
                                    all_metrics.append(metrics)

                        except Exception as e:
                            print(f"Error processing individual sample: {e}")
                            continue

            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        avg_loss = total_loss / max(num_samples, 1)

        # Average metrics
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                avg_metrics[key] = np.mean(values) if values else 0.0

        return avg_loss, avg_metrics

    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """
        Validate for one epoch with support for both batched and individual processing

        Args:
            dataloader: Validation data loader

        Returns:
            Tuple of (average_loss, average_metrics)
        """
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        all_metrics = []

        with torch.no_grad():
            for batch_data in dataloader:
                try:
                    if batch_data.get('use_pyg_batch', False):
                        # Process as a PyG batch
                        loss, metrics = self._process_pyg_batch(batch_data)
                        current_batch_size = batch_data['batch_size']

                        if loss > 0:
                            total_loss += loss.item()
                            num_samples += current_batch_size

                            if metrics:
                                all_metrics.append(metrics)

                    else:
                        # Process samples individually (fallback or single sample)
                        samples = batch_data['samples']

                        for sample in samples:
                            try:
                                # Process single sample
                                loss, metrics = self._process_sample(sample)

                                if loss > 0:
                                    total_loss += loss.item()
                                    num_samples += 1

                                    if metrics:
                                        all_metrics.append(metrics)

                            except Exception as e:
                                if self.device.type == 'mps' and 'Placeholder tensor' in str(e):
                                    print(f"MPS error - skipping sample")
                                    if hasattr(torch.mps, 'empty_cache'):
                                        torch.mps.empty_cache()
                                else:
                                    print(f"Error processing validation sample: {e}")
                                continue

                except Exception as e:
                    print(f"Error processing validation batch: {e}")
                    continue

        avg_loss = total_loss / max(num_samples, 1)

        # Average metrics
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                avg_metrics[key] = np.mean(values) if values else 0.0

        return avg_loss, avg_metrics

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              save_path: str = 'supervised_trust_model.pth',
              patience: int = 20) -> Dict:
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_path: Path to save best model
            patience: Early stopping patience (epochs without improvement)

        Returns:
            Training history dictionary
        """
        print(f"🚀 Starting supervised trust training for {epochs} epochs...")
        print(f"📊 Device: {self.device}")
        print(f"⏰ Early stopping patience: {patience} epochs")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)

            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Progress logging - print every epoch
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Print detailed metrics every 10 epochs or on last epoch
            if epoch % 10 == 0 or epoch == epochs - 1:
                # Print overall metrics for agents
                if 'agent_accuracy' in train_metrics:
                    print(f"  Agent Overall: Train Acc={train_metrics['agent_accuracy']:.3f}, Val Acc={val_metrics.get('agent_accuracy', 0):.3f}")
                    print(f"                 Train F1={train_metrics['agent_f1']:.3f}, Val F1={val_metrics.get('agent_f1', 0):.3f}")

                # Print detailed agent metrics (adversarial vs honest)
                if 'adversarial_agent_accuracy' in train_metrics:
                    print(f"  Adversarial:   Train Acc={train_metrics['adversarial_agent_accuracy']:.3f}, Val Acc={val_metrics.get('adversarial_agent_accuracy', 0):.3f}")
                if 'honest_agent_accuracy' in train_metrics:
                    print(f"  Honest:        Train Acc={train_metrics['honest_agent_accuracy']:.3f}, Val Acc={val_metrics.get('honest_agent_accuracy', 0):.3f}")

                # Print overall metrics for tracks if available
                if 'track_accuracy' in train_metrics:
                    print(f"  Track Overall: Train Acc={train_metrics['track_accuracy']:.3f}, Val Acc={val_metrics.get('track_accuracy', 0):.3f}")
                    print(f"                 Train F1={train_metrics['track_f1']:.3f}, Val F1={val_metrics.get('track_f1', 0):.3f}")

                # Print detailed track metrics (ground truth vs false positive)
                if 'ground_truth_track_accuracy' in train_metrics:
                    print(f"  GT Tracks:     Train Acc={train_metrics['ground_truth_track_accuracy']:.3f}, Val Acc={val_metrics.get('ground_truth_track_accuracy', 0):.3f}")
                if 'false_positive_track_accuracy' in train_metrics:
                    print(f"  FP Tracks:     Train Acc={train_metrics['false_positive_track_accuracy']:.3f}, Val Acc={val_metrics.get('false_positive_track_accuracy', 0):.3f}")

                print("-" * 50)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }, save_path)

                print(f"✅ Saved best model (val_loss: {best_val_loss:.4f}) to {save_path}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"⚠️ Early stopping triggered after {patience} epochs without improvement")
                break

        print(f"🎉 Training completed! Best validation loss: {best_val_loss:.4f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }

def load_dataset(data_path: str) -> List[SupervisedDataSample]:
    """Load dataset from pickle file with parameter diversity metadata"""
    print(f"📂 Loading dataset from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Extract samples from new dataset format
    dataset = data['samples']
    episode_params = data.get('episode_parameters', [])
    param_ranges = data.get('parameter_ranges', {})
    statistics = data.get('statistics', {})

    print(f"✅ Loaded {len(dataset)} samples with parameter diversity")
    print(f"📊 Dataset generated with parameter ranges:")
    for param_name, param_range in param_ranges.items():
        print(f"   - {param_name}: {param_range}")

    if statistics and 'parameter_diversity' in statistics:
        param_div = statistics['parameter_diversity']
        print(f"🎲 Actual parameter diversity across {len(episode_params)} episodes:")
        for param_name, stats in param_div.items():
            print(f"   - {param_name}: {stats['min']:.3f} - {stats['max']:.3f} (avg: {stats['avg']:.3f})")

    return dataset


def split_dataset(dataset: List[SupervisedDataSample],
                 train_ratio: float = 0.8) -> Tuple[List[SupervisedDataSample], List[SupervisedDataSample]]:
    """Split dataset into train and validation sets"""
    np.random.shuffle(dataset)
    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    print(f"📊 Dataset split: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data


def plot_training_results(history: Dict, save_path: str = 'supervised_training_results.png'):
    """Plot training results"""
    _, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    axes[0, 0].plot(history['train_losses'], label='Train Loss', alpha=0.8)
    axes[0, 0].plot(history['val_losses'], label='Val Loss', alpha=0.8)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Agent accuracy
    agent_train_acc = [m.get('agent_accuracy', 0) for m in history['train_metrics']]
    agent_val_acc = [m.get('agent_accuracy', 0) for m in history['val_metrics']]
    axes[0, 1].plot(agent_train_acc, label='Train Accuracy', alpha=0.8)
    axes[0, 1].plot(agent_val_acc, label='Val Accuracy', alpha=0.8)
    axes[0, 1].set_title('Agent Classification Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Agent F1 score
    agent_train_f1 = [m.get('agent_f1', 0) for m in history['train_metrics']]
    agent_val_f1 = [m.get('agent_f1', 0) for m in history['val_metrics']]
    axes[1, 0].plot(agent_train_f1, label='Train F1', alpha=0.8)
    axes[1, 0].plot(agent_val_f1, label='Val F1', alpha=0.8)
    axes[1, 0].set_title('Agent F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Track metrics (if available)
    track_train_acc = [m.get('track_accuracy', 0) for m in history['train_metrics']]
    track_val_acc = [m.get('track_accuracy', 0) for m in history['val_metrics']]
    if any(x > 0 for x in track_train_acc):
        axes[1, 1].plot(track_train_acc, label='Train Accuracy', alpha=0.8)
        axes[1, 1].plot(track_val_acc, label='Val Accuracy', alpha=0.8)
        axes[1, 1].set_title('Track Classification Accuracy')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Track Data\nAvailable',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Track Classification Accuracy')

    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Training results saved to {save_path}")


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train supervised trust prediction model')
    parser.add_argument('--data', type=str, default='supervised_trust_dataset.pkl',
                       help='Path to dataset file')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training (supports both batched and individual processing)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/mps/auto)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of DataLoader workers (default: 0, use 2-4 for MPS/CUDA)')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU training (useful if MPS has issues)')
    parser.add_argument('--patience', type=int, default=100,
                       help='Early stopping patience - epochs without improvement (default: 100)')
    parser.add_argument('--output', type=str, default='supervised_trust_model.pth',
                       help='Output model path')

    args = parser.parse_args()

    # Setup device with GPU acceleration support
    if args.force_cpu:
        device = 'cpu'
        print(f"🖥️  CPU training forced by user")
    elif args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print(f"🚀 Apple Silicon MPS detected")
            print(f"💡 If you encounter MPS issues, use --force-cpu")


            # Increase batch size for better MPS utilization
            if args.batch_size <= 128:
                original_batch_size = args.batch_size
                args.batch_size = 256  # Even larger batch size for MPS
                print(f"💡 Auto-increasing batch size from {original_batch_size} to {args.batch_size} for MPS")

            # Enable MPS memory optimizations
            try:
                if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                    torch.mps.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()  # Clear any cached memory
                print("💡 MPS memory optimizations enabled")
            except Exception as e:
                print(f"⚠️  Could not enable MPS memory optimizations: {e}")
        else:
            device = 'cpu'
            print(f"⚠️  No GPU acceleration available, using CPU")
    else:
        device = args.device

    print(f"🖥️  Using device: {device}")

    # Load and split dataset
    if not os.path.exists(args.data):
        print(f"❌ Dataset file not found: {args.data}")
        print("Please run generate_supervised_data.py first to create the dataset")
        return

    dataset = load_dataset(args.data)
    train_data, val_data = split_dataset(dataset)

    # Create data loaders with MPS/CUDA optimizations
    train_dataset = SupervisedTrustDataset(train_data)
    val_dataset = SupervisedTrustDataset(val_data)

    # Optimize DataLoader settings based on device
    pin_memory = 'cuda' in device
    num_workers = args.num_workers if args.num_workers > 0 else (2 if 'cuda' in device else (0 if device == 'mps' else 0))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, collate_fn=collate_batch,
                             num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_batch,
                           num_workers=num_workers, pin_memory=pin_memory)

    # Create model with NEW DESIGN: continuous ratio-based features (trust-free)
    model = SupervisedTrustGNN(
        agent_features=6,  # 6D continuous: observed_count, fused_count, expected_count, partner_count, detection_ratio, validator_ratio
        track_features=2,  # 2D continuous: detector_count, detector_ratio
        hidden_dim=64
    )

    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"👷 DataLoader workers: {num_workers}")
    print(f"📌 Pin memory: {pin_memory}")

    # Create trainer
    trainer = SupervisedTrustTrainer(model, device=device, learning_rate=args.lr)

    # Train model
    history = trainer.train(train_loader, val_loader, epochs=args.epochs, save_path=args.output, patience=args.patience)

    # Plot results
    plot_training_results(history)

    print("✅ Training completed successfully!")


if __name__ == "__main__":
    main()
