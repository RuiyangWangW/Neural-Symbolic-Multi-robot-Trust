#!/usr/bin/env python3
"""
Curriculum Learning for Supervised Trust GNN

This module implements curriculum learning by sorting training samples from easy to hard,
where "easy" means clear distinction between legitimate/adversarial agents and GT/FP tracks.

Strategy:
1. Easy: High contrast - adversarial agents have many contradictions and FP detections
2. Hard: Low contrast - adversarial agents behave more like legitimate agents
"""

import torch
import numpy as np
import pickle
from typing import List, Dict, Tuple
from dataclasses import dataclass
from generate_supervised_data import SupervisedDataSample


@dataclass
class SampleDifficulty:
    """Stores difficulty metrics for a training sample"""
    sample_idx: int
    sample: SupervisedDataSample

    # Agent-level metrics
    agent_contradiction_contrast: float  # Difference in contradiction edges between legit and adv
    agent_fp_detection_clarity: float    # How clearly adversarial agents detect FPs

    # Track-level metrics
    track_fov_only_contrast: float       # Difference in in_fov_only edges between GT and FP
    track_detection_clarity: float       # How clearly FP tracks have more fov_only than detections

    # Overall difficulty score (lower = easier)
    difficulty_score: float


def compute_sample_difficulty(sample: SupervisedDataSample, sample_idx: int) -> SampleDifficulty:
    """
    Compute difficulty metrics for a training sample.

    Easy samples have:
    - Adversarial agents with many contradiction edges
    - FP tracks detected only by adversarial agents
    - Clear separation between legitimate and adversarial patterns

    Hard samples have:
    - Adversarial agents behaving like legitimate agents
    - FP tracks detected by multiple agents
    - Ambiguous patterns

    Args:
        sample: Training sample
        sample_idx: Index in dataset

    Returns:
        SampleDifficulty object with computed metrics
    """

    # Extract edge information
    edge_index_dict = sample.edge_index_dict
    agent_labels = sample.agent_labels  # 1 = legitimate, 0 = adversarial
    track_labels = sample.track_labels  # 1 = GT, 0 = FP

    num_agents = sample.num_agents
    num_tracks = sample.num_tracks

    # Initialize metrics
    agent_contradiction_contrast = 0.0
    agent_fp_detection_clarity = 0.0
    track_fov_only_contrast = 0.0
    track_detection_clarity = 0.0

    # ========================================================================
    # METRIC 1: Agent Contradiction Contrast
    # ========================================================================
    # Easy: Adversarial agents have many contradictions, legitimate have few

    if ('agent', 'contradicts', 'agent') in edge_index_dict:
        contra_edges = edge_index_dict[('agent', 'contradicts', 'agent')]

        # Count contradictions per agent
        agent_contra_count = torch.zeros(num_agents)
        for i in range(contra_edges.shape[1]):
            src = contra_edges[0, i].item()
            agent_contra_count[src] += 1

        # Average contradictions for legitimate vs adversarial
        # Handle both 1D and 2D label tensors
        agent_labels_flat = agent_labels.flatten() if agent_labels.dim() > 1 else agent_labels

        legit_mask = agent_labels_flat == 1
        adv_mask = agent_labels_flat == 0

        if legit_mask.any() and adv_mask.any():
            legit_avg_contra = agent_contra_count[legit_mask].mean().item()
            adv_avg_contra = agent_contra_count[adv_mask].mean().item()

            # Contrast: higher difference = easier
            # Normalize by total agents to make comparable across samples
            agent_contradiction_contrast = abs(adv_avg_contra - legit_avg_contra) / max(num_agents, 1)

    # ========================================================================
    # METRIC 2: Agent FP Detection Clarity
    # ========================================================================
    # Easy: Only adversarial agents detect FP tracks

    if ('agent', 'in_fov_and_observed', 'track') in edge_index_dict:
        detection_edges = edge_index_dict[('agent', 'in_fov_and_observed', 'track')]

        # Count FP detections by agent type
        legit_fp_detections = 0
        adv_fp_detections = 0

        for i in range(detection_edges.shape[1]):
            agent_idx = detection_edges[0, i].item()
            track_idx = detection_edges[1, i].item()

            if track_idx < len(track_labels):  # Safety check
                is_fp = track_labels[track_idx].item() == 0
                is_adv = agent_labels[agent_idx].item() == 0

                if is_fp:
                    if is_adv:
                        adv_fp_detections += 1
                    else:
                        legit_fp_detections += 1

        # Clarity: high if only adversarial detect FPs
        total_fp_detections = legit_fp_detections + adv_fp_detections
        if total_fp_detections > 0:
            # Ratio of adversarial FP detections to total FP detections
            agent_fp_detection_clarity = adv_fp_detections / total_fp_detections
        else:
            # No FP detections at all - neutral difficulty
            agent_fp_detection_clarity = 0.5

    # ========================================================================
    # METRIC 3: Track FOV-Only Contrast
    # ========================================================================
    # Easy: FP tracks have many in_fov_only edges, GT tracks have few

    if ('agent', 'in_fov_only', 'track') in edge_index_dict:
        fov_only_edges = edge_index_dict[('agent', 'in_fov_only', 'track')]

        # Count in_fov_only per track
        track_fov_only_count = torch.zeros(num_tracks)
        for i in range(fov_only_edges.shape[1]):
            track_idx = fov_only_edges[1, i].item()
            track_fov_only_count[track_idx] += 1

        # Average for GT vs FP
        # Handle both 1D and 2D label tensors
        track_labels_flat = track_labels.flatten() if track_labels.dim() > 1 else track_labels

        gt_mask = track_labels_flat == 1
        fp_mask = track_labels_flat == 0

        if gt_mask.any() and fp_mask.any():
            gt_avg_fov_only = track_fov_only_count[gt_mask].mean().item()
            fp_avg_fov_only = track_fov_only_count[fp_mask].mean().item()

            # Contrast: higher difference = easier
            track_fov_only_contrast = abs(fp_avg_fov_only - gt_avg_fov_only) / max(num_tracks, 1)

    # ========================================================================
    # METRIC 4: Track Detection Clarity
    # ========================================================================
    # Easy: FP tracks have high ratio of in_fov_only to detections

    if ('agent', 'in_fov_and_observed', 'track') in edge_index_dict and \
       ('agent', 'in_fov_only', 'track') in edge_index_dict:

        detection_edges = edge_index_dict[('agent', 'in_fov_and_observed', 'track')]
        fov_only_edges = edge_index_dict[('agent', 'in_fov_only', 'track')]

        # Count per track
        track_detections = torch.zeros(num_tracks)
        track_fov_only = torch.zeros(num_tracks)

        for i in range(detection_edges.shape[1]):
            track_idx = detection_edges[1, i].item()
            track_detections[track_idx] += 1

        for i in range(fov_only_edges.shape[1]):
            track_idx = fov_only_edges[1, i].item()
            track_fov_only[track_idx] += 1

        # For FP tracks: ratio of fov_only to detections
        # High ratio = easy (many agents see but don't detect)
        # Handle both 1D and 2D label tensors
        track_labels_flat = track_labels.flatten() if track_labels.dim() > 1 else track_labels
        fp_mask = track_labels_flat == 0
        if fp_mask.any():
            fp_fov_only = track_fov_only[fp_mask]
            fp_detections = track_detections[fp_mask]

            # Avoid division by zero
            fp_detections_safe = torch.clamp(fp_detections, min=1.0)

            # Ratio: higher = easier
            fp_ratios = fp_fov_only / fp_detections_safe
            track_detection_clarity = fp_ratios.mean().item()

    # ========================================================================
    # COMPUTE OVERALL DIFFICULTY SCORE
    # ========================================================================
    # Combine metrics into single score (lower = easier)

    # Weights for each metric
    w1 = 0.3  # Agent contradiction contrast
    w2 = 0.3  # Agent FP detection clarity
    w3 = 0.2  # Track FOV-only contrast
    w4 = 0.2  # Track detection clarity

    # Invert clarity metrics (higher clarity = lower difficulty)
    agent_contra_difficulty = 1.0 / (1.0 + agent_contradiction_contrast)
    agent_fp_difficulty = 1.0 - agent_fp_detection_clarity  # Already 0-1
    track_fov_difficulty = 1.0 / (1.0 + track_fov_only_contrast)
    track_det_difficulty = 1.0 / (1.0 + track_detection_clarity)

    difficulty_score = (
        w1 * agent_contra_difficulty +
        w2 * agent_fp_difficulty +
        w3 * track_fov_difficulty +
        w4 * track_det_difficulty
    )

    return SampleDifficulty(
        sample_idx=sample_idx,
        sample=sample,
        agent_contradiction_contrast=agent_contradiction_contrast,
        agent_fp_detection_clarity=agent_fp_detection_clarity,
        track_fov_only_contrast=track_fov_only_contrast,
        track_detection_clarity=track_detection_clarity,
        difficulty_score=difficulty_score
    )


def sort_dataset_by_difficulty(dataset: List[SupervisedDataSample]) -> Tuple[List[SupervisedDataSample], List[SampleDifficulty]]:
    """
    Sort dataset from easiest to hardest samples.

    Args:
        dataset: List of training samples

    Returns:
        Tuple of (sorted_dataset, difficulty_metrics)
    """
    print(f"Computing difficulty metrics for {len(dataset)} samples...")

    # Compute difficulty for each sample
    difficulties = []
    for idx, sample in enumerate(dataset):
        if idx % 1000 == 0:
            print(f"  Processing sample {idx}/{len(dataset)}...")

        difficulty = compute_sample_difficulty(sample, idx)
        difficulties.append(difficulty)

    # Sort by difficulty score (ascending = easy to hard)
    difficulties.sort(key=lambda d: d.difficulty_score)

    # Extract sorted samples
    sorted_dataset = [d.sample for d in difficulties]

    print(f"✅ Dataset sorted by difficulty")
    print(f"   Easiest sample difficulty: {difficulties[0].difficulty_score:.4f}")
    print(f"   Hardest sample difficulty: {difficulties[-1].difficulty_score:.4f}")
    print(f"   Median sample difficulty: {difficulties[len(difficulties)//2].difficulty_score:.4f}")

    return sorted_dataset, difficulties


def create_curriculum_schedule(num_samples: int, num_epochs: int, strategy: str = 'performance') -> List[int]:
    """
    Create curriculum schedule placeholder for performance-based strategy.

    For performance-based curriculum, the actual schedule is determined dynamically
    during training based on achieving performance thresholds. This function creates
    a simple step-based placeholder schedule (10% → 20% → 30% → ... → 100%).

    Args:
        num_samples: Total number of training samples
        num_epochs: Total number of training epochs
        strategy: Should be 'performance' (only supported strategy)

    Returns:
        List of sample counts for each epoch (placeholder, not actively used)
    """
    if strategy != 'performance':
        raise ValueError(f"Only 'performance' strategy is supported, got: {strategy}")

    schedule = []

    # Create step-based placeholder: 10%, 20%, 30%, ..., 100%
    # This is not actively used - actual curriculum advancement is performance-based
    for epoch in range(num_epochs):
        # Start at 10%, increase in 10% steps
        # Spread the steps across epochs
        step_size = 0.10
        num_steps = int(1.0 / step_size)  # 10 steps total (10% to 100%)
        epochs_per_step = max(1, num_epochs // num_steps)

        current_step = min(num_steps - 1, epoch // epochs_per_step)
        pct = min(1.0, (current_step + 1) * step_size)

        num_samples_epoch = int(num_samples * pct)
        schedule.append(num_samples_epoch)

    return schedule


class CurriculumDataLoader:
    """
    DataLoader for performance-based curriculum learning.

    This class sorts the dataset by difficulty and provides access to the sorted dataset.
    The actual curriculum advancement (deciding how much data to use) is handled by the
    training loop based on achieving performance thresholds.
    """

    def __init__(self, dataset: List[SupervisedDataSample], batch_size: int,
                 num_epochs: int, strategy: str = 'performance', shuffle_within_curriculum: bool = True):
        """
        Initialize curriculum dataloader.

        Args:
            dataset: Training dataset (will be sorted by difficulty internally)
            batch_size: Batch size
            num_epochs: Total number of training epochs
            strategy: Curriculum strategy (only 'performance' is supported)
            shuffle_within_curriculum: Whether to shuffle samples within current difficulty level
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.strategy = strategy
        self.shuffle_within_curriculum = shuffle_within_curriculum

        # Sort dataset by difficulty
        print("Initializing performance-based curriculum learning...")
        self.sorted_dataset, self.difficulties = sort_dataset_by_difficulty(dataset)

        # Create placeholder schedule (not actively used for performance-based)
        self.schedule = create_curriculum_schedule(
            len(self.sorted_dataset), num_epochs, strategy
        )

        print(f"\n📚 Performance-Based Curriculum:")
        print(f"   Dataset sorted by difficulty: {len(self.sorted_dataset)} samples")
        print(f"   Starting with: 10% of dataset")
        print(f"   Advancement: Based on achieving performance thresholds")
        print(f"   Step size: 10% increments")
        print()

    def get_epoch_data(self, epoch: int) -> List[SupervisedDataSample]:
        """
        Get training data for a specific epoch based on curriculum.

        Args:
            epoch: Epoch number (0-indexed)

        Returns:
            List of samples for this epoch
        """
        if epoch >= len(self.schedule):
            # Use full dataset after curriculum ends
            num_samples = len(self.sorted_dataset)
        else:
            num_samples = self.schedule[epoch]

        # Get easiest N samples
        epoch_data = self.sorted_dataset[:num_samples]

        # Optionally shuffle within this subset
        if self.shuffle_within_curriculum:
            indices = torch.randperm(len(epoch_data)).tolist()
            epoch_data = [epoch_data[i] for i in indices]

        return epoch_data

    def get_statistics(self) -> Dict:
        """Get curriculum statistics for monitoring"""
        return {
            'total_samples': len(self.sorted_dataset),
            'strategy': self.strategy,
            'schedule': self.schedule,
            'easiest_difficulty': self.difficulties[0].difficulty_score,
            'hardest_difficulty': self.difficulties[-1].difficulty_score,
            'median_difficulty': self.difficulties[len(self.difficulties)//2].difficulty_score
        }


def analyze_curriculum_difficulty(dataset_path: str):
    """
    Analyze and visualize difficulty distribution in dataset.

    Args:
        dataset_path: Path to dataset pickle file
    """
    import matplotlib.pyplot as plt

    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    # Handle different dataset formats
    if isinstance(data, dict) and 'samples' in data:
        dataset = data['samples']
        print(f"Loaded dataset dictionary")
        if 'statistics' in data:
            print(f"  Statistics: {data['statistics']}")
    elif isinstance(data, list):
        dataset = data
    else:
        raise ValueError(f"Unknown dataset format: {type(data)}")

    print(f"Number of samples: {len(dataset)}")
    print()

    # Compute difficulties
    _, difficulties = sort_dataset_by_difficulty(dataset)

    # Extract metrics
    difficulty_scores = [d.difficulty_score for d in difficulties]
    agent_contra = [d.agent_contradiction_contrast for d in difficulties]
    agent_fp = [d.agent_fp_detection_clarity for d in difficulties]
    track_fov = [d.track_fov_only_contrast for d in difficulties]
    track_det = [d.track_detection_clarity for d in difficulties]

    # Plot distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Overall difficulty
    axes[0, 0].hist(difficulty_scores, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Overall Difficulty Distribution')
    axes[0, 0].set_xlabel('Difficulty Score (lower = easier)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(np.median(difficulty_scores), color='red', linestyle='--', label='Median')
    axes[0, 0].legend()

    # Agent contradiction contrast
    axes[0, 1].hist(agent_contra, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[0, 1].set_title('Agent Contradiction Contrast')
    axes[0, 1].set_xlabel('Contrast (higher = easier)')
    axes[0, 1].set_ylabel('Count')

    # Agent FP detection clarity
    axes[0, 2].hist(agent_fp, bins=50, alpha=0.7, edgecolor='black', color='blue')
    axes[0, 2].set_title('Agent FP Detection Clarity')
    axes[0, 2].set_xlabel('Clarity (higher = easier)')
    axes[0, 2].set_ylabel('Count')

    # Track FOV-only contrast
    axes[1, 0].hist(track_fov, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].set_title('Track FOV-Only Contrast')
    axes[1, 0].set_xlabel('Contrast (higher = easier)')
    axes[1, 0].set_ylabel('Count')

    # Track detection clarity
    axes[1, 1].hist(track_det, bins=50, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 1].set_title('Track Detection Clarity')
    axes[1, 1].set_xlabel('Clarity (higher = easier)')
    axes[1, 1].set_ylabel('Count')

    # Difficulty over sorted samples
    axes[1, 2].plot(range(len(difficulty_scores)), difficulty_scores, alpha=0.7)
    axes[1, 2].set_title('Difficulty Progression (Sorted)')
    axes[1, 2].set_xlabel('Sample Index (sorted)')
    axes[1, 2].set_ylabel('Difficulty Score')
    axes[1, 2].axhline(np.median(difficulty_scores), color='red', linestyle='--', label='Median')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig('curriculum_difficulty_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved difficulty analysis: curriculum_difficulty_analysis.png")
    plt.close()

    # Print statistics
    print("\n" + "="*80)
    print("CURRICULUM DIFFICULTY STATISTICS")
    print("="*80)
    print(f"\nOverall Difficulty:")
    print(f"  Min: {min(difficulty_scores):.4f}")
    print(f"  Max: {max(difficulty_scores):.4f}")
    print(f"  Mean: {np.mean(difficulty_scores):.4f}")
    print(f"  Median: {np.median(difficulty_scores):.4f}")
    print(f"  Std: {np.std(difficulty_scores):.4f}")

    print(f"\nEasiest 10% samples:")
    print(f"  Avg difficulty: {np.mean(difficulty_scores[:len(difficulty_scores)//10]):.4f}")

    print(f"\nHardest 10% samples:")
    print(f"  Avg difficulty: {np.mean(difficulty_scores[-len(difficulty_scores)//10:]):.4f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "supervised_trust_dataset.pkl"

    analyze_curriculum_difficulty(dataset_path)
