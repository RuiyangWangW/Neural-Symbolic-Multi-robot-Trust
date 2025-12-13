#!/usr/bin/env python3
"""
Curriculum Learning for Supervised Trust GNN

This module implements curriculum learning by sorting training samples from easy to hard,
based on the EGO ROBOT'S OWN EDGE PATTERNS.

Strategy (focusing on ego robot's behavioral signature):

For ADVERSARIAL ego robots:
  - Easy: Many contradicts edges + few co_detection edges (clear adversarial signature)
  - Hard: Few contradicts edges + many co_detection edges (behaving like legitimate)

For LEGITIMATE ego robots:
  - Easy: Many co_detection edges + few contradicts edges (clear collaborative signature)
  - Hard: Few co_detection edges + many contradicts edges (behaving suspiciously)

Each sample is an ego-centric graph. Difficulty is based solely on the ego robot's
own edges, not averaged across all agents.
"""

import torch
import numpy as np
import pickle
from typing import List, Dict, Tuple
from dataclasses import dataclass
from generate_supervised_data import SupervisedDataSample


@dataclass
class SampleDifficulty:
    """
    Stores difficulty metrics for a training sample (ego-centric graph).

    Difficulty is based solely on the EGO ROBOT'S OWN EDGES (ego robot is at index 0).
    """
    sample_idx: int
    sample: SupervisedDataSample

    # Ego robot's own edge counts (stored for analysis, renamed for clarity)
    agent_contradiction_contrast: float  # Now stores: ego robot's contradiction edge count
    agent_fp_detection_clarity: float    # Now stores: ego robot's co_detection edge count

    # Track-level metrics (kept for backward compatibility, set to 0, NOT used for difficulty)
    track_fov_only_contrast: float       # Not used (set to 0)
    track_detection_clarity: float       # Not used (set to 0)

    # Overall difficulty score (lower = easier)
    # Based on ego robot's signature strength:
    #   - Adversarial ego: low difficulty if many contradicts + few codetection
    #   - Legitimate ego: low difficulty if many codetection + few contradicts
    difficulty_score: float


def compute_sample_difficulty(sample: SupervisedDataSample, sample_idx: int) -> SampleDifficulty:
    """
    Compute difficulty based solely on the EGO ROBOT'S OWN EDGE PATTERNS.

    The ego robot is always at index 0 in the ego-centric graph. We measure difficulty
    by how clearly the ego robot's edges match its true type.

    For ADVERSARIAL ego robots:
      - Easy: Many contradicts edges + few co_detection edges
        → Clear adversarial signature (conflicts with others, doesn't collaborate)
      - Hard: Few contradicts edges + many co_detection edges
        → Behaves like legitimate (collaborates, doesn't conflict)

    For LEGITIMATE ego robots:
      - Easy: Many co_detection edges + few contradicts edges
        → Clear legitimate signature (collaborates well, doesn't conflict)
      - Hard: Few co_detection edges + many contradicts edges
        → Behaves suspiciously (conflicts with others, doesn't collaborate)

    Args:
        sample: Training sample (ego-centric graph, ego robot at index 0)
        sample_idx: Index in dataset

    Returns:
        SampleDifficulty object with difficulty score based on ego robot's behavioral signature
    """

    # Extract edge information
    edge_index_dict = sample.edge_index_dict
    agent_labels = sample.agent_labels  # 1 = legitimate, 0 = adversarial

    num_agents = sample.num_agents

    # Ego robot is always at index 0 in ego-centric graph
    EGO_IDX = 0
    ego_label = agent_labels[EGO_IDX].item()  # 1 = legitimate, 0 = adversarial
    is_ego_adversarial = (ego_label == 0)

    # ========================================================================
    # METRIC 1: Ego Robot's Contradiction Edges
    # ========================================================================
    # Count contradiction edges where ego robot is the source

    ego_contradiction_count = 0
    if ('agent', 'contradicts', 'agent') in edge_index_dict:
        contra_edges = edge_index_dict[('agent', 'contradicts', 'agent')]
        for i in range(contra_edges.shape[1]):
            src = contra_edges[0, i].item()
            if src == EGO_IDX:
                ego_contradiction_count += 1

    # ========================================================================
    # METRIC 2: Ego Robot's Co-detection Edges
    # ========================================================================
    # Count co_detection edges where ego robot is the source

    ego_codetection_count = 0
    if ('agent', 'co_detection', 'agent') in edge_index_dict:
        codet_edges = edge_index_dict[('agent', 'co_detection', 'agent')]
        for i in range(codet_edges.shape[1]):
            src = codet_edges[0, i].item()
            if src == EGO_IDX:
                ego_codetection_count += 1

    # ========================================================================
    # COMPUTE DIFFICULTY BASED ON EGO ROBOT'S BEHAVIORAL SIGNATURE
    # ========================================================================
    #
    # For ADVERSARIAL ego robots:
    #   - Strong signature: many contradicts edges, few co_detection edges → EASY
    #   - Weak signature: few contradicts edges, many co_detection edges → HARD
    #
    # For LEGITIMATE ego robots:
    #   - Strong signature: many co_detection edges, few contradicts edges → EASY
    #   - Weak signature: few co_detection edges, many contradicts edges → HARD
    #
    # We normalize by number of other agents to make scores comparable across graphs

    num_other_agents = max(1, num_agents - 1)  # Exclude ego itself

    # Normalize edge counts by number of other agents (max possible edges)
    norm_contradicts = ego_contradiction_count / num_other_agents
    norm_codetection = ego_codetection_count / num_other_agents

    if is_ego_adversarial:
        # For adversarial: High contradicts + low codetection = easy (clear adversarial signature)
        # We want high contradicts to decrease difficulty, low codetection to decrease difficulty

        # Signature strength: high contradicts = good, high codetection = bad
        signature_strength = norm_contradicts - norm_codetection

        # Convert to difficulty: strong signature (high value) → low difficulty
        # Use sigmoid-like transformation to map to [0, 1] range
        # Positive signature_strength → low difficulty
        # Negative signature_strength → high difficulty
        difficulty_score = 1.0 / (1.0 + np.exp(2.0 * signature_strength))

    else:
        # For legitimate: High codetection + low contradicts = easy (clear legitimate signature)
        # We want high codetection to decrease difficulty, low contradicts to decrease difficulty

        # Signature strength: high codetection = good, high contradicts = bad
        signature_strength = norm_codetection - norm_contradicts

        # Convert to difficulty: strong signature → low difficulty
        difficulty_score = 1.0 / (1.0 + np.exp(2.0 * signature_strength))

    # Store metrics for backward compatibility
    # agent_contradiction_contrast now stores ego's contradiction count
    # agent_fp_detection_clarity now stores ego's codetection count
    agent_contradiction_contrast = float(ego_contradiction_count)
    agent_fp_detection_clarity = float(ego_codetection_count)

    # Track metrics set to 0 (not used, kept for backward compatibility)
    track_fov_only_contrast = 0.0
    track_detection_clarity = 0.0

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
    Sort dataset from easiest to hardest samples based on ego robot classification difficulty.

    IMPORTANT: Adversarial and legitimate samples are sorted SEPARATELY to maintain
    class balance during curriculum learning. When sampling for training, samples are
    drawn equally from both sorted lists.

    Difficulty focuses solely on how easy it is to classify the ego robot correctly.

    Args:
        dataset: List of training samples (each is an ego-centric graph)

    Returns:
        Tuple of (interleaved_sorted_dataset, all_difficulty_metrics)
        The returned dataset interleaves adversarial and legitimate samples to maintain balance.
    """
    print(f"Computing difficulty metrics for {len(dataset)} samples...")

    # Compute difficulty for each sample
    difficulties = []
    for idx, sample in enumerate(dataset):
        if idx % 1000 == 0:
            print(f"  Processing sample {idx}/{len(dataset)}...")

        difficulty = compute_sample_difficulty(sample, idx)
        difficulties.append(difficulty)

    # Separate into adversarial and legitimate samples
    adv_difficulties = []
    leg_difficulties = []

    for d in difficulties:
        ego_label = d.sample.agent_labels[0].item()
        if ego_label == 0:  # Adversarial
            adv_difficulties.append(d)
        else:  # Legitimate
            leg_difficulties.append(d)

    # Sort each class separately by difficulty (easy to hard)
    adv_difficulties.sort(key=lambda d: d.difficulty_score)
    leg_difficulties.sort(key=lambda d: d.difficulty_score)

    # Interleave to create balanced sorted dataset
    # Take samples alternately from adversarial and legitimate lists
    interleaved_dataset = []
    interleaved_difficulties = []

    max_len = max(len(adv_difficulties), len(leg_difficulties))

    for i in range(max_len):
        # Add from adversarial list if available
        if i < len(adv_difficulties):
            interleaved_dataset.append(adv_difficulties[i].sample)
            interleaved_difficulties.append(adv_difficulties[i])

        # Add from legitimate list if available
        if i < len(leg_difficulties):
            interleaved_dataset.append(leg_difficulties[i].sample)
            interleaved_difficulties.append(leg_difficulties[i])

    print(f"✅ Dataset sorted by difficulty (class-balanced)")
    print(f"   Adversarial samples: {len(adv_difficulties)}")
    print(f"     Easiest: {adv_difficulties[0].difficulty_score:.4f}")
    print(f"     Hardest: {adv_difficulties[-1].difficulty_score:.4f}")
    print(f"     Median:  {adv_difficulties[len(adv_difficulties)//2].difficulty_score:.4f}")
    print(f"   Legitimate samples: {len(leg_difficulties)}")
    print(f"     Easiest: {leg_difficulties[0].difficulty_score:.4f}")
    print(f"     Hardest: {leg_difficulties[-1].difficulty_score:.4f}")
    print(f"     Median:  {leg_difficulties[len(leg_difficulties)//2].difficulty_score:.4f}")

    return interleaved_dataset, interleaved_difficulties


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
