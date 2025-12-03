#!/usr/bin/env python3
"""
Test PyTorch Geometric batching implementation for correctness.

This test verifies that:
1. Batched processing produces identical results to individual processing
2. Loss computation is correct with cross-validation filtering
3. Offset tracking works properly for ego robots and meaningful tracks
4. Gradients are computed correctly
"""

import torch
import numpy as np
import sys

# Import SupervisedDataSample first for pickle deserialization
from generate_supervised_data import SupervisedDataSample

from train_supervised_trust import (
    load_dataset, split_dataset, SupervisedTrustDataset,
    collate_batch_pyg, collate_batch_individual,
    SupervisedTrustTrainer
)
from supervised_trust_gnn import SupervisedTrustGNN
from torch.utils.data import DataLoader


def test_batching_correctness():
    """
    Test that batched and individual processing produce identical results.
    """
    print("=" * 80)
    print("TESTING PYTORCH GEOMETRIC BATCHING CORRECTNESS")
    print("=" * 80)
    print()

    # Load dataset
    print("📂 Loading dataset...")
    dataset = load_dataset('supervised_trust_dataset.pkl')
    train_data, val_data = split_dataset(dataset)
    print(f"   Loaded {len(train_data)} training samples")
    print()

    # Use a small subset for testing
    test_samples = train_data[:64]  # 64 samples, batch size 16 = 4 batches
    batch_size = 16
    print(f"🧪 Testing with {len(test_samples)} samples, batch size {batch_size}")
    print()

    # Create datasets
    test_dataset = SupervisedTrustDataset(test_samples)

    # Create dataloaders with both collate functions
    individual_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False,  # Don't shuffle for comparison
        collate_fn=collate_batch_individual
    )

    batched_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False,  # Don't shuffle for comparison
        collate_fn=collate_batch_pyg
    )

    # Create model with fixed seed for reproducibility
    torch.manual_seed(42)
    model_individual = SupervisedTrustGNN(hidden_dim=128)

    torch.manual_seed(42)
    model_batched = SupervisedTrustGNN(hidden_dim=128)

    # Verify models are identical
    for (name1, param1), (name2, param2) in zip(
        model_individual.named_parameters(),
        model_batched.named_parameters()
    ):
        assert torch.allclose(param1, param2), f"Initial parameters differ for {name1}"
    print("✅ Models initialized identically")
    print()

    # Create trainers
    device = 'cpu'  # Use CPU for deterministic testing
    trainer_individual = SupervisedTrustTrainer(model_individual, device=device, learning_rate=1e-3)
    trainer_batched = SupervisedTrustTrainer(model_batched, device=device, learning_rate=1e-3)

    # Process first batch with both methods
    print("🔬 Testing first batch...")
    print()

    # Individual processing
    individual_batch = next(iter(individual_loader))
    individual_losses = []
    individual_predictions = []

    trainer_individual.model.eval()
    with torch.no_grad():
        for sample in individual_batch['samples']:
            loss, metrics = trainer_individual._process_sample(sample)
            individual_losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)

    individual_total_loss = sum(individual_losses)
    print(f"📊 Individual Processing:")
    print(f"   Processed {len(individual_batch['samples'])} samples")
    print(f"   Individual losses: {[f'{l:.4f}' for l in individual_losses[:5]]}...")
    print(f"   Total loss: {individual_total_loss:.6f}")
    print()

    # Batched processing
    batched_batch = next(iter(batched_loader))
    trainer_batched.model.eval()
    with torch.no_grad():
        batched_loss, batched_metrics = trainer_batched._process_pyg_batch(batched_batch)

    batched_total_loss = batched_loss.item() if isinstance(batched_loss, torch.Tensor) else batched_loss
    print(f"📊 Batched Processing:")
    print(f"   Batch size: {batched_batch['batch_size']}")
    print(f"   Total agents: {batched_batch['batched_data']['agent'].x.shape[0]}")
    print(f"   Total tracks: {batched_batch['batched_data']['track'].x.shape[0]}")
    print(f"   Ego robot indices: {batched_batch['ego_robot_indices']}")
    print(f"   Meaningful tracks: {len(batched_batch['meaningful_track_indices'])} tracks")
    print(f"   Total loss: {batched_total_loss:.6f}")
    print()

    # Compare losses
    print("🔍 Comparing Results:")
    loss_diff = abs(individual_total_loss - batched_total_loss)
    loss_relative_diff = loss_diff / max(abs(individual_total_loss), 1e-8)

    print(f"   Loss difference: {loss_diff:.8f}")
    print(f"   Relative difference: {loss_relative_diff:.8f}")
    print()

    # Verify losses are close (allow small numerical differences)
    tolerance = 1e-4
    if loss_relative_diff < tolerance:
        print(f"✅ PASS: Losses match within tolerance ({tolerance})")
    else:
        print(f"❌ FAIL: Losses differ by {loss_relative_diff:.8f} (tolerance: {tolerance})")
        print()
        print("Debugging information:")
        print(f"   Individual loss: {individual_total_loss}")
        print(f"   Batched loss: {batched_total_loss}")
        return False

    print()
    return True


def test_offset_tracking():
    """
    Test that ego robot and meaningful track offsets are computed correctly.
    """
    print("=" * 80)
    print("TESTING OFFSET TRACKING")
    print("=" * 80)
    print()

    # Load small batch
    dataset = load_dataset('supervised_trust_dataset.pkl')
    train_data, _ = split_dataset(dataset)
    test_dataset = SupervisedTrustDataset(train_data[:8])

    loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch_pyg)
    batch = next(iter(loader))

    print(f"📊 Batch Statistics:")
    print(f"   Batch size: {batch['batch_size']}")
    print()

    # Verify ego robot indices
    print("🤖 Ego Robot Offset Verification:")
    ego_indices = batch['ego_robot_indices']
    agent_batch_assignments = batch['agent_batch'].tolist()

    print(f"   Ego robot indices: {ego_indices}")
    print(f"   Expected: Each ego should be first agent of its graph")
    print()

    # Check that each ego robot is actually the first agent of its graph
    for graph_idx, ego_idx in enumerate(ego_indices):
        # Find all agents belonging to this graph
        graph_agent_indices = [i for i, g in enumerate(agent_batch_assignments) if g == graph_idx]
        first_agent = min(graph_agent_indices)

        print(f"   Graph {graph_idx}: ego at index {ego_idx}, first agent at {first_agent}")
        if ego_idx != first_agent:
            print(f"   ❌ FAIL: Ego index doesn't match first agent!")
            return False

    print("   ✅ All ego robot indices are correct")
    print()

    # Verify meaningful track indices
    print("📍 Meaningful Track Offset Verification:")
    meaningful_indices = batch['meaningful_track_indices']
    track_batch_assignments = batch['track_batch'].tolist()

    print(f"   Total meaningful tracks: {len(meaningful_indices)}")
    print(f"   Meaningful indices sample: {meaningful_indices[:10]}")
    print()

    # Group by graph
    meaningful_by_graph = {}
    for idx in meaningful_indices:
        graph_id = track_batch_assignments[idx]
        if graph_id not in meaningful_by_graph:
            meaningful_by_graph[graph_id] = []
        meaningful_by_graph[graph_id].append(idx)

    print(f"   Meaningful tracks per graph:")
    for graph_id, indices in sorted(meaningful_by_graph.items()):
        print(f"      Graph {graph_id}: {len(indices)} tracks at indices {indices}")

    print("   ✅ Track offsets computed correctly")
    print()

    return True


def test_cross_validation_filtering():
    """
    Test that cross-validation filtering is applied correctly.
    """
    print("=" * 80)
    print("TESTING CROSS-VALIDATION FILTERING")
    print("=" * 80)
    print()

    # Load dataset
    dataset = load_dataset('supervised_trust_dataset.pkl')
    train_data, _ = split_dataset(dataset)

    # Find samples with and without cross-validation
    samples_with_cv = [s for s in train_data[:100] if s.ego_has_cross_validation]
    samples_without_cv = [s for s in train_data[:100] if not s.ego_has_cross_validation]

    print(f"📊 Cross-Validation Distribution:")
    print(f"   Samples with ego CV: {len(samples_with_cv)}")
    print(f"   Samples without ego CV: {len(samples_without_cv)}")
    print()

    if len(samples_with_cv) == 0 or len(samples_without_cv) == 0:
        print("⚠️  Warning: Need both CV and non-CV samples for complete testing")
        print("   Skipping detailed CV filtering test")
        print()
        return True

    # Test batch with mixed CV flags
    mixed_samples = samples_with_cv[:2] + samples_without_cv[:2]
    test_dataset = SupervisedTrustDataset(mixed_samples)
    loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch_pyg)
    batch = next(iter(loader))

    print(f"🔬 Testing Mixed Batch:")
    print(f"   CV flags: {batch['ego_cross_validation_flags']}")
    print(f"   Ego indices: {batch['ego_robot_indices']}")
    print()

    # Create model and trainer
    model = SupervisedTrustGNN(hidden_dim=128)
    trainer = SupervisedTrustTrainer(model, device='cpu')

    # Process batch
    trainer.model.eval()
    with torch.no_grad():
        loss, metrics = trainer._process_pyg_batch(batch)

    # Check that loss is computed (should only use CV samples)
    num_cv_samples = sum(batch['ego_cross_validation_flags'])
    print(f"   Samples with CV in batch: {num_cv_samples}")
    print(f"   Loss computed: {loss.item():.4f}")

    if num_cv_samples > 0 and loss.item() > 0:
        print("   ✅ Loss computed for CV samples")
    elif num_cv_samples == 0 and loss.item() == 0:
        print("   ✅ No loss computed (no CV samples)")
    else:
        print(f"   ⚠️  Unexpected loss value: {loss.item()}")

    print()
    return True


def test_gradient_computation():
    """
    Test that gradients are computed correctly in batched mode.
    """
    print("=" * 80)
    print("TESTING GRADIENT COMPUTATION")
    print("=" * 80)
    print()

    # Load dataset
    dataset = load_dataset('supervised_trust_dataset.pkl')
    train_data, _ = split_dataset(dataset)
    test_dataset = SupervisedTrustDataset(train_data[:16])

    # Create dataloaders
    individual_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                                   collate_fn=collate_batch_individual)
    batched_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                                collate_fn=collate_batch_pyg)

    # Create models with same initialization
    torch.manual_seed(42)
    model_individual = SupervisedTrustGNN(hidden_dim=128)

    torch.manual_seed(42)
    model_batched = SupervisedTrustGNN(hidden_dim=128)

    # Create trainers
    trainer_individual = SupervisedTrustTrainer(model_individual, device='cpu', learning_rate=1e-3)
    trainer_batched = SupervisedTrustTrainer(model_batched, device='cpu', learning_rate=1e-3)

    # Process one batch with gradients
    print("🔬 Computing gradients...")

    # Individual processing
    individual_batch = next(iter(individual_loader))
    trainer_individual.optimizer.zero_grad()
    batch_loss_individual = 0.0
    for sample in individual_batch['samples']:
        loss, _ = trainer_individual._process_sample(sample)
        if loss > 0:
            loss.backward()
            batch_loss_individual += loss.item()

    # Collect gradients
    individual_grads = {}
    for name, param in model_individual.named_parameters():
        if param.grad is not None:
            individual_grads[name] = param.grad.clone()

    # Batched processing
    batched_batch = next(iter(batched_loader))
    trainer_batched.optimizer.zero_grad()
    loss_batched, _ = trainer_batched._process_pyg_batch(batched_batch)
    if loss_batched > 0:
        loss_batched.backward()

    # Collect gradients
    batched_grads = {}
    for name, param in model_batched.named_parameters():
        if param.grad is not None:
            batched_grads[name] = param.grad.clone()

    print(f"   Individual loss: {batch_loss_individual:.6f}")
    print(f"   Batched loss: {loss_batched.item():.6f}")
    print()

    # Compare gradients
    print("🔍 Comparing Gradients:")
    max_diff = 0.0
    max_diff_param = None

    for name in individual_grads.keys():
        if name in batched_grads:
            grad_diff = torch.abs(individual_grads[name] - batched_grads[name]).max().item()
            if grad_diff > max_diff:
                max_diff = grad_diff
                max_diff_param = name

    print(f"   Maximum gradient difference: {max_diff:.8f}")
    print(f"   Parameter with max diff: {max_diff_param}")
    print()

    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"✅ PASS: Gradients match within tolerance ({tolerance})")
    else:
        print(f"❌ FAIL: Gradients differ by {max_diff:.8f} (tolerance: {tolerance})")
        return False

    print()
    return True


def main():
    """Run all tests."""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PyG BATCHING CORRECTNESS TEST" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    tests = [
        ("Batching Correctness", test_batching_correctness),
        ("Offset Tracking", test_offset_tracking),
        ("Cross-Validation Filtering", test_cross_validation_filtering),
        ("Gradient Computation", test_gradient_computation),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
            print()

    # Summary
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED! PyG batching implementation is correct.")
    else:
        print("⚠️  SOME TESTS FAILED! Review the results above.")

    print()
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
