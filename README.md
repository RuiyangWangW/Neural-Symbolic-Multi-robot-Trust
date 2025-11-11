# Neural-Symbolic Multi-Robot Trust System with IQL

A neural-symbolic framework for distributed trust management in multi-robot systems using Graph Neural Networks (GNNs) and Implicit Q-Learning (IQL) for adaptive, continuous trust assessment.

## System Overview

This system implements an end-to-end trust management framework for multi-robot collaborative scenarios:

1. **Evidence Extraction** (Supervised GNN): Processes ego-centric heterogeneous graphs to extract trust evidence scores for robots and their detected tracks
2. **Adaptive Trust Updates** (IQL Policy): Learns optimal step scales for updating Beta-distributed trust values based on environmental context
3. **Trust Representation**: Each entity (robot/track) maintains trust as Beta distribution Beta(α, β) with natural exponential decay
4. **Event-Based MDP**: Robots log state transitions when they detect tracks, enabling efficient learning from sparse but informative events

## Key Design Decisions

### 1. Robot-Only MDP Architecture

**Design**: Only robots are RL agents with learned policies. Tracks use fixed step scales.

**Rationale**:
- **Robots are persistent**: Live entire episode (100 timesteps), enabling long-term strategy learning
- **Robots are active**: Continuously detect and evaluate tracks (~82 events per episode)
- **Robot trust is complex**: Depends on long-term behavioral patterns requiring adaptive step scales
- **Track trust is simpler**: Binary classification (real/fake) works well with fixed step scale (0.5)

**Implementation**:
- Robots: 5D state features, learned step scales via IQL
- Tracks: Fixed step scale of 0.5 for all trust updates

### 2. Event-Based State Transitions

**Design**: Log transitions only when robots detect tracks (new or existing).

**Rationale**:
- **Sparse but informative**: Detections are decision points requiring trust updates
- **Natural event structure**: Robots act when they have evidence (detected tracks)
- **Efficient learning**: Avoids redundant "no-op" transitions when robots detect nothing

**Implementation**:
- Event triggers: `robot.get_current_timestep_tracks()` returns non-empty list
- Average: ~82 events per robot per 100-timestep episode
- Time gaps: γ^Δt discounting handles variable temporal spacing (avg Δt ≈ 1.2 timesteps)

### 3. Beta Trust Distribution with Exponential Decay

**Design**: Trust represented as Beta(α, β) with exponential decay γ per timestep.

**Rationale**:
- **Principled uncertainty**: Beta distribution naturally models binary trust (legitimate vs adversarial)
- **Forgetting mechanism**: Exponential decay (γ=0.995) ensures old evidence fades
- **Continuous evolution**: Trust decays between updates, captured by γ^Δt

**Update Rule**:
```
α_new = α_old · γ^Δt + step_scale · evidence
β_new = β_old · γ^Δt + step_scale · (1 - evidence)
τ = α / (α + β)  # Mean trust value
```

### 4. 5D State Features (Robots Only)

**State representation for each robot**:
1. **evidence** (e): GNN trust score ∈ [0, 1]
2. **tau** (τ): Mean trust value = α/(α+β)
3. **kappa** (κ): Trust mass = α + β (confidence indicator)
4. **expected_visible**: Count of tracks in robot's FOV **with trust > 0.6** (potential collaborators)
5. **trusted_count**: Count of **detected** tracks with trust > 0.6 (current observations)

**Feature Consistency**:
- Both `expected_visible` and `trusted_count` use the **same trust threshold (0.6)** filter
- **expected_visible**: ALL tracks in FOV with high trust (broader context - what's available)
- **trusted_count**: DETECTED tracks with high trust (narrower context - what's being used)
- This makes the features interpretable: "X trustworthy tracks available, Y currently being observed"

**Why 5D (not 6D)**:
- Removed `entity_type` feature (was always 1.0 for robots, redundant)
- Simplified network architecture
- Tracks not in MDP, so no need to distinguish entity types

## Technical Architecture

### Pipeline Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                         │
└──────────────────────────────────────────────────────────────┘
1. Generate Supervised Data
   └─> robot-track pairs with ground truth labels

2. Train Supervised GNN
   └─> Evidence extractor: graph → trust scores

3. Generate Offline RL Dataset (Event-Based)
   └─> Robot trajectories: (state, action, reward, next_state, Δt)

4. Train IQL Policy
   └─> Learn optimal step scales for trust updates

5. Deploy & Evaluate
   └─> Compare: Paper baseline vs RL policy

┌──────────────────────────────────────────────────────────────┐
│                   RUNTIME ARCHITECTURE                        │
└──────────────────────────────────────────────────────────────┘
For each timestep:
  For each robot (ego):
    1. Build ego-centric graph (robot + visible peers + tracks)
    2. GNN extracts evidence scores
    3. IQL policy predicts step scale for robot
    4. Update robot trust: α/β += step_scale * evidence
    5. Tracks updated with FIXED step scale = 0.5
    6. Apply exponential decay: α/β *= γ
```

### IQL Training Architecture

**Networks**:
- **Q-Critic**: Estimates Q(s,a) for state-action pairs
- **V-Network**: Estimates state value V(s) via expectile regression
- **Beta Policy**: Outputs step scales ∈ [0,1] using Beta distribution

**Loss Functions**:
1. Q-loss: `MSE(Q(s,a), r + γ^Δt · V_target(s'))`
2. V-loss: `Expectile_loss(Q_target(s,a) - V(s))` with τ=0.7
3. Actor-loss: `Advantage-weighted regression` with temperature β=0.3

**Key Hyperparameters**:
- Discount: γ = 0.995 (per timestep, variable spacing via γ^Δt)
- Expectile: τ = 0.7 (for V-network)
- Temperature: β = 0.3 (for actor, tuned to prevent mode collapse)
- Detection weight: λ = 5.0 (oversample detection events during training)

### Reward Function

**Calibration-improvement reward** (100x scaled):
```python
def calibration_improvement_reward(tau_pre, tau_post, ground_truth):
    y = ground_truth  # 0 for adversarial, 1 for legitimate
    pre_error = (tau_pre - y)^2
    post_error = (tau_post - y)^2
    reward = (pre_error - post_error) * 100.0  # Scaled
    return clip(reward, -100, 100)
```

**Why 100x scaling**: Prevents numerical collapse in Q/V networks. Original rewards ~0.003 are too small for stable learning.

## Quick Start

### Prerequisites

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Automated Pipeline (Recommended)

```bash
# Run complete training pipeline
source .venv/bin/activate
chmod +x run_iql_pipeline.sh
./run_iql_pipeline.sh
```

This executes all 5 steps sequentially:
1. Generate supervised dataset (200 episodes)
2. Train supervised GNN
3. Generate offline RL dataset (100 episodes)
4. Train IQL policy (100K updates)
5. Run three-scenario comparison

### Manual Step-by-Step

```bash
source .venv/bin/activate

# Step 1: Generate supervised training data
python generate_supervised_data.py --episodes 200

# Step 2: Train supervised GNN for evidence extraction
python train_supervised_trust.py \
  --data supervised_trust_dataset.pkl \
  --epochs 100

# Step 3: Generate event-based offline RL dataset
python generate_offline_dataset_eventbased.py \
  --episodes 100 \
  --supervised-model supervised_trust_model.pth \
  --output offline_dataset_eventbased.npz

# Step 4: Train IQL policy
python train_iql_trust.py \
  --dataset offline_dataset_eventbased.npz \
  --num-updates 100000 \
  --device cpu

# Step 5: Evaluate on three scenarios
python compare_trust_methods.py
```

## Dataset Structure

### Event-Based Offline Dataset

**File**: `offline_dataset_eventbased.npz`

**Contents**:
```python
{
  'states': (N, 5) float32,           # Robot state features
  'actions': (N,) float32,            # Step scales [0,1]
  'rewards': (N,) float32,            # Calibration improvement
  'next_states': (N, 5) float32,      # Next state features
  'dones': (N,) float32,              # 1=terminal, 0=continue
  'delta_prevs': (N,) int32,          # Time since last event
  'delta_nexts': (N,) int32,          # Time to next event
  'trajectory_starts': (T,) int64,    # Start index per trajectory
  'trajectory_lengths': (T,) int64,   # Length per trajectory
  'gamma': 0.995,                     # Decay factor
  'feature_means': (5,) float32,      # For normalization
  'feature_stds': (5,) float32,       # For normalization
}
```

**Statistics** (100 episodes):
```
Total transitions: ~200K (100 episodes × 24 robots × 82 events)
Total trajectories: ~2,400 robots
Average trajectory length: ~82 events per robot
Average time gap: Δt ≈ 1.2 timesteps
Terminal state proportion: ~1.2% (1/82)
```

### Feature Normalization

**Critical for IQL inference**: Features [1-4] are z-score normalized during training:
```python
normalized[i] = (features[i] - mean[i]) / (std[i] + 1e-8)
```

Normalization parameters saved in dataset and checkpoint, automatically applied during:
- Training: `IQLDataset` applies normalization
- Inference: `LearnableUpdater._normalize_features()` applies normalization

## Implementation Details

### Robot Event Logging

**When robot logs event**:
```python
for step_idx in range(100):  # Episode length
    env.step()

    # CRITICAL: Must call this to populate current_timestep_tracks
    for robot in env.robots:
        robot.update_current_timestep_tracks()

    for robot in env.robots:
        current_tracks = robot.get_current_timestep_tracks()

        if not current_tracks:
            continue  # Skip if no detections

        # Robot has detections → compute evidence and log event
        evidence_scores = gnn.predict(robot, all_robots)
        robot_evidence = evidence_scores.agent_scores[robot.id]
        step_scale = policy.predict(robot_features)

        # Log transition: (state, action, reward, next_state, delta_t)
        log_robot_event(robot, robot_evidence, step_scale, ...)
```

**Key insight**: Each robot can log 0-100 events per episode depending on detection frequency. Active robots (many detections) → long trajectories.

### Track Updates (Fixed Step Scale)

**Runtime behavior**:
```python
# Tracks not in MDP, use fixed step scale
for track in robot.get_current_timestep_tracks():
    evidence = gnn_scores.track_scores[track.track_id]
    step_scale = 0.5  # FIXED, not learned

    track.alpha += step_scale * evidence
    track.beta += step_scale * (1 - evidence)
```

### Trust Decay

**Lazy evaluation** (efficient):
```python
# At event time t, compute decayed values from last event at t_prev
delta = t - t_prev
alpha_decayed = alpha * (gamma ** delta)
beta_decayed = beta * (gamma ** delta)
```

**Equivalent to** eager per-timestep decay: `alpha *= gamma` at every timestep.

### Expected Visible and Trusted Count (Feature Computation)

**For robots - expected_visible** (potential collaborators):
```python
# Count ALL tracks in FOV with trust > threshold
expected_visible = sum(1 for track in all_tracks
                      if robot.is_in_fov(track.position) and
                         track.trust_value > 0.6)
```

**For robots - trusted_count** (current observations):
```python
# Count DETECTED tracks with trust > threshold
trusted_count = sum(1 for track in robot.current_tracks
                   if track.trust_value > 0.6)
```

**Key difference**: `expected_visible` includes ALL tracks in FOV (even if not detected this timestep), while `trusted_count` only includes currently detected tracks. Both use the same trust threshold (0.6) for consistency.

**For tracks** (if we were to log them):
```python
# Count robots detecting SAME OBJECT with trust > threshold
object_id = track.object_id
detecting_robots = object_detectors[object_id]  # All robots detecting this object
trusted_count = sum(1 for robot_id in detecting_robots
                   if robots[robot_id].trust_value > 0.6)
```

## File Structure

### Core Implementation

**Trust System**:
- `rl_trust_system.py` - Main trust update coordinator (ego-sweep)
- `rl_updater.py` - IQL policy wrapper (inference)
- `rl_evidence.py` - Evidence extraction (GNN wrapper)
- `rl_feature_extraction.py` - Unified robot feature extraction (5D, runtime + offline)

**Training Pipeline**:
- `generate_offline_dataset_eventbased.py` - Event-based dataset generation ⭐
- `train_iql_trust.py` - IQL training with event-based variable discounting ⭐
- `iql_networks.py` - Q, V, Beta policy networks

**Supervised Learning**:
- `generate_supervised_data.py` - Generate GNN training data
- `train_supervised_trust.py` - Train supervised GNN
- `supervised_trust_gnn.py` - Heterogeneous GNN architecture

**Environment & Simulation**:
- `simulation_environment.py` - Multi-robot simulation world
- `robot_track_classes.py` - Robot and Track classes with trust
- `rl_scenario_generator.py` - Scenario parameter sampling

**Evaluation**:
- `compare_trust_methods.py` - Compare paper/baseline/RL on 3 scenarios
- `comprehensive_trust_benchmark.py` - Extensive benchmark suite
- `visualize_trust_updates.py` - Visualization tools

**Configuration & Utilities**:
- `rl_config.py` - Training configuration
- `paper_trust_algorithm.py` - Baseline from original paper

### Testing

```bash
# Test feature extraction consistency
python test_feature_extraction.py

# Test IQL implementation
python test_iql_implementation.py

# Test IQL compatibility
python test_iql_compatibility.py
```

## Performance Expectations

### Healthy IQL Training

**Good signs**:
```
Update 10000: Q_loss=0.28 | V_loss=0.15 | Actor_loss=-2.15 (obj=+2.15)
  FQE Value Estimate: 0.32

Update 50000: Q_loss=0.19 | V_loss=0.11 | Actor_loss=-4.58 (obj=+4.58)
  FQE Value Estimate: 0.58

Update 100000: Q_loss=0.16 | V_loss=0.08 | Actor_loss=-7.23 (obj=+7.23)
  FQE Value Estimate: 0.72
```

**What to monitor**:
- ✅ V-loss stays > 0.05 (not collapsing)
- ✅ FQE increases from ~0.15 to ~0.65+
- ✅ Q-loss and V-loss in 0.1-1.0 range
- ✅ **Actor loss becoming MORE negative** (objective increasing) - see note below

**Bad signs** (if you see these, something is wrong):
```
Update 10000: Q_loss=0.002 | V_loss=0.001 | Actor_loss=-5.23
  FQE Value Estimate: 0.002
```
- ❌ V-loss collapses to near-zero
- ❌ FQE decreases or stays near zero
- ❌ Actor loss stays constant or becomes less negative (policy not improving)
- ❌ Actor loss becomes extremely negative (< -50, possible mode collapse)

**Note on Actor Loss Sign**:

The actor loss uses **Advantage-Weighted Regression (AWR)**: `Loss = -E[exp(A/α) * log π(a|s)]`

**Key insight**: For Beta distributions, the PDF can exceed 1, so `log π(a|s)` can be POSITIVE.

As the policy improves:
1. It assigns **higher probability density** to good actions (PDF >> 1)
2. `log π(a|s)` becomes more positive for those actions
3. The weighted term `exp(A/α) * log π(a|s)` becomes more positive
4. The loss `-E[...]` becomes **MORE negative** ✅

**This is CORRECT!** A more negative actor loss means:
- The policy is more confident (sharper Beta distributions)
- Higher probability assigned to actions with positive advantages
- The AWR objective `E[exp(A/α) * log π(a|s)]` is increasing

Think of it as maximizing an objective rather than minimizing a traditional loss. The training logs show both:
- `Actor_loss`: The minimization target (negative and decreasing)
- `obj`: The actual objective being maximized (positive and increasing)

For complete mathematical details, see [ACTOR_LOSS_EXPLANATION.md](ACTOR_LOSS_EXPLANATION.md)

### Runtime Performance

**Expected trust convergence**:
- Legitimate robots: Trust → 0.7-0.9 (high trust)
- Adversarial robots: Trust → 0.1-0.3 (low trust)
- Convergence time: ~50-100 timesteps depending on scenario

## Documentation

### Implementation Guides
- **[ROBOT_ONLY_MDP_RESULTS.md](ROBOT_ONLY_MDP_RESULTS.md)** - Robot-only design rationale and results
- **[ROBOT_ONLY_SUCCESS_SUMMARY.md](ROBOT_ONLY_SUCCESS_SUMMARY.md)** - Complete success metrics
- **[BUG_FIX_SUMMARY.md](BUG_FIX_SUMMARY.md)** - Critical bug fix: update_current_timestep_tracks
- **[EVENT_LOGGING_ANALYSIS.md](EVENT_LOGGING_ANALYSIS.md)** - Event-based design explanation

### Training & Fixes
- **[ACTOR_LOSS_EXPLANATION.md](ACTOR_LOSS_EXPLANATION.md)** - Why actor loss becomes more negative (AWR with Beta distributions)
- **[REWARD_SCALING_FIX.md](REWARD_SCALING_FIX.md)** - Why 100x reward scaling
- **[DATASET_STRUCTURE_ANALYSIS.md](DATASET_STRUCTURE_ANALYSIS.md)** - Dataset statistics analysis
- **[TRAIN_TEST_CONSISTENCY_FIX.md](TRAIN_TEST_CONSISTENCY_FIX.md)** - Feature consistency fix
- **[FEATURE_EXTRACTION_CONSOLIDATION.md](FEATURE_EXTRACTION_CONSOLIDATION.md)** - Shared feature extraction

### Verification & Testing
- **[IMPLEMENTATION_VERIFICATION.md](IMPLEMENTATION_VERIFICATION.md)** - Specification compliance
- **[COMPLETE_FIX_SUMMARY.md](COMPLETE_FIX_SUMMARY.md)** - All bugs fixed summary
- **[test_feature_extraction.py](test_feature_extraction.py)** - Feature consistency tests

## Common Issues & Solutions

### Issue 1: Training Collapse (V-loss → 0)

**Symptoms**: V-loss drops to near-zero, FQE doesn't increase

**Causes & Fixes**:
1. ✅ **Reward too small**: Fixed with 100x scaling
2. ✅ **Heavy discounting**: Fixed with γ=0.995 (was 0.99)
3. ✅ **Short trajectories**: Fixed with robot-only MDP (82 events vs 5.8)
4. ✅ **Missing update call**: Fixed with `update_current_timestep_tracks()`

### Issue 2: Low Event Count (~6 events/robot)

**Symptom**: Dataset has only ~6 events per robot instead of ~82

**Cause**: Missing `robot.update_current_timestep_tracks()` call after `env.step()`

**Fix**: Added in [generate_offline_dataset_eventbased.py:411-413](generate_offline_dataset_eventbased.py#L411-L413)

### Issue 3: Feature Dimension Mismatch

**Symptom**: `RuntimeError: Expected 6D features, got 5D`

**Cause**: Old checkpoints use 6D (with entity_type), new code uses 5D

**Fix**:
```bash
rm iql_final.pth iql_checkpoint_*.pth
./run_iql_pipeline.sh  # Regenerate with 5D
```

### Issue 4: Train-Test Inconsistency

**Symptom**: Policy works in training but fails at inference

**Causes**:
1. ✅ **Different feature extraction**: Fixed with shared `rl_feature_extraction.py`
2. ✅ **Missing normalization**: Fixed by saving `feature_means/stds` in checkpoint
3. ✅ **Different trusted_count**: Fixed with consistent `object_detectors` mapping

## Citation

If you use this code, please cite:

```bibtex
@software{neural_symbolic_trust_2025,
  title = {Neural-Symbolic Multi-Robot Trust System with IQL},
  author = {Your Name},
  year = {2025},
  note = {Robot-only MDP with event-based IQL training}
}
```

## License

[Your License Here]

## Acknowledgments

- IQL implementation based on ["Offline Reinforcement Learning with Implicit Q-Learning"](https://arxiv.org/abs/2110.06169)
- GNN architecture inspired by PyTorch Geometric
- Trust representation using Beta distributions
