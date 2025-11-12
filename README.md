# Neural-Symbolic Multi-Robot Trust System with IQL

A neural-symbolic framework for distributed trust management in multi-robot systems using Graph Neural Networks (GNNs) and Implicit Q-Learning (IQL) for adaptive, continuous trust assessment.

## System Overview

This system implements an end-to-end trust management framework for multi-robot collaborative scenarios with a two-stage learning pipeline:

### Stage 1: Supervised Learning (Evidence Extraction)
**Heterogeneous GNN** trained on labeled ego-centric graphs to predict trustworthiness:
- **Input**: Ego-centric graph with neural-symbolic binary predicates (3 features each for robots and tracks)
- **Output**: Trust probability scores ∈ [0, 1] for robots and tracks
- **Architecture**: 3-layer GAT with heterogeneous message passing and skip connections
- **Training**: Binary cross-entropy on balanced 50/50 adversarial/legitimate samples

### Stage 2: Reinforcement Learning (Adaptive Updates)
**IQL Policy** that learns optimal trust update step scales:
- **Input**: Robot state (6D features including GNN evidence score)
- **Output**: Dual step scales (α_scale, β_scale) ∈ [0, 1]² for asymmetric trust updates
- **Architecture**: Q-network, V-network, and Dual Beta policy network
- **Training**: Implicit Q-Learning on offline event-based trajectories

### Trust Representation
Each entity (robot/track) maintains trust as **Beta distribution** Beta(α, β):
- **Mean trust**: τ = α/(α+β) ∈ [0, 1]
- **Confidence**: κ = α + β (higher = more certain)
- **Decay**: Exponential forgetting with γ = 0.995 per timestep
- **Updates**: Dual-action with separate step scales
  - α += α_scale × evidence (learned for robots, fixed 0.5 for tracks)
  - β += β_scale × (1 - evidence) (learned for robots, fixed 0.5 for tracks)

## Key Design Decisions

### 1. Robot-Only MDP Architecture

**Design**: Only robots are RL agents with learned policies. Tracks use fixed step scales.

**Rationale**:
- **Robots are persistent**: Live entire episode (100 timesteps), enabling long-term strategy learning
- **Robots are active**: Continuously detect and evaluate tracks (~82 events per episode)
- **Robot trust is complex**: Depends on long-term behavioral patterns requiring adaptive step scales
- **Track trust is simpler**: Binary classification (real/fake) works well with fixed step scale (0.5)

**Implementation**:
- Robots: 6D state features, learned dual step scales (α_scale, β_scale) via IQL
- Tracks: Fixed dual step scales (0.5, 0.5) for all trust updates

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

**Update Rule** (Dual-Action):
```
α_new = α_old · γ^Δt + α_scale · evidence
β_new = β_old · γ^Δt + β_scale · (1 - evidence)
τ = α / (α + β)  # Mean trust value

Where:
  α_scale: Step scale for positive evidence updates (learned by IQL policy)
  β_scale: Step scale for negative evidence updates (learned by IQL policy)
  This allows asymmetric update strategies (e.g., cautious with negative evidence)
```

### 4. 6D State Features (Robots Only)

**State representation for each robot**:
1. **evidence** (e): GNN trust score ∈ [0, 1]
2. **tau** (τ): Mean trust value = α/(α+β)
3. **kappa** (κ): Trust mass = α + β (confidence indicator)
4. **expected_visible**: Count of tracks in robot's FOV **with trust > 0.6** (potential collaborators)
5. **trusted_count**: Count of **detected** tracks with trust > 0.6 (current observations)
6. **proximal_count**: Count of robots within proximal range (validation opportunities)

**Feature Consistency**:
- Both `expected_visible` and `trusted_count` use the **same trust threshold (0.6)** filter
- **expected_visible**: ALL tracks in FOV with high trust (broader context - what's available)
- **trusted_count**: DETECTED tracks with high trust (narrower context - what's being used)
- **proximal_count**: Indicates collaborative sensing opportunities
- This makes the features interpretable: "X trustworthy tracks available, Y currently being observed, Z robots nearby"

**Action Space**: Dual Beta distributions
- **Action 1**: α_scale ∈ [0, 1] for positive evidence updates
- **Action 2**: β_scale ∈ [0, 1] for negative evidence updates
- Allows model to learn asymmetric update strategies (e.g., aggressive with positive evidence, cautious with negative)

## Supervised Learning: Neural-Symbolic GNN

### Overview

The supervised GNN serves as the **evidence extractor** in the trust pipeline. It processes ego-centric heterogeneous graphs and outputs trust probability scores for all robots and tracks in the graph.

**Key Design Philosophy**: Use **behavioral features only** (no trust values) to ensure train-test consistency during deployment.

### Trust-Free Continuous Features

**NEW DESIGN**: Ratio-based features that are much more informative than binary features. All features are behavioral and completely trust-free, ensuring perfect train-test consistency.

#### Robot Features (6D Continuous)

1. **observed_count** (Feature 0)
   ```python
   observed_count = number of unique objects observed at current timestep
   # Deduplicated by object_id
   ```
   - **Type**: Integer count ≥ 0
   - **Interpretation**: How actively is this robot detecting objects?
   - **Behavioral indicator**: Detection activity level

2. **fused_count** (Feature 1)
   ```python
   fused_count = number of fused tracks within observed tracks
   # Only counts fused tracks that this robot is observing
   ```
   - **Type**: Integer count ≥ 0
   - **Interpretation**: How many collaborative validations is this robot participating in?
   - **Behavioral indicator**: Fusion participation rate

3. **expected_count** (Feature 2)
   ```python
   expected_count = objects in ego graph that are within this robot's FoV
   # From all objects detected by all robots, how many could this robot see?
   ```
   - **Type**: Integer count ≥ 0
   - **Interpretation**: How many objects could this robot potentially detect?
   - **Behavioral indicator**: Coverage potential

4. **partner_count** (Feature 3)
   ```python
   partner_count = number of robots in ego graph with fused tracks with this robot
   # How many robots has this robot co-detected objects with?
   ```
   - **Type**: Integer count ≥ 0
   - **Interpretation**: How collaborative is this robot?
   - **Behavioral indicator**: Collaboration network size

5. **detection_ratio** (Feature 4)
   ```python
   detection_ratio = observed_count / (expected_count + ε)
   # ε = 1e-8 to avoid division by zero
   ```
   - **Type**: Float ratio ∈ [0, ∞)
   - **Interpretation**: What fraction of detectable objects is this robot actually detecting?
   - **Behavioral indicator**: Detection efficiency
   - **Adversarial pattern**: Legitimate robots have high ratios, adversarial robots may miss real objects

6. **validator_ratio** (Feature 5)
   ```python
   validator_ratio = partner_count / (robots_with_detections_in_fov + ε)
   # How many of the nearby active robots does this robot collaborate with?
   ```
   - **Type**: Float ratio ∈ [0, ∞)
   - **Interpretation**: What fraction of nearby robots validate this robot's detections?
   - **Behavioral indicator**: Collaboration rate
   - **Adversarial pattern**: Legitimate robots collaborate frequently, adversarial robots may be isolated

#### Track Features (2D Continuous)

1. **detector_count** (Feature 0)
   ```python
   detector_count = number of robots in ego graph that detected this track at current timestep
   # For fused tracks: all contributing robots
   # For individual tracks: always 1
   ```
   - **Type**: Integer count ≥ 1
   - **Interpretation**: How many robots independently observed this object?
   - **Behavioral indicator**: Multi-robot consensus
   - **Adversarial pattern**: Real objects detected by multiple robots, false positives usually solo

2. **detector_ratio** (Feature 1)
   ```python
   detector_ratio = detector_count / (robots_with_fov_containing_track + ε)
   # What fraction of robots that could see this object actually detected it?
   ```
   - **Type**: Float ratio ∈ [0, ∞)
   - **Interpretation**: Agreement rate among robots that could see this object
   - **Behavioral indicator**: Detection consistency
   - **Adversarial pattern**: Real objects have high ratios, false positives have low ratios

### Design Philosophy: Trust-Free = Train-Test Consistent

**Why Continuous Ratio Features?**

✅ **Much more informative than binary features**:
- Binary: `HasFusedTracks=1` (robot has at least 1 fused track)
- Continuous: `fused_count=5` and `observed_count=8` → 62.5% fusion rate
- The continuous version provides fine-grained information about behavior patterns

✅ **Completely trust-free**:
- NO features depend on trust values
- All features based on observable events (detections, fusions, FoV geometry)
- Features work identically during training and deployment

✅ **Perfect train-test consistency**:
- **Training**: Model learns from behavioral patterns (counts, ratios, collaboration)
- **Deployment**: SAME behavioral patterns observable regardless of trust initialization
- No distribution shift between training and deployment

✅ **Adversarial discrimination**:
- Legitimate robots: high detection_ratio, high validator_ratio, frequent fusion
- Adversarial robots: low detection_ratio (miss real objects), low validator_ratio (isolated), inject false positives
- Model learns these patterns WITHOUT using trust values

### GNN Architecture

**Network Type**: Heterogeneous Graph Attention Network (GAT)

**Node Types**:
- `agent`: Robots in ego-centric graph
- `track`: Tracks detected by robots

**Edge Types**:
1. `('agent', 'in_fov_and_observed', 'track')`: Robot detects track
2. `('track', 'observed_and_in_fov_by', 'agent')`: Reverse of above
3. `('agent', 'in_fov_only', 'track')`: Track in robot's FOV but not detected
4. `('track', 'in_fov_only_by', 'agent')`: Reverse of above
5. `('agent', 'co_detection', 'agent')`: **TRUST-FREE** robot-robot edges (robots detecting same object)

**Architecture Details**:
```python
Input: x_agent ∈ R^(N_agents × 6), x_track ∈ R^(N_tracks × 2)

# Layer 1: Embedding
h_agent^(0) = Linear(6 → 64)(x_agent)
h_track^(0) = Linear(2 → 64)(x_track)

# Layers 1-3: Heterogeneous GAT with skip connections
for layer in [1, 2, 3]:
    h^(layer) = GAT(h^(layer-1), edges) + h^(layer-2)  # Skip connection
    h^(layer) = ReLU(BatchNorm(h^(layer)))

# Classification heads (separate for agents and tracks)
# Agent classifier:
agent_trust = Sigmoid(Linear(64 → 16 → 1)(h_agent^(3)))

# Track classifier:
track_trust = Sigmoid(Linear(64 → 16 → 1)(h_track^(3)))

Output: p_agent ∈ [0,1]^(N_agents), p_track ∈ [0,1]^(N_tracks)
```

**Key Components**:
- **GAT Attention**: 4 heads, concat=False (average), dropout=0.1
- **Skip Connections**: Stabilize deep network training
- **BatchNorm**: Normalize activations between layers
- **Dropout**: 0.1 in GAT, 0.2/0.1 in classifiers (regularization)
- **LayerNorm**: In classifier MLPs for stable gradients

**Model Size**: ~91,000 parameters

### Training Procedure

**Dataset Generation**:
```bash
python generate_supervised_data.py \
  --episodes 200 \
  --output supervised_trust_dataset.pkl
```

**Process**:
1. Run 200 diverse simulation episodes (varied robot density, adversarial ratio, FP/FN rates)
2. Assign ground-truth trust with 20% noise:
   - 80%: Perfect assignment (legitimate: 0.7-1.0, adversarial: 0.0-0.3)
   - 20%: Random noise (0.0-1.0 regardless of ground truth)
3. Build ego-centric graph for each robot at each timestep
4. Extract neural-symbolic features (3 per robot, 3 per track)
5. Generate binary labels (1=legitimate/ground-truth, 0=adversarial/false-positive)
6. Balance dataset: 50% adversarial samples, 50% legitimate samples

**Typical Dataset**: ~140,000 samples from 200 episodes

**Model Training**:
```bash
python train_supervised_trust.py \
  --data supervised_trust_dataset.pkl \
  --epochs 100 \
  --batch-size 256 \
  --force-cpu \
  --output supervised_trust_model.pth
```

**Training Configuration**:
- **Loss**: Binary Cross-Entropy (BCE) with mean reduction
  ```python
  # Loss per node type
  agent_loss = BCE(predictions['agent'], agent_labels) / N_agents
  track_loss = BCE(predictions['track'], track_labels) / N_tracks

  # Combined loss (average across node types)
  total_loss = (agent_loss + track_loss) / 2
  ```
- **Normalization**: Each sample contributes equally regardless of graph size
- **Optimizer**: Adam, lr=0.001
- **Batch size**: 256 samples
- **Early stopping**: Patience 20 epochs on validation loss
- **Train/Val split**: 80/20

**Expected Performance**:
```
Training (100 epochs):
  Agent: ~85-90% accuracy, ~87-92% F1
  Track: ~85-90% accuracy, ~87-92% F1

Validation:
  Agent: ~85-90% accuracy, ~87-92% F1
  Track: ~85-90% accuracy, ~87-92% F1
```

**Note**: Accuracy is lower than with HighlyTrusted feature (~97%), but the model **generalizes correctly** to deployment scenarios.

### Ego-Centric Graph Construction

**For each robot (ego)**:
```python
# 1. Find proximal robots (within 50.0 units)
proximal_robots = [r for r in all_robots
                   if distance(ego, r) <= proximal_range]

# 2. Collect all tracks from proximal robots
proximal_tracks = []
for robot in proximal_robots:
    proximal_tracks.extend(robot.get_all_tracks())

# 3. Perform track fusion (merge tracks of same object by different robots)
fused_tracks = fusion_algorithm(proximal_tracks)

# 4. Build heterogeneous graph
graph = {
    'agents': proximal_robots,  # Ego + nearby robots
    'tracks': fused_tracks,
    'edges': {
        'in_fov_and_observed': [...],
        'in_fov_only': [...],
        'more_trustworthy_than': [...]
    }
}

# 5. Extract features for all nodes
agent_features = calculate_agent_features(proximal_robots, fused_tracks)
track_features = calculate_track_features(fused_tracks, proximal_robots)
```

**Key Properties**:
- **Ego-centric**: Each robot has its own view of the world
- **Local**: Only includes robots within proximal range (scalable)
- **Heterogeneous**: Two node types with different features
- **Relational**: Captures observation relationships and robot comparisons

### Runtime Inference

**During deployment** (in `rl_trust_system.py`):
```python
for timestep in simulation:
    for ego_robot in all_robots:
        # 1. Build ego graph
        graph = build_ego_graph(ego_robot, all_robots)

        # 2. Extract features (3 per robot, 3 per track)
        x_dict = {
            'agent': agent_features,  # (N_agents, 3)
            'track': track_features   # (N_tracks, 3)
        }

        # 3. Run GNN inference
        predictions = supervised_gnn(x_dict, edge_index_dict)
        #   predictions['agent']: (N_agents,) probabilities
        #   predictions['track']: (N_tracks,) probabilities

        # 4. Use evidence scores in RL policy
        ego_evidence = predictions['agent'][0]  # Ego robot's score
        step_scale = rl_policy.predict(ego_robot, ego_evidence)

        # 5. Update trust
        ego_robot.alpha += step_scale * ego_evidence
        ego_robot.beta += step_scale * (1 - ego_evidence)
```

**Inference Time**: ~5-10ms per ego graph on CPU (fast enough for real-time)

## Technical Architecture

### Pipeline Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                           │
└────────────────────────────────────────────────────────────────────┘

STAGE 1: SUPERVISED LEARNING (Evidence Extraction)
─────────────────────────────────────────────────
1. Generate Supervised Data
   Input:  200 simulation episodes
   Output: ~140K ego-graph samples with binary labels
   └─> Robot features: HasFusedTracks, HighConnectivity, ReliableDetector (3D)
   └─> Track features: DetectedByReliable, MultiRobot, MajorityReliable (3D)
   └─> Labels: 1=legitimate/ground-truth, 0=adversarial/false-positive
   └─> Balanced: 50% adversarial, 50% legitimate

2. Train Supervised GNN
   Input:  supervised_trust_dataset.pkl (~140K samples)
   Output: supervised_trust_model.pth (~91K parameters)
   └─> Architecture: 3-layer heterogeneous GAT
   └─> Training: BCE loss, Adam optimizer, 100 epochs
   └─> Performance: ~85-90% accuracy on agents and tracks

STAGE 2: REINFORCEMENT LEARNING (Adaptive Trust Updates)
─────────────────────────────────────────────────────────
3. Generate Offline RL Dataset (Event-Based)
   Input:  100 simulation episodes + trained supervised GNN
   Output: ~200K robot event transitions
   └─> Features: (evidence, tau, kappa, expected_visible, trusted_count, proximal_count) (6D)
   └─> Actions: (α_scale, β_scale) ∈ [0,1]² (dual Beta distributions for asymmetric updates)
   └─> Rewards: calibration improvement (scaled 100×)
   └─> Event-based: Log only when robots detect tracks (~82 events/robot)

4. Train IQL Policy
   Input:  offline_dataset_balanced.npz (~200K transitions)
   Output: iql_final.pth (Q, V, policy networks)
   └─> Architecture: Q-critic, V-network, Beta policy
   └─> Training: IQL with expectile regression (100K updates)
   └─> Performance: FQE ~0.65-0.75

5. Deploy & Evaluate
   └─> Compare: Paper baseline vs Supervised+Fixed vs RL policy
   └─> Metrics: Trust accuracy, convergence speed, AUC

┌────────────────────────────────────────────────────────────────────┐
│                      RUNTIME ARCHITECTURE                          │
└────────────────────────────────────────────────────────────────────┘

For each timestep t:
  Apply exponential decay: α, β *= γ^1

  For each robot (ego):
    current_tracks = robot.get_current_timestep_tracks()

    if not current_tracks:
      continue  # Skip if no detections

    # STAGE 1: Evidence Extraction (Supervised GNN)
    ┌─────────────────────────────────────────────────┐
    │ 1. Build ego-centric heterogeneous graph       │
    │    - Nodes: proximal robots + their tracks     │
    │    - Features: 3D binary predicates            │
    │    - Edges: observation + comparison relations │
    ├─────────────────────────────────────────────────┤
    │ 2. Run supervised GNN forward pass             │
    │    agent_probs, track_probs = GNN(graph)       │
    │    ego_evidence = agent_probs[ego_index]       │
    └─────────────────────────────────────────────────┘

    # STAGE 2: Adaptive Update (IQL Dual-Action Policy)
    ┌─────────────────────────────────────────────────┐
    │ 3. Extract robot state features (6D)           │
    │    state = [evidence, tau, kappa,              │
    │             expected_visible, trusted_count,   │
    │             proximal_count]                    │
    ├─────────────────────────────────────────────────┤
    │ 4. Predict dual step scales via IQL policy     │
    │    α_scale, β_scale = IQL_policy(state)        │
    │    # Two independent Beta distributions        │
    │    # Both outputs ∈ [0,1]                      │
    └─────────────────────────────────────────────────┘

    # Trust Update (Dual-Action)
    ┌─────────────────────────────────────────────────┐
    │ 5. Update robot's trust distribution           │
    │    alpha += α_scale × evidence                 │
    │    beta  += β_scale × (1 - evidence)           │
    │    # Asymmetric updates: different rates       │
    │    # for positive vs negative evidence         │
    │                                                 │
    │ 6. Update tracks with FIXED dual step scales   │
    │    For each track in current_tracks:           │
    │      track.alpha += 0.5 × track_evidence       │
    │      track.beta  += 0.5 × (1 - track_evidence) │
    └─────────────────────────────────────────────────┘

Result: Trust values τ = α/(α+β) for all robots and tracks
```

### IQL Training Architecture

**Networks**:
- **Q-Critic**: Estimates Q(s,a) for state-action pairs (action_dim=2 for dual actions)
- **V-Network**: Estimates state value V(s) via expectile regression
- **Dual Beta Policy**: Outputs TWO independent step scales (α_scale, β_scale) ∈ [0,1]² using two Beta distributions
  - Shared backbone (2-layer MLP with 128 hidden units)
  - Separate heads for each action's Beta parameters (4 heads total)
  - Allows learning asymmetric update strategies

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
  'states': (N, 6) float32,           # Robot state features (6D)
  'actions': (N, 2) float32,          # Dual step scales [α_scale, β_scale] ∈ [0,1]²
  'rewards': (N,) float32,            # Calibration improvement
  'next_states': (N, 6) float32,      # Next state features (6D)
  'dones': (N,) float32,              # 1=terminal, 0=continue
  'delta_prevs': (N,) int32,          # Time since last event
  'delta_nexts': (N,) int32,          # Time to next event
  'trajectory_starts': (T,) int64,    # Start index per trajectory
  'trajectory_lengths': (T,) int64,   # Length per trajectory
  'gamma': 0.995,                     # Decay factor
  'feature_means': (6,) float32,      # For normalization
  'feature_stds': (6,) float32,       # For normalization
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

**Critical for IQL inference**: Features [1-5] are z-score normalized during training:
```python
normalized[i] = (features[i] - mean[i]) / (std[i] + 1e-8)
# Normalizes: tau, kappa, expected_visible, trusted_count, proximal_count
# Keeps evidence [0] unchanged (already in [0,1] from GNN)
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
        α_scale, β_scale = policy.predict(robot_features)  # Dual actions

        # Log transition: (state, action, reward, next_state, delta_t)
        log_robot_event(robot, robot_evidence, (α_scale, β_scale), ...)
```

**Key insight**: Each robot can log 0-100 events per episode depending on detection frequency. Active robots (many detections) → long trajectories.

### Track Updates (Fixed Step Scale)

**Runtime behavior**:
```python
# Tracks not in MDP, use fixed dual step scales
for track in robot.get_current_timestep_tracks():
    evidence = gnn_scores.track_scores[track.track_id]
    α_scale, β_scale = 0.5, 0.5  # FIXED, not learned

    track.alpha += α_scale * evidence
    track.beta += β_scale * (1 - evidence)
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

**Supervised Learning** (Evidence Extraction):
- `generate_supervised_data.py` - Generate GNN training data with ground truth labels
  - Builds ego-centric graphs for each robot at each timestep
  - Extracts 3D neural-symbolic features (binary predicates)
  - Balances adversarial/legitimate samples (50/50)
  - Adds 20% label noise to prevent overfitting
- `train_supervised_trust.py` - Train supervised GNN
  - Binary classification with BCE loss
  - Trains on ~140K samples from 200 episodes
  - Early stopping with validation monitoring
  - Saves best model checkpoint
- `supervised_trust_gnn.py` - Heterogeneous GNN architecture
  - `TrustFeatureCalculator`: Computes 3D binary features for robots and tracks
  - `EgoGraphBuilder`: Constructs local heterogeneous graphs
  - `SupervisedTrustGNN`: 3-layer GAT with skip connections (~91K params)
  - `SupervisedTrustPredictor`: Inference wrapper for deployment

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
