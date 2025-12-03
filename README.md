# Neural-Symbolic Multi-Robot Trust System

A Graph Neural Network framework for distributed trust assessment in multi-robot systems using **structure-only learning** with **cross-validation constraints**.

## System Overview

This system implements supervised trust prediction for collaborative multi-robot scenarios where some robots may be adversarial. The key innovation is using graph structure alone to predict trustworthiness, with strict cross-validation requirements to ensure meaningful trust updates.

### Core Features

- **Structure-Only Learning**: No node features - learns purely from graph topology
- **Cross-Validation Constraints**: Trust updates require sufficient cross-validation evidence
- **Heterogeneous Graph Neural Network**: 6 edge types capture different relationships
- **Mass-Based Trust Updates**: Beta distribution (α, β) with confidence gating
- **Ego-Centric Architecture**: Each robot builds its own local graph for inference

---

## Architecture

### Supervised Trust GNN

The model is a single heterogeneous Graph Attention Network (GAT) that predicts trust labels from graph structure:

```
┌──────────────────────────────────────────────────────────┐
│ INPUT: Ego-graph (N agents, M tracks, 6 edge types)     │
│   • Agents (robots): Zero-initialized node embeddings   │
│   • Tracks (detections): Zero-initialized node embeddings│
│   • Edges: 6 heterogeneous relation types               │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│ STRUCTURE-ONLY GNN                                       │
│   1. Learnable type embeddings (agent_type, track_type)  │
│   2. 3-layer Heterogeneous GAT with attention           │
│   3. Message passing across 6 edge types                │
│   4. Per-node trust score prediction                    │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│ CROSS-VALIDATION FILTERING                               │
│   • Only update ego robot if it has cross-validation    │
│   • Only update tracks with ≥2 robot observations       │
│   • Skip updates for isolated nodes                     │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│ OUTPUT: Trust predictions for meaningful nodes          │
│   • Ego robot evidence score (if cross-validated)       │
│   • Track evidence scores (if cross-validated)          │
└──────────────────────────────────────────────────────────┘
```

### Model Statistics

- **Total Parameters**: 1,222,914 (structure-only, 128-dim hidden)
  - Conv layer 1: 400,128 params
  - Conv layer 2: 400,128 params
  - Conv layer 3: 400,128 params
  - BatchNorm layers: 1,536 params
  - Classifiers: 20,994 params
- **Hidden Dimension**: 128
- **GAT Layers**: 3 independent layers with 4 attention heads each
- **Dropout**: 0.1-0.2 (GAT layers and classifiers)
- **Architecture**: PyTorch Geometric HeteroData + GAT

---

## Cross-Validation Constraints

A critical feature of this system is requiring **sufficient cross-validation** before updating trust. This prevents premature trust updates from isolated observations.

### Requirements for Trust Updates

**For Ego Robot (the robot performing inference)**:
- Must have **co_detection** OR **contradicts** edges with other robots
- Without cross-validation: Skip trust update for this robot

**For Tracks (detected objects)**:
- Must be **currently detected by ego robot**
- Must have edges to **≥2 robots** (cross-validation requirement)
- Without sufficient evidence: Skip trust update for this track

### How It Works

```python
# During data collection (generate_supervised_data.py):
1. Build ego-graph for each robot
2. Check ego robot cross-validation:
   - Has co_detection edges? ✓
   - Has contradicts edges? ✓
   - If neither: SKIP this ego-graph
3. Identify meaningful tracks:
   - Currently detected by ego? ✓
   - Has edges to ≥2 robots? ✓
   - If no: SKIP this track
4. Save: full ego-graph + meaningful_track_indices

# During training (train_supervised_trust.py):
1. Load sample with cross-validation metadata
2. Compute loss ONLY for:
   - Ego robot (index 0) if has cross-validation
   - Tracks in meaningful_track_indices
3. Masked loss prevents learning from unvalidated nodes

# During inference (supervised_trust_gnn.py):
1. Build ego-graph for robot
2. Check cross-validation constraints
3. If constraints met:
   - Predict ego robot trust
   - Predict meaningful track trust
4. If constraints not met:
   - Return None (no trust update)
```

### Why Cross-Validation Matters

Without cross-validation:
- ❌ Single robot observations are unreliable
- ❌ Adversarial robots can manipulate trust
- ❌ False positives/negatives aren't detected

With cross-validation:
- ✅ Multiple observations provide consensus
- ✅ Contradictions expose adversarial behavior
- ✅ Trust updates are more reliable
- ✅ System is robust to deception

---

## Structure-Only Learning

### No Node Features

All nodes are initialized with **zeros** - the model learns purely from graph structure:

```python
# During forward pass
x_dict = {
    'agent': torch.zeros(num_agents, 128),  # No features!
    'track': torch.zeros(num_tracks, 128)   # No features!
}

# Learnable type embeddings added
x_dict['agent'] += model.agent_type_embedding
x_dict['track'] += model.track_type_embedding

# Message passing through GAT layers
for layer in gat_layers:
    x_dict = layer(x_dict, edge_index_dict)  # Learn from structure
```

### What the Model Learns From

Since there are no handcrafted node features, the model learns from:

1. **Graph Topology**: Which nodes connect to which nodes
2. **Edge Types**: 6 different heterogeneous relations
3. **Neighborhood Patterns**: Local structure around each node
4. **Message Passing**: Information aggregation via GAT attention
5. **Type Embeddings**: Learnable 128-dim embeddings for agent/track types

---

## Edge Types

The model uses **6 heterogeneous edge types** to capture different relationships:

### 1. `('agent', 'in_fov_and_observed', 'track')`
Robot detects this track (observed AND in field of view)

### 2. `('track', 'observed_and_in_fov_by', 'agent')`
Reverse of above - track is observed by this robot

### 3. `('agent', 'in_fov_only', 'track')`
Track is in robot's FoV but NOT observed (miss detection)
- **Key signal**: Robot *should* see this but doesn't
- Indicates potential false negative

### 4. `('track', 'in_fov_only_by', 'agent')`
Reverse of above

### 5. `('agent', 'co_detection', 'agent')`
Two robots detected the same object (collaboration signal)
- **Key for cross-validation**: Multiple observations of same object
- Provides consensus signal

### 6. `('agent', 'contradicts', 'agent')` ⭐
Robot A detects track that Robot B *should* see (in B's FoV) but doesn't
- **Key signal**: Inconsistent observations between robots
- Exposes potential deception or false positives
- **Critical for cross-validation**: Identifies conflicting evidence

---

## Node Indexing and Mappings

### Ego Robot (Always Index 0)

When building an ego-graph:
```python
# In EgoGraphBuilder.build_ego_graph()
proximal_robots = [ego_robot]  # Ego ALWAYS first
for robot in other_robots:
    if within_comm_range(ego_robot, robot):
        proximal_robots.append(robot)

# Node mappings
agent_nodes = {robot.id: idx for idx, robot in enumerate(proximal_robots)}
# Result: {ego_robot.id: 0, other_robot_1.id: 1, ...}
```

**Key Property**: Ego robot always has graph index 0

### Track Mappings

```python
# After track fusion
all_tracks = fused_tracks + individual_tracks
track_nodes = {track.track_id: idx for idx, track in enumerate(all_tracks)}

# Stored in graph
graph_data.agent_nodes = agent_nodes  # {robot_id → graph_index}
graph_data.track_nodes = track_nodes  # {track_id → graph_index}
```

### Meaningful Track Indices

```python
# During cross-validation filtering
meaningful_track_indices = []  # Graph indices, not IDs!

for track_idx, track in enumerate(all_tracks):
    # Check 1: Currently detected by ego?
    if track.track_id in ego_detected_track_ids:
        # Check 2: Has edges to ≥2 robots?
        if count_robots_with_edges(track_idx) >= 2:
            meaningful_track_indices.append(track_idx)  # Graph index

# Used for loss masking
loss = compute_loss(
    predictions['agent'][0],           # Ego robot (index 0)
    predictions['track'][meaningful_track_indices]  # Only meaningful tracks
)
```

### Example

```python
# Ego-graph with:
# - Robots: robot_5 (ego), robot_2, robot_7
# - Tracks: track_A, track_B, track_C, track_D

agent_nodes = {
    'robot_5': 0,  # Ego (always first)
    'robot_2': 1,
    'robot_7': 2
}

track_nodes = {
    'track_A': 0,
    'track_B': 1,
    'track_C': 2,
    'track_D': 3
}

# Cross-validation identifies tracks B and D as meaningful
meaningful_track_indices = [1, 3]  # Graph indices

# During inference:
ego_trust = predictions['agent'][0]  # robot_5 trust
track_B_trust = predictions['track'][1]  # track_B trust
track_D_trust = predictions['track'][3]  # track_D trust
# Tracks A and C are ignored (not in meaningful_track_indices)
```

---

## Trust Representation

After GNN evidence extraction, trust is represented using **Beta distribution** Beta(α, β):

### Beta Distribution Properties

- **Mean trust**: τ = α/(α+β) ∈ [0, 1]
- **Confidence**: κ = α + β (higher = more certain)
- **Initialization**: α=1, β=1 (uniform prior, τ=0.5)

### Mass-Based Updates

```python
# Evidence scores from GNN
evidence_score = model.predict(ego_graph)  # ∈ [0, 1]

# Convert to mass updates
if evidence_score >= 0.5:
    Δα = confidence_factor * evidence_score
    Δβ = 0
else:
    Δα = 0
    Δβ = confidence_factor * (1 - evidence_score)

# Apply trust cap (10% max change per timestep)
capped_Δα, capped_Δβ = apply_trust_cap(Δα, Δβ, current_α, current_β)

# Update with exponential decay
α_new = α_old * 0.99 + capped_Δα
β_new = β_old * 0.99 + capped_Δβ

# New trust value
τ_new = α_new / (α_new + β_new)
```

### Trust Cap

Limits maximum trust change to 10% per timestep:
- Prevents sudden trust oscillations
- Provides stability in dynamic environments
- Applies to both α and β updates

---

## Dataset Generation

### Command

```bash
python generate_supervised_data.py \
  --episodes 15000 \
  --steps 200 \
  --step-interval 5 \
  --robot-density 0.0010,0.0015 \
  --output supervised_trust_dataset.pkl
```

### What It Does

1. **Simulate Episodes**: Run 15,000 episodes with diverse parameters
2. **Temporal Sampling**: Sample every 5 timesteps (avoids redundancy)
3. **Build Ego-Graphs**: For **all robots** at sampled timesteps, build local graph
4. **Apply Cross-Validation**: Filter ego-graphs without sufficient cross-validation
5. **Label Nodes**: Ground truth from simulation (legitimate=1, adversarial=0)
6. **Save Dataset**: Pickle file with ~1.8M samples and metadata

### Parameter Diversity

Each episode uses randomly sampled parameters:

```python
Parameters:
  • robot_density: 0.001 - 0.002
  • target_density: 0.001 - 0.004
  • adversarial_ratio: 0.2 - 0.4
  • false_positive_rate: 0.1 - 0.6
  • false_negative_rate: 0.0 - 0.2
  • world_size: 100.0 × 100.0
  • proximal_range: 50.0 (fixed)
```

### Dataset Statistics

**Recommended Settings** (for good model performance):
```bash
python generate_supervised_data.py \
  --episodes 15000 \
  --steps 200 \
  --step-interval 5 \
  --robot-density 0.0010,0.0015
```

**Expected Output**:
```
File: supervised_trust_dataset.pkl
Samples: ~1,800,000 ego-graphs (after cross-validation filtering)
Training points: ~18,000,000 node labels
Samples-to-parameters ratio: 14.7× (meets 10-20× requirement)
Train/Val Split: 80% / 20%
Source: 15,000 episodes × 40 timesteps × 12 robots (avg)
Generation time: ~8-12 hours
```

**Per Sample**:
- Agents: 5-15 robots (within proximal range)
- Tracks: 10-40 tracks (fused + individual)
- Edges: 50-300 edges (across 6 types)
- Labels: ~10 meaningful nodes (1 ego + 9 tracks avg)

### Data Generation Breakdown

**Per Episode** (200 steps, step_interval=5):
```
Timesteps sampled: 200 / 5 = 40 timesteps
Robots per timestep: ~12 (all robots)
Potential samples: 40 × 12 = 480 samples
After cross-validation (~25%): ~100-120 samples per episode
```

**Total** (15,000 episodes):
```
Potential samples: 15,000 × 480 = 7,200,000
After cross-validation: ~1,800,000 samples
Training points: 1,800,000 × 10 labels = 18,000,000
```

### Alternative: Minimal Configuration

**For faster experimentation** (lower performance):
```bash
python generate_supervised_data.py \
  --episodes 10000 \
  --steps 200 \
  --step-interval 5 \
  --robot-density 0.0010,0.0015
```

**Expected**:
- Samples: ~1,200,000
- Training points: ~12,000,000
- Ratio: 9.8× (just below 10× minimum)
- Generation time: ~5-7 hours

### Cross-Validation Filtering Statistics

```
Before Filtering: ~2,400,000 potential ego-graphs
After Filtering: ~600,000 samples (25% pass rate)

Filtered Out:
  • ~60% - Ego robot has no cross-validation
  • ~15% - No meaningful tracks (all isolated)
```

**Why 25% pass rate?**
- Early timesteps (0-50): Low pass rate (~10%) - robots haven't interacted yet
- Mid-late timesteps (50-200): High pass rate (~35-40%) - robots have cross-validation
- Average across episode: ~25% pass rate

---

## Training

### Command

```bash
python train_supervised_trust.py \
  --data supervised_trust_dataset.pkl \
  --epochs 1000 \
  --batch-size 32 \
  --lr 1e-3 \
  --device auto \
  --patience 100 \
  --output supervised_trust_model.pth
```

### Training Configuration

- **Model**: `SupervisedTrustGNN(hidden_dim=128, dropout=0.3)`
- **Loss**: Binary cross-entropy (BCELoss) with mean reduction
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.7, patience=10)
- **Batch Processing**: Individual samples with gradient accumulation
- **Early Stopping**: Patience of 100 epochs

### Per-Sample Processing

```python
# Each sample is processed individually (not batched)
for sample in batch:
    # 1. Transfer to device
    num_agents, num_tracks, edge_index_dict, labels, cv_data = transfer(sample)

    # 2. Forward pass
    predictions = model(num_agents, num_tracks, edge_index_dict)

    # 3. Compute loss ONLY for cross-validated nodes
    loss = compute_loss(
        predictions,
        labels,
        ego_has_cross_validation=sample.ego_has_cross_validation,
        meaningful_track_indices=sample.meaningful_track_indices
    )

    # 4. Backward (accumulate gradients)
    loss.backward()

# 5. Update weights once per batch
optimizer.step()
```

### Expected Performance

**Convergence**: 50-150 epochs (depends on data size)

**Accuracy** (on meaningful nodes only):
- Agent classification: 80-90%
- Track classification: 80-90%
- F1 scores: 0.75-0.85

**Loss Curves**:
- Training loss: Steady decrease
- Validation loss: Should track training (not increase)

---

## Inference

### During Simulation

```python
from supervised_trust_gnn import SupervisedTrustPredictor

# Initialize predictor
predictor = SupervisedTrustPredictor(
    model_path='supervised_trust_model.pth',
    proximal_range=50.0
)

# For each timestep, for each robot:
for robot in robots:
    # 1. Build ego-graph
    result = predictor.predict_from_robots_tracks(robot, all_robots)

    # 2. Check cross-validation
    if result is None:
        # No cross-validation - skip update
        continue

    # 3. Extract predictions
    predictions = result['predictions']
    meaningful_track_indices = result['meaningful_track_indices']

    # 4. Update ego robot trust
    ego_evidence = predictions['agent']['trust_scores'][0]
    update_robot_trust(robot, ego_evidence)

    # 5. Update meaningful track trust
    for track_idx in meaningful_track_indices:
        track_id = get_track_id_from_index(track_idx)
        track_evidence = predictions['track']['trust_scores'][track_idx]
        update_track_trust(track_id, track_evidence)
```

### Cross-Validation at Inference

Same constraints as training:
1. Ego robot must have co_detection or contradicts edges
2. Tracks must be ego-detected AND have edges to ≥2 robots
3. Returns `None` if constraints not met

---

## Evaluation

### Command

```bash
python compare_trust_methods.py
```

### Comparison Setup

Runs 3 test scenarios:
1. **Scenario 1**: Low FP (10%), Low FN (5%)
2. **Scenario 2**: High FP (50%), Low FN (5%)
3. **Scenario 3**: Low FP (10%), High FN (50%)

For each scenario:
- 10 robots, 30% adversarial
- 500 timesteps
- 100×100 world

### Metrics Tracked

**Per Robot**:
- Trust value over time (τ)
- Confidence (α + β)
- True positive rate (adversarial detected)
- False positive rate (legitimate flagged)

**Per Track**:
- Trust value over time
- Detection accuracy
- False positive/negative rates

### Expected Behavior

**Legitimate Robots**:
- τ → 0.7-0.9 (high trust)
- Steady confidence growth
- Stable trust values

**Adversarial Robots**:
- τ → 0.1-0.3 (low trust)
- Detected within 50-100 timesteps
- Trust decreases when contradictions observed

---

## File Structure

### Core Implementation

**GNN Model**:
- `supervised_trust_gnn.py` - SupervisedTrustGNN model and predictor

**Data Generation**:
- `generate_supervised_data.py` - Generate training data with cross-validation filtering

**Training**:
- `train_supervised_trust.py` - Train supervised trust model with masked loss

**Trust System**:
- `mass_based_trust_update.py` - Mass-based trust accumulation

**Evaluation**:
- `compare_trust_methods.py` - Compare trust methods across scenarios

**Simulation**:
- `simulation_environment.py` - Multi-robot simulation environment
- `robot_track_classes.py` - Robot/Track classes with trust attributes

**Utilities**:
- `paper_trust_algorithm.py` - Baseline trust algorithm

### Documentation

- `README.md` - This file
- `SIMPLIFIED_CROSS_VALIDATION.md` - Cross-validation design details
- `TRUST_CAP_BUG_EXPLAINED.md` - Trust cap implementation

---

## Key Design Decisions

### 1. Structure-Only Learning

**Decision**: No handcrafted node features - learn from graph topology alone

**Rationale**:
- ✅ **Conceptually pure**: Trust emerges from relational patterns
- ✅ **No feature engineering**: Don't need to design features
- ✅ **Generalizable**: Works with any graph structure
- ✅ **PyG compatible**: Standard heterogeneous GAT

**Trade-off**:
- More parameters needed (~3.2M)
- Requires more training data

### 2. Cross-Validation Constraints

**Decision**: Require ≥2 observations before trust updates

**Rationale**:
- ✅ **Robust to deception**: Single observations unreliable
- ✅ **Consensus-based**: Multiple robots must agree
- ✅ **Detects contradictions**: Exposes inconsistent behavior
- ✅ **Prevents premature updates**: Wait for sufficient evidence

**Trade-off**:
- Slower initial trust convergence
- Some samples filtered during training

### 3. Ego-Centric Architecture

**Decision**: Each robot builds its own local graph

**Rationale**:
- ✅ **Scalable**: O(N) graphs instead of O(N²)
- ✅ **Distributed**: Each robot operates independently
- ✅ **Realistic**: Matches real-world constraints
- ✅ **Privacy-preserving**: Robots only share observations within comm range

### 4. Mass-Based Trust Updates

**Decision**: Accumulate α and β directly (not log-sum)

**Rationale**:
- ✅ **Intuitive**: Direct mass accumulation matches Beta semantics
- ✅ **Stable**: Trust cap prevents oscillations
- ✅ **Confidence-aware**: Tracks certainty of estimates
- ✅ **Simple**: Easier to understand and debug

---

## Common Issues

### Issue 1: Low Training Accuracy

**Symptom**: Both train and val accuracy <75%

**Possible Causes**:
1. Insufficient data diversity
2. Model too small (hidden_dim=128)
3. Label imbalance

**Solutions**:
- Generate more episodes (5000+)
- Increase hidden_dim (128 → 256)
- Check label distribution (should be ~50/50)
- Verify edge types exist in data

### Issue 2: Overfitting

**Symptom**: Train loss ↓, Val loss ↑

**Solutions**:
- Increase dropout (0.3 → 0.5)
- Increase weight decay (1e-4 → 1e-3)
- Lower learning rate (1e-3 → 5e-4)
- Generate more diverse data

### Issue 3: Few Samples Pass Cross-Validation

**Symptom**: <50% of samples pass filtering

**Causes**:
- Robots too far apart (increase proximal_range)
- Too few robots (increase robot_density)
- Not enough detection overlap

**Solutions**:
- Increase proximal_range (50 → 75)
- Increase robot_density (0.001 → 0.002)
- Verify co_detection edges exist

### Issue 4: Slow Trust Convergence

**Symptom**: Trust takes >200 timesteps to converge

**Causes**:
- Cross-validation constraints too strict
- Evidence scores near 0.5 (low confidence)
- Trust cap too restrictive

**Solutions**:
- Check evidence scores (should be <0.3 or >0.7)
- Increase confidence_factor in mass updates
- Relax trust cap (10% → 15%)

---

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torch-geometric numpy matplotlib scikit-learn
```

### Complete Pipeline

```bash
# 1. Generate dataset with cross-validation filtering (recommended: 15000 episodes)
python generate_supervised_data.py \
  --episodes 15000 \
  --steps 200 \
  --step-interval 5 \
  --robot-density 0.0010,0.0015 \
  --output supervised_trust_dataset.pkl

# 2. Train model with masked loss (~1.8M samples, 18M training points)
python train_supervised_trust.py \
  --data supervised_trust_dataset.pkl \
  --epochs 1000 \
  --batch-size 32 \
  --lr 1e-3 \
  --patience 100 \
  --device auto

# 3. Evaluate across scenarios
python compare_trust_methods.py
```

### Using Trained Model

```python
from supervised_trust_gnn import SupervisedTrustPredictor

# Load model
predictor = SupervisedTrustPredictor(
    model_path='supervised_trust_model.pth',
    device='cpu',
    proximal_range=50.0
)

# Predict with cross-validation
result = predictor.predict_from_robots_tracks(ego_robot, all_robots)

if result is not None:
    # Cross-validation constraints met
    predictions = result['predictions']
    meaningful_tracks = result['meaningful_track_indices']

    # Update trust values
    ego_trust = predictions['agent']['trust_scores'][0]
    track_trust = predictions['track']['trust_scores'][meaningful_tracks]
else:
    # No cross-validation - skip update
    pass
```

---

## Citation

If you use this code, please cite:

```bibtex
@software{neural_symbolic_trust_2025,
  title = {Neural-Symbolic Multi-Robot Trust System},
  author = {Your Name},
  year = {2025},
  note = {Structure-only GNN with cross-validation constraints for distributed trust assessment}
}
```

---

## License

[Your License Here]

---

## Acknowledgments

- PyTorch Geometric for heterogeneous GNN framework
- Beta distribution for trust representation
- Graph attention networks for message passing
