# Neural-Symbolic Multi-Robot Trust System

A neural-symbolic framework for distributed trust management in multi-robot systems using Graph Neural Networks (GNNs) and mass-based trust accumulation for adaptive, continuous trust assessment.

## System Overview

This system implements a two-stage trust management framework for multi-robot collaborative scenarios:

### Stage 1: Supervised Learning (Evidence Extraction)
**Heterogeneous GNN** trained on labeled ego-centric graphs to predict trustworthiness:
- **Input**: Ego-centric graph with continuous behavioral features (6 for robots, 2 for tracks)
- **Output**: Trust probability scores ∈ [0, 1] for robots and tracks
- **Architecture**: 3-layer GAT with heterogeneous message passing and skip connections
- **Training**: Binary cross-entropy on balanced 50/50 adversarial/legitimate samples

### Stage 2: Mass-Based Trust Updates
**Mass accumulation system** with cross-validation for trust updates:
- **Input**: GNN evidence scores + cross-validation context
- **Output**: Trust parameter updates (Δα, Δβ) with confidence gating
- **Cross-validation**: Simple validator counting for robots, none for tracks
- **Trust cap**: Binary search to limit trust changes to 10% per timestep

### Trust Representation
Each entity (robot/track) maintains trust as **Beta distribution** Beta(α, β):
- **Mean trust**: τ = α/(α+β) ∈ [0, 1]
- **Confidence**: κ = α + β (higher = more certain)
- **Decay**: Exponential forgetting with γ = 0.99 per timestep
- **Updates**: Mass-based accumulation with confidence gating

---

## Key Components

### 1. Supervised GNN (Evidence Extraction)

The supervised GNN serves as the **evidence extractor** in the trust pipeline. It processes ego-centric heterogeneous graphs and outputs trust probability scores.

#### Features: Trust-Free and Continuous

**Design Philosophy**: Use **behavioral features only** (no trust values) to ensure train-test consistency.

**Robot Features (6D Continuous)**:

1. **observed_count**: Number of unique objects observed at current timestep
2. **fused_count**: Number of fused tracks within observed tracks
3. **expected_count**: Objects in ego graph within this robot's FoV
4. **partner_count**: Number of robots with fused tracks with this robot
5. **detection_ratio**: `observed_count / (expected_count + ε)`
6. **validator_ratio**: `partner_count / (robots_with_detections_in_fov + ε)`

**Track Features (2D Continuous)**:

1. **detector_count**: Number of robots that detected this track
2. **detector_ratio**: `detector_count / (robots_with_fov_containing_track + ε)`

**Why Continuous Ratio Features?**
- ✅ Much more informative than binary features
- ✅ Completely trust-free (no train-test distribution shift)
- ✅ Captures fine-grained behavioral patterns
- ✅ Adversarial discrimination: legitimate robots have high ratios, adversarials have low ratios

#### GNN Architecture

**Network Type**: Heterogeneous Graph Attention Network (GAT)

**Node Types**:
- `agent`: Robots in ego-centric graph
- `track`: Tracks detected by robots

**Edge Types**:
1. `('agent', 'in_fov_and_observed', 'track')`: Robot detects track
2. `('track', 'observed_and_in_fov_by', 'agent')`: Reverse
3. `('agent', 'in_fov_only', 'track')`: Track in robot's FOV but not detected
4. `('track', 'in_fov_only_by', 'agent')`: Reverse
5. `('agent', 'co_detection', 'agent')`: Robots detecting same object

**Architecture**:
```
Input: x_agent ∈ R^(N×6), x_track ∈ R^(N×2)
↓
Embedding: Linear(6→64), Linear(2→64)
↓
3× GAT Layers (64→64) with skip connections
↓
Classification: Linear(64→16→1) with Sigmoid
↓
Output: p_agent ∈ [0,1]^N, p_track ∈ [0,1]^N
```

**Model Size**: ~91,000 parameters

#### Training

**Dataset Generation**:
```bash
python generate_supervised_data.py --episodes 200 --output supervised_trust_dataset.pkl
```

**Model Training**:
```bash
python train_supervised_trust.py \
  --data supervised_trust_dataset.pkl \
  --epochs 100 \
  --batch-size 256 \
  --output supervised_trust_model.pth
```

**Expected Performance**:
- Agent: ~85-90% accuracy, ~87-92% F1
- Track: ~85-90% accuracy, ~87-92% F1

---

### 2. Mass-Based Trust Update System

The mass-based system uses GNN evidence scores with cross-validation to update trust distributions.

#### Core Concept: Mass Accumulation

Trust updates are computed using **mass accumulation** with **confidence gating**:

```python
# Base update
log_κ = log(α + β)
g(κ) = 1 / (1 + exp(-(log_κ - μ) / σ))  # Confidence gate
Δmass = c_mass × κ × g(κ)  # Mass proposal

# Split mass by evidence
Δα = Δmass × evidence
Δβ = Δmass × (1 - evidence)
```

**Key parameters**:
- `c_mass = 0.1`: Mass coefficient
- `μ = 2.0`: Confidence gate center (log scale)
- `σ = 1.0`: Confidence gate width
- `γ = 0.99`: Decay factor

**Confidence Gate Intuition**:
- Low confidence (small κ): Gate closes → small updates
- High confidence (large κ): Gate opens → larger updates
- Prevents unstable updates when trust is uncertain

#### Simplified Cross-Validation

**For Robots**: Only count validators
```python
# Count total validators across all tracks
total_validators = 0
for track in robot_tracks:
    num_detectors = len(robots_detecting_this_object)
    num_validators = num_detectors - 1  # Exclude self
    total_validators += num_validators

# Scale update by validator count
Δα *= total_validators
Δβ *= total_validators

# If no validators → (0, 0) update (pure decay)
# If validators → update scaled proportionally
```

**For Tracks**: No cross-validation
```python
# Simple base update using GNN evidence
Δα, Δβ = compute_base_update(α, β, evidence)
```

**Why This Works**:
- Simple and interpretable: more validators → stronger update
- No validators → no update (requires peer validation)
- Naturally bounded (validators typically 0-10)

#### Trust Cap with Binary Search

To limit trust changes to 10% per timestep, we use **binary search** to find the correct scaling factor:

```python
def apply_trust_cap(α, β, Δα, Δβ, delta_tau_max=0.1):
    τ_old = α / (α + β)
    τ_new = (α×γ + Δα) / (α×γ + β×γ + Δα + Δβ)

    if |τ_new - τ_old| > delta_tau_max:
        # Binary search for scaling factor k
        # such that |τ(k×Δα, k×Δβ) - τ_old| = delta_tau_max
        k = binary_search(...)
        Δα *= k
        Δβ *= k

    return Δα, Δβ
```

**Why Binary Search?**
- Trust τ = α/(α+β) is **nonlinear**
- Linear scaling of (Δα, Δβ) does NOT linearly scale Δτ
- Binary search finds exact scaling for desired cap
- See [TRUST_CAP_BUG_EXPLAINED.md](TRUST_CAP_BUG_EXPLAINED.md) for details

#### Update Process

**For each timestep**:

1. **Save original trust values** (used for all calculations)
2. **Collect evidence** from all ego graphs
3. **Average evidence** per robot from all ego graphs that include it
4. **Compute cross-validation** context (validator count)
5. **Compute updates** with mass accumulation
6. **Apply trust cap** using binary search
7. **Apply updates** with decay: `α_new = α_old × γ + Δα`

**Example**:
```python
# Robot R3 with 3 tracks
# Track T1: [R3, R5] → 1 validator
# Track T2: [R3, R5, R7] → 2 validators
# Track T3: [R3] only → 0 validators
# Total: 1 + 2 + 0 = 3 validators

# Get averaged evidence from all ego graphs
avg_evidence = 0.75

# Compute base update
Δα, Δβ = compute_base_update(α, β, 0.75)  # (0.5, 0.3)

# Scale by validators
Δα *= 3  # 1.5
Δβ *= 3  # 0.9

# Apply trust cap (binary search)
Δα, Δβ = apply_trust_cap(α, β, 1.5, 0.9)

# Update trust
R3.α = α × 0.99 + Δα
R3.β = β × 0.99 + Δβ
```

---

## Quick Start

### Prerequisites

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torch-geometric numpy matplotlib
```

### Training Pipeline

#### Step 1: Generate Supervised Training Data

```bash
python generate_supervised_data.py \
  --episodes 200 \
  --output supervised_trust_dataset.pkl
```

**Output**: ~140K ego-graph samples with binary labels from 200 simulation episodes

#### Step 2: Train Supervised GNN

```bash
python train_supervised_trust.py \
  --data supervised_trust_dataset.pkl \
  --epochs 100 \
  --batch-size 256 \
  --output supervised_trust_model.pth
```

**Output**: Trained GNN model (~91K parameters)

#### Step 3: Run Comparison

```bash
python compare_trust_methods.py
```

**Output**: Comparison of three trust methods:
- Paper baseline (original algorithm)
- Fixed-step baseline (GNN + fixed step scale)
- Mass-based (GNN + mass accumulation + cross-validation)

Results saved to:
- `trust_comparison_Scenario_X.json`
- `trust_comparison_Scenario_X.png`
- `three_scenario_comparison.json`

---

## System Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
└─────────────────────────────────────────────────────────────┘

STAGE 1: SUPERVISED LEARNING
─────────────────────────────
1. Generate Dataset
   └─> 200 episodes → ~140K ego-graph samples
   └─> Features: 6D robots, 2D tracks (continuous, trust-free)
   └─> Labels: 1=legitimate, 0=adversarial
   └─> Balanced: 50/50 split

2. Train GNN
   └─> 3-layer heterogeneous GAT
   └─> BCE loss, Adam optimizer
   └─> ~85-90% accuracy on agents/tracks

┌─────────────────────────────────────────────────────────────┐
│                   RUNTIME ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────┘

For each timestep t:

  Step 1: Ego-Sweep (Collect Evidence)
  ────────────────────────────────────
  For each robot (ego):
    1. Build ego-centric graph (proximal robots + tracks)
    2. Extract behavioral features (6D robots, 2D tracks)
    3. Run GNN forward pass → evidence scores
    4. Store evidence for this robot

  Step 2: Compute Updates
  ───────────────────────
  For each robot:
    1. Average evidence from all ego graphs
    2. Count validators (robots detecting same objects)
    3. Compute base update with confidence gating
    4. Scale by validator count
    5. Apply trust cap (binary search)

  For each track:
    1. Use GNN evidence (no cross-validation)
    2. Compute base update with confidence gating
    3. Apply trust cap (binary search)

  Step 3: Apply Updates
  ─────────────────────
  For all robots and tracks:
    α_new = α_old × γ + Δα
    β_new = β_old × γ + Δβ
    τ = α / (α + β)
```

---

## File Structure

### Core Implementation

**Trust System**:
- `mass_based_trust_update.py` - Mass-based trust update system
- `supervised_trust_gnn.py` - Heterogeneous GNN for evidence extraction

**Training**:
- `generate_supervised_data.py` - Generate GNN training data
- `train_supervised_trust.py` - Train supervised GNN

**Evaluation**:
- `compare_trust_methods.py` - Compare trust methods on 3 scenarios
- `visualize_trust_updates.py` - Visualization tools

**Environment**:
- `simulation_environment.py` - Multi-robot simulation world
- `robot_track_classes.py` - Robot and Track classes with trust

**Utilities**:
- `paper_trust_algorithm.py` - Baseline from original paper

---

## Comparison: Three Trust Methods

The system compares three trust update approaches:

### 1. Paper Baseline
Original algorithm from the paper (rule-based)

### 2. Fixed-Step Baseline
- GNN evidence extraction
- Fixed step scale (0.5)
- No cross-validation
- Exponential decay (γ=0.99)

### 3. Mass-Based (Ours)
- GNN evidence extraction
- Mass accumulation with confidence gating
- Simple cross-validation (validator counting)
- Trust cap with binary search
- Exponential decay (γ=0.99)

**Three Test Scenarios**:
1. Low FP (0.3), Low FN (0.1)
2. High FP (0.8), Low FN (0.1)
3. Low FP (0.3), High FN (0.3)

**Metrics**:
- Final trust values (legitimate vs adversarial)
- Trust convergence speed
- Method correlations

---

## Key Design Decisions

### 1. Trust-Free Features

**Decision**: Use only behavioral features (no trust values in GNN input)

**Rationale**:
- **Train-test consistency**: Features identical during training and deployment
- **No distribution shift**: Trust initialization doesn't affect GNN input
- **Behavioral focus**: Model learns from observable actions, not trust values

### 2. Simplified Cross-Validation

**Decision**: Only count validators for robots, none for tracks

**Rationale**:
- **Simplicity**: One parameter instead of 7 complex penalties
- **Interpretability**: More validators → stronger update
- **Natural bounds**: Validator count typically 0-10 (no saturation)
- **Peer validation**: Trust based on agreement with other robots

### 3. Trust Cap with Binary Search

**Decision**: Use binary search to enforce trust cap, not linear scaling

**Rationale**:
- **Correctness**: τ = α/(α+β) is nonlinear, linear scaling doesn't work
- **Accuracy**: Binary search finds exact scaling for desired cap
- **Direction preservation**: Update direction always preserved (monotonic)
- See [TRUST_CAP_BUG_EXPLAINED.md](TRUST_CAP_BUG_EXPLAINED.md) for details

### 4. Averaged Evidence

**Decision**: Average evidence from all ego graphs, compute ONE update per robot

**Rationale**:
- **Simplicity**: Clean data structures (tuples, not lists)
- **Semantics**: Averaging observations is natural
- **Trust cap**: Applied once per robot, not per ego graph

---

## Performance Expectations

### Supervised GNN Training

**Typical results**:
```
Training (100 epochs):
  Agent: ~85-90% accuracy, ~87-92% F1
  Track: ~85-90% accuracy, ~87-92% F1

Validation:
  Agent: ~85-90% accuracy, ~87-92% F1
  Track: ~85-90% accuracy, ~87-92% F1
```

### Trust Convergence

**Expected behavior**:
- Legitimate robots: Trust → 0.7-0.9 (high trust)
- Adversarial robots: Trust → 0.1-0.3 (low trust)
- Convergence time: ~50-100 timesteps

### Method Comparison

**Mass-Based vs Baselines**:
- Better calibration (closer to ground truth)
- Smoother convergence (trust cap prevents oscillations)
- Robust to adversarial scenarios (cross-validation)

---

## Documentation

### Implementation Details
- **[SIMPLIFIED_CROSS_VALIDATION.md](SIMPLIFIED_CROSS_VALIDATION.md)** - Simplified cross-validation design
- **[SIMPLIFIED_EVIDENCE_AVERAGING.md](SIMPLIFIED_EVIDENCE_AVERAGING.md)** - Evidence averaging approach
- **[MASS_BASED_LOGIC_WALKTHROUGH.md](MASS_BASED_LOGIC_WALKTHROUGH.md)** - Complete logic walkthrough

### Trust Cap
- **[TRUST_CAP_BUG_EXPLAINED.md](TRUST_CAP_BUG_EXPLAINED.md)** - Trust cap bug and fix
- **[TRUST_CAP_DIRECTION_PRESERVATION.md](TRUST_CAP_DIRECTION_PRESERVATION.md)** - Direction preservation proof

### Diagnostics
- **[diagnose_trust_cap_saturation.py](diagnose_trust_cap_saturation.py)** - Check if trust cap saturates
- **[verify_trust_cap_direction.py](verify_trust_cap_direction.py)** - Verify direction preservation

---

## Common Issues

### Issue 1: Trust Cap Saturation

**Symptom**: Increasing penalties by 100x has no effect

**Cause**: Both small and large penalties exceed the trust cap

**Solution**: Increase `delta_tau_max` from 0.1 to 0.3:
```python
params = MassBasedParams(delta_tau_max=0.3)
```

### Issue 2: No Validators

**Symptom**: Robots get no trust updates (only decay)

**Cause**: All detections are unique (no co-detection)

**Solution**:
- Check detection parameters (FoV, range)
- Ensure multiple robots can observe same objects
- Verify co-detection edges in ego graph

### Issue 3: GNN Poor Performance

**Symptom**: GNN accuracy < 75%

**Cause**: Insufficient training data or feature issues

**Solution**:
- Generate more episodes (200+)
- Verify features are trust-free and behavioral
- Check data balance (should be 50/50)
- Increase training epochs

---

## Citation

If you use this code, please cite:

```bibtex
@software{neural_symbolic_trust_2025,
  title = {Neural-Symbolic Multi-Robot Trust System},
  author = {Your Name},
  year = {2025},
  note = {Mass-based trust updates with GNN evidence extraction}
}
```

---

## License

[Your License Here]

---

## Acknowledgments

- GNN architecture inspired by PyTorch Geometric
- Trust representation using Beta distributions
- Mass-based updates inspired by Bayesian inference
