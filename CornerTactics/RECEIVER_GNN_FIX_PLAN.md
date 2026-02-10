# Receiver GNN Fix Plan

This document outlines the fixes needed for the GNN receiver prediction model, in priority order.

## Summary of Issues

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| 1 | Edge features too weak | Critical - model can't learn spatial relationships | Medium |
| 2 | Role encoding over-compressed | High - losing tactical signal | Low |
| 3 | Baselines incomplete | High - thesis credibility | Low |
| 4 | D2 augmentation validity | Medium - potential noise injection | Medium |
| 5 | Frame selection comparison | Low - minor optimization | Low |
| 6 | No-receiver class naming | Low - thesis clarity | Trivial |

---

## Phase 1: Edge Features (Critical)

**Problem**: Current edge features are `[is_same_team, is_opponent]` — literally 1 bit of information encoded as 2 floats. All same-team edges are identical. All opponent edges are identical. GATv2 attention has nothing meaningful to differentiate edges.

**Files to modify**:
- `grf/scripts/prepare_gnn_data.py`
- `grf/scripts/prepare_receiver_data.py`
- `models/receiver_model.py` (update `edge_dim` default)

**Changes**:

### 1.1 Update `build_graph_data()` in `prepare_gnn_data.py`

Replace the edge feature computation (lines 221-245) with spatial edge features:

```python
def compute_edge_features(node_features, edges_src, edges_dst):
    """
    Compute rich edge features for all edges.

    Edge features (8-dim):
    - rel_x, rel_y: relative position (dst - src)
    - distance: Euclidean distance
    - angle: angle from src to dst (radians)
    - rel_vx, rel_vy: relative velocity (dst - src)
    - is_same_team, is_opponent: team relationship
    """
    edge_attr = []
    for src, dst in zip(edges_src, edges_dst):
        src_pos = node_features[src, :2]
        dst_pos = node_features[dst, :2]
        src_vel = node_features[src, 2:4]
        dst_vel = node_features[dst, 2:4]

        # Relative position
        rel_pos = dst_pos - src_pos
        distance = np.linalg.norm(rel_pos)

        # Relative velocity
        rel_vel = dst_vel - src_vel

        # Angle from src to dst
        angle = np.arctan2(rel_pos[1], rel_pos[0])

        # Team relationship
        src_is_left = src < 11
        dst_is_left = dst < 11
        src_is_ball = src == 22
        dst_is_ball = dst == 22

        if src_is_ball or dst_is_ball:
            is_same_team, is_opponent = 0.5, 0.5
        elif src_is_left == dst_is_left:
            is_same_team, is_opponent = 1.0, 0.0
        else:
            is_same_team, is_opponent = 0.0, 1.0

        edge_attr.append([
            rel_pos[0], rel_pos[1],
            distance,
            angle,
            rel_vel[0], rel_vel[1],
            is_same_team, is_opponent
        ])

    return np.array(edge_attr, dtype=np.float32)
```

### 1.2 Update metadata

- Change `num_edge_features` from 2 to 8
- Update `edge_feature_names` list

### 1.3 Update model defaults

In `receiver_model.py`, change:
```python
DEFAULT_NUM_EDGE_FEATURES = 8  # was 2
```

### 1.4 Normalize edge features

Add edge normalization in `compute_normalization_stats()` and `normalize_features()`:
- Normalize `rel_x, rel_y, distance, rel_vx, rel_vy`
- Keep `angle` in radians (already bounded)
- Keep `is_same_team, is_opponent` as-is (binary)

**Validation**: After implementing, verify edge_attr shape is `(506, 8)` in saved .npz files.

---

## Phase 2: Role Encoding (High Impact, Low Effort)

**Problem**: Collapsing 10 GRF roles to 4 categories loses tactical signal. CB and LB behave very differently during corners.

**Files to modify**:
- `grf/scripts/prepare_gnn_data.py`

**Changes**:

### 2.1 Replace `role_to_onehot()`

```python
def role_to_onehot(role):
    """
    10-dim one-hot for GRF roles.

    GRF roles: GK=0, CB=1, LB=2, RB=3, DM=4, CM=5, LM=6, RM=7, AM=8, CF=9
    """
    encoding = [0.0] * 10
    if 0 <= role <= 9:
        encoding[role] = 1.0
    return encoding
```

### 2.2 Update feature indices

```python
# New feature layout (19-dim):
# [x, y, vx, vy, tired_factor, ball_possession,
#  role_GK, role_CB, role_LB, role_RB, role_DM, role_CM, role_LM, role_RM, role_AM, role_CF,
#  is_attacker, is_defender, is_ball]

FEAT_X, FEAT_Y = 0, 1
FEAT_VX, FEAT_VY = 2, 3
FEAT_TIRED = 4
FEAT_BALL_POSSESSION = 5
FEAT_ROLE_START = 6  # roles are indices 6-15
FEAT_IS_ATTACKER = 16
FEAT_IS_DEFENDER = 17
FEAT_IS_BALL = 18

NUM_NODE_FEATURES = 19  # was 13
```

### 2.3 Update `build_graph_data()` to use new layout

### 2.4 Update metadata

- Change `num_node_features` from 13 to 19
- Update `node_feature_names` list

**Validation**: Verify node_features shape is `(23, 19)` in saved .npz files.

---

## Phase 3: Complete Baselines

**Problem**: Missing frequency-weighted random baseline and top-k accuracy for baselines.

**Files to modify**:
- `models/receiver_model.py`

**Changes**:

### 3.1 Add frequency-weighted baseline

```python
def frequency_weighted_baseline(train_labels, n_samples, seed=42):
    """
    Random predictions weighted by training class frequency.
    """
    np.random.seed(seed)
    counts = np.bincount(train_labels, minlength=23).astype(float)
    probs = counts / counts.sum()
    return np.random.choice(23, size=n_samples, p=probs)
```

### 3.2 Add top-k accuracy for baselines

```python
def baseline_topk_accuracy(train_labels, test_labels, k=3):
    """
    Top-k accuracy assuming baseline predicts top-k most frequent classes.
    """
    counts = np.bincount(train_labels, minlength=23)
    top_k_classes = set(np.argsort(counts)[-k:])
    return float(np.mean([label in top_k_classes for label in test_labels]))
```

### 3.3 Update `train_receiver()` to report all baselines

Print table:
```
--- Baselines ---
                    Top-1    Top-3    Top-5
Majority class      0.XXX    0.XXX    0.XXX
Freq-weighted       0.XXX    0.XXX    0.XXX
Nearest player      0.XXX    N/A      N/A
Uniform random      0.043    0.130    0.217
```

---

## Phase 4: Validate D2 Augmentation

**Problem**: Horizontal flip (negate x) treats left-corner and right-corner as equivalent. Corner side affects inswing/outswing dynamics.

**Files to modify**:
- `grf/scripts/prepare_receiver_data.py`
- New analysis script

**Changes**:

### 4.1 Add corner side tracking to metadata

In `process_episode_with_receiver()`:
```python
# Determine corner side from ball y position in first frame
ball_y = frames[0].get('ball_position', [0, 0, 0])[1]
corner_side = 'near' if ball_y < 0 else 'far'  # relative to standard view
```

### 4.2 Create validation experiment

Run two training runs:
1. With full D2 augmentation (current)
2. With only vertical flip + identity (no x-flip)

Compare:
- Overall top-1, top-3 accuracy
- Performance on near-side vs far-side corners separately

### 4.3 Decision rule

If performance differs significantly between corner sides when using x-flip augmentation, remove it.

---

## Phase 5: Frame Selection Comparison

**Problem**: Using "delivery frame" (ball starts moving). TacticAI uses frame just before delivery. One frame = 100ms of player movement.

**Files to modify**:
- `grf/scripts/prepare_receiver_data.py`

**Changes**:

### 5.1 Add `pre_delivery` frame option

```python
elif use_frame == 'pre_delivery':
    # Frame just before ball starts moving
    frame_idx = 0
    for i, f in enumerate(frames):
        ball_vel = f.get('ball_velocity', [0, 0, 0])
        if np.linalg.norm(ball_vel[:2]) > 0.01:
            frame_idx = max(0, i - 1)
            break
```

### 5.2 Run comparison experiment

Train with `--use_frame delivery` vs `--use_frame pre_delivery` and compare metrics.

---

## Phase 6: Thesis Clarity

**Problem**: Class 22 is labeled as "ball node" but semantically means "no first touch".

**Changes**:

### 6.1 Update documentation

In thesis and code comments, clarify:
- Nodes 0-10: Attacking team players
- Nodes 11-21: Defending team players
- Node 22: Ball (features only, not a prediction target)
- Class 22 in receiver prediction: "No first touch before out-of-play"

### 6.2 Consider renaming in code

```python
NUM_PLAYER_NODES = 22
BALL_NODE_IDX = 22
NO_RECEIVER_CLASS = 22  # Distinct semantic meaning from BALL_NODE_IDX
```

---

## Execution Order

```
Week 1:
  [x] Phase 1: Edge features
  [x] Phase 2: Role encoding
  [ ] Regenerate data with new features

Week 2:
  [x] Phase 3: Complete baselines
  [ ] Retrain model with new features
  [ ] Compare results

Week 3:
  [ ] Phase 4: D2 augmentation validation
  [ ] Phase 5: Frame selection comparison
  [ ] Final model selection

Week 4:
  [ ] Phase 6: Thesis documentation
  [ ] Final results compilation
```

---

## Success Criteria

| Metric | Current | Target | Rationale |
|--------|---------|--------|-----------|
| Top-1 accuracy | ~0.35? | >0.45 | TacticAI-comparable |
| Top-3 accuracy | ~0.55? | >0.70 | Primary thesis metric |
| Attacker vs Defender accuracy gap | Unknown | <0.10 | Balanced predictions |
| Beat frequency-weighted baseline | N/A | Yes | Proves model learns signal |

---

## Commands

### Regenerate data after Phase 1+2
```bash
conda activate grf_new
cd /home/mseo/football/CornerTactics

# Regenerate with new features
python grf/scripts/prepare_receiver_data.py \
    --input_file grf/data/raw_11v11/corners_merged_10k.json \
    --output_dir grf/data/gnn_receiver_v2 \
    --d2_augment

# Verify shapes
python -c "import numpy as np; d=np.load('grf/data/gnn_receiver_v2/train.npz'); print('node_features:', d['node_features'].shape); print('edge_attr:', d['edge_attr'].shape)"
```

### Train with new features
```bash
python models/receiver_model.py \
    --data_dir grf/data/gnn_receiver_v2 \
    --output_dir results_11v11/receiver_v2 \
    --model gat \
    --epochs 150
```
