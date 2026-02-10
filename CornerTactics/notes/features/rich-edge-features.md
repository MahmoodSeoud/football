# Rich Edge Features Implementation

## Overview

This implements Phase 1 from RECEIVER_GNN_FIX_PLAN.md.

## Problem

Current edge features are `[is_same_team, is_opponent]` - literally 1 bit of information encoded as 2 floats. All same-team edges are identical. All opponent edges are identical. GATv2 attention has nothing meaningful to differentiate edges.

## Solution

Implement 8-dimensional edge features:
1. `rel_x`, `rel_y`: relative position (dst - src)
2. `distance`: Euclidean distance
3. `angle`: angle from src to dst (radians)
4. `rel_vx`, `rel_vy`: relative velocity (dst - src)
5. `is_same_team`, `is_opponent`: team relationship (preserved from original)

## Files to Modify

1. `grf/scripts/prepare_gnn_data.py`:
   - Add `compute_edge_features()` function
   - Update `build_graph_data()` to use it
   - Update normalization to handle edge features
   - Update metadata

2. `grf/scripts/prepare_receiver_data.py`:
   - Update D2 augmentation to handle new edge features
   - Update metadata

## Validation Criteria

- Edge attr shape: `(506, 8)` in saved .npz files
- Node features unchanged: `(23, 13)`
- All edge feature values are finite (no NaN/Inf)
- Normalization applied correctly to continuous edge features

## Progress

- [x] Write failing tests for edge feature computation
- [x] Implement `compute_edge_features()` function
- [x] Update `build_graph_data()` to return 8-dim edge features
- [x] Add edge feature normalization
- [x] Update D2 augmentation in prepare_receiver_data.py
- [x] Update metadata in both scripts
- [x] Verify with integration test

## Implementation Notes

All 17 tests pass. Key changes:
1. Added `compute_edge_features()` function in prepare_gnn_data.py
2. Updated `compute_normalization_stats()` to return 4 values (node_mean, node_std, edge_mean, edge_std)
3. Updated `normalize_features()` to normalize continuous edge features (indices 0,1,2,4,5)
4. Updated D2 augmentation to flip edge features along with node features
5. Updated metadata to include edge normalization stats and new feature names
