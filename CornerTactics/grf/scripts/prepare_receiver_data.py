#!/usr/bin/env python3
"""
Prepare receiver-labeled GNN data from raw JSON corner kick episodes.

Extends prepare_gnn_data.py by adding receiver labels:
- Scans post-delivery frames for first ball_owned_team >= 0 where game_mode != 4
- Maps receiver to normalized node index (0-21) or 22 for no-receiver
- Uses delivery frame for graph features (same as existing GNN data)
- Same split logic: np.random.seed(42), 70/15/15

Node ordering after normalization:
  Nodes 0-10: Left team (attacking)
  Nodes 11-21: Right team (defending)
  Node 22: Ball

Node features (13-dim TacticAI-style):
  [x, y, vx, vy, tired_factor, ball_possession, role_GK, role_DEF, role_MID, role_FWD,
   is_attacker, is_defender, is_ball]

Edge features (2-dim):
  [is_same_team, is_opponent]

Receiver mapping after normalization:
  attacking_team=0: ball_owned_team=0, player=p -> node p
                    ball_owned_team=1, player=p -> node 11+p
  attacking_team=1: ball_owned_team=1, player=p -> node p
                    ball_owned_team=0, player=p -> node 11+p

Usage:
    python grf/scripts/prepare_receiver_data.py \
        --input_file grf/data/raw_11v11/corners_merged_10k.json \
        --output_dir grf/data/gnn_receiver
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm

# Reuse core functions from prepare_gnn_data
from prepare_gnn_data import (
    normalize_frame_for_attacking_team,
    build_graph_data,
    compute_normalization_stats,
    normalize_features,
    NUM_NODE_FEATURES,
    NUM_EDGE_FEATURES,
    FEAT_X, FEAT_Y, FEAT_VX, FEAT_VY,
    EDGE_REL_X, EDGE_REL_Y, EDGE_DISTANCE, EDGE_ANGLE,
    EDGE_REL_VX, EDGE_REL_VY,
)


def find_receiver(episode):
    """
    Find the first player to receive the ball after corner delivery.

    Scans frames for the first occurrence of ball_owned_team >= 0
    where game_mode != 4 (not still in corner kick mode).

    Args:
        episode: Raw episode dict with 'frames' and 'metadata'.

    Returns:
        Tuple (ball_owned_team, ball_owned_player, frame_idx) or
        (None, None, None) if no receiver found.
    """
    frames = episode.get('frames', [])
    for i, f in enumerate(frames):
        game_mode = f.get('game_mode', 4)
        ball_owned_team = f.get('ball_owned_team', -1)
        ball_owned_player = f.get('ball_owned_player', -1)

        if game_mode != 4 and ball_owned_team >= 0 and ball_owned_player >= 0:
            return ball_owned_team, ball_owned_player, i

    return None, None, None


def map_receiver_to_node(ball_owned_team, ball_owned_player, attacking_team):
    """
    Map a receiver (team, player) to the normalized node index.

    After normalize_frame_for_attacking_team:
    - Left team = attacking team -> nodes 0-10
    - Right team = defending team -> nodes 11-21

    When attacking_team=0:
      ball_owned_team=0 (attacker), player=p -> node p
      ball_owned_team=1 (defender), player=p -> node 11+p

    When attacking_team=1 (teams get swapped during normalization):
      ball_owned_team=1 (attacker), player=p -> node p
      ball_owned_team=0 (defender), player=p -> node 11+p

    Args:
        ball_owned_team: Team index (0 or 1) from raw frame.
        ball_owned_player: Player index (0-10) from raw frame.
        attacking_team: Which team is attacking (from episode metadata).

    Returns:
        Node index (0-21).
    """
    if ball_owned_team == attacking_team:
        # Attacker receives -> left team nodes (0-10)
        return ball_owned_player
    else:
        # Defender receives -> right team nodes (11-21)
        return 11 + ball_owned_player


def process_episode_with_receiver(episode, use_frame='delivery'):
    """
    Process a single episode into graph data with receiver label.

    Uses delivery frame for graph features (same as existing GNN pipeline).
    Scans all post-delivery frames to find the receiver.

    Args:
        episode: Episode dict with frames, outcome, metadata.
        use_frame: Frame selection method ('delivery', 'first', int).

    Returns:
        dict with node_features, edge_index, edge_attr, shot_label, goal_label,
        receiver_label (0-22), receiver_team ('attacker'/'defender'/'none'),
        frame_idx, receiver_frame_idx.
        Returns None if episode is invalid (no frames).
    """
    frames = episode.get('frames', [])
    if len(frames) == 0:
        return None

    attacking_team = episode.get('metadata', {}).get('attacking_team', 0)

    # Select frame for graph features (replicates process_episode logic)
    if use_frame == 'first':
        frame_idx = 0
    elif use_frame == 'delivery':
        frame_idx = 0
        for i, f in enumerate(frames):
            ball_vel = f.get('ball_velocity', [0, 0, 0])
            if np.linalg.norm(ball_vel[:2]) > 0.01:
                frame_idx = i
                break
    elif use_frame == 'post_delivery':
        frame_idx = 0
        for i, f in enumerate(frames):
            ball_vel = f.get('ball_velocity', [0, 0, 0])
            if np.linalg.norm(ball_vel[:2]) > 0.01:
                frame_idx = min(i + 10, len(frames) - 1)
                break
    else:
        frame_idx = min(int(use_frame), len(frames) - 1)

    frame = frames[frame_idx]
    frame = normalize_frame_for_attacking_team(frame, attacking_team)

    # Build graph from the selected frame (now returns edge_attr too)
    node_features, edge_index, edge_attr = build_graph_data(frame)

    # Get outcome labels
    outcome = episode.get('outcome', {})
    shot_label = 1 if outcome.get('shot_occurred', False) or outcome.get('goal_scored', False) else 0
    goal_label = 1 if outcome.get('goal_scored', False) else 0

    # Find receiver
    bot, bop, recv_frame_idx = find_receiver(episode)
    if bot is not None:
        receiver_node = map_receiver_to_node(bot, bop, attacking_team)
        receiver_team = 'attacker' if bot == attacking_team else 'defender'
    else:
        receiver_node = 22  # no-receiver class
        receiver_team = 'none'
        recv_frame_idx = -1

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'shot_label': shot_label,
        'goal_label': goal_label,
        'receiver_label': receiver_node,
        'receiver_team': receiver_team,
        'frame_idx': frame_idx,
        'receiver_frame_idx': recv_frame_idx,
    }


def apply_d2_augmentation(graph):
    """
    Apply D2 symmetry group augmentations to a graph.

    D2 symmetry consists of:
    - Identity (no change)
    - Horizontal flip (mirror across center-x)
    - Vertical flip (mirror across center-y)
    - 180° rotation (both flips combined)

    For GRF pitch coordinates:
    - x: [-1, 1] (left goal to right goal)
    - y: [-0.42, 0.42] (sideline to sideline)

    Both node features and edge features are flipped accordingly.

    Args:
        graph: Dict with 'node_features' (23, 13), 'edge_attr' (506, 8), 'receiver_label', etc.
               Node features: [x, y, vx, vy, tired_factor, ball_possession, role_GK, role_DEF,
                               role_MID, role_FWD, is_attacker, is_defender, is_ball]
               Edge features: [rel_x, rel_y, distance, angle, rel_vx, rel_vy,
                               is_same_team, is_opponent]

    Returns:
        List of 4 augmented graphs (including original).
    """
    augmented = []
    node_features = graph['node_features']  # (23, 13)
    edge_attr = graph['edge_attr']  # (506, 8)

    # Node feature indices
    X_IDX, Y_IDX = FEAT_X, FEAT_Y
    VX_IDX, VY_IDX = FEAT_VX, FEAT_VY

    # Augmentation 1: Identity
    augmented.append(graph.copy())

    # Augmentation 2: Horizontal flip (negate x and vx)
    h_flip = graph.copy()
    h_flip_features = node_features.copy()
    h_flip_features[:, X_IDX] = -h_flip_features[:, X_IDX]
    h_flip_features[:, VX_IDX] = -h_flip_features[:, VX_IDX]
    h_flip['node_features'] = h_flip_features
    # Edge features: negate rel_x, rel_vx, and adjust angle
    h_flip_edges = edge_attr.copy()
    h_flip_edges[:, EDGE_REL_X] = -h_flip_edges[:, EDGE_REL_X]
    h_flip_edges[:, EDGE_REL_VX] = -h_flip_edges[:, EDGE_REL_VX]
    # Angle adjustment: x -> -x means angle -> pi - angle (reflected across y-axis)
    h_flip_edges[:, EDGE_ANGLE] = np.pi - h_flip_edges[:, EDGE_ANGLE]
    # Normalize angle to [-pi, pi]
    h_flip_edges[:, EDGE_ANGLE] = np.arctan2(
        np.sin(h_flip_edges[:, EDGE_ANGLE]),
        np.cos(h_flip_edges[:, EDGE_ANGLE])
    )
    h_flip['edge_attr'] = h_flip_edges
    augmented.append(h_flip)

    # Augmentation 3: Vertical flip (negate y and vy)
    v_flip = graph.copy()
    v_flip_features = node_features.copy()
    v_flip_features[:, Y_IDX] = -v_flip_features[:, Y_IDX]
    v_flip_features[:, VY_IDX] = -v_flip_features[:, VY_IDX]
    v_flip['node_features'] = v_flip_features
    # Edge features: negate rel_y, rel_vy, and adjust angle
    v_flip_edges = edge_attr.copy()
    v_flip_edges[:, EDGE_REL_Y] = -v_flip_edges[:, EDGE_REL_Y]
    v_flip_edges[:, EDGE_REL_VY] = -v_flip_edges[:, EDGE_REL_VY]
    # Angle adjustment: y -> -y means angle -> -angle (reflected across x-axis)
    v_flip_edges[:, EDGE_ANGLE] = -v_flip_edges[:, EDGE_ANGLE]
    v_flip['edge_attr'] = v_flip_edges
    augmented.append(v_flip)

    # Augmentation 4: 180° rotation (negate x, y, vx, vy)
    rot180 = graph.copy()
    rot180_features = node_features.copy()
    rot180_features[:, X_IDX] = -rot180_features[:, X_IDX]
    rot180_features[:, Y_IDX] = -rot180_features[:, Y_IDX]
    rot180_features[:, VX_IDX] = -rot180_features[:, VX_IDX]
    rot180_features[:, VY_IDX] = -rot180_features[:, VY_IDX]
    rot180['node_features'] = rot180_features
    # Edge features: negate rel_x, rel_y, rel_vx, rel_vy, and rotate angle by pi
    rot180_edges = edge_attr.copy()
    rot180_edges[:, EDGE_REL_X] = -rot180_edges[:, EDGE_REL_X]
    rot180_edges[:, EDGE_REL_Y] = -rot180_edges[:, EDGE_REL_Y]
    rot180_edges[:, EDGE_REL_VX] = -rot180_edges[:, EDGE_REL_VX]
    rot180_edges[:, EDGE_REL_VY] = -rot180_edges[:, EDGE_REL_VY]
    # Angle adjustment: 180° rotation means angle -> angle + pi (or angle - pi)
    rot180_edges[:, EDGE_ANGLE] = rot180_edges[:, EDGE_ANGLE] + np.pi
    # Normalize angle to [-pi, pi]
    rot180_edges[:, EDGE_ANGLE] = np.arctan2(
        np.sin(rot180_edges[:, EDGE_ANGLE]),
        np.cos(rot180_edges[:, EDGE_ANGLE])
    )
    rot180['edge_attr'] = rot180_edges
    augmented.append(rot180)

    return augmented


def save_data(graphs, output_path, split_name, node_mean=None, node_std=None,
              edge_mean=None, edge_std=None):
    """Save processed data as .npz file with receiver labels and edge attributes."""
    node_features = np.array([g['node_features'] for g in graphs])
    edge_index = graphs[0]['edge_index']
    # Edge attributes vary per graph now due to position-dependent features
    edge_attr = np.array([g['edge_attr'] for g in graphs])
    shot_labels = np.array([g['shot_label'] for g in graphs])
    goal_labels = np.array([g['goal_label'] for g in graphs])
    receiver_labels = np.array([g['receiver_label'] for g in graphs])

    save_dict = {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'shot_labels': shot_labels,
        'goal_labels': goal_labels,
        'receiver_labels': receiver_labels,
    }

    if node_mean is not None:
        save_dict['norm_mean'] = node_mean
        save_dict['norm_std'] = node_std
    if edge_mean is not None:
        save_dict['edge_norm_mean'] = edge_mean
        save_dict['edge_norm_std'] = edge_std

    np.savez(output_path / f'{split_name}.npz', **save_dict)
    print(f"Saved {len(graphs)} graphs to {output_path / f'{split_name}.npz'}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare receiver-labeled GNN data from corner kick episodes'
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSON file with collected episodes')
    parser.add_argument('--output_dir', type=str, default='./grf/data/gnn_receiver',
                        help='Output directory for processed data')
    parser.add_argument('--use_frame', type=str, default='delivery',
                        help="Frame to use: 'first', 'delivery', 'post_delivery', or int")
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d2_augment', action='store_true',
                        help='Apply D2 symmetry augmentation (4x training data)')

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    use_frame = args.use_frame
    if use_frame.isdigit():
        use_frame = int(use_frame)

    # Load episodes
    print(f"Loading episodes from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        episodes = json.load(f)
    print(f"Loaded {len(episodes)} episodes")

    # Process all episodes
    print("Processing episodes into graphs with receiver labels...")
    graphs = []
    for ep in tqdm(episodes, desc="Building graphs"):
        graph = process_episode_with_receiver(ep, use_frame=use_frame)
        if graph is not None:
            graphs.append(graph)

    print(f"Processed {len(graphs)} valid graphs")

    # Receiver distribution analysis
    receiver_labels = [g['receiver_label'] for g in graphs]
    receiver_teams = [g['receiver_team'] for g in graphs]

    n_total = len(graphs)
    n_attacker = sum(1 for t in receiver_teams if t == 'attacker')
    n_defender = sum(1 for t in receiver_teams if t == 'defender')
    n_no_recv = sum(1 for t in receiver_teams if t == 'none')

    print(f"\nReceiver distribution:")
    print(f"  Attacker receives: {n_attacker} ({n_attacker / n_total:.1%})")
    print(f"  Defender receives: {n_defender} ({n_defender / n_total:.1%})")
    print(f"  No receiver:       {n_no_recv} ({n_no_recv / n_total:.1%})")

    # Shot rate by receiver group
    for group_name, group_filter in [('attacker', 'attacker'), ('defender', 'defender'), ('no_recv', 'none')]:
        group_shots = [g['shot_label'] for g in graphs if g['receiver_team'] == group_filter]
        if group_shots:
            rate = np.mean(group_shots)
            print(f"  Shot rate ({group_name}): {rate:.1%} (n={len(group_shots)})")

    # Per-node receiver counts
    node_counts = Counter(receiver_labels)
    print(f"\nReceiver node distribution (top 10):")
    for node, count in node_counts.most_common(10):
        label = f"node {node}" if node < 22 else "no-receiver"
        role = "attacker" if node < 11 else ("defender" if node < 22 else "n/a")
        print(f"  {label} ({role}): {count} ({count / n_total:.1%})")

    # Class balance info
    shot_rate = np.mean([g['shot_label'] for g in graphs])
    goal_rate = np.mean([g['goal_label'] for g in graphs])
    print(f"\nShot rate: {shot_rate:.2%}")
    print(f"Goal rate: {goal_rate:.2%}")

    # Split data (same logic as prepare_gnn_data.py)
    np.random.seed(args.seed)
    indices = np.random.permutation(len(graphs))

    n_train = int(len(graphs) * args.train_ratio)
    n_val = int(len(graphs) * args.val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    # Apply D2 augmentation to training set only (4x data)
    if args.d2_augment:
        print(f"\nApplying D2 symmetry augmentation to training set...")
        augmented_train = []
        for g in train_graphs:
            augmented_train.extend(apply_d2_augmentation(g))
        train_graphs = augmented_train
        print(f"  Training set augmented: {len(train_idx)} -> {len(train_graphs)} samples")

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_graphs)}{' (with D2 augmentation)' if args.d2_augment else ''}")
    print(f"  Val:   {len(val_graphs)}")
    print(f"  Test:  {len(test_graphs)}")

    # Compute normalization stats from training set only
    node_mean, node_std, edge_mean, edge_std = compute_normalization_stats(train_graphs)
    print(f"\nNormalization stats (from training set):")
    print(f"  Node mean: {node_mean}")
    print(f"  Node std:  {node_std}")
    print(f"  Edge mean: {edge_mean}")
    print(f"  Edge std:  {edge_std}")

    # Apply normalization
    train_graphs = normalize_features(train_graphs, node_mean, node_std, edge_mean, edge_std)
    val_graphs = normalize_features(val_graphs, node_mean, node_std, edge_mean, edge_std)
    test_graphs = normalize_features(test_graphs, node_mean, node_std, edge_mean, edge_std)

    # Save data
    save_data(train_graphs, output_path, 'train', node_mean, node_std, edge_mean, edge_std)
    save_data(val_graphs, output_path, 'val')
    save_data(test_graphs, output_path, 'test')

    # Compute receiver class weights (for training)
    # Use sqrt of inverse frequency to dampen extreme weights, then normalize
    train_receivers = np.array([g['receiver_label'] for g in train_graphs])
    class_counts = np.bincount(train_receivers, minlength=23)
    n_active_classes = np.sum(class_counts > 0)
    n_total_samples = len(train_receivers)

    # Sqrt of inverse frequency (dampens extreme weights for rare classes)
    class_weights = np.zeros(23, dtype=np.float32)
    for c in range(23):
        if class_counts[c] > 0:
            # sqrt dampens the extreme values while still upweighting rare classes
            class_weights[c] = np.sqrt(n_total_samples / class_counts[c])

    # Normalize so mean of active weights = 1.0
    active_weights = class_weights[class_weights > 0]
    if len(active_weights) > 0:
        class_weights[class_weights > 0] /= active_weights.mean()

    # Cap at 5.0 and re-normalize after capping
    class_weights = np.minimum(class_weights, 5.0)
    active_weights = class_weights[class_weights > 0]
    if len(active_weights) > 0:
        class_weights[class_weights > 0] /= active_weights.mean()

    print(f"\nClass weights (n_active={n_active_classes}):")
    print(f"  Range: [{class_weights[class_weights > 0].min():.3f}, {class_weights.max():.3f}]")
    print(f"  Mean (active): {class_weights[class_weights > 0].mean():.3f}")

    # Save metadata
    metadata = {
        'input_file': str(args.input_file),
        'use_frame': str(use_frame),
        'd2_augmented': args.d2_augment,
        'num_episodes': len(graphs),
        'train_size': len(train_graphs),
        'train_size_original': len(train_idx),
        'val_size': len(val_graphs),
        'test_size': len(test_graphs),
        'shot_rate': float(shot_rate),
        'goal_rate': float(goal_rate),
        'norm_mean': node_mean.tolist(),
        'norm_std': node_std.tolist(),
        'edge_norm_mean': edge_mean.tolist(),
        'edge_norm_std': edge_std.tolist(),
        'num_nodes': 23,
        'num_node_features': NUM_NODE_FEATURES,
        'num_edge_features': NUM_EDGE_FEATURES,
        'num_receiver_classes': 23,
        'node_feature_names': [
            'x', 'y', 'vx', 'vy', 'tired_factor', 'ball_possession',
            'role_GK', 'role_DEF', 'role_MID', 'role_FWD',
            'is_attacker', 'is_defender', 'is_ball'
        ],
        'edge_feature_names': [
            'rel_x', 'rel_y', 'distance', 'angle',
            'rel_vx', 'rel_vy', 'is_same_team', 'is_opponent'
        ],
        'feature_version': 'v3_rich_edges',
        'receiver_distribution': {
            'attacker_receives': n_attacker,
            'defender_receives': n_defender,
            'no_receiver': n_no_recv,
            'per_node': {str(k): int(v) for k, v in sorted(node_counts.items())},
        },
        'receiver_class_weights': class_weights.tolist(),
        'shot_rate_by_receiver': {
            'attacker': float(np.mean([g['shot_label'] for g in graphs if g['receiver_team'] == 'attacker'])) if n_attacker > 0 else 0.0,
            'defender': float(np.mean([g['shot_label'] for g in graphs if g['receiver_team'] == 'defender'])) if n_defender > 0 else 0.0,
            'no_receiver': float(np.mean([g['shot_label'] for g in graphs if g['receiver_team'] == 'none'])) if n_no_recv > 0 else 0.0,
        },
    }
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {output_path / 'metadata.json'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
