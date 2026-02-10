#!/usr/bin/env python3
"""
Prepare GNN data from raw JSON corner kick episodes.

Converts raw episode data into PyTorch Geometric format:
- 23 nodes (11 left team + 11 right team + 1 ball)
- 13 features per node (TacticAI-style):
  [x, y, vx, vy, tired_factor, ball_possession, role_GK, role_DEF, role_MID, role_FWD,
   is_attacker, is_defender, is_ball]
- Edge features: [is_same_team, is_opponent] (2 features per edge)
- Fully connected edges (506 directed edges)
- Binary label: shot_occurred

Role grouping (GRF role indices):
- GK (role 0) → role_GK=1
- DEF (roles 1-3: CB, LB, RB) → role_DEF=1
- MID (roles 4-8: DM, CM, LM, RM, AM) → role_MID=1
- FWD (role 9: CF) → role_FWD=1

Usage:
    python prepare_gnn_data.py \
        --input_file grf/data/raw_11v11/corners_merged_10k.json \
        --output_dir grf/data/gnn
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


# Feature indices for the 13-dimensional node features
FEAT_X, FEAT_Y = 0, 1
FEAT_VX, FEAT_VY = 2, 3
FEAT_TIRED = 4
FEAT_BALL_POSSESSION = 5
FEAT_ROLE_GK, FEAT_ROLE_DEF, FEAT_ROLE_MID, FEAT_ROLE_FWD = 6, 7, 8, 9
FEAT_IS_ATTACKER, FEAT_IS_DEFENDER, FEAT_IS_BALL = 10, 11, 12

NUM_NODE_FEATURES = 13

# Edge feature indices for the 8-dimensional edge features
EDGE_REL_X, EDGE_REL_Y = 0, 1
EDGE_DISTANCE = 2
EDGE_ANGLE = 3
EDGE_REL_VX, EDGE_REL_VY = 4, 5
EDGE_IS_SAME_TEAM, EDGE_IS_OPPONENT = 6, 7

NUM_EDGE_FEATURES = 8


def compute_edge_features(node_features, edges_src, edges_dst):
    """
    Compute rich edge features for all edges.

    Edge features (8-dim):
    - rel_x, rel_y: relative position (dst - src)
    - distance: Euclidean distance
    - angle: angle from src to dst (radians)
    - rel_vx, rel_vy: relative velocity (dst - src)
    - is_same_team, is_opponent: team relationship

    Args:
        node_features: (N, F) array where F >= 4 (x, y, vx, vy in first 4 cols)
        edges_src: list of source node indices
        edges_dst: list of destination node indices

    Returns:
        (E, 8) array of edge features
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

        # Angle from src to dst (handle zero distance case)
        if distance > 1e-8:
            angle = np.arctan2(rel_pos[1], rel_pos[0])
        else:
            angle = 0.0

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


def role_to_onehot(role):
    """
    Convert GRF role index to 4-dim one-hot encoding.

    GRF roles: GK=0, CB=1, LB=2, RB=3, DM=4, CM=5, LM=6, RM=7, AM=8, CF=9

    Returns: [role_GK, role_DEF, role_MID, role_FWD]
    """
    if role == 0:
        return [1.0, 0.0, 0.0, 0.0]  # GK
    elif role in [1, 2, 3]:
        return [0.0, 1.0, 0.0, 0.0]  # DEF (CB, LB, RB)
    elif role in [4, 5, 6, 7, 8]:
        return [0.0, 0.0, 1.0, 0.0]  # MID (DM, CM, LM, RM, AM)
    elif role == 9:
        return [0.0, 0.0, 0.0, 1.0]  # FWD (CF)
    else:
        return [0.0, 0.0, 0.0, 0.0]  # Unknown


def normalize_frame_for_attacking_team(frame, attacking_team):
    """
    Normalize frame so attacking team always attacks toward x=1.

    When flipping (attacking_team=1):
    - x coordinates: x -> -x
    - x velocities: vx -> -vx
    - Swap left_team <-> right_team (including roles and tired factors)
    """
    if attacking_team == 0:
        return frame

    normalized = {}

    # Flip ball
    ball_pos = list(frame['ball_position'])
    ball_pos[0] = -ball_pos[0]
    normalized['ball_position'] = ball_pos

    ball_vel = list(frame['ball_velocity'])
    ball_vel[0] = -ball_vel[0]
    normalized['ball_velocity'] = ball_vel

    # Swap and flip teams - positions
    right_pos = np.array(frame['right_team_positions'])
    left_pos = np.array(frame['left_team_positions'])
    right_pos[:, 0] = -right_pos[:, 0]
    left_pos[:, 0] = -left_pos[:, 0]

    normalized['left_team_positions'] = right_pos.tolist()
    normalized['right_team_positions'] = left_pos.tolist()

    # Swap and flip teams - velocities
    right_vel = np.array(frame['right_team_velocities'])
    left_vel = np.array(frame['left_team_velocities'])
    right_vel[:, 0] = -right_vel[:, 0]
    left_vel[:, 0] = -left_vel[:, 0]

    normalized['left_team_velocities'] = right_vel.tolist()
    normalized['right_team_velocities'] = left_vel.tolist()

    # Swap teams - roles (if present)
    if 'left_team_roles' in frame and 'right_team_roles' in frame:
        normalized['left_team_roles'] = frame['right_team_roles']
        normalized['right_team_roles'] = frame['left_team_roles']

    # Swap teams - tired factors (if present)
    if 'left_team_tired' in frame and 'right_team_tired' in frame:
        normalized['left_team_tired'] = frame['right_team_tired']
        normalized['right_team_tired'] = frame['left_team_tired']

    for key in frame:
        if key not in normalized:
            normalized[key] = frame[key]

    return normalized


def build_graph_data(frame, corner_taker_node=None):
    """
    Build graph representation from a single frame.

    Args:
        frame: Normalized frame dict with positions, velocities, roles, tired factors.
        corner_taker_node: Node index of the player taking the corner (for ball_possession).
                           If None, will attempt to infer from ball position proximity.

    Returns:
        node_features: (23, 13) array
        edge_index: (2, 506) array (fully connected, both directions)
        edge_attr: (506, 8) array with rich edge features
    """
    left_pos = np.array(frame['left_team_positions'])      # (11, 2)
    right_pos = np.array(frame['right_team_positions'])    # (11, 2)
    left_vel = np.array(frame['left_team_velocities'])     # (11, 2)
    right_vel = np.array(frame['right_team_velocities'])   # (11, 2)
    ball_pos = np.array(frame['ball_position'][:2])        # (2,)
    ball_vel = np.array(frame['ball_velocity'][:2])        # (2,)

    # Get roles and tired factors (with defaults for backward compatibility)
    left_roles = frame.get('left_team_roles', [0] + [5] * 10)  # Default: GK + all CM
    right_roles = frame.get('right_team_roles', [0] + [5] * 10)
    left_tired = frame.get('left_team_tired', [0.0] * 11)
    right_tired = frame.get('right_team_tired', [0.0] * 11)

    # Infer corner taker if not provided (player closest to ball in attacking team)
    if corner_taker_node is None:
        # Left team (attacking) takes the corner - find closest player to ball
        left_dists = np.linalg.norm(left_pos - ball_pos, axis=1)
        corner_taker_node = np.argmin(left_dists)

    # Build node features: 13 dims
    # [x, y, vx, vy, tired_factor, ball_possession, role_GK, role_DEF, role_MID, role_FWD,
    #  is_attacker, is_defender, is_ball]
    node_features = []

    # Left team (attacking): nodes 0-10
    for i in range(11):
        role_onehot = role_to_onehot(left_roles[i])
        ball_possession = 1.0 if i == corner_taker_node else 0.0
        features = [
            left_pos[i, 0], left_pos[i, 1],     # x, y
            left_vel[i, 0], left_vel[i, 1],     # vx, vy
            float(left_tired[i]),               # tired_factor
            ball_possession,                     # ball_possession (1 for corner taker)
            role_onehot[0], role_onehot[1],     # role_GK, role_DEF
            role_onehot[2], role_onehot[3],     # role_MID, role_FWD
            1.0,                                 # is_attacker
            0.0,                                 # is_defender
            0.0,                                 # is_ball
        ]
        node_features.append(features)

    # Right team (defending): nodes 11-21
    for i in range(11):
        role_onehot = role_to_onehot(right_roles[i])
        features = [
            right_pos[i, 0], right_pos[i, 1],   # x, y
            right_vel[i, 0], right_vel[i, 1],   # vx, vy
            float(right_tired[i]),              # tired_factor
            0.0,                                 # ball_possession (defender never takes corner)
            role_onehot[0], role_onehot[1],     # role_GK, role_DEF
            role_onehot[2], role_onehot[3],     # role_MID, role_FWD
            0.0,                                 # is_attacker
            1.0,                                 # is_defender
            0.0,                                 # is_ball
        ]
        node_features.append(features)

    # Ball: node 22
    ball_features = [
        ball_pos[0], ball_pos[1],               # x, y
        ball_vel[0], ball_vel[1],               # vx, vy
        0.0,                                     # tired_factor (N/A for ball)
        0.0,                                     # ball_possession (N/A)
        0.0, 0.0, 0.0, 0.0,                     # role one-hot (N/A)
        0.0,                                     # is_attacker
        0.0,                                     # is_defender
        1.0,                                     # is_ball
    ]
    node_features.append(ball_features)

    node_features = np.array(node_features, dtype=np.float32)

    # Build fully connected edge index (directed)
    num_nodes = 23
    edges_src = []
    edges_dst = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges_src.append(i)
                edges_dst.append(j)

    edge_index = np.array([edges_src, edges_dst], dtype=np.int64)

    # Build rich edge attributes using compute_edge_features
    edge_attr = compute_edge_features(node_features, edges_src, edges_dst)

    return node_features, edge_index, edge_attr


def process_episode(episode, use_frame='delivery'):
    """
    Process a single episode into graph data.

    Args:
        episode: Episode dict with frames and outcome
        use_frame: 'first', 'delivery', or int

    Returns:
        dict with node_features, edge_index, edge_attr, labels, or None if invalid
    """
    frames = episode.get('frames', [])
    if len(frames) == 0:
        return None

    attacking_team = episode.get('metadata', {}).get('attacking_team', 0)

    # Select frame
    if use_frame == 'first':
        frame_idx = 0
    elif use_frame == 'delivery':
        # Find ball delivery, then advance 5 frames for players to react
        frame_idx = 0
        for i, f in enumerate(frames):
            ball_vel = f.get('ball_velocity', [0, 0, 0])
            if np.linalg.norm(ball_vel[:2]) > 0.01:
                frame_idx = i
                break
    elif use_frame == 'post_delivery':
        # Use frame ~10 steps after delivery when all players are moving
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

    # Build graph with new features
    node_features, edge_index, edge_attr = build_graph_data(frame)

    # Get label
    outcome = episode.get('outcome', {})
    shot_label = 1 if outcome.get('shot_occurred', False) or outcome.get('goal_scored', False) else 0

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'shot_label': shot_label,
        'goal_label': 1 if outcome.get('goal_scored', False) else 0,
        'frame_idx': frame_idx,
    }


def compute_normalization_stats(all_graphs):
    """Compute mean and std for continuous features in nodes and edges.

    Node features: Only normalizes features 0-4: [x, y, vx, vy, tired_factor]
                   Features 5-12 are binary/one-hot indicators.

    Edge features: Only normalizes continuous features 0,1,2,4,5:
                   [rel_x, rel_y, distance, rel_vx, rel_vy]
                   Feature 3 (angle) is already bounded in radians.
                   Features 6,7 (team flags) are binary indicators.

    Returns:
        (node_mean, node_std, edge_mean, edge_std)
    """
    # Node feature stats
    all_node_features = np.vstack([g['node_features'] for g in all_graphs])
    node_mean = np.mean(all_node_features[:, :5], axis=0)
    node_std = np.std(all_node_features[:, :5], axis=0)
    node_std[node_std < 1e-6] = 1.0  # Avoid division by zero

    # Edge feature stats - only for continuous features (indices 0,1,2,4,5)
    # Stack all edge attributes
    if 'edge_attr' in all_graphs[0]:
        all_edge_features = np.vstack([g['edge_attr'] for g in all_graphs])
        # Indices for continuous edge features
        edge_continuous_idx = [0, 1, 2, 4, 5]  # rel_x, rel_y, dist, rel_vx, rel_vy
        edge_mean = np.mean(all_edge_features[:, edge_continuous_idx], axis=0)
        edge_std = np.std(all_edge_features[:, edge_continuous_idx], axis=0)
        edge_std[edge_std < 1e-6] = 1.0
    else:
        # Fallback for graphs without edge_attr
        edge_mean = np.zeros(5, dtype=np.float32)
        edge_std = np.ones(5, dtype=np.float32)

    return node_mean, node_std, edge_mean, edge_std


def normalize_features(graphs, node_mean, node_std, edge_mean=None, edge_std=None):
    """Apply normalization to continuous features in nodes and edges.

    Args:
        graphs: List of graph dicts with 'node_features' and 'edge_attr'.
        node_mean, node_std: Stats for node features (first 5 features).
        edge_mean, edge_std: Stats for edge features (indices 0,1,2,4,5).
                             If None, edge features are not normalized.
    """
    # Indices for continuous edge features
    edge_continuous_idx = [0, 1, 2, 4, 5]

    for g in graphs:
        # Normalize node features
        g['node_features'][:, :5] = (g['node_features'][:, :5] - node_mean) / node_std

        # Normalize edge features if stats provided
        if edge_mean is not None and edge_std is not None and 'edge_attr' in g:
            g['edge_attr'][:, edge_continuous_idx] = (
                (g['edge_attr'][:, edge_continuous_idx] - edge_mean) / edge_std
            )

    return graphs


def save_data(graphs, output_path, split_name, node_mean=None, node_std=None,
              edge_mean=None, edge_std=None):
    """Save processed data as .npz file."""
    node_features = np.array([g['node_features'] for g in graphs])
    edge_index = graphs[0]['edge_index']  # Same for all graphs
    # Edge attributes vary per graph now due to position-dependent features
    edge_attr = np.array([g['edge_attr'] for g in graphs])
    shot_labels = np.array([g['shot_label'] for g in graphs])
    goal_labels = np.array([g['goal_label'] for g in graphs])

    save_dict = {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'shot_labels': shot_labels,
        'goal_labels': goal_labels,
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
    parser = argparse.ArgumentParser(description='Prepare GNN data from corner kick episodes')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSON file with collected episodes')
    parser.add_argument('--output_dir', type=str, default='./grf/data/gnn',
                        help='Output directory for processed data')
    parser.add_argument('--use_frame', type=str, default='delivery',
                        help="Frame to use: 'first', 'delivery', or int")
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splits')

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse use_frame
    use_frame = args.use_frame
    if use_frame.isdigit():
        use_frame = int(use_frame)

    # Load episodes
    print(f"Loading episodes from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        episodes = json.load(f)
    print(f"Loaded {len(episodes)} episodes")

    # Process all episodes
    print("Processing episodes into graphs...")
    graphs = []
    for ep in tqdm(episodes, desc="Building graphs"):
        graph = process_episode(ep, use_frame=use_frame)
        if graph is not None:
            graphs.append(graph)

    print(f"Processed {len(graphs)} valid graphs")

    # Compute class balance
    shot_rate = np.mean([g['shot_label'] for g in graphs])
    goal_rate = np.mean([g['goal_label'] for g in graphs])
    print(f"Shot rate: {shot_rate:.2%}")
    print(f"Goal rate: {goal_rate:.2%}")

    # Split data
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

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_graphs)}")
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

    # Save metadata
    metadata = {
        'input_file': str(args.input_file),
        'use_frame': str(use_frame),
        'num_episodes': len(graphs),
        'train_size': len(train_graphs),
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
    }
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {output_path / 'metadata.json'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
