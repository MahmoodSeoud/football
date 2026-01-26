#!/usr/bin/env python3
"""
Extract features from collected GRF corner kick episodes.

Creates two feature sets:
1. Position-only features (replicates 7.5 ECTS StatsBomb approach)
2. Position + velocity features (tests temporal dynamics hypothesis)

Usage:
    python extract_features.py --input_file episodes.json --output_dir ./processed
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize_frame_for_attacking_team(frame, attacking_team):
    """
    Normalize frame coordinates so the attacking team is always attacking toward x=1.

    For 11v11 data, corners can be taken by either team:
    - Left team (attacking_team=0): attacks toward x=1 (no change needed)
    - Right team (attacking_team=1): attacks toward x=-1 (need to flip)

    When flipping:
    - x coordinates: x -> -x
    - x velocities: vx -> -vx
    - Swap left_team <-> right_team (so "attacking" team is always in left_team slots)
    """
    if attacking_team == 0:
        # Left team attacking toward x=1, no normalization needed
        return frame

    # Right team attacking toward x=-1, need to flip
    normalized = {}

    # Flip ball position (x only)
    ball_pos = frame['ball_position'].copy() if hasattr(frame['ball_position'], 'copy') else list(frame['ball_position'])
    ball_pos[0] = -ball_pos[0]
    normalized['ball_position'] = ball_pos

    # Flip ball velocity (x only)
    ball_vel = frame['ball_velocity'].copy() if hasattr(frame['ball_velocity'], 'copy') else list(frame['ball_velocity'])
    ball_vel[0] = -ball_vel[0]
    normalized['ball_velocity'] = ball_vel

    # Swap and flip team positions
    # Right team becomes "attacking" (left_team slot), left team becomes "defending"
    right_pos = np.array(frame['right_team_positions'])
    left_pos = np.array(frame['left_team_positions'])

    # Flip x coordinates
    right_pos[:, 0] = -right_pos[:, 0]
    left_pos[:, 0] = -left_pos[:, 0]

    # Swap: attacking team (right) goes to left_team, defending (left) goes to right_team
    normalized['left_team_positions'] = right_pos.tolist()
    normalized['right_team_positions'] = left_pos.tolist()

    # Swap and flip team velocities
    right_vel = np.array(frame['right_team_velocities'])
    left_vel = np.array(frame['left_team_velocities'])

    # Flip x velocities
    right_vel[:, 0] = -right_vel[:, 0]
    left_vel[:, 0] = -left_vel[:, 0]

    # Swap
    normalized['left_team_velocities'] = right_vel.tolist()
    normalized['right_team_velocities'] = left_vel.tolist()

    # Copy other fields unchanged
    for key in frame:
        if key not in normalized:
            normalized[key] = frame[key]

    return normalized


def compute_position_features(frame):
    """
    Compute position-only features from a single frame.
    Mirrors the StatsBomb aggregate features from 7.5 ECTS work.

    Features:
    - Zone counts (attackers/defenders in box, near goal, etc.)
    - Spatial density measures
    - Distances to goal
    - Numerical advantages
    """
    left_pos = np.array(frame['left_team_positions'])   # Attacking team
    right_pos = np.array(frame['right_team_positions']) # Defending team
    ball_pos = np.array(frame['ball_position'][:2])     # x, y only

    # GRF pitch is normalized: x in [-1, 1], y in [-0.42, 0.42]
    # Goal is at x = 1.0, y = 0.0
    goal_pos = np.array([1.0, 0.0])

    # Define zones (normalized coordinates)
    # Penalty box roughly: x > 0.8, |y| < 0.2
    penalty_x_thresh = 0.8
    penalty_y_thresh = 0.2

    # Six-yard box: x > 0.9, |y| < 0.1
    six_yard_x_thresh = 0.9
    six_yard_y_thresh = 0.1

    features = {}

    # --- Zone counts ---
    # Attackers (left team, excluding goalkeeper at index 0)
    attackers = left_pos[1:]  # Skip GK
    in_box_att = np.sum((attackers[:, 0] > penalty_x_thresh) &
                        (np.abs(attackers[:, 1]) < penalty_y_thresh))
    near_goal_att = np.sum((attackers[:, 0] > six_yard_x_thresh) &
                           (np.abs(attackers[:, 1]) < six_yard_y_thresh))

    features['attacking_in_box'] = int(in_box_att)
    features['attacking_near_goal'] = int(near_goal_att)
    features['total_attacking'] = len(attackers)

    # Defenders (right team, excluding goalkeeper at index 0)
    defenders = right_pos[1:]  # Skip GK
    in_box_def = np.sum((defenders[:, 0] > penalty_x_thresh) &
                        (np.abs(defenders[:, 1]) < penalty_y_thresh))
    near_goal_def = np.sum((defenders[:, 0] > six_yard_x_thresh) &
                           (np.abs(defenders[:, 1]) < six_yard_y_thresh))

    features['defending_in_box'] = int(in_box_def)
    features['defending_near_goal'] = int(near_goal_def)
    features['total_defending'] = len(defenders)

    # --- Numerical advantage ---
    features['numerical_advantage'] = int(in_box_att - in_box_def)
    features['attacker_defender_ratio'] = in_box_att / max(in_box_def, 1)

    # --- Distance features ---
    att_distances = np.linalg.norm(attackers - goal_pos, axis=1)
    def_distances = np.linalg.norm(defenders - goal_pos, axis=1)

    features['attacking_to_goal_dist_mean'] = float(np.mean(att_distances))
    features['attacking_to_goal_dist_min'] = float(np.min(att_distances))
    features['defending_to_goal_dist_mean'] = float(np.mean(def_distances))
    features['defending_to_goal_dist_min'] = float(np.min(def_distances))

    # Goalkeeper position
    gk_pos = right_pos[0]
    features['keeper_distance_to_goal'] = float(np.linalg.norm(gk_pos - goal_pos))
    features['keeper_x'] = float(gk_pos[0])
    features['keeper_y'] = float(gk_pos[1])

    # --- Density features ---
    # Variance of positions as proxy for spread/density
    features['attacking_density'] = float(1.0 / (np.var(att_distances) + 0.01))
    features['defending_density'] = float(1.0 / (np.var(def_distances) + 0.01))
    features['defending_depth'] = float(np.std(defenders[:, 1]))  # Y-spread

    # --- Ball position ---
    features['ball_x'] = float(ball_pos[0])
    features['ball_y'] = float(ball_pos[1])

    return features


def compute_velocity_features(frame):
    """
    Compute velocity features from a single frame.
    These are the ADDITIONAL features that position-only doesn't have.
    """
    left_vel = np.array(frame['left_team_velocities'])
    right_vel = np.array(frame['right_team_velocities'])
    ball_vel = np.array(frame['ball_velocity'][:2])

    features = {}

    # --- Speed magnitudes ---
    att_speeds = np.linalg.norm(left_vel[1:], axis=1)  # Skip GK
    def_speeds = np.linalg.norm(right_vel[1:], axis=1)

    features['attacking_speed_mean'] = float(np.mean(att_speeds))
    features['attacking_speed_max'] = float(np.max(att_speeds))
    features['attacking_speed_std'] = float(np.std(att_speeds))

    features['defending_speed_mean'] = float(np.mean(def_speeds))
    features['defending_speed_max'] = float(np.max(def_speeds))
    features['defending_speed_std'] = float(np.std(def_speeds))

    # --- Direction features (are players moving toward goal?) ---
    goal_direction = np.array([1.0, 0.0])  # Unit vector toward goal

    # Dot product with goal direction = component of velocity toward goal
    att_toward_goal = np.array([np.dot(v, goal_direction) for v in left_vel[1:]])
    def_toward_goal = np.array([np.dot(v, goal_direction) for v in right_vel[1:]])

    features['attacking_toward_goal_mean'] = float(np.mean(att_toward_goal))
    features['attacking_toward_goal_max'] = float(np.max(att_toward_goal))
    features['defending_toward_goal_mean'] = float(np.mean(def_toward_goal))

    # Count players moving toward goal
    features['attackers_moving_forward'] = int(np.sum(att_toward_goal > 0.01))
    features['defenders_moving_forward'] = int(np.sum(def_toward_goal > 0.01))

    # --- Ball velocity ---
    features['ball_speed'] = float(np.linalg.norm(ball_vel))
    features['ball_vx'] = float(ball_vel[0])
    features['ball_vy'] = float(ball_vel[1])

    # --- Goalkeeper movement ---
    gk_vel = right_vel[0]
    features['keeper_speed'] = float(np.linalg.norm(gk_vel))
    features['keeper_moving_toward_ball'] = float(np.dot(gk_vel, ball_vel) > 0)

    # --- Relative velocities (closing speed) ---
    # Are attackers closing in on defenders?
    features['speed_differential'] = float(np.mean(att_speeds) - np.mean(def_speeds))

    return features


def extract_episode_features(episode, use_frame='first'):
    """
    Extract features from an episode.

    Args:
        episode: Episode dict with frames and outcome
        use_frame: 'first' (t=0), 'delivery' (when ball moves), or int index

    Returns:
        dict with position_features, velocity_features, and label
    """
    frames = episode['frames']

    if len(frames) == 0:
        return None

    # Get attacking team from metadata (for 11v11 data)
    # Default to 0 (left team) for academy_corner data which doesn't have this field
    attacking_team = episode.get('metadata', {}).get('attacking_team', 0)

    # Select which frame to use
    if use_frame == 'first':
        frame_idx = 0
    elif use_frame == 'delivery':
        # Find first frame where ball is moving
        for i, f in enumerate(frames):
            if f.get('ball_velocity') and np.linalg.norm(f['ball_velocity'][:2]) > 0.01:
                frame_idx = i
                break
        else:
            frame_idx = 0
    else:
        frame_idx = min(int(use_frame), len(frames) - 1)

    frame = frames[frame_idx]

    # Normalize coordinates so attacking team always attacks toward x=1
    frame = normalize_frame_for_attacking_team(frame, attacking_team)

    # Extract features
    pos_features = compute_position_features(frame)
    vel_features = compute_velocity_features(frame)

    # Determine label (binary: shot occurred or not)
    outcome = episode['outcome']
    shot_label = 1 if outcome.get('shot_occurred', False) or outcome.get('goal_scored', False) else 0
    goal_label = 1 if outcome.get('goal_scored', False) else 0

    return {
        'position_features': pos_features,
        'velocity_features': vel_features,
        'shot_label': shot_label,
        'goal_label': goal_label,
        'outcome_type': outcome.get('outcome', 'unknown'),
        'frame_used': frame_idx,
    }


def process_all_episodes(episodes, output_dir, use_frame='delivery'):
    """
    Process all episodes and create feature datasets.

    Args:
        episodes: List of episode dicts
        output_dir: Directory to save output files
        use_frame: Which frame to extract features from:
            - 'first': Initial setup (t=0), velocities will be ~0
            - 'delivery': When ball starts moving (meaningful velocities)
            - int: Specific frame index
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_data = []
    frame_indices = []

    for i, episode in enumerate(tqdm(episodes, desc="Extracting features")):
        features = extract_episode_features(episode, use_frame=use_frame)
        if features is None:
            continue

        features['episode_id'] = i
        all_data.append(features)
        frame_indices.append(features['frame_used'])

    if len(all_data) == 0:
        print("No valid episodes found!")
        return None, None

    # Create DataFrames
    position_cols = list(all_data[0]['position_features'].keys())
    velocity_cols = list(all_data[0]['velocity_features'].keys())

    # Position-only dataset
    pos_data = []
    for d in all_data:
        row = {'episode_id': d['episode_id'], 'shot_label': d['shot_label'], 'goal_label': d['goal_label']}
        row.update(d['position_features'])
        pos_data.append(row)

    df_position = pd.DataFrame(pos_data)

    # Position + velocity dataset
    full_data = []
    for d in all_data:
        row = {'episode_id': d['episode_id'], 'shot_label': d['shot_label'], 'goal_label': d['goal_label']}
        row.update(d['position_features'])
        row.update(d['velocity_features'])
        full_data.append(row)

    df_full = pd.DataFrame(full_data)

    # Save datasets
    df_position.to_csv(output_path / 'features_position_only.csv', index=False)
    df_full.to_csv(output_path / 'features_position_velocity.csv', index=False)

    # Save feature lists for reference
    feature_info = {
        'position_features': position_cols,
        'velocity_features': velocity_cols,
        'num_episodes': len(all_data),
        'shot_rate': float(df_position['shot_label'].mean()),
        'goal_rate': float(df_position['goal_label'].mean()),
        'use_frame': use_frame,
        'avg_frame_index': float(np.mean(frame_indices)),
    }
    with open(output_path / 'feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)

    print(f"\nFeature extraction complete!")
    print(f"  Episodes processed: {len(all_data)}")
    print(f"  Position features: {len(position_cols)}")
    print(f"  Velocity features: {len(velocity_cols)}")
    print(f"  Frame selection: '{use_frame}' (avg index: {np.mean(frame_indices):.1f})")
    print(f"  Shot rate: {feature_info['shot_rate']:.2%}")
    print(f"  Goal rate: {feature_info['goal_rate']:.2%}")

    return df_position, df_full


def main():
    parser = argparse.ArgumentParser(description='Extract features from GRF episodes')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSON file with collected episodes')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help='Output directory for feature CSVs')
    parser.add_argument('--use_frame', type=str, default='delivery',
                        help="Frame to extract features from: 'first' (t=0), 'delivery' (ball moving), or int")

    args = parser.parse_args()

    # Parse use_frame argument
    use_frame = args.use_frame
    if use_frame.isdigit():
        use_frame = int(use_frame)

    print(f"Loading episodes from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        episodes = json.load(f)

    print(f"Loaded {len(episodes)} episodes")
    print(f"Using frame: {use_frame}")

    process_all_episodes(episodes, args.output_dir, use_frame=use_frame)


if __name__ == '__main__':
    main()
