# Corner Kick Prediction with Google Research Football
## Implementation Plan for 15 ECTS Thesis Extension

**Author:** Mahmood Mohammed Seoud  
**Supervisor:** Stella Grasshof  
**Institution:** IT University of Copenhagen

---

## Executive Summary

**Objective:** Beat the AUC ≈ 0.50 (random chance) baseline established in the 7.5 ECTS work by demonstrating that velocity/temporal features enable corner kick outcome prediction.

**Approach:** Use Google Research Football (GRF) simulator to generate synthetic corner kick data with full tracking (positions + velocities), then show that models with velocity features outperform position-only models.

**Success Criteria:** 
- Model with velocity features achieves AUC > 0.55 (statistically significant improvement over 0.50)
- Position-only model on same GRF data achieves AUC ≈ 0.50 (confirming 7.5 ECTS finding transfers to synthetic domain)

---

## Project Structure

```
CornerTactics/
├── grf/
│   ├── docker/
│   │   └── Dockerfile              # Custom GRF Docker image
│   ├── scripts/
│   │   ├── collect_corners.py      # Main data collection script
│   │   ├── extract_features.py     # Feature engineering
│   │   └── utils.py                # Helper functions
│   ├── slurm/
│   │   ├── build_container.sh      # One-time container build
│   │   ├── collect_data.slurm      # Data collection job
│   │   └── train_model.slurm       # Model training job
│   └── data/
│       ├── raw/                    # Raw episode data
│       ├── processed/              # Extracted features
│       └── splits/                 # Train/val/test splits
├── models/
│   ├── baseline_position.py        # Position-only model (replicates 7.5 ECTS)
│   ├── velocity_model.py           # Position + velocity model
│   └── evaluation.py               # Metrics and comparison
├── notebooks/
│   └── analysis.ipynb              # Results visualization
└── requirements.txt
```

---

## Phase 1: Environment Setup (Local Machine)

### Task 1.1: Create Custom GRF Dockerfile

GRF's official Docker image uses TensorFlow 1.15 (EOL). Create a minimal image for data collection only.

**File:** `grf/docker/Dockerfile`

```dockerfile
FROM ubuntu:20.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libgl1-mesa-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-ttf-dev \
    libsdl2-gfx-dev \
    libboost-all-dev \
    libdirectfb-dev \
    libst-dev \
    mesa-utils \
    xvfb \
    x11vnc \
    python3.8 \
    python3-pip \
    python3.8-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install GRF and dependencies
RUN pip install gfootball \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    tqdm \
    h5py

# Create working directory
WORKDIR /app

# Set display for headless rendering
ENV DISPLAY=:99

# Default command: start Xvfb and run Python
ENTRYPOINT ["bash", "-c", "Xvfb :99 -screen 0 1024x768x24 &> /dev/null & exec \"$@\"", "--"]
CMD ["python"]
```

### Task 1.2: Build and Export Docker Image (Local Machine)

```bash
# Navigate to docker directory
cd grf/docker

# Build the image
docker build -t grf-corner-collector:latest .

# Verify it works locally
docker run --rm grf-corner-collector:latest python -c "import gfootball; print('GRF OK')"

# List images to find the IMAGE_ID
docker images

# Example output:
# REPOSITORY              TAG       IMAGE ID       CREATED         SIZE
# grf-corner-collector    latest    abc123def456   2 minutes ago   3.2GB

# Save to tar file using the IMAGE_ID
sudo docker save abc123def456 -o grf-corner-collector.tar

# Check file size (should be ~2-4GB)
ls -lh grf-corner-collector.tar
```

### Task 1.3: Transfer to HPC

```bash
# Transfer tar file to HPC (replace with your HPC details)
scp grf-corner-collector.tar username@hpc.itu.dk:/home/username/containers/

# Or use rsync for resumable transfer (better for large files)
rsync -avP grf-corner-collector.tar username@hpc.itu.dk:/home/username/containers/
```

---

## Phase 2: HPC Container Setup

### Task 2.1: Build Apptainer Image on HPC

**IMPORTANT:** Run this interactively on the login node, NOT as a SLURM job.

**File:** `grf/slurm/build_container.sh`

```bash
#!/bin/bash
# Run this interactively on HPC login node (not as SLURM job)

# Create containers directory if it doesn't exist
mkdir -p /home/$USER/containers

# Navigate to containers directory
cd /home/$USER/containers

# Build Apptainer .sif from Docker archive
# NOTE: Use docker-archive:// prefix to tell Apptainer it's a Docker image
# This may take 10-20 minutes
apptainer build grf-corner-collector.sif docker-archive://grf-corner-collector.tar

# Verify the container works
apptainer exec grf-corner-collector.sif python -c "import gfootball; print('GRF loaded successfully')"

# Test with Xvfb (headless display)
apptainer exec grf-corner-collector.sif bash -c "Xvfb :99 -screen 0 1024x768x24 &> /dev/null & DISPLAY=:99 python -c \"import gfootball.env as football_env; env = football_env.create_environment(env_name='academy_corner', render=False); print('Environment created'); env.close()\""

echo "Container ready at: /home/$USER/containers/grf-corner-collector.sif"
```

### Task 2.2: Verify GRF Corner Scenarios

Before mass data collection, verify what scenarios are available.

```bash
# Interactive test on HPC
apptainer exec grf-corner-collector.sif python -c "
import gfootball.env as football_env

# List available scenarios
scenarios = [
    'academy_corner',
    '11_vs_11_competition', 
    '11_vs_11_stochastic',
]

for scenario in scenarios:
    try:
        env = football_env.create_environment(
            env_name=scenario,
            representation='raw',
            render=False
        )
        obs = env.reset()
        print(f'{scenario}: OK - obs shape: {type(obs)}')
        env.close()
    except Exception as e:
        print(f'{scenario}: FAILED - {e}')
"
```

---

## Phase 3: Data Collection

### Task 3.1: Corner Kick Data Collector Script

**File:** `grf/scripts/collect_corners.py`

```python
#!/usr/bin/env python3
"""
Collect corner kick episodes from Google Research Football.

Each episode captures:
- Player positions (x, y) for all 22 players at each timestep
- Player velocities (vx, vy) derived from position changes
- Ball position and velocity
- Game state (score, time, etc.)
- Outcome label (goal, shot, clearance, etc.)

Usage:
    python collect_corners.py --num_episodes 1000 --output_dir /path/to/data
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import gfootball.env as football_env
import numpy as np
from tqdm import tqdm


def create_corner_env(render=False):
    """Create a GRF environment for corner kick scenarios."""
    env = football_env.create_environment(
        env_name='academy_corner',
        representation='raw',  # Get full game state
        rewards='scoring,checkpoints',
        render=render,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        write_video=False,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
    )
    return env


def extract_frame_data(obs):
    """
    Extract structured data from a single observation frame.
    
    GRF 'raw' observation contains:
    - ball: [x, y, z] position
    - ball_direction: [vx, vy, vz] velocity
    - left_team: [[x, y], ...] for 11 players
    - left_team_direction: [[vx, vy], ...] velocities
    - right_team: [[x, y], ...] for 11 players  
    - right_team_direction: [[vx, vy], ...] velocities
    - left_team_active: which player is controlled
    - game_mode: current game state (corner, penalty, etc.)
    - score: [left_score, right_score]
    """
    # Handle observation format (may be nested)
    if isinstance(obs, list):
        obs = obs[0]
    
    frame = {
        'ball_position': obs['ball'].tolist() if hasattr(obs['ball'], 'tolist') else list(obs['ball']),
        'ball_velocity': obs['ball_direction'].tolist() if hasattr(obs['ball_direction'], 'tolist') else list(obs['ball_direction']),
        'left_team_positions': obs['left_team'].tolist() if hasattr(obs['left_team'], 'tolist') else [list(p) for p in obs['left_team']],
        'left_team_velocities': obs['left_team_direction'].tolist() if hasattr(obs['left_team_direction'], 'tolist') else [list(v) for v in obs['left_team_direction']],
        'right_team_positions': obs['right_team'].tolist() if hasattr(obs['right_team'], 'tolist') else [list(p) for p in obs['right_team']],
        'right_team_velocities': obs['right_team_direction'].tolist() if hasattr(obs['right_team_direction'], 'tolist') else [list(v) for v in obs['right_team_direction']],
        'game_mode': int(obs['game_mode']),
        'score': list(obs['score']),
    }
    
    # Add ball ownership if available
    if 'ball_owned_team' in obs:
        frame['ball_owned_team'] = int(obs['ball_owned_team'])
    if 'ball_owned_player' in obs:
        frame['ball_owned_player'] = int(obs['ball_owned_player'])
    
    return frame


def determine_outcome(episode_frames, final_reward):
    """
    Determine the outcome of a corner kick episode.
    
    Returns:
        dict with:
            - outcome: 'goal', 'shot_saved', 'shot_missed', 'clearance', 'possession_lost', 'timeout'
            - shot_occurred: bool
            - goal_scored: bool
    """
    outcome = {
        'shot_occurred': False,
        'goal_scored': False,
        'outcome': 'unknown'
    }
    
    if final_reward > 0:
        outcome['goal_scored'] = True
        outcome['shot_occurred'] = True
        outcome['outcome'] = 'goal'
    elif final_reward < 0:
        # Opponent scored (shouldn't happen in corner scenario)
        outcome['outcome'] = 'opponent_goal'
    else:
        # Check game mode changes to infer outcome
        game_modes = [f['game_mode'] for f in episode_frames]
        
        # Game modes in GRF:
        # 0: Normal, 1: KickOff, 2: GoalKick, 3: FreeKick, 
        # 4: Corner, 5: ThrowIn, 6: Penalty
        
        if 2 in game_modes[1:]:  # GoalKick after corner = shot missed or saved
            outcome['shot_occurred'] = True
            outcome['outcome'] = 'shot_missed'
        elif 5 in game_modes[1:]:  # ThrowIn = ball went out
            outcome['outcome'] = 'out_of_play'
        elif 3 in game_modes[1:]:  # FreeKick = foul
            outcome['outcome'] = 'foul'
        else:
            # Ball was cleared or possession changed
            # Check if ball ownership changed
            final_frame = episode_frames[-1]
            if final_frame.get('ball_owned_team', 0) == 1:  # Right team (defending)
                outcome['outcome'] = 'clearance'
            else:
                outcome['outcome'] = 'possession_retained'
    
    return outcome


def collect_episode(env, max_steps=100):
    """
    Collect a single corner kick episode.
    
    Returns:
        dict containing all frames and metadata
    """
    obs = env.reset()
    
    episode = {
        'frames': [],
        'actions': [],
        'rewards': [],
        'metadata': {
            'start_time': datetime.now().isoformat(),
            'max_steps': max_steps,
        }
    }
    
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Extract current frame data
        frame_data = extract_frame_data(obs)
        frame_data['step'] = step
        episode['frames'].append(frame_data)
        
        # Take random action (we're collecting data, not training an agent)
        # Action space: 0-18 for different movements and kicks
        action = env.action_space.sample()
        
        obs, reward, done, info = env.step(action)
        
        episode['actions'].append(int(action))
        episode['rewards'].append(float(reward))
        total_reward += reward
        step += 1
    
    # Determine outcome
    episode['outcome'] = determine_outcome(episode['frames'], total_reward)
    episode['metadata']['total_steps'] = step
    episode['metadata']['total_reward'] = total_reward
    episode['metadata']['end_time'] = datetime.now().isoformat()
    
    return episode


def collect_corners(num_episodes, output_dir, save_every=100):
    """
    Collect multiple corner kick episodes.
    
    Args:
        num_episodes: Number of episodes to collect
        output_dir: Directory to save data
        save_every: Save checkpoint every N episodes
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    env = create_corner_env(render=False)
    
    all_episodes = []
    outcome_counts = {'goal': 0, 'shot_missed': 0, 'clearance': 0, 'other': 0}
    
    try:
        for i in tqdm(range(num_episodes), desc="Collecting episodes"):
            episode = collect_episode(env, max_steps=150)
            all_episodes.append(episode)
            
            # Track outcome distribution
            outcome = episode['outcome']['outcome']
            if outcome in outcome_counts:
                outcome_counts[outcome] += 1
            else:
                outcome_counts['other'] += 1
            
            # Periodic checkpoint
            if (i + 1) % save_every == 0:
                checkpoint_file = output_path / f"episodes_checkpoint_{i+1}.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(all_episodes[-save_every:], f)
                print(f"\nCheckpoint saved: {checkpoint_file}")
                print(f"Outcome distribution so far: {outcome_counts}")
    
    finally:
        env.close()
    
    # Save final dataset
    final_file = output_path / f"corner_episodes_n{num_episodes}.json"
    with open(final_file, 'w') as f:
        json.dump(all_episodes, f)
    
    # Save metadata
    meta_file = output_path / "collection_metadata.json"
    metadata = {
        'num_episodes': num_episodes,
        'outcome_distribution': outcome_counts,
        'collection_date': datetime.now().isoformat(),
    }
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nCollection complete!")
    print(f"Total episodes: {num_episodes}")
    print(f"Outcome distribution: {outcome_counts}")
    print(f"Data saved to: {final_file}")
    
    return all_episodes


def main():
    parser = argparse.ArgumentParser(description='Collect corner kick data from GRF')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Number of episodes to collect')
    parser.add_argument('--output_dir', type=str, default='./data/raw',
                        help='Output directory for collected data')
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save checkpoint every N episodes')
    
    args = parser.parse_args()
    
    print(f"Starting corner kick data collection...")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Output: {args.output_dir}")
    
    collect_corners(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        save_every=args.save_every
    )


if __name__ == '__main__':
    main()
```

### Task 3.2: Feature Extraction Script

**File:** `grf/scripts/extract_features.py`

```python
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


def process_all_episodes(episodes, output_dir):
    """
    Process all episodes and create feature datasets.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    for i, episode in enumerate(tqdm(episodes, desc="Extracting features")):
        features = extract_episode_features(episode, use_frame='first')
        if features is None:
            continue
        
        features['episode_id'] = i
        all_data.append(features)
    
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
        'shot_rate': df_position['shot_label'].mean(),
        'goal_rate': df_position['goal_label'].mean(),
    }
    with open(output_path / 'feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"\nFeature extraction complete!")
    print(f"  Episodes processed: {len(all_data)}")
    print(f"  Position features: {len(position_cols)}")
    print(f"  Velocity features: {len(velocity_cols)}")
    print(f"  Shot rate: {feature_info['shot_rate']:.2%}")
    print(f"  Goal rate: {feature_info['goal_rate']:.2%}")
    
    return df_position, df_full


def main():
    parser = argparse.ArgumentParser(description='Extract features from GRF episodes')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSON file with collected episodes')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help='Output directory for feature CSVs')
    
    args = parser.parse_args()
    
    print(f"Loading episodes from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        episodes = json.load(f)
    
    print(f"Loaded {len(episodes)} episodes")
    
    process_all_episodes(episodes, args.output_dir)


if __name__ == '__main__':
    main()
```

### Task 3.3: SLURM Job Script for Data Collection

**File:** `grf/slurm/collect_data.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=grf-corner-collect
#SBATCH --output=logs/grf-collect-%j.out
#SBATCH --error=logs/grf-collect-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=brown

# Create log directory if needed
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

# Set paths
CONTAINER_PATH="/home/$USER/containers/grf-corner-collector.sif"
SCRIPT_PATH="/home/$USER/CornerTactics/grf/scripts/collect_corners.py"
OUTPUT_DIR="/home/$USER/CornerTactics/grf/data/raw"

# Number of episodes to collect
NUM_EPISODES=10000

# Run data collection inside container
# -B or --bind mounts the project directory inside the container
apptainer exec \
    -B /home/$USER/CornerTactics:/app \
    $CONTAINER_PATH \
    bash -c "
        # Start virtual display
        Xvfb :99 -screen 0 1024x768x24 &> /dev/null &
        export DISPLAY=:99
        sleep 2
        
        # Run collection script
        cd /app
        python grf/scripts/collect_corners.py \
            --num_episodes $NUM_EPISODES \
            --output_dir $OUTPUT_DIR \
            --save_every 500
    "

echo "End time: $(date)"
echo "Data collection complete!"
```

---

## Phase 4: Model Training

### Task 4.1: Baseline Position-Only Model

**File:** `models/baseline_position.py`

```python
#!/usr/bin/env python3
"""
Position-only baseline model.
Replicates the 7.5 ECTS approach on GRF synthetic data.
Expected result: AUC ≈ 0.50 (random chance)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def load_data(filepath):
    """Load feature CSV and split into X, y."""
    df = pd.read_csv(filepath)
    
    # Separate features and labels
    label_cols = ['episode_id', 'shot_label', 'goal_label']
    feature_cols = [c for c in df.columns if c not in label_cols]
    
    X = df[feature_cols].values
    y = df['shot_label'].values
    
    return X, y, feature_cols


def train_and_evaluate(X, y, feature_names, model_name='XGBoost'):
    """Train model and evaluate performance."""
    
    # Train/test split (match-based would be ideal, but episodes are independent in GRF)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model
    if model_name == 'XGBoost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1),
            random_state=42,
            eval_metric='auc'
        )
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
    elif model_name == 'LogisticRegression':
        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    results = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'train_size': len(y_train),
        'test_size': len(y_test),
        'positive_rate_train': y_train.mean(),
        'positive_rate_test': y_test.mean(),
    }
    
    # Cross-validation AUC
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    results['cv_auc_mean'] = cv_scores.mean()
    results['cv_auc_std'] = cv_scores.std()
    
    # Feature importance (for tree models)
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(feature_names, model.feature_importances_))
        results['feature_importance'] = dict(sorted(importance.items(), key=lambda x: -x[1])[:10])
    
    return results, model


def permutation_test(X, y, n_permutations=100):
    """
    Permutation test to verify no signal in data.
    Train on shuffled labels and compare to real labels.
    """
    from sklearn.model_selection import cross_val_score
    
    # Real label performance
    model = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42, eval_metric='auc')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    real_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
    real_auc = real_scores.mean()
    
    # Shuffled label performances
    shuffled_aucs = []
    for i in range(n_permutations):
        y_shuffled = np.random.permutation(y)
        scores = cross_val_score(model, X_scaled, y_shuffled, cv=5, scoring='roc_auc')
        shuffled_aucs.append(scores.mean())
    
    shuffled_aucs = np.array(shuffled_aucs)
    
    # Statistical test
    p_value = np.mean(shuffled_aucs >= real_auc)
    z_score = (real_auc - shuffled_aucs.mean()) / shuffled_aucs.std()
    
    return {
        'real_auc': real_auc,
        'shuffled_auc_mean': shuffled_aucs.mean(),
        'shuffled_auc_std': shuffled_aucs.std(),
        'p_value': p_value,
        'z_score': z_score,
        'significant': p_value < 0.05
    }


def main():
    parser = argparse.ArgumentParser(description='Train position-only baseline')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to features_position_only.csv')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--permutation_test', action='store_true',
                        help='Run permutation test')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    X, y, feature_names = load_data(args.data_file)
    print(f"  Samples: {len(y)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Positive rate: {y.mean():.2%}")
    
    # Train models
    all_results = {}
    for model_name in ['XGBoost', 'RandomForest', 'LogisticRegression']:
        print(f"\nTraining {model_name}...")
        results, model = train_and_evaluate(X, y, feature_names, model_name)
        all_results[model_name] = results
        
        print(f"  AUC: {results['auc']:.4f}")
        print(f"  F1:  {results['f1']:.4f}")
        print(f"  CV AUC: {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
    
    # Permutation test
    if args.permutation_test:
        print("\nRunning permutation test (this may take a few minutes)...")
        perm_results = permutation_test(X, y, n_permutations=100)
        all_results['permutation_test'] = perm_results
        
        print(f"  Real AUC: {perm_results['real_auc']:.4f}")
        print(f"  Shuffled AUC: {perm_results['shuffled_auc_mean']:.4f} ± {perm_results['shuffled_auc_std']:.4f}")
        print(f"  p-value: {perm_results['p_value']:.4f}")
        print(f"  Significant: {perm_results['significant']}")
    
    # Save results
    results_file = output_path / 'baseline_position_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
```

### Task 4.2: Velocity-Enhanced Model

**File:** `models/velocity_model.py`

```python
#!/usr/bin/env python3
"""
Position + Velocity model.
Tests whether velocity features improve prediction beyond position-only baseline.
Expected result: AUC > 0.55 (significant improvement)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def load_data(filepath):
    """Load feature CSV."""
    df = pd.read_csv(filepath)
    label_cols = ['episode_id', 'shot_label', 'goal_label']
    feature_cols = [c for c in df.columns if c not in label_cols]
    
    X = df[feature_cols].values
    y = df['shot_label'].values
    
    return X, y, feature_cols, df


def ablation_study(X_full, y, feature_names, position_feature_count):
    """
    Compare position-only vs position+velocity performance.
    This is the KEY experiment for the thesis.
    """
    # Split features
    X_position = X_full[:, :position_feature_count]
    X_velocity = X_full[:, position_feature_count:]
    
    position_features = feature_names[:position_feature_count]
    velocity_features = feature_names[position_feature_count:]
    
    # Train/test split (same split for fair comparison)
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_pos = X_train_full[:, :position_feature_count]
    X_test_pos = X_test_full[:, :position_feature_count]
    
    # Standardize
    scaler_full = StandardScaler()
    scaler_pos = StandardScaler()
    
    X_train_full_scaled = scaler_full.fit_transform(X_train_full)
    X_test_full_scaled = scaler_full.transform(X_test_full)
    
    X_train_pos_scaled = scaler_pos.fit_transform(X_train_pos)
    X_test_pos_scaled = scaler_pos.transform(X_test_pos)
    
    results = {}
    
    # Train position-only model
    print("\n--- Position-Only Model ---")
    model_pos = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1),
        random_state=42, eval_metric='auc'
    )
    model_pos.fit(X_train_pos_scaled, y_train)
    
    y_pred_pos = model_pos.predict_proba(X_test_pos_scaled)[:, 1]
    auc_pos = roc_auc_score(y_test, y_pred_pos)
    
    cv_pos = cross_val_score(model_pos, X_train_pos_scaled, y_train, cv=5, scoring='roc_auc')
    
    results['position_only'] = {
        'auc': auc_pos,
        'cv_auc_mean': cv_pos.mean(),
        'cv_auc_std': cv_pos.std(),
        'num_features': len(position_features),
    }
    print(f"  AUC: {auc_pos:.4f}")
    print(f"  CV AUC: {cv_pos.mean():.4f} ± {cv_pos.std():.4f}")
    
    # Train position + velocity model
    print("\n--- Position + Velocity Model ---")
    model_full = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1),
        random_state=42, eval_metric='auc'
    )
    model_full.fit(X_train_full_scaled, y_train)
    
    y_pred_full = model_full.predict_proba(X_test_full_scaled)[:, 1]
    auc_full = roc_auc_score(y_test, y_pred_full)
    
    cv_full = cross_val_score(model_full, X_train_full_scaled, y_train, cv=5, scoring='roc_auc')
    
    results['position_velocity'] = {
        'auc': auc_full,
        'cv_auc_mean': cv_full.mean(),
        'cv_auc_std': cv_full.std(),
        'num_features': len(feature_names),
    }
    print(f"  AUC: {auc_full:.4f}")
    print(f"  CV AUC: {cv_full.mean():.4f} ± {cv_full.std():.4f}")
    
    # Statistical comparison
    print("\n--- Comparison ---")
    auc_improvement = auc_full - auc_pos
    relative_improvement = (auc_full - auc_pos) / (auc_pos - 0.5 + 1e-6) * 100 if auc_pos > 0.5 else float('inf')
    
    results['comparison'] = {
        'auc_improvement': auc_improvement,
        'relative_improvement_pct': relative_improvement,
        'velocity_helps': auc_full > auc_pos + 0.02,  # >2% improvement threshold
    }
    
    print(f"  AUC Improvement: {auc_improvement:+.4f}")
    print(f"  Velocity helps: {results['comparison']['velocity_helps']}")
    
    # Feature importance for velocity features
    importance = dict(zip(feature_names, model_full.feature_importances_))
    velocity_importance = {k: v for k, v in importance.items() if k in velocity_features}
    results['velocity_feature_importance'] = dict(sorted(velocity_importance.items(), key=lambda x: -x[1])[:10])
    
    print(f"\n  Top velocity features:")
    for feat, imp in list(results['velocity_feature_importance'].items())[:5]:
        print(f"    {feat}: {imp:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train position+velocity model')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to features_position_velocity.csv')
    parser.add_argument('--feature_info', type=str, required=True,
                        help='Path to feature_info.json')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load feature info
    with open(args.feature_info, 'r') as f:
        feature_info = json.load(f)
    
    position_feature_count = len(feature_info['position_features'])
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    X, y, feature_names, df = load_data(args.data_file)
    print(f"  Samples: {len(y)}")
    print(f"  Position features: {position_feature_count}")
    print(f"  Velocity features: {len(feature_names) - position_feature_count}")
    print(f"  Positive rate: {y.mean():.2%}")
    
    # Run ablation study
    results = ablation_study(X, y, feature_names, position_feature_count)
    
    # Save results
    results_file = output_path / 'velocity_ablation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")
    
    # Print thesis-relevant conclusion
    print("\n" + "="*60)
    print("THESIS CONCLUSION")
    print("="*60)
    
    if results['comparison']['velocity_helps']:
        print("✓ Velocity features IMPROVE corner kick prediction")
        print(f"  Position-only AUC: {results['position_only']['auc']:.4f}")
        print(f"  Position+Velocity AUC: {results['position_velocity']['auc']:.4f}")
        print(f"  Improvement: {results['comparison']['auc_improvement']:+.4f}")
        print("\nThis explains why the 7.5 ECTS baseline failed:")
        print("  Static positioning lacks the temporal dynamics needed for prediction.")
        print("  TacticAI's 25Hz tracking provides velocity information that enables prediction.")
    else:
        print("✗ Velocity features did NOT significantly improve prediction")
        print("  Further investigation needed...")


if __name__ == '__main__':
    main()
```

### Task 4.3: SLURM Job Script for Training

**File:** `grf/slurm/train_model.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=grf-corner-train
#SBATCH --output=logs/grf-train-%j.out
#SBATCH --error=logs/grf-train-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=brown

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

# Paths
CONTAINER_PATH="/home/$USER/containers/grf-corner-collector.sif"
PROJECT_DIR="/home/$USER/CornerTactics"
DATA_DIR="$PROJECT_DIR/grf/data"
RESULTS_DIR="$PROJECT_DIR/results"

mkdir -p $RESULTS_DIR

# Step 1: Extract features from raw data
echo "=== Step 1: Feature Extraction ==="
apptainer exec \
    -B $PROJECT_DIR:/app \
    $CONTAINER_PATH \
    python /app/grf/scripts/extract_features.py \
        --input_file /app/grf/data/raw/corner_episodes_n10000.json \
        --output_dir /app/grf/data/processed

# Step 2: Train position-only baseline
echo "=== Step 2: Position-Only Baseline ==="
apptainer exec \
    -B $PROJECT_DIR:/app \
    $CONTAINER_PATH \
    python /app/models/baseline_position.py \
        --data_file /app/grf/data/processed/features_position_only.csv \
        --output_dir /app/results \
        --permutation_test

# Step 3: Train position+velocity model and compare
echo "=== Step 3: Position+Velocity Ablation ==="
apptainer exec \
    -B $PROJECT_DIR:/app \
    $CONTAINER_PATH \
    python /app/models/velocity_model.py \
        --data_file /app/grf/data/processed/features_position_velocity.csv \
        --feature_info /app/grf/data/processed/feature_info.json \
        --output_dir /app/results

echo "End time: $(date)"
echo "Training complete! Check results in $RESULTS_DIR"
```

---

## Phase 5: Execution Checklist

### On Local Machine:

```bash
# 1. Create project structure
mkdir -p CornerTactics/grf/{docker,scripts,slurm,data/{raw,processed}}
mkdir -p CornerTactics/{models,results,notebooks}

# 2. Copy all files above to appropriate locations
# (Create Dockerfile, collect_corners.py, extract_features.py, etc.)

# 3. Build Docker image
cd CornerTactics/grf/docker
docker build -t grf-corner-collector:latest .

# 4. Test locally (optional)
docker run --rm grf-corner-collector:latest python -c "import gfootball; print('OK')"

# 5. Get image ID and save
docker images
# Look for grf-corner-collector, note the IMAGE ID (e.g., abc123def456)

sudo docker save abc123def456 -o grf-corner-collector.tar

# 6. Transfer to HPC
scp grf-corner-collector.tar username@hpc.itu.dk:/home/username/containers/
scp -r CornerTactics username@hpc.itu.dk:/home/username/
```

### On HPC:

```bash
# 1. Build Apptainer image (run interactively, NOT via SLURM)
cd /home/$USER/containers

# Build from docker archive (use docker-archive:// prefix!)
apptainer build grf-corner-collector.sif docker-archive://grf-corner-collector.tar

# 2. Verify container works
apptainer exec grf-corner-collector.sif python -c "import gfootball; print('GRF OK')"

# 3. Submit data collection job
cd /home/$USER/CornerTactics
sbatch grf/slurm/collect_data.slurm

# 4. Monitor job
squeue -u $USER
tail -f logs/grf-collect-*.out

# 5. Once data collection completes, submit training job
sbatch grf/slurm/train_model.slurm

# 6. Check results
cat results/baseline_position_results.json
cat results/velocity_ablation_results.json
```

---

## Quick Reference: HPC Container Commands

### Building from Docker Archive
```bash
# On HPC, convert .tar to .sif
apptainer build container_name.sif docker-archive://file_name.tar
```

### Pulling from DockerHub
```bash
# Alternative: pull directly from Docker registry
apptainer pull grf.sif docker://some_registry/gfootball:latest
```

### Running Container
```bash
# Basic execution
apptainer exec container.sif python script.py

# With GPU support
apptainer exec --nv container.sif python script.py

# With directory mounting
apptainer exec -B /home/user/data:/data container.sif python script.py

# Multiple binds
apptainer exec -B /path1:/mount1 -B /path2:/mount2 container.sif command
```

### SLURM Job Template
```bash
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --output=logs/my-job-%j.out
#SBATCH --error=logs/my-job-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=brown
# #SBATCH --gres=gpu              # Uncomment for GPU
# #SBATCH --gres-flags=enforce-binding

echo "Running on $(hostname)"

apptainer exec -B /home/$USER/project:/app container.sif python /app/script.py
```

---

## Expected Results

| Model | Expected AUC | Interpretation |
|-------|--------------|----------------|
| Position-only (GRF) | ~0.50-0.52 | Confirms 7.5 ECTS finding transfers to synthetic domain |
| Position+Velocity (GRF) | ~0.55-0.65 | Velocity features enable prediction |
| Random baseline | 0.50 | Chance level |

### Success Criteria:

1. **Position-only AUC ≈ 0.50** → Validates that static positioning alone cannot predict outcomes
2. **Position+Velocity AUC > 0.55** → Demonstrates velocity features matter
3. **Permutation test p > 0.05 for position-only** → No signal in positions alone
4. **Permutation test p < 0.05 for position+velocity** → Real signal exists with velocity

---

## Thesis Narrative

The results tell this story:

1. **7.5 ECTS work showed:** Static freeze-frame data yields AUC ≈ 0.50 (random chance) on real StatsBomb data.

2. **This work shows:** 
   - The same finding holds in synthetic GRF data (position-only fails)
   - Adding velocity features enables prediction (AUC > 0.55)
   - This explains WHY TacticAI works: their 25Hz tracking provides velocity information

3. **Conclusion:** Corner kick prediction requires temporal dynamics. Static positioning is fundamentally insufficient, regardless of model architecture (traditional ML or GNN).

---

## Timeline Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 1: Local setup | 2-4 hours | Docker build, testing |
| Phase 2: HPC setup | 1-2 hours | Container conversion, verification |
| Phase 3: Data collection | 4-8 hours | 10k episodes, depends on HPC queue |
| Phase 4: Training | 1-2 hours | Feature extraction + model training |
| Phase 5: Analysis | 2-4 hours | Results interpretation, visualization |

**Total: ~2-3 days** (including HPC queue wait times)

---

## Files Summary

| File | Purpose |
|------|---------|
| `grf/docker/Dockerfile` | Custom GRF container image |
| `grf/scripts/collect_corners.py` | Data collection from GRF |
| `grf/scripts/extract_features.py` | Feature engineering |
| `grf/slurm/build_container.sh` | One-time Apptainer build |
| `grf/slurm/collect_data.slurm` | SLURM job for data collection |
| `grf/slurm/train_model.slurm` | SLURM job for training |
| `models/baseline_position.py` | Position-only model (replicates 7.5 ECTS) |
| `models/velocity_model.py` | Position+velocity model (tests hypothesis) |

