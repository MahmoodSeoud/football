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
