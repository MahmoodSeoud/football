#!/usr/bin/env python3
"""
Collect corner kick data from full 11v11 matches in Google Research Football.

Unlike academy_corner (which starts at a corner with weak defense), this script:
1. Plays full 11v11 matches with stochastic AI opponents
2. Detects when corner kicks occur naturally (game_mode == 4)
3. Captures the corner kick sequence until it resolves
4. Records realistic outcomes (goal rate should be ~3-5%, not 77%)

Usage:
    python collect_corners_11v11.py --num_corners 1000 --output_dir ./data/raw_11v11
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import gfootball.env as football_env
import numpy as np
from tqdm import tqdm


# GRF Game Modes
GAME_MODE_NORMAL = 0
GAME_MODE_KICKOFF = 1
GAME_MODE_GOALKICK = 2
GAME_MODE_FREEKICK = 3
GAME_MODE_CORNER = 4
GAME_MODE_THROWIN = 5
GAME_MODE_PENALTY = 6


def create_11v11_env(render=False):
    """Create a full 11v11 match environment with stochastic AI."""
    env = football_env.create_environment(
        env_name='11_vs_11_stochastic',
        representation='raw',
        rewards='scoring',
        render=render,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        write_video=False,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
    )
    return env


def extract_frame_data(obs):
    """Extract structured data from a single observation frame."""
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

    if 'ball_owned_team' in obs:
        frame['ball_owned_team'] = int(obs['ball_owned_team'])
    if 'ball_owned_player' in obs:
        frame['ball_owned_player'] = int(obs['ball_owned_player'])

    return frame


def get_game_mode(obs):
    """Extract game mode from observation."""
    if isinstance(obs, list):
        obs = obs[0]
    return int(obs['game_mode'])


def get_score(obs):
    """Extract score from observation."""
    if isinstance(obs, list):
        obs = obs[0]
    return list(obs['score'])


def determine_corner_outcome(corner_frames, post_corner_frames, score_before, score_after,
                             attacking_team):
    """
    Determine the outcome of a corner kick sequence.

    Args:
        corner_frames: Frames during game_mode == CORNER
        post_corner_frames: Frames after corner until set piece ends
        score_before: [left, right] score when corner started
        score_after: [left, right] score when sequence ended
        attacking_team: 0 for left team, 1 for right team

    Returns:
        dict with outcome information
    """
    outcome = {
        'shot_occurred': False,
        'goal_scored': False,
        'outcome': 'unknown',
        'attacking_team': attacking_team,
    }

    # Check for goal
    if attacking_team == 0:  # Left team attacking
        if score_after[0] > score_before[0]:
            outcome['goal_scored'] = True
            outcome['shot_occurred'] = True
            outcome['outcome'] = 'goal'
            return outcome
    else:  # Right team attacking
        if score_after[1] > score_before[1]:
            outcome['goal_scored'] = True
            outcome['shot_occurred'] = True
            outcome['outcome'] = 'goal'
            return outcome

    # Check game mode transitions in post-corner frames
    if post_corner_frames:
        post_modes = [f['game_mode'] for f in post_corner_frames]

        if GAME_MODE_GOALKICK in post_modes:
            # Goal kick = shot went wide or was saved
            outcome['shot_occurred'] = True
            outcome['outcome'] = 'shot_missed'
        elif GAME_MODE_CORNER in post_modes:
            # Another corner = shot was deflected for corner
            outcome['shot_occurred'] = True
            outcome['outcome'] = 'shot_deflected_corner'
        elif GAME_MODE_THROWIN in post_modes:
            outcome['outcome'] = 'out_of_play'
        elif GAME_MODE_FREEKICK in post_modes:
            outcome['outcome'] = 'foul'
        elif GAME_MODE_KICKOFF in post_modes:
            # Kickoff without score change = own goal or timeout (rare)
            outcome['outcome'] = 'kickoff_no_goal'
        else:
            # Normal play resumed = ball was cleared
            outcome['outcome'] = 'clearance'
    else:
        outcome['outcome'] = 'clearance'

    return outcome


def detect_attacking_team_for_corner(obs):
    """
    Determine which team is taking the corner based on ball position.

    In GRF, pitch is x: [-1, 1], y: [-0.42, 0.42]
    Left team attacks toward x=1, right team toward x=-1
    Corner at x > 0.9 = left team corner (attacking right goal)
    Corner at x < -0.9 = right team corner (attacking left goal)
    """
    if isinstance(obs, list):
        obs = obs[0]

    ball_x = obs['ball'][0]

    if ball_x > 0.9:
        return 0  # Left team taking corner (attacking right goal)
    elif ball_x < -0.9:
        return 1  # Right team taking corner (attacking left goal)
    else:
        # Fallback: check which side of pitch
        return 0 if ball_x > 0 else 1


def collect_corner_from_match(env, corner_data, max_match_steps=3000, max_corner_frames=200):
    """
    Play a match and collect any corner kicks that occur.

    Args:
        env: GRF environment
        corner_data: List to append collected corners to
        max_match_steps: Maximum steps per match
        max_corner_frames: Maximum frames to capture per corner sequence

    Returns:
        Number of corners collected from this match
    """
    obs = env.reset()
    corners_this_match = 0

    in_corner = False
    corner_frames = []
    post_corner_frames = []
    score_at_corner_start = None
    attacking_team = None
    corner_start_step = 0

    for step in range(max_match_steps):
        game_mode = get_game_mode(obs)
        current_score = get_score(obs)

        # Detect corner kick start
        if game_mode == GAME_MODE_CORNER and not in_corner:
            in_corner = True
            corner_frames = []
            post_corner_frames = []
            score_at_corner_start = current_score.copy()
            attacking_team = detect_attacking_team_for_corner(obs)
            corner_start_step = step

        # Capture frames during corner
        if in_corner:
            frame_data = extract_frame_data(obs)
            frame_data['step'] = step - corner_start_step

            if game_mode == GAME_MODE_CORNER:
                corner_frames.append(frame_data)
            else:
                post_corner_frames.append(frame_data)

            # Check if corner sequence is over
            corner_sequence_done = False

            if game_mode == GAME_MODE_CORNER:
                # Still in corner mode
                pass
            elif len(post_corner_frames) > 30:
                # Enough post-corner frames to determine outcome
                corner_sequence_done = True
            elif game_mode in [GAME_MODE_KICKOFF, GAME_MODE_GOALKICK,
                               GAME_MODE_FREEKICK, GAME_MODE_THROWIN,
                               GAME_MODE_PENALTY]:
                # Clear set piece transition
                corner_sequence_done = True
            elif game_mode == GAME_MODE_CORNER and len(post_corner_frames) > 0:
                # New corner (deflection)
                corner_sequence_done = True

            # Timeout check
            if len(corner_frames) + len(post_corner_frames) > max_corner_frames:
                corner_sequence_done = True

            if corner_sequence_done and len(corner_frames) > 0:
                # Save this corner
                outcome = determine_corner_outcome(
                    corner_frames, post_corner_frames,
                    score_at_corner_start, current_score,
                    attacking_team
                )

                # Collect corners from BOTH teams
                # With random actions, our team rarely wins corners, so we'd miss most data
                # The physics are identical - we just track which team is attacking
                # and can normalize coordinates during feature extraction
                corner_episode = {
                    'frames': corner_frames + post_corner_frames,
                    'corner_frames_count': len(corner_frames),
                    'post_corner_frames_count': len(post_corner_frames),
                    'outcome': outcome,
                    'metadata': {
                        'match_step': corner_start_step,
                        'attacking_team': attacking_team,
                        'score_before': score_at_corner_start,
                        'score_after': current_score,
                    }
                }
                corner_data.append(corner_episode)
                corners_this_match += 1

                # Reset for next corner
                in_corner = False
                corner_frames = []
                post_corner_frames = []

        # Take action (use built-in AI by taking action 0 = idle, or random)
        # For more varied play, use random actions
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        if done:
            break

    return corners_this_match


def collect_corners_11v11(num_corners, output_dir, save_every=100):
    """
    Collect corner kicks from full 11v11 matches.

    Args:
        num_corners: Target number of corners to collect
        output_dir: Output directory
        save_every: Save checkpoint every N corners
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env = create_11v11_env(render=False)

    all_corners = []
    outcome_counts = {
        'goal': 0,
        'shot_missed': 0,
        'shot_deflected_corner': 0,
        'clearance': 0,
        'out_of_play': 0,
        'foul': 0,
        'other': 0
    }

    matches_played = 0

    try:
        with tqdm(total=num_corners, desc="Collecting corners") as pbar:
            while len(all_corners) < num_corners:
                corners_before = len(all_corners)

                corners_this_match = collect_corner_from_match(
                    env, all_corners,
                    max_match_steps=3000,
                    max_corner_frames=200
                )

                matches_played += 1

                # Update outcome counts for new corners
                for corner in all_corners[corners_before:]:
                    outcome = corner['outcome']['outcome']
                    if outcome in outcome_counts:
                        outcome_counts[outcome] += 1
                    else:
                        outcome_counts['other'] += 1

                # Update progress
                new_corners = len(all_corners) - corners_before
                pbar.update(new_corners)

                # Periodic checkpoint
                if len(all_corners) > 0 and len(all_corners) % save_every < new_corners:
                    checkpoint_num = (len(all_corners) // save_every) * save_every
                    checkpoint_file = output_path / f"corners_11v11_checkpoint_{checkpoint_num}.json"
                    with open(checkpoint_file, 'w') as f:
                        json.dump(all_corners[:checkpoint_num], f)

                    goal_rate = outcome_counts['goal'] / len(all_corners) if all_corners else 0
                    tqdm.write(f"\nCheckpoint {checkpoint_num}: {matches_played} matches, "
                              f"goal rate: {goal_rate:.1%}")
                    tqdm.write(f"Outcomes: {outcome_counts}")

    finally:
        env.close()

    # Save final dataset
    final_file = output_path / f"corners_11v11_n{len(all_corners)}.json"
    with open(final_file, 'w') as f:
        json.dump(all_corners, f)

    # Save metadata
    meta_file = output_path / "collection_11v11_metadata.json"
    metadata = {
        'num_corners': len(all_corners),
        'matches_played': matches_played,
        'corners_per_match': len(all_corners) / matches_played if matches_played > 0 else 0,
        'outcome_distribution': outcome_counts,
        'goal_rate': outcome_counts['goal'] / len(all_corners) if all_corners else 0,
        'collection_date': datetime.now().isoformat(),
        'scenario': '11_vs_11_stochastic',
    }
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCollection complete!")
    print(f"Total corners: {len(all_corners)}")
    print(f"Matches played: {matches_played}")
    print(f"Corners per match: {metadata['corners_per_match']:.2f}")
    print(f"Goal rate: {metadata['goal_rate']:.1%}")
    print(f"Outcome distribution: {outcome_counts}")
    print(f"Data saved to: {final_file}")

    return all_corners


def main():
    parser = argparse.ArgumentParser(
        description='Collect corner kicks from 11v11 GRF matches'
    )
    parser.add_argument('--num_corners', type=int, default=1000,
                        help='Target number of corners to collect')
    parser.add_argument('--output_dir', type=str, default='./data/raw_11v11',
                        help='Output directory')
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save checkpoint every N corners')

    args = parser.parse_args()

    print(f"Starting 11v11 corner collection...")
    print(f"  Target corners: {args.num_corners}")
    print(f"  Output: {args.output_dir}")
    print(f"  Note: This may require many matches - corners are rare events")

    collect_corners_11v11(
        num_corners=args.num_corners,
        output_dir=args.output_dir,
        save_every=args.save_every
    )


if __name__ == '__main__':
    main()
