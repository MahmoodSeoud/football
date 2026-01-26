#!/usr/bin/env python3
"""
Utility functions for GRF corner kick data collection and analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# GRF Game Mode constants
class GameMode:
    NORMAL = 0
    KICKOFF = 1
    GOALKICK = 2
    FREEKICK = 3
    CORNER = 4
    THROWIN = 5
    PENALTY = 6


# GRF Action constants
class Action:
    IDLE = 0
    LEFT = 1
    TOP_LEFT = 2
    TOP = 3
    TOP_RIGHT = 4
    RIGHT = 5
    BOTTOM_RIGHT = 6
    BOTTOM = 7
    BOTTOM_LEFT = 8
    LONG_PASS = 9
    HIGH_PASS = 10
    SHORT_PASS = 11
    SHOT = 12
    SPRINT = 13
    RELEASE_DIRECTION = 14
    RELEASE_SPRINT = 15
    SLIDING = 16
    DRIBBLE = 17
    RELEASE_DRIBBLE = 18


# GRF pitch dimensions (normalized coordinates)
PITCH_X_MIN = -1.0
PITCH_X_MAX = 1.0
PITCH_Y_MIN = -0.42
PITCH_Y_MAX = 0.42
GOAL_X = 1.0
GOAL_Y = 0.0
PENALTY_BOX_X = 0.8
PENALTY_BOX_Y = 0.2
SIX_YARD_BOX_X = 0.9
SIX_YARD_BOX_Y = 0.1


def load_episodes(filepath: str) -> List[Dict]:
    """Load episodes from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_episodes(episodes: List[Dict], filepath: str):
    """Save episodes to a JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(episodes, f)


def get_distance_to_goal(position: np.ndarray) -> float:
    """Calculate distance from a position to the goal center."""
    goal = np.array([GOAL_X, GOAL_Y])
    return float(np.linalg.norm(position - goal))


def is_in_penalty_box(position: np.ndarray) -> bool:
    """Check if a position is inside the penalty box."""
    return position[0] > PENALTY_BOX_X and abs(position[1]) < PENALTY_BOX_Y


def is_in_six_yard_box(position: np.ndarray) -> bool:
    """Check if a position is inside the six-yard box."""
    return position[0] > SIX_YARD_BOX_X and abs(position[1]) < SIX_YARD_BOX_Y


def find_delivery_frame(frames: List[Dict], velocity_threshold: float = 0.01) -> int:
    """
    Find the frame index where the ball starts moving (delivery moment).

    Args:
        frames: List of frame dicts from an episode
        velocity_threshold: Minimum ball velocity to consider as moving

    Returns:
        Frame index of delivery, or 0 if ball never moves
    """
    for i, frame in enumerate(frames):
        ball_vel = frame.get('ball_velocity', [0, 0, 0])
        if np.linalg.norm(ball_vel[:2]) > velocity_threshold:
            return i
    return 0


def compute_player_speeds(velocities: np.ndarray) -> np.ndarray:
    """Compute speed magnitudes from velocity vectors."""
    return np.linalg.norm(velocities, axis=1)


def count_players_in_zone(
    positions: np.ndarray,
    x_min: float = -np.inf,
    x_max: float = np.inf,
    y_min: float = -np.inf,
    y_max: float = np.inf
) -> int:
    """Count players within a rectangular zone."""
    in_zone = (
        (positions[:, 0] > x_min) &
        (positions[:, 0] < x_max) &
        (positions[:, 1] > y_min) &
        (positions[:, 1] < y_max)
    )
    return int(np.sum(in_zone))


def summarize_outcome_distribution(episodes: List[Dict]) -> Dict[str, int]:
    """Summarize the outcome distribution across episodes."""
    counts = {}
    for ep in episodes:
        outcome = ep.get('outcome', {}).get('outcome', 'unknown')
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def get_episode_stats(episodes: List[Dict]) -> Dict:
    """Get summary statistics for a collection of episodes."""
    if not episodes:
        return {}

    outcomes = summarize_outcome_distribution(episodes)
    total = len(episodes)

    goal_count = sum(1 for ep in episodes if ep.get('outcome', {}).get('goal_scored', False))
    shot_count = sum(1 for ep in episodes if ep.get('outcome', {}).get('shot_occurred', False))

    steps = [ep.get('metadata', {}).get('total_steps', 0) for ep in episodes]

    return {
        'total_episodes': total,
        'outcome_distribution': outcomes,
        'goal_rate': goal_count / total if total > 0 else 0,
        'shot_rate': shot_count / total if total > 0 else 0,
        'avg_steps': np.mean(steps) if steps else 0,
        'min_steps': min(steps) if steps else 0,
        'max_steps': max(steps) if steps else 0,
    }


def print_episode_stats(episodes: List[Dict]):
    """Print summary statistics for episodes."""
    stats = get_episode_stats(episodes)
    if not stats:
        print("No episodes to summarize")
        return

    print(f"\nEpisode Statistics:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Goal rate: {stats['goal_rate']:.2%}")
    print(f"  Shot rate: {stats['shot_rate']:.2%}")
    print(f"  Steps: {stats['avg_steps']:.1f} avg, {stats['min_steps']}-{stats['max_steps']} range")
    print(f"  Outcome distribution:")
    for outcome, count in sorted(stats['outcome_distribution'].items()):
        pct = count / stats['total_episodes'] * 100
        print(f"    {outcome}: {count} ({pct:.1f}%)")
