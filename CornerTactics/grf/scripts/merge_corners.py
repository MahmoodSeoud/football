#!/usr/bin/env python3
"""
Merge corner kick data from parallel collection jobs.

Usage:
    python merge_corners.py --input_dirs job_1 job_2 job_3 job_4 --output_file merged.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def merge_corner_files(input_dirs, output_file):
    """Merge corner data from multiple directories."""
    all_corners = []
    total_matches = 0
    combined_outcomes = {}

    for input_dir in input_dirs:
        input_path = Path(input_dir)

        # Find the main data file
        json_files = list(input_path.glob("corners_11v11_n*.json"))
        if not json_files:
            print(f"Warning: No data file found in {input_dir}")
            continue

        data_file = json_files[0]
        print(f"Loading {data_file}...")

        with open(data_file, 'r') as f:
            corners = json.load(f)

        all_corners.extend(corners)

        # Load metadata if available
        meta_file = input_path / "collection_11v11_metadata.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            total_matches += meta.get('matches_played', 0)

            for outcome, count in meta.get('outcome_distribution', {}).items():
                combined_outcomes[outcome] = combined_outcomes.get(outcome, 0) + count

    print(f"\nMerged {len(all_corners)} corners from {len(input_dirs)} directories")

    # Save merged data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_corners, f)

    # Save merged metadata
    meta_output = output_path.parent / "collection_merged_metadata.json"
    merged_meta = {
        'num_corners': len(all_corners),
        'matches_played': total_matches,
        'corners_per_match': len(all_corners) / total_matches if total_matches > 0 else 0,
        'outcome_distribution': combined_outcomes,
        'goal_rate': combined_outcomes.get('goal', 0) / len(all_corners) if all_corners else 0,
        'source_directories': [str(d) for d in input_dirs],
        'merge_date': datetime.now().isoformat(),
    }
    with open(meta_output, 'w') as f:
        json.dump(merged_meta, f, indent=2)

    print(f"Saved merged data to: {output_path}")
    print(f"Saved metadata to: {meta_output}")
    print(f"Total corners: {len(all_corners)}")
    print(f"Goal rate: {merged_meta['goal_rate']:.1%}")
    print(f"Outcomes: {combined_outcomes}")

    return all_corners


def main():
    parser = argparse.ArgumentParser(description='Merge corner data from parallel jobs')
    parser.add_argument('--input_dirs', nargs='+', required=True,
                        help='Input directories from parallel jobs')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file for merged data')

    args = parser.parse_args()
    merge_corner_files(args.input_dirs, args.output_file)


if __name__ == '__main__':
    main()
