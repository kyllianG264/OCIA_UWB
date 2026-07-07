"""
Tracking sur les positions projetees, avec export CSV compatible solver.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from solver_lps.features.cv.review.generation.raw_tracking.data.tracking_output import (
    POSITIONS_RAW_COLUMNS,
    TRACKING_ASSIGNMENTS_COLUMNS,
    configure_utf8_stdout,
    ensure_dir,
    read_csv_rows,
    require_file,
    write_csv_rows,
)
from solver_lps.features.cv.review.generation.raw_tracking.domain.tracking_core import (
    group_projected_positions,
    run_tracking,
)


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Tracking sur positions projetees")
    parser.add_argument("--input_csv", required=True, help="CSV de positions projetees")
    parser.add_argument("--output", default="./resultats", help="Dossier de sortie CSV")
    parser.add_argument("--max_track_distance", type=float, default=120.0, help="Distance max entre deux frames pour conserver un track")
    parser.add_argument("--max_idle_frames", type=int, default=15, help="Nombre max de frames sans match avant suppression du track")
    args = parser.parse_args()

    ensure_dir(args.output)
    require_file(args.input_csv, "positions projetees CSV")
    grouped = group_projected_positions(read_csv_rows(args.input_csv))

    csv_path = os.path.join(args.output, "positions_raw.csv")
    assignments_path = os.path.join(args.output, "tracking_assignments.csv")
    meta_path = os.path.join(args.output, "run_metadata.json")
    positions_rows, assignment_rows, output_counts = run_tracking(
        grouped,
        max_track_distance=args.max_track_distance,
        max_idle_frames=args.max_idle_frames,
    )
    write_csv_rows(csv_path, POSITIONS_RAW_COLUMNS, positions_rows)
    write_csv_rows(assignments_path, TRACKING_ASSIGNMENTS_COLUMNS, assignment_rows)

    for frame_idx in sorted(grouped):
        if frame_idx % 100 == 0:
            timestamp_s = grouped[frame_idx]["timestamp_s"]
            print(f"  Frame {frame_idx:6d} | t={timestamp_s:7.1f}s")

    summary = {
        "input_csv": args.input_csv,
        "csv_path": csv_path,
        "assignments_path": assignments_path,
        "max_track_distance": args.max_track_distance,
        "max_idle_frames": args.max_idle_frames,
        "output_counts": output_counts,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 60)
    print("Tracking termine")
    print(f"CSV exporte : {csv_path}")
    print(f"Assignations exportees : {assignments_path}")
    print(f"Metadonnees : {meta_path}")
    print(f"Points suivis : gauche={output_counts['gauche']} droite={output_counts['droite']}")


if __name__ == "__main__":
    main()
