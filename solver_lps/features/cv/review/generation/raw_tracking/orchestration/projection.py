"""
Projection des detections image sur le terrain 2D.
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

from solver_lps.features.cv.review.generation.raw_tracking.data.calibration_input import (
    load_calibration,
)
from solver_lps.features.cv.review.generation.raw_tracking.data.tracking_output import (
    PROJECTED_POSITIONS_COLUMNS,
    configure_utf8_stdout,
    ensure_dir,
    read_csv_rows,
    require_file,
    write_csv_rows,
)
from solver_lps.features.cv.review.generation.raw_tracking.domain.projection_core import (
    apply_undistort_view,
    project,
    terrain_contains,
    undistort_point,
)


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Projection des detections sur le terrain")
    parser.add_argument("--input_csv", required=True, help="CSV de detections image")
    parser.add_argument("--calibration", required=True, help="Fichier calibration.json")
    parser.add_argument("--output", default="./resultats", help="Dossier de sortie CSV")
    parser.add_argument(
        "--disable_bounds_filter",
        action="store_true",
        help="Option legacy. Les points hors terrain sont maintenant conserves et marques via la colonne on_terrain.",
    )
    args = parser.parse_args()

    ensure_dir(args.output)
    require_file(args.input_csv, "detections CSV")
    calibration = load_calibration(args.calibration)

    csv_path = os.path.join(args.output, "projected_positions.csv")
    meta_path = os.path.join(args.output, "run_metadata.json")
    kept_counts = {"gauche": 0, "droite": 0}
    outside_counts = {"gauche": 0, "droite": 0}
    projected_rows = []

    for row in read_csv_rows(args.input_csv):
        camera = row["cam"].strip().lower()
        if camera == "gauche":
            h_matrix = calibration["H_g"]
            bounds = calibration["bounds_g"]
            distortion = calibration.get("distortion_g")
            undistort_view = calibration.get("undistort_view_g")
        elif camera == "droite":
            h_matrix = calibration["H_d"]
            bounds = calibration["bounds_d"]
            distortion = calibration.get("distortion_d")
            undistort_view = calibration.get("undistort_view_d")
        else:
            continue

        foot_x = float(row["foot_x"])
        foot_y = float(row["foot_y"])
        undistorted_x, undistorted_y = undistort_point(foot_x, foot_y, distortion)
        view_x, view_y = apply_undistort_view(undistorted_x, undistorted_y, undistort_view)
        projected_x, projected_y = project(h_matrix, view_x, view_y)
        inside_bounds = terrain_contains(bounds, projected_x, projected_y)

        if not inside_bounds:
            outside_counts[camera] += 1

        projected_rows.append(
            [
                row["frame"],
                row["timestamp_s"],
                row["timestamp_unix"],
                camera,
                row["detection_id"],
                int(foot_x),
                int(foot_y),
                projected_x,
                projected_y,
                1 if inside_bounds else 0,
                row.get("conf", ""),
            ]
        )
        kept_counts[camera] += 1

    write_csv_rows(csv_path, PROJECTED_POSITIONS_COLUMNS, projected_rows)

    summary = {
        "input_csv": args.input_csv,
        "calibration": args.calibration,
        "csv_path": csv_path,
        "bounds_filter_enabled": False,
        "kept_counts": kept_counts,
        "outside_counts": outside_counts,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 60)
    print("Projection terminee")
    print(f"CSV exporte : {csv_path}")
    print(f"Metadonnees : {meta_path}")
    print(f"Points gardes : gauche={kept_counts['gauche']} droite={kept_counts['droite']}")
    print(f"Points hors terrain marques : gauche={outside_counts['gauche']} droite={outside_counts['droite']}")


if __name__ == "__main__":
    main()
