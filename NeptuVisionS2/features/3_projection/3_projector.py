"""
Projection des detections image sur le terrain 2D.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.utils import configure_utf8_stdout, ensure_dir, load_calibration, require_file


def project(h_matrix: np.ndarray, px: float, py: float) -> tuple[int, int]:
    point = np.array([[[px, py]]], dtype=float)
    projected = cv2.perspectiveTransform(point, h_matrix)
    return int(projected[0][0][0]), int(projected[0][0][1])


def undistort_point(px: float, py: float, distortion: dict | None) -> tuple[float, float]:
    if not distortion or not distortion.get("enabled"):
        return float(px), float(py)
    width = float(distortion.get("width", 1.0) or 1.0)
    height = float(distortion.get("height", 1.0) or 1.0)
    cx = float(distortion.get("cx", width / 2.0))
    cy = float(distortion.get("cy", height / 2.0))
    scale = max(width, height, 1.0)
    k1 = float(distortion.get("k1", 0.0))
    k2 = float(distortion.get("k2", 0.0))
    x = (float(px) - cx) / scale
    y = (float(py) - cy) / scale
    r2 = x * x + y * y
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    if abs(factor) < 1e-6:
        factor = 1.0
    xu = x / factor
    yu = y / factor
    return xu * scale + cx, yu * scale + cy


def apply_undistort_view(px: float, py: float, view_transform: dict | None) -> tuple[float, float]:
    if not view_transform:
        return float(px), float(py)
    scale = float(view_transform.get("scale", 1.0) or 1.0)
    offset_x = float(view_transform.get("offset_x", 0.0))
    offset_y = float(view_transform.get("offset_y", 0.0))
    return float(px) * scale + offset_x, float(py) * scale + offset_y


def terrain_contains(bounds: dict, px: int, py: int) -> bool:
    return bounds["x_min"] <= px <= bounds["x_max"] and bounds["y_min"] <= py <= bounds["y_max"]


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

    with open(args.input_csv, encoding="utf-8") as source_handle, open(csv_path, "w", newline="", encoding="utf-8") as target_handle:
        reader = csv.DictReader(source_handle)
        writer = csv.writer(target_handle)
        writer.writerow(
            [
                "frame",
                "timestamp_s",
                "timestamp_unix",
                "cam",
                "detection_id",
                "source_foot_x",
                "source_foot_y",
                "X",
                "Y",
                "on_terrain",
                "conf",
            ]
        )

        for row in reader:
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

            writer.writerow(
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
    print(f"Projection terminee")
    print(f"CSV exporte : {csv_path}")
    print(f"Metadonnees : {meta_path}")
    print(f"Points gardes : gauche={kept_counts['gauche']} droite={kept_counts['droite']}")
    print(f"Points hors terrain marques : gauche={outside_counts['gauche']} droite={outside_counts['droite']}")


if __name__ == "__main__":
    main()
