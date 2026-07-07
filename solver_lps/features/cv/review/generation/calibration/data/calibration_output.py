from __future__ import annotations

import json
import os

import numpy as np


def save_calibration(
    output_dir: str,
    *,
    terrain_path: str,
    terrain_width: int,
    terrain_height: int,
    left_video_points: list[list[int]],
    left_terrain_points: list[list[float]],
    left_homography: np.ndarray,
    left_inliers: list[int],
    left_rms_error: float,
    left_point_errors: list[float],
    left_distortion: dict | None,
    left_view: dict,
    left_distortion_lines: list,
    left_bounds: dict | None,
    right_video_points: list[list[int]],
    right_terrain_points: list[list[float]],
    right_homography: np.ndarray,
    right_inliers: list[int],
    right_rms_error: float,
    right_point_errors: list[float],
    right_distortion: dict | None,
    right_view: dict,
    right_distortion_lines: list,
    right_bounds: dict | None,
    split_y: float | None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    try:
        stored_terrain_path = os.path.relpath(
            os.path.abspath(terrain_path),
            os.path.abspath(output_dir),
        ).replace("\\", "/")
    except ValueError:
        stored_terrain_path = os.path.basename(terrain_path)
    payload = {
        "terrain_image_path": stored_terrain_path,
        "terrain_image_size": [terrain_width, terrain_height],
        "terrain_png_size": [terrain_width, terrain_height],
        "cam_gauche": {
            "frame_points": left_video_points,
            "terrain_points": left_terrain_points,
            "frame_corners": left_video_points[:4],
            "terrain_corners": left_terrain_points[:4],
            "H": left_homography.tolist(),
            "inlier_mask": left_inliers,
            "rms_reprojection_error_px": left_rms_error,
            "point_errors_px": left_point_errors,
            "distortion": left_distortion,
            "undistort_view": left_view,
            "distortion_lines": left_distortion_lines,
        },
        "cam_droite": {
            "frame_points": right_video_points,
            "terrain_points": right_terrain_points,
            "frame_corners": right_video_points[:4],
            "terrain_corners": right_terrain_points[:4],
            "H": right_homography.tolist(),
            "inlier_mask": right_inliers,
            "rms_reprojection_error_px": right_rms_error,
            "point_errors_px": right_point_errors,
            "distortion": right_distortion,
            "undistort_view": right_view,
            "distortion_lines": right_distortion_lines,
        },
        "terrain_bounds": {
            "gauche": left_bounds,
            "droite": right_bounds,
        },
        "split_y": split_y,
        "split_axis": "y",
        "H_g": left_homography.tolist(),
        "H_d": right_homography.tolist(),
        "bounds_g": left_bounds,
        "bounds_d": right_bounds,
        "undistort_view_g": left_view,
        "undistort_view_d": right_view,
        "calibration_method": "findHomography(RANSAC)",
    }
    path = os.path.join(output_dir, "calibration.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
