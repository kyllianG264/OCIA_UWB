from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def compute_homography(source_points: list[list[int]], target_points: list[list[float]]) -> tuple[np.ndarray, list[int], float, list[float]]:
    if len(source_points) != len(target_points):
        raise ValueError("Video and terrain must have the same number of points.")
    if len(source_points) < 4:
        raise ValueError("At least 4 point pairs are required.")
    source = np.array(source_points, dtype=np.float32)
    target = np.array(target_points, dtype=np.float32)
    method = cv2.RANSAC if len(source_points) >= 5 else 0
    homography, mask = cv2.findHomography(source, target, method=method, ransacReprojThreshold=6.0)
    if homography is None:
        raise ValueError("Homography computation failed.")
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - target, axis=1)
    rms_error = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else 0.0
    inlier_mask = [int(value) for value in (mask.reshape(-1).tolist() if mask is not None else [1] * len(source_points))]
    return homography, inlier_mask, rms_error, [float(value) for value in errors.tolist()]


def bounds_from_points(terrain_points: list[list[float]]) -> dict[str, int]:
    x_values = [point[0] for point in terrain_points]
    y_values = [point[1] for point in terrain_points]
    return {
        "x_min": int(round(min(x_values))),
        "x_max": int(round(max(x_values))),
        "y_min": int(round(min(y_values))),
        "y_max": int(round(max(y_values))),
    }


def compute_split_y(left_points: list[list[float]], right_points: list[list[float]]) -> Optional[float]:
    if not left_points or not right_points:
        return None
    left_bounds = bounds_from_points(left_points)
    right_bounds = bounds_from_points(right_points)
    return float((left_bounds["y_max"] + right_bounds["y_min"]) / 2.0)
