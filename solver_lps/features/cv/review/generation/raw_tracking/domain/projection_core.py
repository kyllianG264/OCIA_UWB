from __future__ import annotations

import numpy as np
import cv2


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

