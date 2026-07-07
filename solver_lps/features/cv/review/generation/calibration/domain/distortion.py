from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def apply_radial_correction_to_points(points, distortion):
    if not distortion or not distortion.get("enabled"):
        return [list(point) for point in points]
    width = float(distortion["width"])
    height = float(distortion["height"])
    cx = float(distortion["cx"])
    cy = float(distortion["cy"])
    scale = max(width, height, 1.0)
    k1 = float(distortion.get("k1", 0.0))
    k2 = float(distortion.get("k2", 0.0))
    corrected = []
    for point in points:
        x = (float(point[0]) - cx) / scale
        y = (float(point[1]) - cy) / scale
        r2 = x * x + y * y
        factor = 1.0 + k1 * r2 + k2 * r2 * r2
        if abs(factor) < 1e-6:
            factor = 1.0
        xu = x / factor
        yu = y / factor
        corrected.append([xu * scale + cx, yu * scale + cy])
    return corrected


def invert_radial_correction_to_points(points, distortion, iterations: int = 8):
    if not distortion or not distortion.get("enabled"):
        return [list(point) for point in points]
    width = float(distortion["width"])
    height = float(distortion["height"])
    cx = float(distortion["cx"])
    cy = float(distortion["cy"])
    scale = max(width, height, 1.0)
    k1 = float(distortion.get("k1", 0.0))
    k2 = float(distortion.get("k2", 0.0))
    inverted = []
    for point in points:
        xu = (float(point[0]) - cx) / scale
        yu = (float(point[1]) - cy) / scale
        x = xu
        y = yu
        for _ in range(iterations):
            r2 = x * x + y * y
            factor = 1.0 + k1 * r2 + k2 * r2 * r2
            if abs(factor) < 1e-6:
                factor = 1.0
            x = xu * factor
            y = yu * factor
        inverted.append([x * scale + cx, y * scale + cy])
    return inverted


def apply_view_transform_to_points(points, view_transform):
    if not view_transform:
        return [list(point) for point in points]
    scale = float(view_transform.get("scale", 1.0))
    offset_x = float(view_transform.get("offset_x", 0.0))
    offset_y = float(view_transform.get("offset_y", 0.0))
    return [[float(point[0]) * scale + offset_x, float(point[1]) * scale + offset_y] for point in points]


def invert_view_transform_to_points(points, view_transform):
    if not view_transform:
        return [list(point) for point in points]
    scale = float(view_transform.get("scale", 1.0) or 1.0)
    offset_x = float(view_transform.get("offset_x", 0.0))
    offset_y = float(view_transform.get("offset_y", 0.0))
    return [[(float(point[0]) - offset_x) / scale, (float(point[1]) - offset_y) / scale] for point in points]


def compute_undistort_view_transform(width: int, height: int, distortion: Optional[dict], margin_px: float = 12.0):
    identity = {
        "scale": 1.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "width": int(width),
        "height": int(height),
    }
    if not distortion or not distortion.get("enabled"):
        return identity

    edge_steps = max(8, min(width, height) // 80)
    border_points = []
    for x_value in np.linspace(0.0, max(width - 1, 0), edge_steps + 1):
        border_points.append([x_value, 0.0])
        border_points.append([x_value, float(max(height - 1, 0))])
    for y_value in np.linspace(0.0, max(height - 1, 0), edge_steps + 1):
        border_points.append([0.0, y_value])
        border_points.append([float(max(width - 1, 0)), y_value])

    corrected = apply_radial_correction_to_points(border_points, distortion)
    x_values = [point[0] for point in corrected]
    y_values = [point[1] for point in corrected]
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    usable_width = max(float(width) - 2.0 * margin_px, 1.0)
    usable_height = max(float(height) - 2.0 * margin_px, 1.0)
    scale = min(usable_width / span_x, usable_height / span_y)
    offset_x = margin_px + (usable_width - span_x * scale) / 2.0 - min_x * scale
    offset_y = margin_px + (usable_height - span_y * scale) / 2.0 - min_y * scale
    return {
        "scale": float(scale),
        "offset_x": float(offset_x),
        "offset_y": float(offset_y),
        "width": int(width),
        "height": int(height),
    }


def remap_frame_with_distortion(frame_rgb: np.ndarray, distortion: Optional[dict], view_transform: Optional[dict] = None) -> np.ndarray:
    if not distortion or not distortion.get("enabled"):
        return frame_rgb
    height, width = frame_rgb.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    destination_points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    corrected_points = invert_view_transform_to_points(destination_points.tolist(), view_transform)
    corrected_points = np.asarray(corrected_points, dtype=np.float32)
    source_points = invert_radial_correction_to_points(corrected_points, distortion)
    source_points_array = np.asarray(source_points, dtype=np.float32).reshape(height, width, 2)
    map_x = source_points_array[:, :, 0]
    map_y = source_points_array[:, :, 1]
    return cv2.remap(frame_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def line_fit_error(points) -> float:
    if len(points) < 2:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx = float(vx)
    vy = float(vy)
    x0 = float(x0)
    y0 = float(y0)
    norm = max((vx * vx + vy * vy) ** 0.5, 1e-6)
    return sum((((point[0] - x0) * vy - (point[1] - y0) * vx) / norm) ** 2 for point in points) / len(points)


def distortion_objective(lines, distortion):
    corrected_lines = [apply_radial_correction_to_points(line, distortion) for line in lines if len(line) >= 2]
    if not corrected_lines:
        return 0.0
    return sum(line_fit_error(line) for line in corrected_lines) / len(corrected_lines)


def optimize_distortion(lines, width: int, height: int):
    usable_lines = [line for line in lines if len(line) >= 3]
    if len(usable_lines) < 2:
        return None
    cx = width / 2.0
    cy = height / 2.0
    best = {
        "enabled": True,
        "k1": 0.0,
        "k2": 0.0,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
    }
    best_score = distortion_objective(usable_lines, best)
    for step in (0.25, 0.08, 0.025, 0.008):
        k1_center = best["k1"]
        k2_center = best["k2"]
        k1_values = np.linspace(k1_center - (step * 4), k1_center + (step * 4), 17)
        k2_values = np.linspace(k2_center - (step * 1.5), k2_center + (step * 1.5), 13)
        for k1 in k1_values:
            for k2 in k2_values:
                candidate = {
                    "enabled": True,
                    "k1": float(k1),
                    "k2": float(k2),
                    "cx": cx,
                    "cy": cy,
                    "width": width,
                    "height": height,
                }
                score = distortion_objective(usable_lines, candidate)
                if score < best_score:
                    best_score = score
                    best = candidate
    if best_score >= distortion_objective(usable_lines, {"enabled": False, "k1": 0.0, "k2": 0.0, "cx": cx, "cy": cy, "width": width, "height": height}):
        return {
            "enabled": False,
            "k1": 0.0,
            "k2": 0.0,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
        }
    return best
