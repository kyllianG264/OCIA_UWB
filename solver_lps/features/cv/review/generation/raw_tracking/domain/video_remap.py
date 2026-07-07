"""
Logique de remap/undistortion des frames video.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

REMAP_CACHE = {}


def adapt_distortion_to_frame(
    distortion: dict[str, Any] | None,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any] | None:
    if not distortion:
        return distortion
    src_width = float(distortion.get("width", frame_width) or frame_width or 1.0)
    src_height = float(distortion.get("height", frame_height) or frame_height or 1.0)
    scale_x = float(frame_width) / max(src_width, 1.0)
    scale_y = float(frame_height) / max(src_height, 1.0)
    adapted = dict(distortion)
    adapted["cx"] = float(distortion.get("cx", src_width / 2.0)) * scale_x
    adapted["cy"] = float(distortion.get("cy", src_height / 2.0)) * scale_y
    adapted["width"] = int(frame_width)
    adapted["height"] = int(frame_height)
    return adapted


def adapt_view_transform_to_frame(
    view_transform: dict[str, Any] | None,
    source_width: int,
    source_height: int,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any] | None:
    if not view_transform:
        return view_transform
    src_width = float(source_width or frame_width or 1.0)
    src_height = float(source_height or frame_height or 1.0)
    scale_x = float(frame_width) / max(src_width, 1.0)
    scale_y = float(frame_height) / max(src_height, 1.0)
    adapted = dict(view_transform)
    adapted["offset_x"] = float(view_transform.get("offset_x", 0.0)) * scale_x
    adapted["offset_y"] = float(view_transform.get("offset_y", 0.0)) * scale_y
    adapted["width"] = int(frame_width)
    adapted["height"] = int(frame_height)
    return adapted


def _remap_cache_key(width, height, distortion, view_transform):
    if not distortion or not distortion.get("enabled"):
        return None
    return (
        int(width),
        int(height),
        round(float(distortion.get("k1", 0.0)), 8),
        round(float(distortion.get("k2", 0.0)), 8),
        round(float(distortion.get("cx", 0.0)), 4),
        round(float(distortion.get("cy", 0.0)), 4),
        round(float(view_transform.get("scale", 1.0)) if view_transform else 1.0, 8),
        round(float(view_transform.get("offset_x", 0.0)) if view_transform else 0.0, 4),
        round(float(view_transform.get("offset_y", 0.0)) if view_transform else 0.0, 4),
    )


def _build_remap_maps(width, height, distortion, view_transform):
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    if view_transform:
        view_scale = float(view_transform.get("scale", 1.0) or 1.0)
        offset_x = float(view_transform.get("offset_x", 0.0))
        offset_y = float(view_transform.get("offset_y", 0.0))
        corrected_x = (grid_x - offset_x) / view_scale
        corrected_y = (grid_y - offset_y) / view_scale
    else:
        corrected_x = grid_x
        corrected_y = grid_y

    width_ref = float(distortion.get("width", width) or width or 1.0)
    height_ref = float(distortion.get("height", height) or height or 1.0)
    cx = float(distortion.get("cx", width_ref / 2.0))
    cy = float(distortion.get("cy", height_ref / 2.0))
    scale = max(width_ref, height_ref, 1.0)
    k1 = float(distortion.get("k1", 0.0))
    k2 = float(distortion.get("k2", 0.0))

    xu = (corrected_x - cx) / scale
    yu = (corrected_y - cy) / scale
    x = xu.copy()
    y = yu.copy()
    for _ in range(8):
        r2 = x * x + y * y
        factor = 1.0 + k1 * r2 + k2 * r2 * r2
        factor = np.where(np.abs(factor) < 1e-6, 1.0, factor)
        x = xu * factor
        y = yu * factor

    map_x = (x * scale + cx).astype(np.float32)
    map_y = (y * scale + cy).astype(np.float32)
    return map_x, map_y


def remap_frame_with_distortion(
    frame_bgr,
    distortion: dict | None,
    view_transform: dict | None = None,
):
    if frame_bgr is None or not distortion or not distortion.get("enabled"):
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    source_width = int(distortion.get("width", width) or width)
    source_height = int(distortion.get("height", height) or height)
    distortion = adapt_distortion_to_frame(distortion, width, height)
    view_transform = adapt_view_transform_to_frame(
        view_transform,
        source_width,
        source_height,
        width,
        height,
    )
    cache_key = _remap_cache_key(width, height, distortion, view_transform)
    if cache_key not in REMAP_CACHE:
        REMAP_CACHE[cache_key] = _build_remap_maps(width, height, distortion, view_transform)
    map_x, map_y = REMAP_CACHE[cache_key]
    return cv2.remap(
        frame_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
