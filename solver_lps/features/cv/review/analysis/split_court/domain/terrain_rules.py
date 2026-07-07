from __future__ import annotations

import math
from typing import Any, Optional

from solver_lps.features.ground.domain.calibration import load_calibration_geometry

COURT_MARGIN = 45.0
RAW_ID_ALIAS_OFFSET = 30
SPLIT_HANDOFF_BAND = 70.0
SPLIT_HANDOFF_EXTRA_PER_FRAME = 8.0
SPAWN_BORDER_MARGIN = 70.0


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def normalize_cam_name(cam: str) -> str:
    return str(cam or "").strip().lower()


def parse_int_id(raw_id: str) -> Optional[int]:
    text = str(raw_id or "").strip()
    if not text.isdigit():
        return None
    return int(text)


def raw_id_alias_match(previous_raw_id: str, new_raw_id: str) -> bool:
    previous = parse_int_id(previous_raw_id)
    new = parse_int_id(new_raw_id)
    if previous is None or new is None:
        return False
    if previous == new:
        return True
    return abs(previous - new) == RAW_ID_ALIAS_OFFSET


def load_split_y(calibration_path: str) -> Optional[float]:
    geometry = load_calibration_geometry(calibration_path)
    return geometry.get("split_y")


def is_near_split(y_value: float, split_y: Optional[float], band: float) -> bool:
    if split_y is None:
        return False
    return abs(float(y_value) - float(split_y)) <= float(band)


def crosses_split(prev_y: float, new_y: float, split_y: Optional[float]) -> bool:
    if split_y is None:
        return False
    return (prev_y <= split_y <= new_y) or (new_y <= split_y <= prev_y)


def is_inside_bounds(x_value: float, y_value: float, bounds, margin: float = 0.0) -> bool:
    if bounds is None:
        return True
    left, right, top, bottom = bounds
    return (
        left - margin <= float(x_value) <= right + margin
        and top - margin <= float(y_value) <= bottom + margin
    )


def is_near_court_border(detection: Any, bounds, margin: float = SPAWN_BORDER_MARGIN) -> bool:
    if bounds is None:
        return True
    left, right, top, bottom = bounds
    return (
        abs(detection.x - left) <= margin
        or abs(detection.x - right) <= margin
        or abs(detection.y - top) <= margin
        or abs(detection.y - bottom) <= margin
    )


def normalize_detection_for_court(detection: Any, bounds, split_y: Optional[float]) -> Optional[Any]:
    if detection.on_terrain:
        detection.spawn_allowed = True
        return detection
    if not is_inside_bounds(detection.x, detection.y, bounds, margin=COURT_MARGIN):
        return None
    if split_y is not None and not is_near_split(detection.y, split_y, SPLIT_HANDOFF_BAND + 20.0):
        return None
    detection.status = "geometry_off_terrain"
    detection.spawn_allowed = True
    return detection


def allow_camera_handoff(
    previous_cam: str,
    new_cam: str,
    prev_y: float,
    new_y: float,
    split_y: Optional[float],
    *,
    frame_gap: int,
    prev_on_terrain: bool,
    new_on_terrain: bool,
) -> bool:
    normalized_previous = normalize_cam_name(previous_cam)
    normalized_new = normalize_cam_name(new_cam)
    if not normalized_previous or not normalized_new:
        return True
    if normalized_previous == normalized_new:
        return True
    if "+" in normalized_previous or "+" in normalized_new:
        return True
    if not prev_on_terrain or not new_on_terrain:
        return False
    dynamic_band = SPLIT_HANDOFF_BAND + max(0, frame_gap - 1) * SPLIT_HANDOFF_EXTRA_PER_FRAME
    if crosses_split(prev_y, new_y, split_y):
        return True
    return is_near_split(prev_y, split_y, dynamic_band) and is_near_split(new_y, split_y, dynamic_band)
