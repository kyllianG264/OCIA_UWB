import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

GRAY_ZONE_GRID_STEP = 8.0
GRAY_ZONE_SPLIT_MARGIN = 54.0
GRAY_ZONE_CENTER_HALF_WIDTH_RATIO = 0.24
GRAY_ZONE_OVERLAP_BONUS = 0.22
GRAY_ZONE_RISK_THRESHOLD = 0.72
DISAPPEARANCE_MIN_TRACK_FRAMES = 5
HANDOFF_SPLIT_MARGIN = 100.0
HANDOFF_CENTER_HALF_WIDTH = 130.0
HANDOFF_MATCH_DISTANCE = 120.0
HANDOFF_FRAME_WINDOW = 5
HANDOFF_DISPLAY_POINT_COUNT = 1200
HANDOFF_FRONTIER_HISTORY = 12
HANDOFF_FRONTIER_ANGLE_BINS = 720
HANDOFF_FRONTIER_MIN_POINTS = 24
HANDOFF_FRONTIER_BAND_GAP = 12.0


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    x_value, y_value = point
    inside = False
    if len(polygon) < 3:
        return False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        intersects = ((current_y > y_value) != (previous_y > y_value)) and (
            x_value < (previous_x - current_x) * (y_value - current_y) / max(previous_y - current_y, 1e-9) + current_x
        )
        if intersects:
            inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _distance_point_to_segment(point: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float]) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    ratio = ((px - ax) * abx + (py - ay) * aby) / denom
    ratio = max(0.0, min(1.0, ratio))
    closest_x = ax + ratio * abx
    closest_y = ay + ratio * aby
    return math.hypot(px - closest_x, py - closest_y)


def _distance_to_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> float:
    if len(polygon) < 2:
        return float("inf")
    best = float("inf")
    previous = polygon[-1]
    for current in polygon:
        best = min(best, _distance_point_to_segment(point, previous, current))
        previous = current
    return best


def _project_terrain_to_view(inv_h: np.ndarray, x_value: float, y_value: float) -> Optional[Tuple[float, float]]:
    projected = inv_h @ np.array([x_value, y_value, 1.0], dtype=float)
    w_value = float(projected[2])
    if abs(w_value) <= 1e-9:
        return None
    return float(projected[0] / w_value), float(projected[1] / w_value)


def _corrected_view_rect(camera_data: dict) -> Tuple[float, float, float, float]:
    distortion = camera_data.get("distortion") or {}
    view = camera_data.get("undistort_view") or {}
    width = _safe_float(view.get("width"), distortion.get("width", 1920.0))
    height = _safe_float(view.get("height"), distortion.get("height", 1080.0))
    scale = _safe_float(view.get("scale"), 1.0)
    offset_x = _safe_float(view.get("offset_x"), 0.0)
    offset_y = _safe_float(view.get("offset_y"), 0.0)
    return (
        offset_x,
        offset_x + width * scale,
        offset_y,
        offset_y + height * scale,
    )


def _view_to_undistorted(view_point: Tuple[float, float], camera_data: dict) -> Tuple[float, float]:
    view = camera_data.get("undistort_view") or {}
    scale = _safe_float(view.get("scale"), 1.0)
    offset_x = _safe_float(view.get("offset_x"), 0.0)
    offset_y = _safe_float(view.get("offset_y"), 0.0)
    if abs(scale) <= 1e-9:
        return view_point
    return (
        (float(view_point[0]) - offset_x) / scale,
        (float(view_point[1]) - offset_y) / scale,
    )


def _distortion_risk(view_point: Tuple[float, float], camera_data: dict) -> float:
    distortion = camera_data.get("distortion") or {}
    if not distortion.get("enabled"):
        return 0.0
    undistorted_x, undistorted_y = _view_to_undistorted(view_point, camera_data)
    width = max(_safe_float(distortion.get("width"), 1.0), 1.0)
    height = max(_safe_float(distortion.get("height"), 1.0), 1.0)
    cx = _safe_float(distortion.get("cx"), width / 2.0)
    cy = _safe_float(distortion.get("cy"), height / 2.0)
    scale = max(width, height, 1.0)
    radius = math.hypot((undistorted_x - cx) / scale, (undistorted_y - cy) / scale)
    if radius <= 0.25:
        return 0.0
    return min(0.42, max(0.0, radius - 0.25) * 0.65)


def _camera_configs(calibration_data: dict) -> List[dict]:
    cameras = []
    for side_key, legacy_key in (("cam_gauche", "H_g"), ("cam_droite", "H_d")):
        camera_data = dict(calibration_data.get(side_key) or {})
        h_value = camera_data.get("H") or calibration_data.get(legacy_key)
        frame_corners = camera_data.get("frame_corners") or []
        if not h_value or len(frame_corners) < 4:
            continue
        cameras.append(
            {
                "name": side_key,
                "inv_h": np.linalg.inv(np.array(h_value, dtype=float)),
                "frame_polygon": [(float(px), float(py)) for px, py in frame_corners],
                "distortion": camera_data.get("distortion") or calibration_data.get(
                    "distortion_g" if side_key.endswith("gauche") else "distortion_d"
                ),
                "undistort_view": camera_data.get("undistort_view") or calibration_data.get(
                    "undistort_view_g" if side_key.endswith("gauche") else "undistort_view_d"
                ),
            }
        )
    return cameras


def _grid_index(x_value: float, y_value: float, bounds: Tuple[float, float, float, float], step: float, shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    left, right, top, bottom = bounds
    if x_value < left or x_value > right or y_value < top or y_value > bottom:
        return None
    col = int((x_value - left) / step)
    row = int((y_value - top) / step)
    if row < 0 or row >= shape[0] or col < 0 or col >= shape[1]:
        return None
    return row, col


def _mask_to_polygons(mask: np.ndarray, bounds: Tuple[float, float, float, float], step: float) -> List[List[Tuple[float, float]]]:
    polygons: List[List[Tuple[float, float]]] = []
    left, _right, top, _bottom = bounds
    for row in range(mask.shape[0]):
        col = 0
        while col < mask.shape[1]:
            if not mask[row, col]:
                col += 1
                continue
            start_col = col
            while col + 1 < mask.shape[1] and mask[row, col + 1]:
                col += 1
            end_col = col
            x0 = left + start_col * step
            x1 = left + (end_col + 1) * step
            y0 = top + row * step
            y1 = top + (row + 1) * step
            polygons.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            col += 1
    return polygons


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    expanded = mask.copy()
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row, col]:
                continue
            row_start = max(0, row - 1)
            row_end = min(mask.shape[0], row + 2)
            col_start = max(0, col - 1)
            col_end = min(mask.shape[1], col + 2)
            if np.any(mask[row_start:row_end, col_start:col_end]):
                expanded[row, col] = True
    return expanded


def _fit_handoff_ellipse(event_points, step):
    if not event_points:
        return [], None
    points = np.array([point["scene_point"] for point in event_points], dtype=float)
    median = np.median(points, axis=0)
    limits = np.maximum(np.percentile(np.abs(points - median), 90, axis=0), float(step) * 2.0)
    retained_mask = np.all(np.abs(points - median) <= limits, axis=1)
    retained_points = points[retained_mask]
    retained_events = [point for point, retained in zip(event_points, retained_mask) if retained]
    if not len(retained_points):
        retained_points = points
        retained_events = list(event_points)
    center = np.mean(retained_points, axis=0)
    if len(retained_points) >= 2:
        covariance = np.cov(retained_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, order]
    else:
        axes = np.eye(2)
    projections = (retained_points - center) @ axes
    radii = np.maximum(np.percentile(np.abs(projections), 90, axis=0), float(step) * 2.0)
    angle_degrees = math.degrees(math.atan2(float(axes[1, 0]), float(axes[0, 0])))
    ellipse = {
        "center": [float(center[0]), float(center[1])],
        "radius_x": float(radii[0]),
        "radius_y": float(radii[1]),
        "angle_degrees": angle_degrees,
        "event_count": len(event_points),
        "retained_event_count": len(retained_events),
        "confidence": min(1.0, len(retained_events) / 500.0) * (len(retained_events) / len(event_points)),
    }
    return retained_events, ellipse


def _ellipse_axes(ellipse):
    angle = math.radians(float(ellipse.get("angle_degrees", 0.0)))
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return cos_angle, sin_angle


def _keep_inner_loss_frontier_events(event_points, court_center):
    if len(event_points) < HANDOFF_FRONTIER_MIN_POINTS:
        return list(event_points)

    center_x, center_y = court_center
    grouped_events = {}
    for event in event_points:
        scene_point = event.get("scene_point") or [center_x, center_y]
        dx = float(scene_point[0]) - center_x
        dy = float(scene_point[1]) - center_y
        angle = math.atan2(dy, dx)
        bin_index = int(((angle + math.pi) / (2.0 * math.pi)) * HANDOFF_FRONTIER_ANGLE_BINS)
        bin_index = max(0, min(HANDOFF_FRONTIER_ANGLE_BINS - 1, bin_index))
        distance = math.hypot(dx, dy)
        if dy < -HANDOFF_FRONTIER_BAND_GAP:
            band = "top"
        elif dy > HANDOFF_FRONTIER_BAND_GAP:
            band = "bottom"
        else:
            band = "center"
        grouped_events.setdefault((band, bin_index), []).append((distance, event))

    frontier_events = []
    for (band, _bin_index), items in grouped_events.items():
        if not items:
            continue
        _distance, event = min(items, key=lambda item: item[0])
        enriched_event = dict(event)
        enriched_event["frontier_band"] = band
        frontier_events.append(enriched_event)

    if len(frontier_events) < HANDOFF_FRONTIER_MIN_POINTS:
        return list(event_points)
    return frontier_events


def _ellipse_theory_overlap(ellipse, mask, bounds, step):
    if not ellipse:
        return 0.0
    cos_angle, sin_angle = _ellipse_axes(ellipse)
    center_x, center_y = ellipse["center"]
    radius_x = max(float(ellipse["radius_x"]), 1.0)
    radius_y = max(float(ellipse["radius_y"]), 1.0)
    inside_count = 0
    overlap_count = 0
    for row in range(mask.shape[0]):
        y_value = bounds[2] + (row + 0.5) * step
        for col in range(mask.shape[1]):
            x_value = bounds[0] + (col + 0.5) * step
            dx = x_value - center_x
            dy = y_value - center_y
            axis_x = dx * cos_angle + dy * sin_angle
            axis_y = -dx * sin_angle + dy * cos_angle
            if (axis_x / radius_x) ** 2 + (axis_y / radius_y) ** 2 > 1.0:
                continue
            inside_count += 1
            if mask[row, col]:
                overlap_count += 1
    return 0.0 if inside_count == 0 else overlap_count / inside_count


def _handoff_loss_zone(raw_rows, bounds, split_y, step, shape):
    active = {}
    pending = []
    camera_history = {"gauche": [], "droite": []}
    current_frame = None
    current_rows = []
    observation_count = 0
    event_points = []
    events_by_direction = {}
    center_x = (bounds[0] + bounds[1]) / 2.0

    def predicted_position(candidate, target_frame):
        delta = float(target_frame - candidate["frame"])
        return (
            candidate["point"][0] + candidate["velocity"][0] * delta,
            candidate["point"][1] + candidate["velocity"][1] * delta,
        )

    def best_match(candidate, observations):
        best = None
        for observation_frame, row in observations:
            if abs(observation_frame - candidate["frame"]) > HANDOFF_FRAME_WINDOW:
                continue
            predicted_x, predicted_y = predicted_position(candidate, observation_frame)
            distance = math.hypot(
                _safe_float(row.get("X")) - predicted_x,
                _safe_float(row.get("Y")) - predicted_y,
            )
            if best is None or distance < best[0]:
                best = (distance, observation_frame)
        return best

    def row_point(row):
        return (_safe_float(row.get("X")), _safe_float(row.get("Y")))

    def row_frame(row):
        return int(_safe_float(row.get("frame")))

    def select_frontier_row(history):
        if not history:
            return None
        if len(history) < 2:
            return history[-1]
        first_x, first_y = row_point(history[0])
        last_x, last_y = row_point(history[-1])
        direction_x = last_x - first_x
        direction_y = last_y - first_y
        norm = math.hypot(direction_x, direction_y)
        if norm <= 1e-9:
            return history[-1]
        direction_x /= norm
        direction_y /= norm
        best_row = history[-1]
        best_projection = -float("inf")
        best_frame = row_frame(best_row)
        for row in history:
            x_value, y_value = row_point(row)
            projection = (x_value - first_x) * direction_x + (y_value - first_y) * direction_y
            frame_value = row_frame(row)
            if projection > best_projection or (
                math.isclose(projection, best_projection, abs_tol=1e-9) and frame_value > best_frame
            ):
                best_projection = projection
                best_frame = frame_value
                best_row = row
        return best_row

    def accept_candidate(candidate, match):
        direction = f"{candidate['camera']}_to_{candidate['opposite_camera']}"
        event_points.append(
            {
                "scene_point": list(candidate["point"]),
                "cam": candidate["camera"],
                "frame": candidate["frame"],
                "matched_frame": match[1],
                "match_distance": match[0],
                "direction": direction,
                "event_type": "handoff_loss",
            }
        )
        events_by_direction[direction] = events_by_direction.get(direction, 0) + 1

    def process_frame(frame, rows):
        seen = {}
        by_camera = {"gauche": [], "droite": []}
        for row in rows:
            camera = str(row.get("cam", "")).strip().lower()
            player_id = str(row.get("player_id", "")).strip()
            if camera and player_id:
                seen[(camera, player_id)] = row
                if camera in by_camera:
                    by_camera[camera].append((frame, row))
        for camera in camera_history:
            camera_history[camera].extend(by_camera[camera])
            camera_history[camera] = [
                item for item in camera_history[camera] if frame - item[0] <= HANDOFF_FRAME_WINDOW
            ]

        unresolved = []
        for candidate in pending:
            match = best_match(candidate, by_camera[candidate["opposite_camera"]])
            if match is not None and match[0] <= HANDOFF_MATCH_DISTANCE:
                accept_candidate(candidate, match)
            elif frame - candidate["frame"] <= HANDOFF_FRAME_WINDOW:
                unresolved.append(candidate)
        pending[:] = unresolved

        for key, track_state in list(active.items()):
            if key in seen:
                continue
            active.pop(key, None)
            if track_state["streak"] < DISAPPEARANCE_MIN_TRACK_FRAMES:
                continue
            previous = select_frontier_row(track_state.get("history") or [track_state["row"]])
            if previous is None:
                continue
            if str(previous.get("on_terrain", "1")).strip().lower() in {"0", "false", "no", "off"}:
                continue
            x_value = _safe_float(previous.get("X"))
            y_value = _safe_float(previous.get("Y"))
            if _grid_index(x_value, y_value, bounds, step, shape) is None:
                continue
            if split_y is None or abs(y_value - float(split_y)) > HANDOFF_SPLIT_MARGIN:
                continue
            if abs(x_value - center_x) > HANDOFF_CENTER_HALF_WIDTH:
                continue
            history = track_state.get("history") or []
            previous_index = history.index(previous) if previous in history else len(history) - 1
            previous_row = history[previous_index - 1] if previous_index > 0 else track_state.get("previous_row")
            if previous_row is None:
                velocity = (0.0, 0.0)
            else:
                frame_delta = max(1.0, _safe_float(previous.get("frame")) - _safe_float(previous_row.get("frame")))
                velocity = (
                    (_safe_float(previous.get("X")) - _safe_float(previous_row.get("X"))) / frame_delta,
                    (_safe_float(previous.get("Y")) - _safe_float(previous_row.get("Y"))) / frame_delta,
                )
            opposite_camera = "droite" if key[0] == "gauche" else "gauche"
            candidate = {
                "camera": key[0],
                "opposite_camera": opposite_camera,
                "frame": int(_safe_float(previous.get("frame"))),
                "point": (x_value, y_value),
                "velocity": velocity,
            }
            match = best_match(candidate, camera_history[opposite_camera])
            if match is not None and match[0] <= HANDOFF_MATCH_DISTANCE:
                accept_candidate(candidate, match)
            else:
                pending.append(candidate)

        for key, row in seen.items():
            previous_state = active.get(key)
            history = [] if previous_state is None else list(previous_state.get("history") or [previous_state["row"]])
            history.append(row)
            history = history[-HANDOFF_FRONTIER_HISTORY:]
            active[key] = {
                "row": row,
                "previous_row": None if previous_state is None else previous_state["row"],
                "history": history,
                "streak": 1 if previous_state is None else previous_state["streak"] + 1,
            }

    for row in raw_rows or ():
        observation_count += 1
        frame = int(_safe_float(row.get("frame")))
        if current_frame is None:
            current_frame = frame
        if frame != current_frame:
            process_frame(current_frame, current_rows)
            current_rows = []
            current_frame = frame
        current_rows.append(row)
    if current_rows:
        process_frame(current_frame, current_rows)

    court_center = ((bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0)
    frontier_events = _keep_inner_loss_frontier_events(event_points, court_center)
    retained_events = list(frontier_events)
    _ellipse_events, ellipse = _fit_handoff_ellipse(retained_events, step)
    sample_step = max(1, math.ceil(len(retained_events) / HANDOFF_DISPLAY_POINT_COUNT))
    display_points = retained_events[::sample_step][:HANDOFF_DISPLAY_POINT_COUNT]
    retained_keys = {
        (
            point.get("cam"),
            point.get("direction"),
            int(point.get("frame", -1)),
            round(float(point.get("scene_point", [0.0, 0.0])[0]), 3),
            round(float(point.get("scene_point", [0.0, 0.0])[1]), 3),
        )
        for point in retained_events
    }
    rejected_events = [
        point
        for point in event_points
        if (
            point.get("cam"),
            point.get("direction"),
            int(point.get("frame", -1)),
            round(float(point.get("scene_point", [0.0, 0.0])[0]), 3),
            round(float(point.get("scene_point", [0.0, 0.0])[1]), 3),
        )
        not in retained_keys
    ]
    disappearance_sample_step = max(1, math.ceil(len(event_points) / HANDOFF_DISPLAY_POINT_COUNT))
    disappearance_display_points = event_points[::disappearance_sample_step][:HANDOFF_DISPLAY_POINT_COUNT]
    rejected_sample_step = max(1, math.ceil(len(rejected_events) / HANDOFF_DISPLAY_POINT_COUNT))
    rejected_display_points = rejected_events[::rejected_sample_step][:HANDOFF_DISPLAY_POINT_COUNT]
    return display_points, disappearance_display_points, rejected_display_points, ellipse, {
        "raw_observation_count": observation_count,
        "disappearance_event_count": len(event_points),
        "handoff_event_count": len(event_points),
        "handoff_retained_event_count": len(retained_events),
        "handoff_rejected_event_count": len(rejected_events),
        "handoff_frontier_event_count": len(frontier_events),
        "handoff_events_by_direction": events_by_direction,
        "disappearance_min_track_frames": DISAPPEARANCE_MIN_TRACK_FRAMES,
        "handoff_frame_window": HANDOFF_FRAME_WINDOW,
        "handoff_split_margin": HANDOFF_SPLIT_MARGIN,
        "handoff_center_half_width": HANDOFF_CENTER_HALF_WIDTH,
        "handoff_match_distance": HANDOFF_MATCH_DISTANCE,
        "handoff_frontier_history": HANDOFF_FRONTIER_HISTORY,
        "handoff_frontier_angle_bins": HANDOFF_FRONTIER_ANGLE_BINS,
        "handoff_frontier_band_gap": HANDOFF_FRONTIER_BAND_GAP,
        "handoff_frontier_rule": "first_loss_from_court_center_by_band_and_angle",
    }


@dataclass
class GrayZone:
    bounds: Tuple[float, float, float, float]
    split_y: Optional[float]
    grid_step: float
    gray_zone_mask: np.ndarray
    gray_zone_polygons: List[List[Tuple[float, float]]]
    risk_map: np.ndarray
    metadata: dict
    disappearance_cells: List[dict] = field(default_factory=list)
    handoff_points: List[dict] = field(default_factory=list)
    disappearance_points: List[dict] = field(default_factory=list)
    handoff_rejected_points: List[dict] = field(default_factory=list)
    handoff_ellipse: Optional[dict] = None

    def is_in_gray_zone(self, x_value: float, y_value: float) -> bool:
        index = _grid_index(float(x_value), float(y_value), self.bounds, self.grid_step, self.gray_zone_mask.shape)
        return bool(index is not None and self.gray_zone_mask[index[0], index[1]])


def compute_gray_zone(
    calibration: Optional[dict],
    calibration_data: Optional[dict],
    *,
    raw_rows: Optional[Iterable[dict]] = None,
    grid_step: float = GRAY_ZONE_GRID_STEP,
) -> GrayZone:
    if calibration is None or calibration.get("bounds") is None:
        empty_mask = np.zeros((1, 1), dtype=bool)
        return GrayZone(
            bounds=(0.0, 0.0, 0.0, 0.0),
            split_y=None,
            grid_step=float(grid_step),
            gray_zone_mask=empty_mask,
            gray_zone_polygons=[],
            risk_map=np.zeros((1, 1), dtype=float),
            metadata={"reason": "missing_calibration", "raw_observation_count": 0},
        )

    cameras = _camera_configs(calibration_data or {})
    bounds = tuple(float(value) for value in calibration["bounds"])
    split_y = calibration.get("split_y")
    center_x = (bounds[0] + bounds[1]) / 2.0
    center_half_width = max(grid_step * 2.0, (bounds[1] - bounds[0]) * GRAY_ZONE_CENTER_HALF_WIDTH_RATIO)
    width = max(1, int(math.ceil((bounds[1] - bounds[0]) / grid_step)))
    height = max(1, int(math.ceil((bounds[3] - bounds[2]) / grid_step)))
    risk_map = np.zeros((height, width), dtype=float)

    for row in range(height):
        y_value = bounds[2] + (row + 0.5) * grid_step
        for col in range(width):
            x_value = bounds[0] + (col + 0.5) * grid_step
            cell_risk = 0.0
            split_proximity = 0.0
            if split_y is not None:
                split_distance = abs(y_value - float(split_y))
                if split_distance <= GRAY_ZONE_SPLIT_MARGIN:
                    split_proximity = 1.0 - (split_distance / GRAY_ZONE_SPLIT_MARGIN)
            center_distance = abs(x_value - center_x)
            center_proximity = 0.0
            if center_distance <= center_half_width:
                center_proximity = 1.0 - (center_distance / center_half_width)

            visible_camera_count = 0
            distortion_risk = 0.0
            for camera in cameras:
                view_point = _project_terrain_to_view(camera["inv_h"], x_value, y_value)
                if view_point is None:
                    continue
                image_left, image_right, image_top, image_bottom = _corrected_view_rect(camera)
                if not (
                    image_left <= view_point[0] <= image_right
                    and image_top <= view_point[1] <= image_bottom
                ):
                    continue
                if not _point_in_polygon(view_point, camera["frame_polygon"]):
                    continue
                visible_camera_count += 1
                distortion_risk = max(distortion_risk, _distortion_risk(view_point, camera))

            # The gray zone is meant to model the central camera handoff corridor,
            # not every unreliable edge around the court.
            if split_proximity > 0.0 and center_proximity > 0.0:
                cell_risk += 0.88 * split_proximity * center_proximity
                if visible_camera_count >= 2:
                    cell_risk += GRAY_ZONE_OVERLAP_BONUS * split_proximity
                cell_risk += distortion_risk * 0.35
            risk_map[row, col] = cell_risk

    gray_zone_mask = risk_map >= GRAY_ZONE_RISK_THRESHOLD
    gray_zone_mask = _dilate_mask(gray_zone_mask)
    gray_zone_polygons = _mask_to_polygons(gray_zone_mask, bounds, float(grid_step))
    handoff_points, disappearance_points, handoff_rejected_points, handoff_ellipse, raw_metadata = _handoff_loss_zone(
        raw_rows,
        bounds,
        split_y,
        float(grid_step),
        gray_zone_mask.shape,
    )
    theory_overlap = _ellipse_theory_overlap(
        handoff_ellipse,
        gray_zone_mask,
        bounds,
        float(grid_step),
    )
    return GrayZone(
        bounds=bounds,
        split_y=None if split_y is None else float(split_y),
        grid_step=float(grid_step),
        gray_zone_mask=gray_zone_mask,
        gray_zone_polygons=gray_zone_polygons,
        risk_map=risk_map,
        metadata={
            "camera_count": len(cameras),
            "source": "camera_visibility",
            **raw_metadata,
            "handoff_theory_overlap_ratio": theory_overlap,
        },
        handoff_points=handoff_points,
        disappearance_points=disappearance_points,
        handoff_rejected_points=handoff_rejected_points,
        handoff_ellipse=handoff_ellipse,
    )
