import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from solver_lps.features.ground.domain.calibration import load_calibration_geometry


_APP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
_DEFAULT_CV_LOG_DIR = os.path.join(_APP_ROOT, "features", "cv", "review", "data", "cv_logs")

DEFAULT_INPUT = os.path.join(_DEFAULT_CV_LOG_DIR, "positions_raw.csv")
DEFAULT_OUTPUT = os.path.join(_DEFAULT_CV_LOG_DIR, "positions_stable_hungarian.csv")
DEFAULT_CALIBRATION = os.path.join(_DEFAULT_CV_LOG_DIR, "calibration.json")
DISPLAY_SMOOTH_ALPHA = 0.24
DISPLAY_TENTATIVE_ALPHA = 0.34
DISPLAY_SMOOTH_DEADZONE = 3.0
DISPLAY_SNAP_DISTANCE = 90.0
VELOCITY_DAMPING = 0.90
SPLIT_HANDOFF_BAND = 70.0
SPLIT_HANDOFF_EXTRA_PER_FRAME = 8.0
RAW_ID_ALIAS_OFFSET = 30
COURT_MARGIN = 45.0
SPAWN_BORDER_MARGIN = 70.0
SPAWN_WARMUP_SECONDS = 5.0
DUPLICATE_SUPPRESSION_DISTANCE = 42.0
OUTPUT_DEDUPE_DISTANCE = 36.0
ACTIVE_SPEED_GATE_PX_PER_FRAME = 18.0
REATTACH_SPEED_GATE_PX_PER_FRAME = 22.0
ALIAS_REATTACH_MAX_FRAME_GAP = 360
ALIAS_REATTACH_MAX_DISTANCE = 700.0
CROSSING_SPEED_MIN_PX_PER_FRAME = 10.0
CROSSING_MATCH_MARGIN = 45.0
CROSSING_MAX_DISTANCE = 520.0


def linear_sum_assignment(cost_matrix: np.ndarray):
    rows, cols = cost_matrix.shape
    transposed = False
    matrix = cost_matrix.tolist()
    if rows > cols:
        transposed = True
        matrix = cost_matrix.T.tolist()
        rows, cols = cols, rows

    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)
    way = [0] * (cols + 1)

    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (cols + 1)
        used = [False] * (cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_ind = []
    col_ind = []
    for j in range(1, cols + 1):
        if p[j] == 0:
            continue
        row = p[j] - 1
        col = j - 1
        if transposed:
            row, col = col, row
        row_ind.append(row)
        col_ind.append(col)
    return np.array(row_ind, dtype=int), np.array(col_ind, dtype=int)


@dataclass
class Detection:
    frame: int
    timestamp_s: float
    timestamp_unix: str
    raw_player_id: str
    x: float
    y: float
    cam: str
    on_terrain: bool
    source_ids: List[str] = field(default_factory=list)
    source_cams: List[str] = field(default_factory=list)
    merged_count: int = 1
    status: str = "ok"
    primary_cam: str = ""
    spawn_allowed: bool = True


@dataclass
class Track:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    last_frame: int
    last_timestamp_s: float
    hits: int = 1
    misses: int = 0
    age: int = 1
    confirmed: bool = False
    last_raw_player_id: str = ""
    last_cam: str = ""
    last_on_terrain: bool = True
    confidence: float = 0.2
    display_pos: Optional[Tuple[float, float]] = None

    def stable_id(self) -> str:
        return f"T{self.track_id:03d}"


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Track stable CV IDs from a raw CSV using Kalman filtering and Hungarian assignment."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION, help="Calibration JSON used to recover the midcourt split.")
    parser.add_argument("--merge-distance", type=float, default=35.0, help="Max distance to merge two camera detections.")
    parser.add_argument("--match-distance", type=float, default=70.0, help="Base max distance to match a detection to a live track.")
    parser.add_argument("--distance-per-frame", type=float, default=18.0, help="Extra allowed distance per missing frame.")
    parser.add_argument("--reattach-distance", type=float, default=240.0, help="Max distance to reattach an older lost track.")
    parser.add_argument("--max-misses", type=int, default=15, help="Frames before a live track becomes lost.")
    parser.add_argument("--max-reattach-gap", type=int, default=120, help="Max frame gap allowed for lost track reattachment.")
    parser.add_argument("--min-hits", type=int, default=3, help="Observations needed before a track is confirmed.")
    parser.add_argument("--expected-players", type=int, default=10, help="Max confirmed players to output per frame.")
    return parser.parse_args(argv)


def default_config(input_path: Optional[str] = None, output_path: Optional[str] = None, **overrides):
    return argparse.Namespace(
        input=input_path or DEFAULT_INPUT,
        output=output_path or DEFAULT_OUTPUT,
        calibration=str(overrides.get("calibration", DEFAULT_CALIBRATION)),
        merge_distance=float(overrides.get("merge_distance", 35.0)),
        match_distance=float(overrides.get("match_distance", 70.0)),
        distance_per_frame=float(overrides.get("distance_per_frame", 18.0)),
        reattach_distance=float(overrides.get("reattach_distance", 240.0)),
        max_misses=int(overrides.get("max_misses", 15)),
        max_reattach_gap=int(overrides.get("max_reattach_gap", 120)),
        min_hits=int(overrides.get("min_hits", 3)),
        expected_players=int(overrides.get("expected_players", 10)),
    )


def to_bool(value: str) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", ""}


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def smooth_display_position(
    current: Optional[Tuple[float, float]],
    target: Tuple[float, float],
    *,
    confirmed: bool,
) -> Tuple[float, float]:
    if current is None:
        return target
    distance = euclidean(current, target)
    if distance <= DISPLAY_SMOOTH_DEADZONE:
        return current
    if distance >= DISPLAY_SNAP_DISTANCE:
        return target
    alpha = DISPLAY_SMOOTH_ALPHA if confirmed else DISPLAY_TENTATIVE_ALPHA
    return (
        current[0] + (target[0] - current[0]) * alpha,
        current[1] + (target[1] - current[1]) * alpha,
    )


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


def is_near_court_border(detection: Detection, bounds, margin: float = SPAWN_BORDER_MARGIN) -> bool:
    if bounds is None:
        return True
    left, right, top, bottom = bounds
    return (
        abs(detection.x - left) <= margin
        or abs(detection.x - right) <= margin
        or abs(detection.y - top) <= margin
        or abs(detection.y - bottom) <= margin
    )


def normalize_detection_for_court(detection: Detection, bounds, split_y: Optional[float]) -> Optional[Detection]:
    if detection.on_terrain:
        detection.spawn_allowed = True
        return detection
    if not is_inside_bounds(detection.x, detection.y, bounds, margin=COURT_MARGIN):
        return None
    if split_y is not None and not is_near_split(detection.y, split_y, SPLIT_HANDOFF_BAND + 20.0):
        return None
    detection.on_terrain = True
    detection.status = "geometry_on_terrain"
    detection.spawn_allowed = False
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


def load_frames(csv_path: str) -> List[List[Detection]]:
    grouped: Dict[int, List[Detection]] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            frame = int(float(row["frame"]))
            detection = Detection(
                frame=frame,
                timestamp_s=float(row.get("timestamp_s") or 0.0),
                timestamp_unix=str(row.get("timestamp_unix") or ""),
                raw_player_id=str(row.get("player_id") or "").strip() or "player",
                x=float(row["X"]),
                y=float(row["Y"]),
                cam=str(row.get("cam") or row.get("demi_terrain") or "").strip(),
                on_terrain=to_bool(row.get("on_terrain", "1")),
            )
            detection.primary_cam = detection.cam
            detection.source_ids = [detection.raw_player_id]
            detection.source_cams = [detection.cam] if detection.cam else []
            grouped.setdefault(frame, []).append(detection)
    return [grouped[frame] for frame in sorted(grouped)]


def merge_camera_duplicates(detections: List[Detection], merge_distance: float, split_y: Optional[float]) -> List[Detection]:
    merged: List[Detection] = []
    used = set()
    for index, detection in enumerate(detections):
        if index in used:
            continue
        cluster = [detection]
        used.add(index)
        for other_index in range(index + 1, len(detections)):
            if other_index in used:
                continue
            other = detections[other_index]
            if detection.cam == other.cam:
                continue
            if detection.on_terrain != other.on_terrain:
                continue
            if detection.on_terrain and not (
                is_near_split(detection.y, split_y, SPLIT_HANDOFF_BAND)
                and is_near_split(other.y, split_y, SPLIT_HANDOFF_BAND)
            ):
                continue
            if euclidean((detection.x, detection.y), (other.x, other.y)) > merge_distance:
                continue
            cluster.append(other)
            used.add(other_index)
        if len(cluster) == 1:
            merged.append(detection)
            continue
        avg_x = sum(item.x for item in cluster) / len(cluster)
        avg_y = sum(item.y for item in cluster) / len(cluster)
        merged.append(
            Detection(
                frame=detection.frame,
                timestamp_s=detection.timestamp_s,
                timestamp_unix=detection.timestamp_unix,
                raw_player_id=detection.raw_player_id,
                x=avg_x,
                y=avg_y,
                cam="+".join(sorted({item.cam for item in cluster if item.cam})),
                on_terrain=detection.on_terrain,
                source_ids=sorted({sid for item in cluster for sid in item.source_ids}),
                source_cams=sorted({cam for item in cluster for cam in item.source_cams}),
                merged_count=len(cluster),
                status="merged",
                spawn_allowed=any(item.spawn_allowed for item in cluster),
            )
        )
    return merged


def create_filter_state(x: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
    state = np.array([[x], [y], [0.0], [0.0]], dtype=float)
    covariance = np.diag([150.0, 150.0, 80.0, 80.0]).astype(float)
    return state, covariance


def kalman_predict(track: Track, frame_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    dt = float(max(1, frame_gap))
    transition = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    process_noise = np.array(
        [
            [4.0 * dt, 0.0, 0.0, 0.0],
            [0.0, 4.0 * dt, 0.0, 0.0],
            [0.0, 0.0, 2.5 * dt, 0.0],
            [0.0, 0.0, 0.0, 2.5 * dt],
        ],
        dtype=float,
    )
    predicted_state = transition @ track.state
    predicted_covariance = transition @ track.covariance @ transition.T + process_noise
    return predicted_state, predicted_covariance


def kalman_update(predicted_state: np.ndarray, predicted_covariance: np.ndarray, detection: Detection):
    measurement = np.array([[detection.x], [detection.y]], dtype=float)
    observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
    measurement_noise = np.diag([30.0, 30.0]).astype(float)
    innovation = measurement - observation @ predicted_state
    innovation_covariance = observation @ predicted_covariance @ observation.T + measurement_noise
    kalman_gain = predicted_covariance @ observation.T @ np.linalg.inv(innovation_covariance)
    updated_state = predicted_state + kalman_gain @ innovation
    updated_covariance = (np.eye(4) - kalman_gain @ observation) @ predicted_covariance
    residual_distance = math.hypot(float(innovation[0, 0]), float(innovation[1, 0]))
    return updated_state, updated_covariance, residual_distance


def track_confidence(track: Track, residual_distance: float) -> float:
    confidence = 0.0
    confidence += min(0.45, track.hits * 0.12)
    confidence += min(0.20, track.age * 0.02)
    confidence -= min(0.25, track.misses * 0.08)
    confidence -= min(0.20, residual_distance / 400.0)
    if track.confirmed:
        confidence += 0.12
    return max(0.0, min(1.0, confidence))


def refresh_display_state(track: Track):
    filtered_position = (float(track.state[0, 0]), float(track.state[1, 0]))
    track.display_pos = smooth_display_position(track.display_pos, filtered_position, confirmed=track.confirmed)
    track.state[2, 0] *= VELOCITY_DAMPING
    track.state[3, 0] *= VELOCITY_DAMPING


def gating_distance(frame_gap: int, base_distance: float, distance_per_frame: float) -> float:
    return base_distance + max(0, frame_gap - 1) * distance_per_frame


def track_speed(track: Track) -> float:
    return math.hypot(float(track.state[2, 0]), float(track.state[3, 0]))


def crossing_motion_budget(
    track: Track,
    detection: Detection,
    predicted_state: np.ndarray,
    frame_gap: int,
    args,
) -> Optional[float]:
    split_y = getattr(args, "split_y", None)
    if split_y is None or frame_gap <= 1:
        return None
    previous_xy = latest_track_position(track)
    predicted_xy = (float(predicted_state[0, 0]), float(predicted_state[1, 0]))
    if not (
        crosses_split(previous_xy[1], predicted_xy[1], split_y)
        or crosses_split(previous_xy[1], detection.y, split_y)
        or (
            is_near_split(previous_xy[1], split_y, SPLIT_HANDOFF_BAND)
            and detection.on_terrain
        )
    ):
        return None
    speed = max(
        track_speed(track),
        euclidean(previous_xy, predicted_xy) / max(1, frame_gap),
    )
    if speed < CROSSING_SPEED_MIN_PX_PER_FRAME:
        return None
    return min(CROSSING_MAX_DISTANCE, speed * max(1, frame_gap) + CROSSING_MATCH_MARGIN)


def match_cost(track: Track, detection: Detection, predicted_state: np.ndarray, frame_gap: int, args, reattach: bool) -> Optional[float]:
    alias_match = raw_id_alias_match(track.last_raw_player_id, detection.raw_player_id)
    previous_xy = latest_track_position(track)
    if reattach and alias_match and frame_gap > args.max_reattach_gap:
        predicted_xy = previous_xy
    else:
        predicted_xy = (float(predicted_state[0, 0]), float(predicted_state[1, 0]))
    distance = euclidean(predicted_xy, (detection.x, detection.y))
    limit = args.reattach_distance if reattach else gating_distance(frame_gap, args.match_distance, args.distance_per_frame)
    speed_gate = (REATTACH_SPEED_GATE_PX_PER_FRAME if reattach else ACTIVE_SPEED_GATE_PX_PER_FRAME) * max(1, frame_gap)
    physical_limit = max(args.match_distance, speed_gate + 20.0)
    crossing_budget = crossing_motion_budget(track, detection, predicted_state, frame_gap, args)
    if crossing_budget is not None:
        limit = max(limit, crossing_budget)
        physical_limit = max(physical_limit, crossing_budget)
    if reattach and alias_match:
        limit = max(limit, min(ALIAS_REATTACH_MAX_DISTANCE, physical_limit))
    limit = min(limit, physical_limit)
    if distance > limit:
        return None
    if not allow_camera_handoff(
        track.last_cam,
        detection.cam,
        previous_xy[1],
        detection.y,
        getattr(args, "split_y", None),
        frame_gap=frame_gap,
        prev_on_terrain=track.last_on_terrain,
        new_on_terrain=detection.on_terrain,
    ):
        return None

    score = distance
    if detection.raw_player_id == track.last_raw_player_id:
        score -= 18.0
    elif alias_match:
        score -= 14.0
    if detection.cam and detection.cam == track.last_cam:
        score -= 6.0
    elif detection.on_terrain and track.last_on_terrain:
        score += 14.0
    if detection.on_terrain and is_near_split(detection.y, getattr(args, "split_y", None), SPLIT_HANDOFF_BAND):
        score -= 4.0
    if detection.on_terrain == track.last_on_terrain:
        score -= 2.0
    if track.confirmed:
        score -= 3.0
    if reattach:
        score -= min(12.0, frame_gap * 0.15)
        score -= min(10.0, track.hits * 0.5)
    return score


def assign_tracks(candidates: List[Tuple[int, int, float]]) -> Dict[int, int]:
    if not candidates:
        return {}
    track_ids = sorted({track_id for track_id, _, _ in candidates})
    detection_ids = sorted({detection_id for _, detection_id, _ in candidates})
    track_index = {track_id: idx for idx, track_id in enumerate(track_ids)}
    detection_index = {detection_id: idx for idx, detection_id in enumerate(detection_ids)}
    cost_matrix = np.full((len(track_ids), len(detection_ids)), 1e9, dtype=float)
    for track_id, detection_id, cost in candidates:
        cost_matrix[track_index[track_id], detection_index[detection_id]] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments: Dict[int, int] = {}
    for row, col in zip(row_ind, col_ind):
        if cost_matrix[row, col] >= 1e8:
            continue
        assignments[detection_ids[col]] = track_ids[row]
    return assignments


def latest_track_position(track: Track) -> Tuple[float, float]:
    if track.display_pos is not None:
        return track.display_pos
    return float(track.state[0, 0]), float(track.state[1, 0])


def advance_track_without_detection(track: Track, target_frame: int, target_timestamp_s: Optional[float], args):
    frame_gap = max(1, int(target_frame) - int(track.last_frame))
    predicted_state, predicted_covariance = kalman_predict(track, frame_gap)
    track.state = predicted_state
    track.covariance = predicted_covariance
    track.last_frame = int(target_frame)
    if target_timestamp_s is not None:
        track.last_timestamp_s = float(target_timestamp_s)
    track.misses += frame_gap
    track.age += frame_gap
    track.confidence = track_confidence(
        track,
        gating_distance(track.misses + 1, args.match_distance, args.distance_per_frame),
    )
    refresh_display_state(track)


def suppresses_duplicate_spawn(detection: Detection, tracks: Dict[int, Track], frame: int) -> bool:
    for track in tracks.values():
        if not track.confirmed:
            continue
        frame_gap = max(1, frame - track.last_frame)
        predicted_state, _ = kalman_predict(track, frame_gap)
        predicted_xy = (float(predicted_state[0, 0]), float(predicted_state[1, 0]))
        if euclidean(predicted_xy, (detection.x, detection.y)) <= DUPLICATE_SUPPRESSION_DISTANCE:
            return True
    return False


def should_spawn_track(detection: Detection, active_tracks: Dict[int, Track], lost_tracks: Dict[int, Track], args, first_timestamp_s: float) -> bool:
    if not detection.spawn_allowed:
        return False
    if suppresses_duplicate_spawn(detection, active_tracks, detection.frame):
        return False
    if suppresses_duplicate_spawn(detection, lost_tracks, detection.frame):
        return False

    elapsed_s = max(0.0, detection.timestamp_s - first_timestamp_s)
    if elapsed_s <= SPAWN_WARMUP_SECONDS:
        return True
    if is_near_court_border(detection, getattr(args, "court_bounds", None)):
        return True
    if is_near_split(detection.y, getattr(args, "split_y", None), SPLIT_HANDOFF_BAND):
        return False
    return False


def row_score(row: Dict[str, object]) -> Tuple[float, float, float, float]:
    return (
        float(row.get("track_hits", 0) or 0),
        float(row.get("confidence", 0.0) or 0.0),
        float(row.get("track_age", 0) or 0),
        -float(row.get("track_misses", 0) or 0),
    )


def dedupe_frame_rows(rows: List[Dict[str, object]], expected_players: Optional[int] = None) -> List[Dict[str, object]]:
    kept: List[Dict[str, object]] = []
    for row in sorted(rows, key=row_score, reverse=True):
        row_pos = (float(row["X"]), float(row["Y"]))
        if any(euclidean(row_pos, (float(existing["X"]), float(existing["Y"]))) <= OUTPUT_DEDUPE_DISTANCE for existing in kept):
            continue
        kept.append(row)
    if expected_players is not None and expected_players > 0 and len(kept) > expected_players:
        kept = sorted(kept, key=row_score, reverse=True)[:expected_players]
    return sorted(kept, key=lambda item: str(item["stable_id"]))


def write_output(rows: List[Dict[str, object]], output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    columns = [
        "frame",
        "timestamp_s",
        "timestamp_unix",
        "stable_id",
        "raw_player_id",
        "X",
        "Y",
        "vx",
        "vy",
        "cam",
        "source_cams",
        "source_ids",
        "merged_count",
        "track_age",
        "track_hits",
        "track_misses",
        "confidence",
        "status",
        "on_terrain",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def rows_to_stable_frames(rows: List[Dict[str, object]]):
    grouped: Dict[int, List[Dict[str, object]]] = {}
    timestamps: Dict[int, float] = {}
    for row in rows:
        frame = int(row["frame"])
        timestamps[frame] = float(row["timestamp_s"])
        grouped.setdefault(frame, []).append(
            {
                "frame": frame,
                "timestamp_s": float(row["timestamp_s"]),
                "player_id": str(row["stable_id"]),
                "raw_player_id": str(row["raw_player_id"]),
                "source_player_ids": [item for item in str(row["source_ids"]).split(",") if item],
                "x": float(row["X"]),
                "y": float(row["Y"]),
                "half": str(row["cam"]),
                "on_terrain": str(row["on_terrain"]).strip() not in ("0", "false", "False", ""),
                "confidence": float(row["confidence"]),
                "track_age": int(row["track_age"]),
                "track_hits": int(row["track_hits"]),
                "track_misses": int(row["track_misses"]),
                "vx": float(row["vx"]),
                "vy": float(row["vy"]),
                "status": str(row["status"]),
            }
        )
    frames = []
    for sequence_index, frame in enumerate(sorted(grouped)):
        players = sorted(grouped[frame], key=lambda item: item["player_id"])
        frames.append(
            {
                "frame": frame,
                "timestamp_s": timestamps[frame],
                "sequence_index": sequence_index,
                "players": players,
            }
        )
    return frames


def run_tracking(args) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    frames = load_frames(args.input)
    geometry = load_calibration_geometry(getattr(args, "calibration", DEFAULT_CALIBRATION))
    args.split_y = geometry.get("split_y")
    args.court_bounds = geometry.get("bounds")
    active_tracks: Dict[int, Track] = {}
    lost_tracks: Dict[int, Track] = {}
    next_track_id = 1
    output_rows: List[Dict[str, object]] = []
    first_timestamp_s = frames[0][0].timestamp_s if frames and frames[0] else 0.0
    stats = {
        "frames": len(frames),
        "detections_in": 0,
        "detections_used": 0,
        "detections_rejected": 0,
        "detections_merged": 0,
        "rows_out": 0,
        "tracks_created": 0,
        "tracks_confirmed": 0,
        "tracks_reattached": 0,
        "split_y": -1 if args.split_y is None else int(round(args.split_y)),
    }

    for frame_detections in frames:
        stats["detections_in"] += len(frame_detections)
        current_frame = frame_detections[0].frame if frame_detections else 0
        current_timestamp_s = frame_detections[0].timestamp_s if frame_detections else None
        usable_detections = []
        for detection in frame_detections:
            normalized = normalize_detection_for_court(detection, args.court_bounds, args.split_y)
            if normalized is None:
                stats["detections_rejected"] += 1
                continue
            usable_detections.append(normalized)
        stats["detections_used"] += len(usable_detections)
        if not usable_detections:
            for track_id in list(active_tracks.keys()):
                track = active_tracks[track_id]
                advance_track_without_detection(track, current_frame, current_timestamp_s, args)
                if track.misses > args.max_misses:
                    lost_tracks[track_id] = active_tracks.pop(track_id)
            continue

        detections = merge_camera_duplicates(usable_detections, args.merge_distance, args.split_y)
        stats["detections_merged"] += sum(item.merged_count - 1 for item in detections)
        frame_output_rows: List[Dict[str, object]] = []

        prediction_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, int]] = {}
        active_candidates: List[Tuple[int, int, float]] = []
        for track_id, track in active_tracks.items():
            frame_gap = max(1, detections[0].frame - track.last_frame)
            predicted_state, predicted_covariance = kalman_predict(track, frame_gap)
            prediction_cache[("active", track_id)] = (predicted_state, predicted_covariance, frame_gap)
            for detection_index, detection in enumerate(detections):
                cost = match_cost(track, detection, predicted_state, frame_gap, args, reattach=False)
                if cost is not None:
                    active_candidates.append((track_id, detection_index, cost))

        active_assignments = assign_tracks(active_candidates)
        assigned_detections = set(active_assignments.keys())
        matched_active_tracks = set(active_assignments.values())

        reattach_candidates: List[Tuple[int, int, float]] = []
        for track_id, track in lost_tracks.items():
            frame_gap = max(1, detections[0].frame - track.last_frame)
            if frame_gap > ALIAS_REATTACH_MAX_FRAME_GAP:
                continue
            predicted_state, predicted_covariance = kalman_predict(track, frame_gap)
            prediction_cache[("lost", track_id)] = (predicted_state, predicted_covariance, frame_gap)
            for detection_index, detection in enumerate(detections):
                if detection_index in assigned_detections:
                    continue
                if frame_gap > args.max_reattach_gap and not raw_id_alias_match(track.last_raw_player_id, detection.raw_player_id):
                    continue
                cost = match_cost(track, detection, predicted_state, frame_gap, args, reattach=True)
                if cost is not None:
                    reattach_candidates.append((track_id, detection_index, cost))

        reattach_assignments = assign_tracks(reattach_candidates)
        assigned_detections.update(reattach_assignments.keys())

        for detection_index, track_id in active_assignments.items():
            detection = detections[detection_index]
            track = active_tracks[track_id]
            predicted_state, predicted_covariance, _ = prediction_cache[("active", track_id)]
            updated_state, updated_covariance, residual_distance = kalman_update(predicted_state, predicted_covariance, detection)
            was_confirmed = track.confirmed
            track.state = updated_state
            track.covariance = updated_covariance
            track.last_frame = detection.frame
            track.last_timestamp_s = detection.timestamp_s
            track.hits += 1
            track.age += 1
            track.misses = 0
            track.last_raw_player_id = detection.raw_player_id
            track.last_cam = detection.cam
            track.last_on_terrain = detection.on_terrain
            if not track.confirmed and track.hits >= args.min_hits:
                track.confirmed = True
                if not was_confirmed:
                    stats["tracks_confirmed"] += 1
            track.confidence = track_confidence(track, residual_distance)
            refresh_display_state(track)

        for detection_index, track_id in reattach_assignments.items():
            detection = detections[detection_index]
            track = lost_tracks.pop(track_id)
            predicted_state, predicted_covariance, frame_gap = prediction_cache[("lost", track_id)]
            updated_state, updated_covariance, residual_distance = kalman_update(predicted_state, predicted_covariance, detection)
            track.state = updated_state
            track.covariance = updated_covariance
            track.last_frame = detection.frame
            track.last_timestamp_s = detection.timestamp_s
            track.hits += 1
            track.age += max(1, frame_gap)
            track.misses = 0
            track.last_raw_player_id = detection.raw_player_id
            track.last_cam = detection.cam
            track.last_on_terrain = detection.on_terrain
            if not track.confirmed and track.hits >= args.min_hits:
                track.confirmed = True
                stats["tracks_confirmed"] += 1
            track.confidence = track_confidence(track, residual_distance)
            refresh_display_state(track)
            active_tracks[track_id] = track
            stats["tracks_reattached"] += 1
            matched_active_tracks.add(track_id)

        for track_id in list(active_tracks.keys()):
            if track_id in matched_active_tracks:
                continue
            track = active_tracks[track_id]
            advance_track_without_detection(track, current_frame, current_timestamp_s, args)
            if track.misses > args.max_misses:
                lost_tracks[track_id] = active_tracks.pop(track_id)

        for detection_index, detection in enumerate(detections):
            if detection_index in active_assignments:
                track = active_tracks[active_assignments[detection_index]]
                row_status = "merged" if detection.status == "merged" else "matched"
            elif detection_index in reattach_assignments:
                track = active_tracks[reattach_assignments[detection_index]]
                row_status = "reattached"
            else:
                if not should_spawn_track(detection, active_tracks, lost_tracks, args, first_timestamp_s):
                    continue
                state, covariance = create_filter_state(detection.x, detection.y)
                track = Track(
                    track_id=next_track_id,
                    state=state,
                    covariance=covariance,
                    last_frame=detection.frame,
                    last_timestamp_s=detection.timestamp_s,
                    confirmed=args.min_hits <= 1,
                    last_raw_player_id=detection.raw_player_id,
                    last_cam=detection.cam,
                    last_on_terrain=detection.on_terrain,
                    confidence=0.18 if detection.on_terrain else 0.10,
                    display_pos=(detection.x, detection.y),
                )
                active_tracks[next_track_id] = track
                next_track_id += 1
                stats["tracks_created"] += 1
                row_status = "spawned"
                if track.confirmed:
                    stats["tracks_confirmed"] += 1

            if not track.confirmed:
                continue

            frame_output_rows.append(
                {
                    "frame": detection.frame,
                    "timestamp_s": f"{detection.timestamp_s:.3f}",
                    "timestamp_unix": detection.timestamp_unix,
                    "stable_id": track.stable_id(),
                    "raw_player_id": detection.raw_player_id,
                    "X": round(float((track.display_pos or (float(track.state[0, 0]), float(track.state[1, 0])))[0]), 2),
                    "Y": round(float((track.display_pos or (float(track.state[0, 0]), float(track.state[1, 0])))[1]), 2),
                    "vx": round(float(track.state[2, 0]), 2),
                    "vy": round(float(track.state[3, 0]), 2),
                    "cam": detection.cam,
                    "source_cams": ",".join(detection.source_cams),
                    "source_ids": ",".join(detection.source_ids),
                    "merged_count": detection.merged_count,
                    "track_age": track.age,
                    "track_hits": track.hits,
                    "track_misses": track.misses,
                    "confidence": round(track.confidence, 3),
                    "status": row_status,
                    "on_terrain": 1 if detection.on_terrain else 0,
                }
            )
        output_rows.extend(dedupe_frame_rows(frame_output_rows, expected_players=getattr(args, "expected_players", 10)))

    stats["rows_out"] = len(output_rows)
    return output_rows, stats


def build_stable_frames(csv_path: str, **overrides):
    args = default_config(input_path=csv_path, output_path=overrides.pop("output_path", DEFAULT_OUTPUT), **overrides)
    rows, stats = run_tracking(args)
    return rows_to_stable_frames(rows), stats


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    rows, stats = run_tracking(args)
    write_output(rows, args.output)
    print(
        "Tracking termine: "
        f"{stats['rows_out']} lignes, "
        f"{stats['tracks_created']} tracks crees, "
        f"{stats['tracks_confirmed']} confirms, "
        f"{stats['tracks_reattached']} reattaches, "
        f"{stats['detections_merged']} fusions camera, "
        f"{stats['detections_rejected']} detections rejetees."
    )
    print(f"CSV genere: {args.output}")


if __name__ == "__main__":
    main()

