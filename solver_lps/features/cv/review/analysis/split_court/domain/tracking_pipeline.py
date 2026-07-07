import logging
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

from solver_lps.session_assets import DEFAULT_SET, DEFAULT_SPORT, SessionAssets
from solver_lps.features.ground.domain.calibration import load_calibration_geometry
from ..data.raw_input import load_frames as load_raw_frames
from ..data.merged_output import rows_to_tracking_frames as export_rows_to_tracking_frames
from .assignment_solver import linear_sum_assignment as solve_linear_sum_assignment
from .continuity_filters import filter_continuity_track
from .duplicate_filters import filter_duplicate_observations
from .ghost_filters import filter_ghost_track
from .terrain_rules import (
    crosses_split,
    euclidean,
    is_inside_bounds,
    is_near_court_border,
    is_near_split,
    normalize_cam_name,
    raw_id_alias_match,
)
from .terrain_filters import filter_terrain_observations
from .traversal_filters import filter_traversal_transition
from .speed_filters import filter_impossible_speed_match
from .reattach_filters import filter_reattach_candidate
from .tracking_lifecycle import (
    BOOTSTRAP_IGNORE_SPAWN_GATE_FRAMES,
    SpawnGuard,
    classify_spawn_category,
    is_bootstrap_spawn_frame,
    should_promote_track,
)

_DEFAULT_SESSION = SessionAssets(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET)
_DEFAULT_OUTPUT = str(_DEFAULT_SESSION.analysis_dir / "positions_tracked_hungarian.csv")
_DEFAULT_CALIBRATION = str(_DEFAULT_SESSION.calibration_path)
DISPLAY_SMOOTH_ALPHA = 0.24
DISPLAY_TENTATIVE_ALPHA = 0.34
DISPLAY_SMOOTH_DEADZONE = 3.0
DISPLAY_SNAP_DISTANCE = 90.0
VELOCITY_DAMPING = 0.90
SPLIT_HANDOFF_BAND = 70.0
SPAWN_WARMUP_SECONDS = 5.0
DUPLICATE_SUPPRESSION_DISTANCE = 42.0
SPAWN_DUPLICATE_CONFIRMED_DISTANCE = 72.0
SPAWN_DUPLICATE_CANDIDATE_DISTANCE = 28.0
SPAWN_DUPLICATE_LOST_DISTANCE = 84.0
OUTPUT_DEDUPE_DISTANCE = 36.0
OFF_TERRAIN_DUPLICATE_DISTANCE = 20.0
ACTIVE_SPEED_GATE_PX_PER_FRAME = 18.0
REATTACH_SPEED_GATE_PX_PER_FRAME = 22.0
ALIAS_REATTACH_MAX_FRAME_GAP = 360
ALIAS_REATTACH_MAX_DISTANCE = 700.0
CROSSING_SPEED_MIN_PX_PER_FRAME = 10.0
CROSSING_MATCH_MARGIN = 45.0
CROSSING_MAX_DISTANCE = 520.0
MAX_TRACK_SPEED_PX_PER_FRAME = 160.0
MIN_TRACK_CONFIRM_HITS = 8


logger = logging.getLogger(__name__)


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
    duplicate_of_track_id: Optional[int] = None


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
    spawn_category: str = "midfield"
    spawn_position: Optional[Tuple[float, float]] = None

    def track_label(self) -> str:
        return f"T{self.track_id:03d}"


def to_bool(value: str) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", ""}


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
    speed = math.hypot(float(track.state[2, 0]), float(track.state[3, 0]))
    if speed > MAX_TRACK_SPEED_PX_PER_FRAME:
        scale = MAX_TRACK_SPEED_PX_PER_FRAME / max(speed, 1e-9)
        track.state[2, 0] *= scale
        track.state[3, 0] *= scale


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
    implied_speed = distance / max(1, frame_gap)
    if implied_speed > MAX_TRACK_SPEED_PX_PER_FRAME:
        return None
    limit = args.reattach_distance if reattach else gating_distance(frame_gap, args.match_distance, args.distance_per_frame)
    speed_gate = (REATTACH_SPEED_GATE_PX_PER_FRAME if reattach else ACTIVE_SPEED_GATE_PX_PER_FRAME) * max(1, frame_gap)
    physical_limit = max(args.match_distance, speed_gate + 20.0)
    crossing_budget = crossing_motion_budget(track, detection, predicted_state, frame_gap, args)
    if crossing_budget is not None:
        limit = max(limit, crossing_budget)
        physical_limit = max(physical_limit, crossing_budget)
    if alias_match and not reattach:
        alias_limit = min(ALIAS_REATTACH_MAX_DISTANCE, gating_distance(frame_gap, args.match_distance + 80.0, args.distance_per_frame))
        limit = max(limit, alias_limit)
        physical_limit = max(physical_limit, alias_limit)
    if reattach and alias_match:
        limit = max(limit, min(ALIAS_REATTACH_MAX_DISTANCE, physical_limit))
    limit = min(limit, physical_limit)
    if distance > limit:
        return None
    if not filter_traversal_transition(
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

    row_ind, col_ind = solve_linear_sum_assignment(cost_matrix)
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


def _dedupe_owned_detections(
    detections: List[Detection],
    active_tracks: Dict[int, Track],
    lost_tracks: Dict[int, Track],
    current_frame: int,
    args,
    stats: Dict[str, int],
    spawn_guard: Optional[SpawnGuard] = None,
) -> List[Detection]:
    return filter_duplicate_observations(
        detections,
        active_tracks,
        lost_tracks,
        current_frame,
        args,
        stats,
        spawn_guard=spawn_guard,
    )


def _confirmed_track_count(active_tracks: Dict[int, Track], lost_tracks: Dict[int, Track]) -> int:
    return sum(1 for track in list(active_tracks.values()) + list(lost_tracks.values()) if track.confirmed)


def _maybe_confirm_track(
    track: Track,
    active_tracks: Dict[int, Track],
    lost_tracks: Dict[int, Track],
    args,
    stats: Dict[str, int],
):
    filter_ghost_track(track, active_tracks, lost_tracks, args, stats)


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
        frame_gap = max(1, frame - track.last_frame)
        predicted_state, _ = kalman_predict(track, frame_gap)
        predicted_xy = (float(predicted_state[0, 0]), float(predicted_state[1, 0]))
        if track.confirmed:
            limit = 18.0
        elif track.misses > 0:
            limit = 20.0
        else:
            limit = 14.0
        if not track.last_on_terrain and not detection.on_terrain and detection.cam == track.last_cam:
            limit = max(limit, OFF_TERRAIN_DUPLICATE_DISTANCE)
        if euclidean(predicted_xy, (detection.x, detection.y)) <= limit:
            return True
    return False


def should_spawn_track(
    detection: Detection,
    active_tracks: Dict[int, Track],
    lost_tracks: Dict[int, Track],
    args,
    first_timestamp_s: float,
    spawn_guard: Optional[SpawnGuard] = None,
    bootstrap_ignore_spawn_gate_frames: int = BOOTSTRAP_IGNORE_SPAWN_GATE_FRAMES,
) -> bool:
    if not detection.spawn_allowed:
        return False
    bootstrap_active = is_bootstrap_spawn_frame(detection.frame, bootstrap_ignore_spawn_gate_frames)
    if spawn_guard is not None:
        if not spawn_guard.should_allow_spawn(
            detection,
            detection.frame,
            first_timestamp_s,
            getattr(args, "court_bounds", None),
            bootstrap_ignore_spawn_gate_frames=bootstrap_ignore_spawn_gate_frames,
        ):
            return False
    elif not bootstrap_active and detection.on_terrain:
        return False
    if suppresses_duplicate_spawn(detection, active_tracks, detection.frame):
        return False
    if suppresses_duplicate_spawn(detection, lost_tracks, detection.frame):
        return False
    return True


def row_score(row: Dict[str, object]) -> Tuple[float, float, float, float]:
    return (
        float(row.get("track_hits", 0) or 0),
        float(row.get("confidence", 0.0) or 0.0),
        float(row.get("track_age", 0) or 0),
        -float(row.get("track_misses", 0) or 0),
    )


def dedupe_frame_rows(rows: List[Dict[str, object]], expected_players: Optional[int] = None) -> List[Dict[str, object]]:
    best_rows: Dict[str, Dict[str, object]] = {}
    for row in sorted(rows, key=row_score, reverse=True):
        stable_id = str(row.get("stable_id", ""))
        if stable_id in best_rows:
            continue
        best_rows[stable_id] = row
    kept = list(best_rows.values())
    if expected_players is not None and expected_players > 0 and len(kept) > expected_players:
        kept = sorted(kept, key=row_score, reverse=True)[:expected_players]
    return sorted(kept, key=lambda item: str(item["stable_id"]))


def _prepare_frame_detections(frame_detections: List[Detection], args, stats: Dict[str, int]) -> Tuple[int, Optional[float], List[Detection]]:
    return filter_terrain_observations(frame_detections, args, stats)


def _build_matching_state(
    active_tracks: Dict[int, Track],
    lost_tracks: Dict[int, Track],
    detections: List[Detection],
    args,
) -> Tuple[
    Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, int]],
    Dict[int, int],
    Dict[int, int],
    set,
]:
    prediction_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, int]] = {}
    active_candidates: List[Tuple[int, int, float]] = []
    current_frame = detections[0].frame

    for track_id, track in active_tracks.items():
        frame_gap = max(1, current_frame - track.last_frame)
        predicted_state, predicted_covariance = kalman_predict(track, frame_gap)
        prediction_cache[("active", track_id)] = (predicted_state, predicted_covariance, frame_gap)
        for detection_index, detection in enumerate(detections):
            cost = filter_impossible_speed_match(track, detection, predicted_state, frame_gap, args, reattach=False)
            if cost is not None:
                active_candidates.append((track_id, detection_index, cost))

    active_assignments = assign_tracks(active_candidates)
    assigned_detections = set(active_assignments.keys())
    matched_active_tracks = set(active_assignments.values())

    reattach_candidates: List[Tuple[int, int, float]] = []
    for track_id, track in lost_tracks.items():
        frame_gap = max(1, current_frame - track.last_frame)
        if frame_gap > ALIAS_REATTACH_MAX_FRAME_GAP:
            continue
        predicted_state, predicted_covariance = kalman_predict(track, frame_gap)
        prediction_cache[("lost", track_id)] = (predicted_state, predicted_covariance, frame_gap)
        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            if frame_gap > args.max_reattach_gap and not raw_id_alias_match(track.last_raw_player_id, detection.raw_player_id):
                continue
            cost = filter_reattach_candidate(track, detection, predicted_state, frame_gap, args)
            if cost is not None:
                reattach_candidates.append((track_id, detection_index, cost))

    reattach_assignments = assign_tracks(reattach_candidates)
    return prediction_cache, active_assignments, reattach_assignments, matched_active_tracks


def _update_assigned_tracks(
    active_tracks: Dict[int, Track],
    lost_tracks: Dict[int, Track],
    detections: List[Detection],
    prediction_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, int]],
    active_assignments: Dict[int, int],
    reattach_assignments: Dict[int, int],
    matched_active_tracks: set,
    args,
    stats: Dict[str, int],
    spawn_guard: Optional[SpawnGuard] = None,
):
    for detection_index, track_id in active_assignments.items():
        detection = detections[detection_index]
        track = active_tracks[track_id]
        predicted_state, predicted_covariance, _ = prediction_cache[("active", track_id)]
        updated_state, updated_covariance, residual_distance = kalman_update(predicted_state, predicted_covariance, detection)
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
        track.confidence = track_confidence(track, residual_distance)
        refresh_display_state(track)
        if spawn_guard is not None:
            spawn_guard.observe(track.last_raw_player_id, detection.frame, detection.on_terrain)
        _maybe_confirm_track(track, active_tracks, lost_tracks, args, stats)

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
        track.confidence = track_confidence(track, residual_distance)
        refresh_display_state(track)
        active_tracks[track_id] = track
        stats["tracks_reattached"] += 1
        matched_active_tracks.add(track_id)
        if spawn_guard is not None:
            spawn_guard.observe(track.last_raw_player_id, detection.frame, detection.on_terrain)
        _maybe_confirm_track(track, active_tracks, lost_tracks, args, stats)


def _age_unmatched_tracks(
    active_tracks: Dict[int, Track],
    lost_tracks: Dict[int, Track],
    current_frame: int,
    current_timestamp_s: Optional[float],
    matched_active_tracks: set,
    args,
):
    for track_id in list(active_tracks.keys()):
        if track_id in matched_active_tracks:
            continue
        track = active_tracks[track_id]
        filter_continuity_track(track, current_frame, current_timestamp_s, args)
        if track.misses > args.max_misses:
            lost_tracks[track_id] = active_tracks.pop(track_id)


def _build_output_row(
    track: Track,
    frame: int,
    timestamp_s: float,
    row_status: str,
    detection: Optional[Detection] = None,
) -> Dict[str, object]:
    source_ids = detection.source_ids if detection is not None else []
    source_cams = detection.source_cams if detection is not None else []
    raw_player_id = detection.raw_player_id if detection is not None else track.last_raw_player_id
    cam = detection.cam if detection is not None else track.last_cam
    merged_count = detection.merged_count if detection is not None else 1
    unix_ts = detection.timestamp_unix if detection is not None else ""
    return {
        "frame": frame,
        "timestamp_s": f"{timestamp_s:.3f}",
        "timestamp_unix": unix_ts,
        "stable_id": track.track_label(),
        "raw_player_id": raw_player_id,
        "X": round(float((track.display_pos or (float(track.state[0, 0]), float(track.state[1, 0])))[0]), 2),
        "Y": round(float((track.display_pos or (float(track.state[0, 0]), float(track.state[1, 0])))[1]), 2),
        "vx": round(float(track.state[2, 0]), 2),
        "vy": round(float(track.state[3, 0]), 2),
        "cam": cam,
        "source_cams": ",".join(source_cams),
        "source_ids": ",".join(source_ids),
        "merged_count": merged_count,
        "track_age": track.age,
        "track_hits": track.hits,
        "track_misses": track.misses,
        "confidence": round(track.confidence, 3),
        "status": row_status,
        "on_terrain": 1 if (detection.on_terrain if detection is not None else track.last_on_terrain) else 0,
    }


def _spawn_track(
    next_track_id: int,
    detection: Detection,
    args,
    stats: Dict[str, int],
    *,
    spawn_category: str,
) -> Tuple[Track, int]:
    state, covariance = create_filter_state(detection.x, detection.y)
    track = Track(
        track_id=next_track_id,
        state=state,
        covariance=covariance,
        last_frame=detection.frame,
        last_timestamp_s=detection.timestamp_s,
        confirmed=False,
        last_raw_player_id=detection.raw_player_id,
        last_cam=detection.cam,
        last_on_terrain=detection.on_terrain,
        confidence=0.18 if detection.on_terrain else 0.10,
        display_pos=(detection.x, detection.y),
        spawn_category=spawn_category,
        spawn_position=(detection.x, detection.y),
    )
    stats["tracks_created"] += 1
    stats["candidate_created"] += 1
    logger.debug(
        "candidate_created track=%s raw=%s category=%s frame=%s pos=(%.1f,%.1f)",
        track.track_label(),
        track.last_raw_player_id,
        spawn_category,
        detection.frame,
        detection.x,
        detection.y,
    )
    return track, next_track_id + 1


def _emit_frame_rows(
    current_frame: int,
    current_timestamp_s: Optional[float],
    detections: List[Detection],
    active_tracks: Dict[int, Track],
    lost_tracks: Dict[int, Track],
    active_assignments: Dict[int, int],
    reattach_assignments: Dict[int, int],
    args,
    stats: Dict[str, int],
    next_track_id: int,
    first_timestamp_s: float,
    spawn_guard: Optional[SpawnGuard] = None,
):
    frame_output_rows: List[Dict[str, object]] = []
    emitted_track_ids = set()

    for detection_index, detection in enumerate(detections):
        if detection_index in active_assignments:
            track = active_tracks[active_assignments[detection_index]]
            row_status = "merged" if detection.status == "merged" else "matched"
        elif detection_index in reattach_assignments:
            track = active_tracks[reattach_assignments[detection_index]]
            row_status = "reattached"
        else:
            if not should_spawn_track(
                detection,
                active_tracks,
                lost_tracks,
                args,
                first_timestamp_s,
                spawn_guard=spawn_guard,
            ):
                continue
            spawn_category = classify_spawn_category(
                detection,
                first_timestamp_s,
                getattr(args, "court_bounds", None),
            )
            track, next_track_id = _spawn_track(
                next_track_id,
                detection,
                args,
                stats,
                spawn_category=spawn_category,
            )
            active_tracks[track.track_id] = track
            row_status = "spawned"

        if spawn_guard is not None:
            spawn_guard.observe(track.last_raw_player_id, current_frame, track.last_on_terrain)
        if not track.confirmed:
            continue
        emitted_track_ids.add(track.track_id)
        frame_output_rows.append(_build_output_row(track, current_frame, detection.timestamp_s, row_status, detection))

    for track_id, track in active_tracks.items():
        if not track.confirmed or track_id in emitted_track_ids:
            continue
        fallback_timestamp_s = current_timestamp_s if current_timestamp_s is not None else track.last_timestamp_s
        frame_output_rows.append(_build_output_row(track, current_frame, fallback_timestamp_s, "predicted"))

    return frame_output_rows, next_track_id


def _interpolate_timestamp(previous_timestamp_s: Optional[float], next_timestamp_s: Optional[float], step_index: int, step_count: int) -> Optional[float]:
    if previous_timestamp_s is None or next_timestamp_s is None or step_count <= 0:
        return next_timestamp_s if next_timestamp_s is not None else previous_timestamp_s
    if step_count == 1:
        return next_timestamp_s
    ratio = float(step_index) / float(step_count)
    return float(previous_timestamp_s) + (float(next_timestamp_s) - float(previous_timestamp_s)) * ratio


def _emit_gap_frame_rows(
    current_frame: int,
    current_timestamp_s: Optional[float],
    active_tracks: Dict[int, Track],
    args,
):
    frame_rows = []
    for track in active_tracks.values():
        if not track.confirmed:
            continue
        fallback_timestamp_s = current_timestamp_s if current_timestamp_s is not None else track.last_timestamp_s
        frame_rows.append(_build_output_row(track, current_frame, fallback_timestamp_s, "predicted"))
    return frame_rows


def _create_tracking_stats(args, frames):
    first_timestamp_s = frames[0][0].timestamp_s if frames and frames[0] else 0.0
    return {
        "frames": len(frames),
        "detections_in": 0,
        "detections_used": 0,
        "detections_rejected": 0,
        "detections_merged": 0,
        "rows_out": 0,
        "tracks_created": 0,
        "tracks_confirmed": 0,
        "tracks_reattached": 0,
        "candidate_created": 0,
        "candidate_promoted": 0,
        "candidate_rejected": 0,
        "candidate_rejected_duplicate_track": 0,
        "detection_dedup_merged": 0,
        "split_y": -1 if args.split_y is None else int(round(args.split_y)),
    }, first_timestamp_s


def _emit_missing_frame_rows(
    previous_frame: Optional[int],
    previous_timestamp_s: Optional[float],
    current_frame: int,
    current_timestamp_s: Optional[float],
    active_tracks: Dict[int, Track],
    lost_tracks: Dict[int, Track],
    args,
    output_rows: List[Dict[str, object]],
):
    if previous_frame is None or current_frame <= previous_frame + 1:
        return

    gap_count = current_frame - previous_frame - 1
    for step_index in range(1, gap_count + 1):
        gap_frame = previous_frame + step_index
        gap_timestamp = _interpolate_timestamp(previous_timestamp_s, current_timestamp_s, step_index, gap_count + 1)
        _age_unmatched_tracks(active_tracks, lost_tracks, gap_frame, gap_timestamp, set(), args)
        gap_rows = _emit_gap_frame_rows(gap_frame, gap_timestamp, active_tracks, args)
        output_rows.extend(dedupe_frame_rows(gap_rows, expected_players=getattr(args, "expected_players", 10)))


def _process_empty_tracking_frame(
    current_frame: int,
    current_timestamp_s: Optional[float],
    active_tracks: Dict[int, Track],
    lost_tracks: Dict[int, Track],
    args,
    output_rows: List[Dict[str, object]],
):
    _age_unmatched_tracks(active_tracks, lost_tracks, current_frame, current_timestamp_s, set(), args)
    gap_rows = _emit_gap_frame_rows(current_frame, current_timestamp_s, active_tracks, args)
    output_rows.extend(dedupe_frame_rows(gap_rows, expected_players=getattr(args, "expected_players", 10)))


def _process_tracking_frame(
    frame_detections: List[Detection],
    args,
    stats: Dict[str, int],
    active_tracks: Dict[int, Track],
    lost_tracks: Dict[int, Track],
    next_track_id: int,
    first_timestamp_s: float,
    spawn_guard: SpawnGuard,
):
    current_frame, current_timestamp_s, detections = _prepare_frame_detections(frame_detections, args, stats)
    detections = _dedupe_owned_detections(
        detections,
        active_tracks,
        lost_tracks,
        current_frame,
        args,
        stats,
        spawn_guard=spawn_guard,
    )
    if not detections:
        return current_frame, current_timestamp_s, next_track_id, []

    prediction_cache, active_assignments, reattach_assignments, matched_active_tracks = _build_matching_state(
        active_tracks,
        lost_tracks,
        detections,
        args,
    )
    _update_assigned_tracks(
        active_tracks,
        lost_tracks,
        detections,
        prediction_cache,
        active_assignments,
        reattach_assignments,
        matched_active_tracks,
        args,
        stats,
        spawn_guard,
    )
    _age_unmatched_tracks(active_tracks, lost_tracks, current_frame, current_timestamp_s, matched_active_tracks, args)
    frame_output_rows, next_track_id = _emit_frame_rows(
        current_frame,
        current_timestamp_s,
        detections,
        active_tracks,
        lost_tracks,
        active_assignments,
        reattach_assignments,
        args,
        stats,
        next_track_id,
        first_timestamp_s,
        spawn_guard,
    )
    return current_frame, current_timestamp_s, next_track_id, frame_output_rows


def run_tracking(args) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    frames = load_raw_frames(args.input)
    progress_callback = getattr(args, "progress_callback", None)
    if progress_callback is not None:
        progress_callback(0, len(frames))
    geometry = load_calibration_geometry(getattr(args, "calibration", _DEFAULT_CALIBRATION))
    args.split_y = geometry.get("split_y")
    args.court_bounds = geometry.get("bounds")
    active_tracks: Dict[int, Track] = {}
    lost_tracks: Dict[int, Track] = {}
    next_track_id = 1
    output_rows: List[Dict[str, object]] = []
    spawn_guard = SpawnGuard()
    stats, first_timestamp_s = _create_tracking_stats(args, frames)

    previous_frame: Optional[int] = None
    previous_timestamp_s: Optional[float] = None
    for frame_index, frame_detections in enumerate(frames, start=1):
        stats["detections_in"] += len(frame_detections)
        current_frame, current_timestamp_s, next_track_id, frame_output_rows = _process_tracking_frame(
            frame_detections,
            args,
            stats,
            active_tracks,
            lost_tracks,
            next_track_id,
            first_timestamp_s,
            spawn_guard,
        )
        _emit_missing_frame_rows(
            previous_frame,
            previous_timestamp_s,
            current_frame,
            current_timestamp_s,
            active_tracks,
            lost_tracks,
            args,
            output_rows,
        )

        output_rows.extend(dedupe_frame_rows(frame_output_rows, expected_players=getattr(args, "expected_players", 10)))
        previous_frame = current_frame
        previous_timestamp_s = current_timestamp_s
        if progress_callback is not None:
            progress_callback(frame_index, len(frames))

    stats["rows_out"] = len(output_rows)
    return output_rows, stats


def _build_tracking_args(csv_path: str, output_path: Optional[str] = None, **overrides):
    return SimpleNamespace(
        input=csv_path,
        output=output_path or _DEFAULT_OUTPUT,
        calibration=str(overrides.pop("calibration", _DEFAULT_CALIBRATION)),
        merge_distance=float(overrides.pop("merge_distance", 35.0)),
        match_distance=float(overrides.pop("match_distance", 70.0)),
        distance_per_frame=float(overrides.pop("distance_per_frame", 18.0)),
        reattach_distance=float(overrides.pop("reattach_distance", 240.0)),
        max_misses=int(overrides.pop("max_misses", 15)),
        max_reattach_gap=int(overrides.pop("max_reattach_gap", 120)),
        min_hits=int(overrides.pop("min_hits", 3)),
        expected_players=int(overrides.pop("expected_players", 10)),
        split_y=overrides.pop("split_y", None),
    )


def default_config(input_path: Optional[str] = None, output_path: Optional[str] = None, **overrides):
    return _build_tracking_args(input_path or "", output_path, **overrides)



def build_tracking_frames(csv_path: str, **overrides):
    args = _build_tracking_args(csv_path, overrides.pop("output_path", None), **overrides)
    rows, stats = run_tracking(args)
    return export_rows_to_tracking_frames(rows), stats
