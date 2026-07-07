from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from .terrain_rules import euclidean, raw_id_alias_match

logger = logging.getLogger(__name__)

OFF_TERRAIN_DUPLICATE_DISTANCE = 20.0
ALIAS_REATTACH_MAX_DISTANCE = 700.0
ALIAS_REATTACH_MAX_FRAME_GAP = 360
ACTIVE_SPEED_GATE_PX_PER_FRAME = 18.0
REATTACH_SPEED_GATE_PX_PER_FRAME = 22.0
MAX_TRACK_SPEED_PX_PER_FRAME = 160.0
CROSSING_SPEED_MIN_PX_PER_FRAME = 10.0
CROSSING_MATCH_MARGIN = 45.0
CROSSING_MAX_DISTANCE = 520.0
SPLIT_HANDOFF_BAND = 70.0
SPLIT_HANDOFF_EXTRA_PER_FRAME = 8.0


def _detection_dedup_rank(item: Tuple[int, object, float, float]) -> Tuple[float, int, int, int, int, int]:
    index, detection, distance, _limit = item
    source_cam_count = len(getattr(detection, "source_cams", []) or [])
    source_id_count = len(getattr(detection, "source_ids", []) or [])
    return (
        float(distance),
        -int(getattr(detection, "merged_count", 1) or 1),
        -source_cam_count,
        -source_id_count,
        0 if getattr(detection, "spawn_allowed", True) else 1,
        index,
    )


def _track_duplicate_distance_limit(track, frame_gap: int, *, same_off_terrain: bool = False) -> float:
    if same_off_terrain:
        if track.confirmed:
            return OFF_TERRAIN_DUPLICATE_DISTANCE
        if track.misses > 0:
            return OFF_TERRAIN_DUPLICATE_DISTANCE
        return OFF_TERRAIN_DUPLICATE_DISTANCE
    if track.confirmed:
        return 18.0 + min(8.0, max(0, frame_gap - 1) * 2.0)
    if track.misses > 0:
        return 20.0
    return 14.0 + min(4.0, max(0, frame_gap - 1) * 1.5)


def _kalman_predict(track, frame_gap: int):
    dt = float(max(1, frame_gap))
    transition = [
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    process_noise = [
        [4.0 * dt, 0.0, 0.0, 0.0],
        [0.0, 4.0 * dt, 0.0, 0.0],
        [0.0, 0.0, 2.5 * dt, 0.0],
        [0.0, 0.0, 0.0, 2.5 * dt],
    ]
    import numpy as np

    transition = np.array(transition, dtype=float)
    process_noise = np.array(process_noise, dtype=float)
    predicted_state = transition @ track.state
    predicted_covariance = transition @ track.covariance @ transition.T + process_noise
    return predicted_state, predicted_covariance


def _latest_track_position(track) -> Tuple[float, float]:
    if track.display_pos is not None:
        return track.display_pos
    return float(track.state[0, 0]), float(track.state[1, 0])


def _gating_distance(frame_gap: int, base_distance: float, distance_per_frame: float) -> float:
    return base_distance + max(0, frame_gap - 1) * distance_per_frame


def _track_speed(track) -> float:
    return math.hypot(float(track.state[2, 0]), float(track.state[3, 0]))


def _crossing_motion_budget(track, detection, predicted_state, frame_gap: int, args) -> Optional[float]:
    split_y = getattr(args, "split_y", None)
    if split_y is None or frame_gap <= 1:
        return None
    previous_xy = _latest_track_position(track)
    predicted_xy = (float(predicted_state[0, 0]), float(predicted_state[1, 0]))
    if not (
        abs(previous_xy[1] - split_y) <= SPLIT_HANDOFF_BAND
        or abs(predicted_xy[1] - split_y) <= SPLIT_HANDOFF_BAND
        or abs(detection.y - split_y) <= SPLIT_HANDOFF_BAND
    ):
        return None
    speed = max(
        _track_speed(track),
        euclidean(previous_xy, predicted_xy) / max(1, frame_gap),
    )
    if speed < CROSSING_SPEED_MIN_PX_PER_FRAME:
        return None
    return min(CROSSING_MAX_DISTANCE, speed * max(1, frame_gap) + CROSSING_MATCH_MARGIN)


def _match_cost(track, detection, predicted_state, frame_gap: int, args, reattach: bool) -> Optional[float]:
    alias_match = raw_id_alias_match(track.last_raw_player_id, detection.raw_player_id)
    previous_xy = _latest_track_position(track)
    if reattach and alias_match and frame_gap > getattr(args, "max_reattach_gap", 0):
        predicted_xy = previous_xy
    else:
        predicted_xy = (float(predicted_state[0, 0]), float(predicted_state[1, 0]))
    distance = euclidean(predicted_xy, (detection.x, detection.y))
    implied_speed = distance / max(1, frame_gap)
    if implied_speed > MAX_TRACK_SPEED_PX_PER_FRAME:
        return None
    limit = getattr(args, "reattach_distance", 240.0) if reattach else _gating_distance(
        frame_gap,
        getattr(args, "match_distance", 70.0),
        getattr(args, "distance_per_frame", 18.0),
    )
    speed_gate = (REATTACH_SPEED_GATE_PX_PER_FRAME if reattach else ACTIVE_SPEED_GATE_PX_PER_FRAME) * max(1, frame_gap)
    physical_limit = max(getattr(args, "match_distance", 70.0), speed_gate + 20.0)
    crossing_budget = _crossing_motion_budget(track, detection, predicted_state, frame_gap, args)
    if crossing_budget is not None:
        limit = max(limit, crossing_budget)
        physical_limit = max(physical_limit, crossing_budget)
    if alias_match and not reattach:
        alias_limit = min(ALIAS_REATTACH_MAX_DISTANCE, _gating_distance(frame_gap, getattr(args, "match_distance", 70.0) + 80.0, getattr(args, "distance_per_frame", 18.0)))
        limit = max(limit, alias_limit)
        physical_limit = max(physical_limit, alias_limit)
    if reattach and alias_match:
        limit = max(limit, min(ALIAS_REATTACH_MAX_DISTANCE, physical_limit))
    limit = min(limit, physical_limit)
    if distance > limit:
        return None
    previous_cam = str(getattr(track, "last_cam", "") or "")
    new_cam = str(getattr(detection, "cam", "") or "")
    if previous_cam and new_cam and previous_cam != new_cam and "+" not in previous_cam and "+" not in new_cam:
        if not getattr(track, "last_on_terrain", True) or not getattr(detection, "on_terrain", True):
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
    split_y = getattr(args, "split_y", None)
    if detection.on_terrain and split_y is not None and abs(detection.y - split_y) <= SPLIT_HANDOFF_BAND:
        score -= 4.0
    if detection.on_terrain == track.last_on_terrain:
        score -= 2.0
    if track.confirmed:
        score -= 3.0
    if reattach:
        score -= min(12.0, frame_gap * 0.15)
        score -= min(10.0, track.hits * 0.5)
    return score


def _same_view_duplicate_distance_limit(a, b) -> float:
    if not a.on_terrain and not b.on_terrain and a.cam == b.cam:
        return 20.0
    return 14.0


def _log_detection_duplicate_of_track(
    detection,
    owner_track_id: int,
    owner_distance: float,
    owner_limit: float,
    kept_detection,
):
    logger.debug(
        "detection_duplicate_of_track frame=%s raw=%s cam=%s x=%.1f y=%.1f track_id=%s distance=%.1f limit=%.1f kept_raw=%s kept_cam=%s kept_x=%.1f kept_y=%.1f",
        detection.frame,
        detection.raw_player_id,
        detection.cam,
        detection.x,
        detection.y,
        owner_track_id,
        owner_distance,
        owner_limit,
        kept_detection.raw_player_id,
        kept_detection.cam,
        kept_detection.x,
        kept_detection.y,
    )


def _log_candidate_rejected_duplicate_track(
    detection,
    owner_track_id: Optional[int],
    owner_distance: float,
    owner_limit: float,
    reason: str,
):
    logger.debug(
        "candidate_rejected_duplicate_track frame=%s raw=%s cam=%s x=%.1f y=%.1f track_id=%s distance=%.1f limit=%.1f reason=%s",
        detection.frame,
        detection.raw_player_id,
        detection.cam,
        detection.x,
        detection.y,
        owner_track_id if owner_track_id is not None else "none",
        owner_distance,
        owner_limit,
        reason,
    )


def _share_same_lineage(a, b) -> bool:
    if raw_id_alias_match(a.raw_player_id, b.raw_player_id):
        return True
    source_ids_a = set(getattr(a, "source_ids", []) or [])
    source_ids_b = set(getattr(b, "source_ids", []) or [])
    if source_ids_a and source_ids_b and source_ids_a.intersection(source_ids_b):
        return True
    return False


def _nearest_existing_track(
    detection,
    tracks: Dict[int, object],
    current_frame: int,
    args,
) -> Tuple[Optional[int], float, float]:
    best_track_id: Optional[int] = None
    best_score = float("inf")
    best_distance = float("inf")
    best_limit = 0.0
    for track_id, track in tracks.items():
        frame_gap = max(1, current_frame - track.last_frame)
        predicted_state, _ = _kalman_predict(track, frame_gap)
        predicted_xy = (float(predicted_state[0, 0]), float(predicted_state[1, 0]))
        distance = euclidean(predicted_xy, (detection.x, detection.y))
        limit = _track_duplicate_distance_limit(
            track,
            frame_gap,
            same_off_terrain=bool(not track.last_on_terrain and not detection.on_terrain and detection.cam == track.last_cam),
        )
        if not track.confirmed:
            limit = min(limit, _gating_distance(frame_gap, args.match_distance, args.distance_per_frame))
        if distance > limit:
            continue
        score = distance
        if track.confirmed:
            score -= 12.0
        elif track.misses > 0:
            score -= 5.0
        if detection.raw_player_id == track.last_raw_player_id:
            score -= 10.0
        elif raw_id_alias_match(track.last_raw_player_id, detection.raw_player_id):
            score -= 8.0
        if best_track_id is None or score < best_score:
            best_track_id = track_id
            best_score = score
            best_distance = distance
            best_limit = limit
    if best_track_id is None:
        return None, float("inf"), 0.0
    return best_track_id, best_distance, best_limit


def dedupe_owned_detections(
    detections: List[object],
    active_tracks: Dict[int, object],
    lost_tracks: Dict[int, object],
    current_frame: int,
    args,
    stats: Dict[str, int],
    spawn_guard=None,
) -> List[object]:
    owned_groups: Dict[int, List[Tuple[int, object, float, float]]] = {}
    passthrough: List[Tuple[int, object]] = []

    for index, detection in enumerate(detections):
        owner_track_id, owner_distance, owner_limit = _nearest_existing_track(
            detection,
            active_tracks,
            current_frame,
            args,
        )
        if owner_track_id is None:
            owner_track_id, owner_distance, owner_limit = _nearest_existing_track(
                detection,
                lost_tracks,
                current_frame,
                args,
            )
        if owner_track_id is None:
            passthrough.append((index, detection))
            continue
        owned_groups.setdefault(owner_track_id, []).append((index, detection, owner_distance, owner_limit))

    kept_entries: List[Tuple[int, object]] = list(passthrough)
    for owner_track_id, items in owned_groups.items():
        keep_index, keep_detection, keep_distance, keep_limit = min(items, key=_detection_dedup_rank)
        kept_entries.append((keep_index, keep_detection))
        for index, detection, owner_distance, owner_limit in items:
            if detection is keep_detection:
                continue
            detection.duplicate_of_track_id = owner_track_id
            stats["detection_dedup_merged"] += 1
            stats["candidate_rejected_duplicate_track"] += 1
            if spawn_guard is not None:
                spawn_guard.mark_duplicate(detection.raw_player_id, current_frame)
            _log_detection_duplicate_of_track(detection, owner_track_id, owner_distance, owner_limit, keep_detection)
            logger.debug(
                "detection_dedup_merged frame=%s raw=%s cam=%s x=%.1f y=%.1f kept_raw=%s kept_cam=%s kept_x=%.1f kept_y=%.1f track_id=%s distance=%.1f limit=%.1f reason=owner_track",
                detection.frame,
                detection.raw_player_id,
                detection.cam,
                detection.x,
                detection.y,
                keep_detection.raw_player_id,
                keep_detection.cam,
                keep_detection.x,
                keep_detection.y,
                owner_track_id,
                owner_distance,
                owner_limit,
            )
            _log_candidate_rejected_duplicate_track(detection, owner_track_id, owner_distance, owner_limit, "owner_track")

    unowned = list(passthrough)
    unowned.sort(key=lambda item: item[0])
    if len(unowned) > 1:
        consumed = set()
        for base_index, base_detection in unowned:
            if base_index in consumed:
                continue
            cluster = [(base_index, base_detection)]
            for other_index, other_detection in unowned:
                if other_index <= base_index or other_index in consumed:
                    continue
                if euclidean((base_detection.x, base_detection.y), (other_detection.x, other_detection.y)) > _same_view_duplicate_distance_limit(base_detection, other_detection):
                    continue
                same_view = base_detection.cam == other_detection.cam and base_detection.on_terrain == other_detection.on_terrain
                if not _share_same_lineage(base_detection, other_detection) and not same_view:
                    continue
                cluster.append((other_index, other_detection))
                consumed.add(other_index)
            if len(cluster) == 1:
                continue
            keep_index, keep_detection = min(
                cluster,
                key=lambda item: (
                    -int(getattr(item[1], "merged_count", 1) or 1),
                    -len(getattr(item[1], "source_cams", []) or []),
                    -len(getattr(item[1], "source_ids", []) or []),
                    0 if getattr(item[1], "spawn_allowed", True) else 1,
                    item[0],
                ),
            )
            kept_entries.append((keep_index, keep_detection))
            for index, detection in cluster:
                if detection is keep_detection:
                    continue
                detection.duplicate_of_track_id = -1
                stats["detection_dedup_merged"] += 1
                stats["candidate_rejected_duplicate_track"] += 1
                if spawn_guard is not None:
                    spawn_guard.mark_duplicate(detection.raw_player_id, current_frame)
                logger.debug(
                    "detection_dedup_merged frame=%s raw=%s cam=%s x=%.1f y=%.1f kept_raw=%s kept_cam=%s kept_x=%.1f kept_y=%.1f distance=%.1f reason=%s",
                    detection.frame,
                    detection.raw_player_id,
                    detection.cam,
                    detection.x,
                    detection.y,
                    keep_detection.raw_player_id,
                    keep_detection.cam,
                    keep_detection.x,
                    keep_detection.y,
                    euclidean((detection.x, detection.y), (keep_detection.x, keep_detection.y)),
                    "same_lineage" if _share_same_lineage(detection, keep_detection) else "same_view_close",
                )
                _log_candidate_rejected_duplicate_track(
                    detection,
                    None,
                    euclidean((detection.x, detection.y), (keep_detection.x, keep_detection.y)),
                    _same_view_duplicate_distance_limit(detection, keep_detection),
                    "same_lineage",
                )
                consumed.add(index)

    kept_entries.sort(key=lambda item: item[0])
    return [detection for _, detection in kept_entries]
