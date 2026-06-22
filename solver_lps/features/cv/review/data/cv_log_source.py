import csv
import bisect
import math
import os
import time

try:
    import cv2
except ImportError:
    cv2 = None

from solver_lps.features.ground.domain.court_geometry import court_bounds
from solver_lps.features.ground.domain.calibration import load_terrain_calibration
from solver_lps.features.cv.review.domain.tracking_lifecycle import BOOTSTRAP_IGNORE_SPAWN_GATE_FRAMES, SpawnGuard
from solver_lps.features.cv.review.domain.tracking_pipeline import (
    build_tracking_frames as build_hungarian_kalman_cv_tracks,
)


_APP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
_WORKSPACE_ROOT = os.path.dirname(_APP_ROOT)
_INTERNAL_CV_LOG_DIR = os.path.join(_APP_ROOT, "features", "cv", "review", "data", "cv_logs")
_EXTERNAL_CV_LOG_DIR = os.path.join(_WORKSPACE_ROOT, "python", "CV Logs")
DEFAULT_COURT_CENTER_X = 1250.0
DEFAULT_COURT_CENTER_Y = 900.0

CV_ALL_PLAYERS = "__all__"
CV_REID_MAX_DISTANCE = 125.0
CV_REID_DISTANCE_PER_FRAME = 18.0
CV_REID_MAX_FRAME_GAP = 30
CV_DUPLICATE_DISTANCE = 32.0
CV_POSITION_SMOOTH_ALPHA = 0.18
CV_VELOCITY_SMOOTH_ALPHA = 0.22
CV_TENTATIVE_POSITION_ALPHA = 0.28
CV_TENTATIVE_VELOCITY_ALPHA = 0.35
CV_RENDER_SMOOTH_ALPHA = 0.16
CV_RENDER_SELECTED_ALPHA = 0.22
CV_RENDER_SMOOTH_DEADZONE = 12.0
CV_RENDER_SNAP_DISTANCE = 240.0
CV_MIDCOURT_BAND = 90.0
CV_CROSS_HALF_EXTRA_BAND_PER_FRAME = 10.0
CV_SMOOTH_DEADZONE = 5.0
CV_MIN_TRACK_OBSERVATIONS = 2
CV_MAX_SPEED_PER_FRAME = 140.0
CV_REATTACH_MAX_FRAME_GAP = 140
CV_REATTACH_MAX_DISTANCE = 240.0
CV_REATTACH_CONFIDENCE_BONUS = 12.0
CV_REATTACH_AGE_BONUS = 0.25
CV_DUPLICATE_SUPPRESSION_DISTANCE = 75.0
CV_DUPLICATE_SUPPRESSION_FRAME_GAP = 160
CV_VISIBLE_DUPLICATE_DISTANCE = 80.0
CV_SPAWN_BORDER_MARGIN = 160.0
CV_SPAWN_WARMUP_SECONDS = 5.0
CV_MIDFIELD_SPAWN_MIN_OBSERVATIONS = 2
CV_HUNGARIAN_MERGE_DISTANCE = 35.0
CV_HUNGARIAN_MATCH_DISTANCE = 70.0
CV_HUNGARIAN_DISTANCE_PER_FRAME = 18.0
CV_HUNGARIAN_MAX_MISSES = 15
CV_HUNGARIAN_MIN_HITS = 3
CV_ON_TERRAIN_OVERRIDE_DISTANCE = 85.0
CV_OFF_TERRAIN_MATCH_PENALTY = 28.0
CV_ON_TERRAIN_MATCH_BONUS = 10.0


def _first_existing_path(*candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def _resolve_default_cv_log_path(external_dir=_EXTERNAL_CV_LOG_DIR, internal_dir=_INTERNAL_CV_LOG_DIR):
    return _first_existing_path(
        os.path.join(external_dir, "positions_raw.csv"),
        os.path.join(internal_dir, "positions_raw.csv"),
        os.path.join(external_dir, "positions_merged.csv"),
        os.path.join(internal_dir, "positions_merged.csv"),
    )


def _resolve_refreshable_cv_log_path(csv_path):
    if not csv_path:
        return csv_path
    root, ext = os.path.splitext(csv_path)
    if ext.lower() == ".csv":
        raw_candidate = root.replace("positions_merged", "positions_raw") + ext
        if os.path.exists(raw_candidate):
            return raw_candidate
        if csv_path.endswith("positions_merged.csv"):
            sibling_raw = os.path.join(os.path.dirname(csv_path), "positions_raw.csv")
            if os.path.exists(sibling_raw):
                return sibling_raw
    return csv_path

DEFAULT_CV_LOG_PATH = _resolve_default_cv_log_path()
DEFAULT_CV_CALIBRATION_PATH = _first_existing_path(
    os.path.join(_EXTERNAL_CV_LOG_DIR, "calibration.json"),
    os.path.join(_INTERNAL_CV_LOG_DIR, "calibration.json"),
)
DEFAULT_CV_VIDEO_PATH = _first_existing_path(
    os.path.join(_EXTERNAL_CV_LOG_DIR, "tracking_composite.mp4"),
    os.path.join(_INTERNAL_CV_LOG_DIR, "tracking_composite.mp4"),
)
DEFAULT_CV_METADATA_PATH = _first_existing_path(
    os.path.join(_EXTERNAL_CV_LOG_DIR, "run_metadata.json"),
    os.path.join(_INTERNAL_CV_LOG_DIR, "run_metadata.json"),
)


def distance_2d(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _normalize_player_id(raw_player_id):
    text = str(raw_player_id).strip()
    return text if text else "player"


def _normalize_expected_player_count(count):
    if count in (None, "", False):
        return None
    try:
        value = int(count)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def _resolve_cv_video_path(video_path=None):
    candidates = []
    if video_path:
        candidates.append(video_path)
    if DEFAULT_CV_LOG_PATH:
        candidates.append(os.path.join(os.path.dirname(DEFAULT_CV_LOG_PATH), "tracking_composite.mp4"))
    candidates.append(DEFAULT_CV_VIDEO_PATH)

    metadata_path = DEFAULT_CV_METADATA_PATH
    if os.path.exists(metadata_path):
        try:
            import json
            with open(metadata_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            csv_path = metadata.get("csv_path")
            if csv_path:
                candidates.append(os.path.join(os.path.dirname(csv_path), "tracking_composite.mp4"))
        except Exception:
            pass

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _resolve_related_path(reference_path, explicit_path, filename, fallback_path):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    if reference_path:
        candidates.append(os.path.join(os.path.dirname(reference_path), filename))
    if fallback_path:
        candidates.append(fallback_path)
    return _first_existing_path(*candidates)


def _resolve_terrain_image_path(image_path):
    if not image_path:
        return None
    directory = os.path.dirname(image_path)
    if directory:
        named_candidate = os.path.join(directory, "terrain-de-basket.jpg")
        if os.path.exists(named_candidate):
            return named_candidate
    root, ext = os.path.splitext(image_path)
    if ext.lower() in {".jpg", ".jpeg"}:
        if os.path.exists(image_path):
            return image_path
        png_candidate = root + ".png"
        if os.path.exists(png_candidate):
            return png_candidate
        return None
    jpg_candidate = root + ".jpg"
    jpeg_candidate = root + ".jpeg"
    if os.path.exists(jpg_candidate):
        return jpg_candidate
    if os.path.exists(jpeg_candidate):
        return jpeg_candidate
    return image_path if os.path.exists(image_path) else None


def _smooth_point(old_point, new_point, alpha):
    if old_point is None:
        return new_point
    if distance_2d(old_point, new_point) <= CV_SMOOTH_DEADZONE:
        return old_point
    return (
        old_point[0] + (new_point[0] - old_point[0]) * alpha,
        old_point[1] + (new_point[1] - old_point[1]) * alpha,
    )


def _is_near_midcourt(y_value, split_y, band=CV_MIDCOURT_BAND):
    if split_y is None:
        return False
    return abs(y_value - split_y) <= band


def _infer_split_y(frames):
    split_candidates = []
    for frame in frames:
        for player in frame["players"]:
            split_candidates.append(player["y"])
    if not split_candidates:
        return None
    return (min(split_candidates) + max(split_candidates)) / 2.0


def _infer_frame_bounds(frames):
    xs = []
    ys = []
    for frame in frames:
        for player in frame["players"]:
            xs.append(player["x"])
            ys.append(player["y"])
    if not xs or not ys:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def _collect_coordinate_stats(frames):
    all_x = []
    all_y = []
    on_x = []
    on_y = []
    for frame in frames:
        for player in frame["players"]:
            x_value = float(player["x"])
            y_value = float(player["y"])
            all_x.append(x_value)
            all_y.append(y_value)
            if player.get("on_terrain", True):
                on_x.append(x_value)
                on_y.append(y_value)
    if not all_x or not all_y:
        return None
    reference_x = on_x if on_x else all_x
    reference_y = on_y if on_y else all_y
    return {
        "all_bounds": (min(all_x), max(all_x), min(all_y), max(all_y)),
        "on_terrain_bounds": (min(reference_x), max(reference_x), min(reference_y), max(reference_y)),
    }


def _terrain_reference_bounds(calibration):
    if not calibration:
        return None
    terrain_size = calibration.get("terrain_png_size")
    if isinstance(terrain_size, (list, tuple)) and len(terrain_size) == 2:
        try:
            width = max(float(terrain_size[0]) - 1.0, 1.0)
            height = max(float(terrain_size[1]) - 1.0, 1.0)
            return 0.0, width, 0.0, height
        except (TypeError, ValueError):
            pass
    return calibration.get("bounds")


def _detect_coordinate_space(calibration, coordinate_stats):
    if not calibration or not coordinate_stats:
        return "solver_world"
    terrain_bounds = _terrain_reference_bounds(calibration)
    if terrain_bounds is None:
        return "solver_world"
    min_x, max_x, min_y, max_y = coordinate_stats["on_terrain_bounds"]
    left, right, top, bottom = terrain_bounds
    width = max(float(right) - float(left), 1.0)
    height = max(float(bottom) - float(top), 1.0)
    margin_x = width * 0.35
    margin_y = height * 0.35
    if (
        min_x >= left - margin_x
        and max_x <= right + margin_x
        and min_y >= top - margin_y
        and max_y <= bottom + margin_y
    ):
        return "terrain_pixels"
    return "solver_world"


def _crosses_split_line(prev_y, new_y, split_y):
    if split_y is None:
        return False
    return (prev_y <= split_y <= new_y) or (new_y <= split_y <= prev_y)


def _allow_cross_half_match(
    prev_half,
    new_half,
    prev_y,
    new_y,
    split_y,
    prev_on_terrain=True,
    new_on_terrain=True,
    frame_gap=1,
):
    if prev_half == new_half:
        return True
    if not prev_on_terrain or not new_on_terrain:
        return False
    dynamic_band = CV_MIDCOURT_BAND + max(0, frame_gap - 1) * CV_CROSS_HALF_EXTRA_BAND_PER_FRAME
    if _crosses_split_line(prev_y, new_y, split_y):
        return True
    if prev_half == "merged" or new_half == "merged":
        return _is_near_midcourt(prev_y, split_y, band=dynamic_band) or _is_near_midcourt(new_y, split_y, band=dynamic_band)
    return _is_near_midcourt(prev_y, split_y, band=dynamic_band) and _is_near_midcourt(new_y, split_y, band=dynamic_band)


def _merge_same_person_candidates(players, split_y):
    if len(players) < 2:
        return players
    merged = []
    used = set()
    for index, player in enumerate(players):
        if index in used:
            continue
        cluster = [player]
        used.add(index)
        for other_index in range(index + 1, len(players)):
            if other_index in used:
                continue
            other = players[other_index]
            if not player.get("on_terrain", True) or not other.get("on_terrain", True):
                continue
            if player["half"] == other["half"]:
                continue
            if not (
                _is_near_midcourt(player["y"], split_y)
                and _is_near_midcourt(other["y"], split_y)
            ):
                continue
            if distance_2d((player["x"], player["y"]), (other["x"], other["y"])) > CV_DUPLICATE_DISTANCE:
                continue
            cluster.append(other)
            used.add(other_index)
        if len(cluster) == 1:
            merged.append(cluster[0])
            continue
        merged.append(
            {
                "frame": player["frame"],
                "timestamp_s": player["timestamp_s"],
                "player_id": cluster[0]["player_id"],
                "raw_player_id": cluster[0]["raw_player_id"],
                "source_player_ids": sorted({item["raw_player_id"] for item in cluster}),
                "x": sum(item["x"] for item in cluster) / len(cluster),
                "y": sum(item["y"] for item in cluster) / len(cluster),
                "half": "merged",
                "on_terrain": True,
            }
        )
    return merged


def _suppress_off_terrain_near_on_terrain(players, radius=CV_ON_TERRAIN_OVERRIDE_DISTANCE):
    if len(players) < 2:
        return players
    on_players = [player for player in players if player.get("on_terrain", True)]
    if not on_players:
        return players
    filtered = []
    for player in players:
        if player.get("on_terrain", True):
            filtered.append(player)
            continue
        keep_player = True
        for on_player in on_players:
            if player["half"] != on_player["half"]:
                continue
            if distance_2d((player["x"], player["y"]), (on_player["x"], on_player["y"])) <= radius:
                keep_player = False
                break
        if keep_player:
            filtered.append(player)
    return filtered


def _track_prediction(track, frame_gap):
    return (
        track["smoothed_pos"][0] + track["velocity"][0] * frame_gap,
        track["smoothed_pos"][1] + track["velocity"][1] * frame_gap,
    )


def _clamp_speed(velocity):
    speed = math.hypot(velocity[0], velocity[1])
    if speed <= CV_MAX_SPEED_PER_FRAME or speed <= 0.0:
        return velocity
    scale = CV_MAX_SPEED_PER_FRAME / speed
    return velocity[0] * scale, velocity[1] * scale


def _track_confidence(track, residual_dist):
    confidence = 0.0
    confidence += min(0.45, track["hits"] * 0.18)
    confidence += min(0.20, track["age"] * 0.04)
    confidence -= min(0.30, track["misses"] * 0.12)
    confidence -= min(0.20, residual_dist / 500.0)
    if track.get("confirmed"):
        confidence += 0.10
    return max(0.0, min(1.0, confidence))


def _track_match_score(track, detection, predicted, frame_gap, *, reattach=False):
    dist = distance_2d(predicted, (detection["x"], detection["y"]))
    if reattach:
        max_distance = CV_REATTACH_MAX_DISTANCE
    else:
        max_distance = CV_REID_MAX_DISTANCE + CV_REID_DISTANCE_PER_FRAME * max(0, frame_gap - 1)
    if dist > max_distance:
        return None

    score = dist
    if detection["raw_player_id"] == track["last_raw_player_id"]:
        score -= 18.0
    if detection["half"] == track["last_half"]:
        score -= 6.0
    if detection.get("on_terrain", True):
        score -= CV_ON_TERRAIN_MATCH_BONUS
    else:
        score += CV_OFF_TERRAIN_MATCH_PENALTY
    if track["confirmed"]:
        score -= 3.0
    else:
        score += 2.0
    if reattach:
        score -= CV_REATTACH_CONFIDENCE_BONUS
        score -= min(15.0, track["age"] * CV_REATTACH_AGE_BONUS)
        score -= min(8.0, track["hits"] * 0.5)
    return score


def _track_snapshot(track, frame, *, raw_detection=None):
    raw_detection = raw_detection or {}
    source_player_ids = list(raw_detection.get("source_player_ids") or [track["last_raw_player_id"]])
    raw_x = raw_detection.get("x", track["smoothed_pos"][0])
    raw_y = raw_detection.get("y", track["smoothed_pos"][1])
    return {
        "frame": frame["frame"],
        "timestamp_s": frame["timestamp_s"],
        "player_id": track["stable_id"],
        "raw_player_id": track["last_raw_player_id"],
        "source_player_ids": source_player_ids,
        "x": track["smoothed_pos"][0],
        "y": track["smoothed_pos"][1],
        "raw_x": raw_x,
        "raw_y": raw_y,
        "vx": track["velocity"][0],
        "vy": track["velocity"][1],
        "confidence": round(track["confidence"], 3),
        "track_age": track["age"],
        "track_hits": track["hits"],
        "track_misses": track["misses"],
        "half": track["last_half"],
        "on_terrain": track.get("last_on_terrain", True),
    }


def _visible_player_score(player):
    return (
        1 if player.get("selected") else 0,
        float(player.get("confidence", 0.0) or 0.0),
        float(player.get("track_age", 0) or 0),
        -float(player.get("track_misses", 0) or 0),
    )


def _tracked_player_index(player):
    player_id = str(player.get("player_id", "") or "")
    if player_id.startswith("P") and player_id[1:].isdigit():
        return int(player_id[1:])
    return 10**9


def _prefer_duplicate_player(a, b):
    if bool(a.get("selected")) != bool(b.get("selected")):
        return a if a.get("selected") else b

    a_index = _tracked_player_index(a)
    b_index = _tracked_player_index(b)
    if a_index != b_index:
        return a if a_index < b_index else b

    a_score = _visible_player_score(a)
    b_score = _visible_player_score(b)
    return a if a_score >= b_score else b


def _limit_visible_players(players, expected_count):
    expected_count = _normalize_expected_player_count(expected_count)
    if expected_count is None or len(players) <= expected_count:
        return players

    selected_players = [player for player in players if player.get("selected")]
    remaining_players = [player for player in players if not player.get("selected")]
    remaining_players.sort(key=_visible_player_score, reverse=True)

    visible = list(selected_players)
    budget = max(0, expected_count - len(visible))
    visible.extend(remaining_players[:budget])

    if len(visible) > expected_count:
        visible = visible[:expected_count]

    return sorted(visible, key=_visible_player_score, reverse=True)


def _dedupe_visible_players(players, duplicate_distance=CV_VISIBLE_DUPLICATE_DISTANCE):
    if len(players) < 2:
        return players

    deduped = []
    for player in players:
        duplicate_index = None
        for kept_index, kept in enumerate(deduped):
            if distance_2d((player["x"], player["y"]), (kept["x"], kept["y"])) <= duplicate_distance:
                duplicate_index = kept_index
                break
        if duplicate_index is None:
            deduped.append(player)
            continue
        deduped[duplicate_index] = _prefer_duplicate_player(deduped[duplicate_index], player)
    return deduped


def _update_track(track, detection, frame_gap):
    predicted = _track_prediction(track, frame_gap)
    measured = (detection["x"], detection["y"])
    residual = (measured[0] - predicted[0], measured[1] - predicted[1])
    residual_dist = math.hypot(residual[0], residual[1])

    position_alpha = CV_POSITION_SMOOTH_ALPHA if track["confirmed"] else CV_TENTATIVE_POSITION_ALPHA
    velocity_alpha = CV_VELOCITY_SMOOTH_ALPHA if track["confirmed"] else CV_TENTATIVE_VELOCITY_ALPHA

    updated_pos = (
        predicted[0] + residual[0] * position_alpha,
        predicted[1] + residual[1] * position_alpha,
    )
    measured_velocity = (
        (measured[0] - track["smoothed_pos"][0]) / max(1, frame_gap),
        (measured[1] - track["smoothed_pos"][1]) / max(1, frame_gap),
    )
    updated_velocity = (
        track["velocity"][0] + (measured_velocity[0] - track["velocity"][0]) * velocity_alpha,
        track["velocity"][1] + (measured_velocity[1] - track["velocity"][1]) * velocity_alpha,
    )
    updated_velocity = _clamp_speed(updated_velocity)

    track["smoothed_pos"] = _smooth_point(track["smoothed_pos"], updated_pos, position_alpha)
    track["velocity"] = updated_velocity
    track["last_frame"] = detection["frame"]
    track["last_timestamp_s"] = detection["timestamp_s"]
    track["last_raw_player_id"] = detection["raw_player_id"]
    track["last_half"] = detection["half"]
    track["last_on_terrain"] = detection.get("on_terrain", True)
    track["hits"] += 1
    track["misses"] = 0
    track["age"] = detection["frame"] - track["birth_frame"] + 1
    track["last_residual"] = residual_dist
    track["confidence"] = _track_confidence(track, residual_dist)
    required_confirmations = int(track.get("min_confirmations", CV_MIN_TRACK_OBSERVATIONS))
    if track["hits"] >= required_confirmations and track["age"] >= required_confirmations:
        track["confirmed"] = True


def _register_new_track(track_id, detection, spawn_category="midfield"):
    min_confirmations = (
        CV_MIN_TRACK_OBSERVATIONS
        if spawn_category in ("warmup", "border")
        else CV_MIDFIELD_SPAWN_MIN_OBSERVATIONS
    )
    return {
        "stable_id": f"P{track_id}",
        "smoothed_pos": (detection["x"], detection["y"]),
        "velocity": (0.0, 0.0),
        "last_frame": detection["frame"],
        "last_timestamp_s": detection["timestamp_s"],
        "last_raw_player_id": detection["raw_player_id"],
        "last_half": detection["half"],
        "last_on_terrain": detection.get("on_terrain", True),
        "birth_frame": detection["frame"],
        "hits": 1,
        "misses": 0,
        "age": 1,
        "last_residual": 0.0,
        "confidence": 0.18 if detection.get("on_terrain", True) else 0.08,
        "confirmed": False,
        "spawn_category": spawn_category,
        "min_confirmations": min_confirmations,
    }


def _reactivate_track(track, detection, frame_gap):
    predicted = _track_prediction(track, frame_gap)
    measured = (detection["x"], detection["y"])
    residual = (measured[0] - predicted[0], measured[1] - predicted[1])
    residual_dist = math.hypot(residual[0], residual[1])

    position_alpha = CV_POSITION_SMOOTH_ALPHA if track["confirmed"] else CV_TENTATIVE_POSITION_ALPHA
    velocity_alpha = CV_VELOCITY_SMOOTH_ALPHA if track["confirmed"] else CV_TENTATIVE_VELOCITY_ALPHA
    updated_pos = (
        predicted[0] + residual[0] * position_alpha,
        predicted[1] + residual[1] * position_alpha,
    )
    measured_velocity = (
        (measured[0] - track["smoothed_pos"][0]) / max(1, frame_gap),
        (measured[1] - track["smoothed_pos"][1]) / max(1, frame_gap),
    )
    updated_velocity = (
        track["velocity"][0] + (measured_velocity[0] - track["velocity"][0]) * velocity_alpha,
        track["velocity"][1] + (measured_velocity[1] - track["velocity"][1]) * velocity_alpha,
    )
    updated_velocity = _clamp_speed(updated_velocity)

    track["smoothed_pos"] = _smooth_point(track["smoothed_pos"], updated_pos, position_alpha)
    track["velocity"] = updated_velocity
    track["last_frame"] = detection["frame"]
    track["last_timestamp_s"] = detection["timestamp_s"]
    track["last_raw_player_id"] = detection["raw_player_id"]
    track["last_half"] = detection["half"]
    track["last_on_terrain"] = detection.get("on_terrain", True)
    track["hits"] += 1
    track["misses"] = 0
    track["age"] = max(track["age"] + frame_gap, detection["frame"] - track["birth_frame"] + 1)
    track["last_residual"] = residual_dist
    track["confidence"] = _track_confidence(track, residual_dist)
    required_confirmations = int(track.get("min_confirmations", CV_MIN_TRACK_OBSERVATIONS))
    if track["hits"] >= required_confirmations and track["age"] >= required_confirmations:
        track["confirmed"] = True


def _track_is_visible(track):
    return track["confirmed"]


def _is_near_spawn_border(detection, bounds, margin=CV_SPAWN_BORDER_MARGIN):
    if bounds is None:
        return True
    left, right, top, bottom = bounds
    x = detection["x"]
    y = detection["y"]
    return (
        abs(x - left) <= margin
        or abs(x - right) <= margin
        or abs(y - top) <= margin
        or abs(y - bottom) <= margin
    )


def _spawn_category(detection, frame, bounds, clip_start_timestamp_s):
    if clip_start_timestamp_s is None:
        return "border" if _is_near_spawn_border(detection, bounds) else "midfield"
    elapsed_s = max(0.0, float(frame["timestamp_s"]) - float(clip_start_timestamp_s))
    if elapsed_s <= CV_SPAWN_WARMUP_SECONDS:
        return "warmup"
    return "border" if _is_near_spawn_border(detection, bounds) else "midfield"


def _should_suppress_new_track(detection, frame, active_tracks, dormant_tracks, split_y):
    for track in active_tracks.values():
        if not track.get("confirmed"):
            continue
        predicted = _track_prediction(track, max(1, frame["frame"] - track["last_frame"]))
        if not _allow_cross_half_match(
            track["last_half"],
            detection["half"],
            predicted[1],
            detection["y"],
            split_y,
            track.get("last_on_terrain", True),
            detection.get("on_terrain", True),
            frame_gap=max(1, frame["frame"] - track["last_frame"]),
        ):
            continue
        if distance_2d(predicted, (detection["x"], detection["y"])) <= CV_DUPLICATE_SUPPRESSION_DISTANCE:
            return True

    for track in dormant_tracks.values():
        if not track.get("confirmed"):
            continue
        frame_gap = frame["frame"] - track["last_frame"]
        if frame_gap <= 0 or frame_gap > CV_DUPLICATE_SUPPRESSION_FRAME_GAP:
            continue
        predicted = _track_prediction(track, frame_gap)
        if not _allow_cross_half_match(
            track["last_half"],
            detection["half"],
            predicted[1],
            detection["y"],
            split_y,
            track.get("last_on_terrain", True),
            detection.get("on_terrain", True),
            frame_gap=frame_gap,
        ):
            continue
        if distance_2d(predicted, (detection["x"], detection["y"])) <= CV_DUPLICATE_SUPPRESSION_DISTANCE:
            return True

    return False


def _build_tracked_cv_frames(frames, split_y=None):
    resolved_split_y = split_y if split_y is not None else _infer_split_y(frames)
    spawn_bounds = _infer_frame_bounds(frames)
    clip_start_timestamp_s = frames[0]["timestamp_s"] if frames else None
    tracked_frames = []
    active_tracks = {}
    dormant_tracks = {}
    next_track_id = 1
    spawn_guard = SpawnGuard()

    for frame in frames:
        detections = _merge_same_person_candidates(frame["players"], resolved_split_y)
        detections = _suppress_off_terrain_near_on_terrain(detections)
        candidates = []
        track_ids = sorted(active_tracks.keys())
        dormant_ids = sorted(dormant_tracks.keys())

        for track_id in track_ids:
            track = active_tracks[track_id]
            frame_gap = frame["frame"] - track["last_frame"]
            if frame_gap <= 0 or frame_gap > CV_REID_MAX_FRAME_GAP:
                continue
            predicted = _track_prediction(track, frame_gap)
            for detection_index, detection in enumerate(detections):
                if not _allow_cross_half_match(
                    track["last_half"],
                    detection["half"],
                    predicted[1],
                    detection["y"],
                    resolved_split_y,
                    track.get("last_on_terrain", True),
                    detection.get("on_terrain", True),
                    frame_gap=frame_gap,
                ):
                    continue
                score = _track_match_score(track, detection, predicted, frame_gap, reattach=False)
                if score is None:
                    continue
                dist = distance_2d(predicted, (detection["x"], detection["y"]))
                candidates.append((score, dist, track_id, detection_index, "active"))

        for track_id in dormant_ids:
            track = dormant_tracks[track_id]
            frame_gap = frame["frame"] - track["last_frame"]
            if frame_gap <= 0 or frame_gap > CV_REATTACH_MAX_FRAME_GAP:
                continue
            predicted = _track_prediction(track, frame_gap)
            for detection_index, detection in enumerate(detections):
                if not _allow_cross_half_match(
                    track["last_half"],
                    detection["half"],
                    predicted[1],
                    detection["y"],
                    resolved_split_y,
                    track.get("last_on_terrain", True),
                    detection.get("on_terrain", True),
                    frame_gap=frame_gap,
                ):
                    continue
                score = _track_match_score(track, detection, predicted, frame_gap, reattach=True)
                if score is None:
                    continue
                dist = distance_2d(predicted, (detection["x"], detection["y"]))
                candidates.append((score, dist, track_id, detection_index, "dormant"))

        assigned_tracks = set()
        assigned_detections = set()
        assignments = {}
        assignment_sources = {}
        for score, dist, track_id, detection_index, source in sorted(candidates):
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            assignments[detection_index] = track_id
            assignment_sources[detection_index] = source
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)

        matched_track_ids = set()
        frame_players = []
        for detection_index, detection in enumerate(detections):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
                reattach_target = None
                reattach_score = None
                for dormant_id in dormant_ids:
                    if dormant_id not in dormant_tracks:
                        continue
                    dormant_track = dormant_tracks[dormant_id]
                    frame_gap = frame["frame"] - dormant_track["last_frame"]
                    if frame_gap <= 0 or frame_gap > CV_REATTACH_MAX_FRAME_GAP:
                        continue
                    predicted = _track_prediction(dormant_track, frame_gap)
                    if not _allow_cross_half_match(
                        dormant_track["last_half"],
                        detection["half"],
                        predicted[1],
                        detection["y"],
                        resolved_split_y,
                        dormant_track.get("last_on_terrain", True),
                        detection.get("on_terrain", True),
                        frame_gap=frame_gap,
                    ):
                        continue
                    score = _track_match_score(dormant_track, detection, predicted, frame_gap, reattach=True)
                    if score is None:
                        continue
                    if reattach_score is None or score < reattach_score:
                        reattach_target = dormant_id
                        reattach_score = score
                if reattach_target is not None:
                    active_tracks[reattach_target] = dormant_tracks.pop(reattach_target)
                    track = active_tracks[reattach_target]
                    frame_gap = max(1, detection["frame"] - track["last_frame"])
                    _reactivate_track(track, detection, frame_gap)
                    track_id = reattach_target
                    matched_track_ids.add(track_id)
                else:
                    if not spawn_guard.should_allow_spawn(
                        detection,
                        frame["frame"],
                        clip_start_timestamp_s or 0.0,
                        spawn_bounds,
                        bootstrap_ignore_spawn_gate_frames=BOOTSTRAP_IGNORE_SPAWN_GATE_FRAMES,
                    ):
                        continue
                    if _should_suppress_new_track(detection, frame, active_tracks, dormant_tracks, resolved_split_y):
                        continue
                    spawn_category = _spawn_category(detection, frame, spawn_bounds, clip_start_timestamp_s)
                    active_tracks[track_id] = _register_new_track(track_id, detection, spawn_category=spawn_category)
            else:
                source = assignment_sources.get(detection_index, "active")
                if source == "dormant" and track_id in dormant_tracks:
                    active_tracks[track_id] = dormant_tracks.pop(track_id)
                track = active_tracks[track_id]
                frame_gap = max(1, detection["frame"] - track["last_frame"])
                _reactivate_track(track, detection, frame_gap)
                matched_track_ids.add(track_id)

            track = active_tracks[track_id]
            spawn_guard.observe(track.last_raw_player_id, detection["frame"], detection.get("on_terrain", True))
            if not _track_is_visible(track):
                continue
            frame_players.append(_track_snapshot(track, frame, raw_detection=detection))

        for track_id in list(active_tracks.keys()):
            if track_id in matched_track_ids:
                continue
            track = active_tracks[track_id]
            frame_gap = frame["frame"] - track["last_frame"]
            if frame_gap <= 0:
                continue
            if frame_gap > CV_REID_MAX_FRAME_GAP:
                if track.get("confirmed"):
                    dormant_tracks[track_id] = active_tracks.pop(track_id)
                else:
                    del active_tracks[track_id]

        frame_players = _dedupe_visible_players(frame_players)

        tracked_frames.append(
            {
                "frame": frame["frame"],
                "timestamp_s": frame["timestamp_s"],
                "sequence_index": frame.get("sequence_index", len(tracked_frames)),
                "players": sorted(frame_players, key=lambda item: item["player_id"]),
            }
        )
    return tracked_frames


def _filter_short_lived_tracks(frames, min_observations=CV_MIN_TRACK_OBSERVATIONS):
    if min_observations <= 1:
        return frames

    counts = {}
    for frame in frames:
        for player in frame["players"]:
            player_id = player["player_id"]
            counts[player_id] = counts.get(player_id, 0) + 1

    allowed_ids = {player_id for player_id, count in counts.items() if count >= min_observations}
    if len(allowed_ids) == len(counts):
        return frames

    filtered_frames = []
    for frame in frames:
        players = [player for player in frame["players"] if player["player_id"] in allowed_ids]
        filtered_frames.append(
            {
                "frame": frame["frame"],
                "timestamp_s": frame["timestamp_s"],
                "sequence_index": frame.get("sequence_index", len(filtered_frames)),
                "players": players,
            }
        )
    return filtered_frames


def _load_cv_frames(csv_path, split_y=None, calibration_path=None, expected_player_count=None):
    try:
        tracked_frames, _stats = build_hungarian_kalman_cv_tracks(
            csv_path,
            calibration=calibration_path or DEFAULT_CV_CALIBRATION_PATH,
            merge_distance=CV_HUNGARIAN_MERGE_DISTANCE,
            match_distance=CV_HUNGARIAN_MATCH_DISTANCE,
            distance_per_frame=CV_HUNGARIAN_DISTANCE_PER_FRAME,
            reattach_distance=CV_REATTACH_MAX_DISTANCE,
            max_misses=CV_HUNGARIAN_MAX_MISSES,
            max_reattach_gap=CV_REATTACH_MAX_FRAME_GAP,
            min_hits=CV_HUNGARIAN_MIN_HITS,
            expected_players=_normalize_expected_player_count(expected_player_count) or 10,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to build CV review frames from {csv_path}") from exc
    return _filter_short_lived_tracks(tracked_frames, min_observations=CV_HUNGARIAN_MIN_HITS)


def _load_cv_calibration(calibration_path):
    return load_terrain_calibration(calibration_path)


class CvLogSource:
    def __init__(self, csv_path, calibration_path=None, player_id=None, video_path=None, expected_player_count=None):
        self.csv_path = _resolve_refreshable_cv_log_path(csv_path)
        resolved_calibration_path = _resolve_related_path(
            self.csv_path,
            calibration_path,
            "calibration.json",
            DEFAULT_CV_CALIBRATION_PATH,
        )
        self.calibration = _load_cv_calibration(resolved_calibration_path)
        self.video_path = _resolve_cv_video_path(
            _resolve_related_path(
                self.csv_path,
                video_path,
                "tracking_composite.mp4",
                DEFAULT_CV_VIDEO_PATH,
            )
        )
        self.video_capture = None
        self.video_fps = None
        self.expected_player_count = _normalize_expected_player_count(expected_player_count)
        split_y = None if self.calibration is None else self.calibration.get("split_y")
        self.frames = _load_cv_frames(
            self.csv_path,
            split_y=split_y,
            calibration_path=calibration_path,
            expected_player_count=self.expected_player_count,
        )
        self.all_player_ids = []
        for frame in self.frames:
            for player in frame["players"]:
                player_id_value = player["player_id"]
                if player_id_value not in self.all_player_ids:
                    self.all_player_ids.append(player_id_value)
        self.player_id = _normalize_player_id(player_id) if player_id else None
        self.frame_index = 0
        self._last_smoothed_frame_index = None
        self._smoothed_scene_positions = {}
        self.court_bounds = court_bounds(DEFAULT_COURT_CENTER_X, DEFAULT_COURT_CENTER_Y)
        self.coordinate_stats = _collect_coordinate_stats(self.frames)
        self.coordinate_space = _detect_coordinate_space(self.calibration, self.coordinate_stats)
        self.playback_started_at = time.monotonic()
        self.playback_position_s = 0.0
        self.playback_paused = False
        self._last_video_sequence_index = None
        self._last_video_frame = None
        self._last_video_read_at = 0.0
        self.video_scrubbing = False
        normalized_timestamps = []
        if self.frames:
            first_timestamp = self.frames[0]["timestamp_s"]
            normalized_timestamps = [max(0.0, frame["timestamp_s"] - first_timestamp) for frame in self.frames]
        self.normalized_timestamps = normalized_timestamps
        positive_deltas = [
            self.normalized_timestamps[index] - self.normalized_timestamps[index - 1]
            for index in range(1, len(self.normalized_timestamps))
            if self.normalized_timestamps[index] > self.normalized_timestamps[index - 1]
        ]
        self.nominal_frame_period_s = min(positive_deltas) if positive_deltas else (1.0 / 25.0)
        self.playback_duration_s = (
            self.normalized_timestamps[-1] + self.nominal_frame_period_s if self.normalized_timestamps else 0.0
        )
        if self.video_path and cv2 is not None:
            capture = cv2.VideoCapture(self.video_path)
            if capture.isOpened():
                self.video_capture = capture
                self.video_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            else:
                capture.release()
                self.video_path = None
        elif cv2 is None:
            self.video_path = None

    @property
    def has_video(self):
        return self.video_capture is not None

    @property
    def playback_frame_count(self):
        return len(self.frames)

    @property
    def status(self):
        if not self.frames:
            return "vision: aucun frame"
        return f"vision: {os.path.basename(self.csv_path)} | frame {self.frame_index + 1}/{len(self.frames)}"

    @property
    def playback_state(self):
        return {
            "paused": self.playback_paused,
            "position_s": self.playback_position_s,
            "duration_s": self.playback_duration_s,
            "frame_index": self.frame_index,
            "frame_count": len(self.frames),
        }

    @property
    def view_config(self):
        if not self.calibration:
            return None
        left, right, top, bottom = self.court_bounds
        return {
            "mode": "world",
            "center_x": (left + right) / 2.0,
            "center_y": (top + bottom) / 2.0,
            "bounds": (left, right, top, bottom),
            "split_y": None,
            "terrain_image_path": _resolve_terrain_image_path(self.calibration.get("terrain_image_path")),
            "coord_transform": "swap_xy_flip_y" if getattr(self, "coordinate_space", None) == "terrain_pixels" else "identity",
        }

    @property
    def analytics_bounds(self):
        if self.calibration:
            return self.court_bounds
        if not self.frames:
            return None
        xs = [player["x"] for frame in self.frames for player in frame["players"]]
        ys = [player["y"] for frame in self.frames for player in frame["players"]]
        return min(xs), max(xs), min(ys), max(ys)

    def _select_player(self, players):
        if not players:
            return None
        if self.player_id == CV_ALL_PLAYERS:
            return None
        if self.player_id is None:
            self.player_id = players[0]["player_id"]
        for player in players:
            if player["player_id"] == self.player_id:
                return player
            if player.get("raw_player_id") == self.player_id:
                self.player_id = player["player_id"]
                return player
            if self.player_id in player.get("source_player_ids", []):
                self.player_id = player["player_id"]
                return player
        return None

    def set_selected_player(self, player_id):
        if player_id in (None, "", CV_ALL_PLAYERS):
            self.player_id = CV_ALL_PLAYERS
            return
        self.player_id = _normalize_player_id(player_id)

    def set_expected_player_count(self, count):
        self.expected_player_count = _normalize_expected_player_count(count)
        return self.expected_player_count

    def cycle_selected_player(self, step=1):
        if not self.all_player_ids:
            return self.player_id
        selectable_ids = [CV_ALL_PLAYERS] + list(self.all_player_ids)
        if self.player_id not in selectable_ids:
            self.player_id = selectable_ids[0]
            return self.player_id
        current_index = selectable_ids.index(self.player_id)
        self.player_id = selectable_ids[(current_index + step) % len(selectable_ids)]
        return self.player_id

    def set_playback(self, position_s=None, paused=None):
        if position_s is not None:
            self.playback_position_s = min(self.playback_duration_s, max(0.0, float(position_s)))
            self.playback_started_at = time.monotonic() - self.playback_position_s
        if paused is not None:
            self.playback_paused = bool(paused)
            if not self.playback_paused:
                self.playback_started_at = time.monotonic() - self.playback_position_s

    def toggle_pause(self):
        self.set_playback(paused=not self.playback_paused)
        return self.playback_paused

    def seek_relative(self, delta_s):
        self.set_playback(position_s=self.playback_position_s + float(delta_s), paused=True)
        return self.playback_position_s

    def seek_absolute(self, position_s, paused=True):
        self.set_playback(position_s=position_s, paused=paused)
        return self.playback_position_s

    def seek_frames(self, delta_frames):
        if not self.frames:
            return self.playback_position_s
        target_index = max(0, min(len(self.frames) - 1, self.frame_index + int(delta_frames)))
        position_s = self.normalized_timestamps[target_index] if self.normalized_timestamps else 0.0
        self.set_playback(position_s=position_s, paused=True)
        self.frame_index = target_index
        return self.playback_position_s

    def _map_position_to_scene(self, x_cv, y_cv):
        if not self.calibration or self.coordinate_space != "terrain_pixels":
            return x_cv, y_cv
        calib_left, calib_right, calib_top, calib_bottom = _terrain_reference_bounds(self.calibration)
        calib_width = max(calib_right - calib_left, 1.0)
        calib_height = max(calib_bottom - calib_top, 1.0)
        court_left, court_right, court_top, court_bottom = self.court_bounds
        court_length = max(court_right - court_left, 1.0)
        court_width = max(court_bottom - court_top, 1.0)

        # The calibration image is a portrait top-down court:
        # image Y follows court length, image X follows court width.
        # We expose the explicit transform as swap_xy_flip_y to keep the orientation consistent
        # with the review canvas and make the mapping visible in the UI.
        normalized_width = min(1.0, max(0.0, (float(x_cv) - calib_left) / calib_width))
        normalized_length = min(1.0, max(0.0, (float(y_cv) - calib_top) / calib_height))
        scene_x = court_left + normalized_length * court_length
        scene_y = court_bottom - normalized_width * court_width
        return scene_x, scene_y

    def _resolve_playback_index(self):
        if not self.frames:
            return 0
        if self.playback_duration_s <= 0.0:
            return min(self.frame_index, len(self.frames) - 1)
        if self.playback_paused:
            elapsed_s = self.playback_position_s
        else:
            elapsed_s = time.monotonic() - self.playback_started_at
            if elapsed_s >= self.playback_duration_s:
                elapsed_s = self.playback_duration_s
                self.playback_paused = True
            self.playback_position_s = elapsed_s
        index = bisect.bisect_right(self.normalized_timestamps, elapsed_s) - 1
        return max(0, min(index, len(self.frames) - 1))

    def _smooth_scene_position(self, player_id, target_point, *, selected=False):
        previous_point = self._smoothed_scene_positions.get(player_id)
        if previous_point is None:
            self._smoothed_scene_positions[player_id] = target_point
            return target_point
        jump_distance = distance_2d(previous_point, target_point)
        if jump_distance <= CV_RENDER_SMOOTH_DEADZONE:
            return previous_point
        if jump_distance >= CV_RENDER_SNAP_DISTANCE:
            self._smoothed_scene_positions[player_id] = target_point
            return target_point
        alpha = CV_RENDER_SELECTED_ALPHA if selected else CV_RENDER_SMOOTH_ALPHA
        smoothed_point = (
            previous_point[0] + (target_point[0] - previous_point[0]) * alpha,
            previous_point[1] + (target_point[1] - previous_point[1]) * alpha,
        )
        self._smoothed_scene_positions[player_id] = smoothed_point
        return smoothed_point

    def _read_video_frame(self, sequence_index):
        if self.video_capture is None or cv2 is None:
            return None
        if self._last_video_sequence_index == sequence_index and self._last_video_frame is not None:
            return self._last_video_frame
        if self.video_scrubbing and self._last_video_frame is not None:
            return self._last_video_frame

        target_index = max(0, int(sequence_index))
        if self._last_video_sequence_index is not None and target_index == (self._last_video_sequence_index + 1):
            ok, frame = self.video_capture.read()
        else:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            ok, frame = self.video_capture.read()
        if not ok or frame is None:
            return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_video_sequence_index = target_index
        self._last_video_frame = rgb_frame
        self._last_video_read_at = time.monotonic()
        return rgb_frame

    def set_video_scrubbing(self, active):
        self.video_scrubbing = bool(active)

    def get_frame_packet(self, include_video=True):
        if not self.frames:
            return {
                "raw": {},
                "valid": False,
                "source": "vision",
                "status": "vision: aucun log charge",
                "cv_positions": [],
                "selected_player_id": None if self.player_id == CV_ALL_PLAYERS else self.player_id,
                "selection_mode": "all" if self.player_id == CV_ALL_PLAYERS else "single",
                "all_player_ids": list(self.all_player_ids),
                "selected_player_visible": False,
                "playback": self.playback_state,
            }
        self.frame_index = self._resolve_playback_index()
        frame = self.frames[self.frame_index]
        if (
            self._last_smoothed_frame_index is None
            or abs(self.frame_index - self._last_smoothed_frame_index) > 3
        ):
            self._smoothed_scene_positions = {}
        self._last_smoothed_frame_index = self.frame_index
        selected = self._select_player(frame["players"])
        video_frame = self._last_video_frame
        if include_video:
            video_frame = self._read_video_frame(frame.get("sequence_index", self.frame_index))
        positions = []
        for player in frame["players"]:
            if not player.get("on_terrain", True):
                continue
            raw_x = player.get("raw_x", player["x"])
            raw_y = player.get("raw_y", player["y"])
            scene_x, scene_y = self._map_position_to_scene(raw_x, raw_y)
            selected_player = selected is not None and player["player_id"] == selected["player_id"]
            if "raw_x" not in player and "raw_y" not in player:
                scene_x, scene_y = self._smooth_scene_position(
                    player["player_id"],
                    (scene_x, scene_y),
                    selected=selected_player,
                )
            else:
                self._smoothed_scene_positions[player["player_id"]] = (scene_x, scene_y)
            positions.append(
                {
                    "player_id": player["player_id"],
                    "raw_player_id": player.get("raw_player_id"),
                    "source_player_ids": list(player.get("source_player_ids", [])),
                    "x": scene_x,
                    "y": scene_y,
                    "raw_x": raw_x,
                    "raw_y": raw_y,
                    "half": player["half"],
                    "on_terrain": True,
                    "confidence": player.get("confidence", 1.0),
                    "track_age": player.get("track_age"),
                    "track_hits": player.get("track_hits"),
                    "track_misses": player.get("track_misses"),
                    "vx": player.get("vx", 0.0),
                    "vy": player.get("vy", 0.0),
                    "selected": selected_player,
                }
            )
        positions = _dedupe_visible_players(positions)
        positions = _limit_visible_players(positions, self.expected_player_count)
        self._smoothed_scene_positions = {
            item["player_id"]: self._smoothed_scene_positions[item["player_id"]]
            for item in positions
            if item["player_id"] in self._smoothed_scene_positions
        }
        self.all_player_ids = [item["player_id"] for item in positions]
        primary_position = None
        selected_visible = selected is not None and selected.get("on_terrain", True)
        if selected_visible:
            selected_in_output = next((item for item in positions if item["player_id"] == selected["player_id"]), None)
            if selected_in_output is not None:
                primary_position = (selected_in_output["x"], selected_in_output["y"])
            else:
                primary_position = self._map_position_to_scene(
                    selected.get("raw_x", selected["x"]),
                    selected.get("raw_y", selected["y"]),
                )
        return {
            "raw": {},
            "valid": selected is not None or self.player_id == CV_ALL_PLAYERS,
            "source": "vision",
            "status": self.status,
            "cv_positions": positions,
            "selected_player_id": None if self.player_id == CV_ALL_PLAYERS else self.player_id,
            "selection_mode": "all" if self.player_id == CV_ALL_PLAYERS else "single",
            "all_player_ids": list(self.all_player_ids),
            "selected_player_visible": selected_visible,
            "primary_position": primary_position,
            "timestamp_s": frame["timestamp_s"],
            "video_frame": video_frame,
            "video_available": video_frame is not None,
            "playback": self.playback_state,
        }

