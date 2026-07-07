from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from solver_lps.features.cv.review.data.session_assets import (
    DEFAULT_SET,
    cv_review_calibration_path,
    default_cv_positions_merged_path,
    default_cv_positions_raw_path,
)
from solver_lps.features.ground.domain.calibration import load_calibration_geometry

try:
    from ..data.csv_raw_reader import Detection, load_frames
    from ..data.track_exporter import rows_to_tracking_frames, write_output
    from .camera_zones import load_camera_zones
    from .camera_selector import select_best_camera_detections
    from .raw_cleaner import keep_useful_detections
except ImportError:
    from solver_lps.features.cv.review.analysis.full_court.data.csv_raw_reader import Detection, load_frames
    from solver_lps.features.cv.review.analysis.full_court.data.track_exporter import rows_to_tracking_frames, write_output
    from solver_lps.features.cv.review.analysis.full_court.domain.camera_zones import load_camera_zones
    from solver_lps.features.cv.review.analysis.full_court.domain.camera_selector import select_best_camera_detections
    from solver_lps.features.cv.review.analysis.full_court.domain.raw_cleaner import keep_useful_detections


DEFAULT_SPORT = "volley"
DEFAULT_INPUT = default_cv_positions_raw_path(DEFAULT_SPORT, DEFAULT_SET)
DEFAULT_OUTPUT = default_cv_positions_merged_path(DEFAULT_SPORT, DEFAULT_SET)
DEFAULT_CALIBRATION = str(cv_review_calibration_path(DEFAULT_SPORT))
DEFAULT_TRACK_MATCH_DISTANCE = 26.0
DEFAULT_ZONE_MARGIN = 0.0


@dataclass
class Track:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    last_frame: int
    last_timestamp_s: float
    hits: int = 1
    misses: int = 0
    age: int = 1

    def label(self) -> str:
        return f"T{self.track_id:03d}"


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Full-court V1 best-camera merged pipeline.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input raw CSV path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output merged CSV path.")
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION, help="Calibration JSON path.")
    parser.add_argument("--expected-players", type=int, default=12, help="Max merged players per frame.")
    parser.add_argument("--zone-margin", type=float, default=DEFAULT_ZONE_MARGIN, help="Extra margin around useful zones.")
    parser.add_argument("--track-match-distance", type=float, default=DEFAULT_TRACK_MATCH_DISTANCE, help="Track matching distance in terrain pixels.")
    return parser.parse_args(argv)


def default_config(input_path: Optional[str] = None, output_path: Optional[str] = None, **overrides):
    return argparse.Namespace(
        input=input_path or DEFAULT_INPUT,
        output=output_path or DEFAULT_OUTPUT,
        calibration=str(overrides.get("calibration", DEFAULT_CALIBRATION)),
        expected_players=int(overrides.get("expected_players", 12)),
        zone_margin=float(overrides.get("zone_margin", DEFAULT_ZONE_MARGIN)),
        track_match_distance=float(overrides.get("track_match_distance", DEFAULT_TRACK_MATCH_DISTANCE)),
    )


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _build_frame_candidates(frame_detections: List[Detection], *, camera_zones, expected_players: int, zone_margin: float) -> List[Detection]:
    kept: List[Detection] = []
    for camera_name, zone in camera_zones.items():
        camera_items = [item for item in frame_detections if str(item.cam).strip().lower() == camera_name]
        kept.extend(keep_useful_detections(camera_items, zone=zone, strong_margin=zone_margin))
    return select_best_camera_detections(kept, camera_zones=camera_zones, expected_players=expected_players)


def _assign_tracks(
    detections: List[Detection],
    tracks: Dict[int, Track],
    *,
    current_frame: int,
    current_timestamp_s: float,
    next_track_id: int,
    match_distance: float,
) -> tuple[List[Dict[str, object]], int]:
    rows: List[Dict[str, object]] = []
    available_track_ids = set(tracks)
    matches: List[tuple[float, int, int]] = []
    for detection_index, detection in enumerate(detections):
        for track_id in available_track_ids:
            track = tracks[track_id]
            matches.append((_distance((track.x, track.y), (detection.x, detection.y)), detection_index, track_id))
    matched_detection_indices = set()
    matched_track_ids = set()
    assignments: Dict[int, int] = {}
    for distance, detection_index, track_id in sorted(matches, key=lambda item: item[0]):
        if distance > match_distance:
            continue
        if detection_index in matched_detection_indices or track_id in matched_track_ids:
            continue
        assignments[detection_index] = track_id
        matched_detection_indices.add(detection_index)
        matched_track_ids.add(track_id)

    for track_id, track in list(tracks.items()):
        if track_id not in matched_track_ids:
            track.misses += 1
            track.age += 1
            if track.misses > 10:
                del tracks[track_id]

    for detection_index, detection in enumerate(detections):
        assigned_track_id = assignments.get(detection_index)
        if assigned_track_id is None:
            assigned_track_id = next_track_id
            next_track_id += 1
            track = Track(
                track_id=assigned_track_id,
                x=float(detection.x),
                y=float(detection.y),
                vx=0.0,
                vy=0.0,
                last_frame=current_frame,
                last_timestamp_s=current_timestamp_s,
            )
            tracks[assigned_track_id] = track
            status = "selected"
        else:
            track = tracks[assigned_track_id]
            dt = max(1e-6, float(current_timestamp_s) - float(track.last_timestamp_s))
            vx = (float(detection.x) - track.x) / dt
            vy = (float(detection.y) - track.y) / dt
            track.vx = vx
            track.vy = vy
            track.x = float(detection.x)
            track.y = float(detection.y)
            track.last_frame = current_frame
            track.last_timestamp_s = current_timestamp_s
            track.hits += 1
            track.age += 1
            track.misses = 0
            status = "selected_tracked"

        rows.append(
            {
                "frame": int(detection.frame),
                "timestamp_s": float(detection.timestamp_s),
                "timestamp_unix": str(detection.timestamp_unix),
                "stable_id": track.label(),
                "raw_player_id": str(detection.raw_player_id),
                "X": round(float(detection.x), 3),
                "Y": round(float(detection.y), 3),
                "vx": round(float(track.vx), 3),
                "vy": round(float(track.vy), 3),
                "cam": str(detection.cam),
                "source_cams": str(detection.cam),
                "source_ids": str(detection.raw_player_id),
                "merged_count": 1,
                "track_age": int(track.age),
                "track_hits": int(track.hits),
                "track_misses": int(track.misses),
                "confidence": 1.0,
                "status": status,
                "on_terrain": 1,
            }
        )
    return rows, next_track_id


def run_tracking(args) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    geometry = load_calibration_geometry(args.calibration)
    camera_zones = load_camera_zones(geometry, margin=float(args.zone_margin))
    frames = load_frames(args.input)
    progress_callback = getattr(args, "progress_callback", None)
    if progress_callback is not None:
        progress_callback(0, len(frames))
    tracks: Dict[int, Track] = {}
    next_track_id = 1
    output_rows: List[Dict[str, object]] = []

    stats = {
        "frames_in": len(frames),
        "rows_in": sum(len(frame) for frame in frames),
        "rows_kept": 0,
        "rows_out": 0,
        "tracks_created": 0,
        "camera_zones": len(camera_zones),
    }

    for frame_index, frame_detections in enumerate(frames, start=1):
        if not frame_detections:
            if progress_callback is not None:
                progress_callback(frame_index, len(frames))
            continue
        current_frame = int(frame_detections[0].frame)
        current_timestamp_s = float(frame_detections[0].timestamp_s)
        candidates = _build_frame_candidates(
            frame_detections,
            camera_zones=camera_zones,
            expected_players=int(args.expected_players),
            zone_margin=float(args.zone_margin),
        )
        stats["rows_kept"] += len(candidates)
        before = next_track_id
        frame_rows, next_track_id = _assign_tracks(
            candidates,
            tracks,
            current_frame=current_frame,
            current_timestamp_s=current_timestamp_s,
            next_track_id=next_track_id,
            match_distance=float(args.track_match_distance),
        )
        stats["tracks_created"] += next_track_id - before
        output_rows.extend(frame_rows)
        if progress_callback is not None:
            progress_callback(frame_index, len(frames))

    stats["rows_out"] = len(output_rows)
    return output_rows, stats


def build_tracking_frames(csv_path: str, **overrides):
    args = default_config(input_path=csv_path, output_path=overrides.pop("output_path", DEFAULT_OUTPUT), **overrides)
    rows, stats = run_tracking(args)
    return rows_to_tracking_frames(rows), stats


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    rows, stats = run_tracking(args)
    write_output(rows, args.output)
    print(
        "Full-court V1 termine: "
        f"{stats['rows_out']} lignes, "
        f"{stats['rows_kept']} detections retenues, "
        f"{stats['tracks_created']} tracks crees."
    )
    print(f"CSV genere: {args.output}")


if __name__ == "__main__":
    main()
