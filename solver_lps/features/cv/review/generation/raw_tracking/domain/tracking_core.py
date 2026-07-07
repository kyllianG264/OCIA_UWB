from __future__ import annotations

import math
from collections import defaultdict


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def group_projected_positions(rows) -> dict[int, dict]:
    grouped = defaultdict(lambda: {"gauche": [], "droite": [], "timestamp_s": 0.0, "timestamp_unix": None})
    for row in rows:
        frame_idx = int(row["frame"])
        camera = row["cam"].strip().lower()
        if camera not in ("gauche", "droite"):
            continue
        grouped[frame_idx][camera].append(
            {
                "x": int(float(row["X"])),
                "y": int(float(row["Y"])),
                "on_terrain": str(row.get("on_terrain", "1")).strip() not in ("0", "false", "False", ""),
                "detection_id": row.get("detection_id", ""),
                "source_foot_x": int(float(row.get("source_foot_x", 0) or 0)),
                "source_foot_y": int(float(row.get("source_foot_y", 0) or 0)),
            }
        )
        grouped[frame_idx]["timestamp_s"] = float(row["timestamp_s"])
        raw_unix = str(row.get("timestamp_unix", "")).strip()
        if raw_unix:
            grouped[frame_idx]["timestamp_unix"] = int(float(raw_unix))
    return grouped


def run_tracking(grouped_frames: dict[int, dict], *, max_track_distance: float, max_idle_frames: int):
    next_track_id = {"gauche": 1, "droite": 1}
    active_tracks = {"gauche": {}, "droite": {}}
    output_counts = {"gauche": 0, "droite": 0}
    positions_rows = []
    assignment_rows = []

    for frame_idx in sorted(grouped_frames):
        payload = grouped_frames[frame_idx]
        timestamp_s = payload["timestamp_s"]
        timestamp_unix = payload["timestamp_unix"]

        for camera in ("gauche", "droite"):
            detections = payload[camera]
            tracks = active_tracks[camera]
            candidates = []

            for track_id, track in tracks.items():
                frame_gap = frame_idx - track["last_frame"]
                if frame_gap <= 0 or frame_gap > max_idle_frames:
                    continue
                max_distance_allowed = max_track_distance * max(1, frame_gap)
                for detection_index, detection in enumerate(detections):
                    dist = distance((track["x"], track["y"]), (detection["x"], detection["y"]))
                    if dist <= max_distance_allowed:
                        candidates.append((dist, track_id, detection_index))

            assignments = {}
            used_tracks = set()
            used_detections = set()
            for dist, track_id, detection_index in sorted(candidates):
                if track_id in used_tracks or detection_index in used_detections:
                    continue
                assignments[detection_index] = track_id
                used_tracks.add(track_id)
                used_detections.add(detection_index)

            seen_tracks = set()
            for detection_index, detection in enumerate(detections):
                track_id = assignments.get(detection_index)
                if track_id is None:
                    track_id = next_track_id[camera]
                    next_track_id[camera] += 1

                tracks[track_id] = {
                    "x": detection["x"],
                    "y": detection["y"],
                    "last_frame": frame_idx,
                    "on_terrain": detection["on_terrain"],
                }
                seen_tracks.add(track_id)

                positions_rows.append(
                    [
                        frame_idx,
                        f"{timestamp_s:.3f}",
                        timestamp_unix or "",
                        track_id,
                        detection["x"],
                        detection["y"],
                        camera,
                        1 if detection["on_terrain"] else 0,
                    ]
                )
                assignment_rows.append(
                    [
                        frame_idx,
                        f"{timestamp_s:.3f}",
                        timestamp_unix or "",
                        camera,
                        track_id,
                        detection["detection_id"],
                        detection["source_foot_x"],
                        detection["source_foot_y"],
                        detection["x"],
                        detection["y"],
                        1 if detection["on_terrain"] else 0,
                    ]
                )
                output_counts[camera] += 1

            for track_id in list(tracks.keys()):
                if track_id in seen_tracks:
                    continue
                if frame_idx - tracks[track_id]["last_frame"] > max_idle_frames:
                    del tracks[track_id]

    return positions_rows, assignment_rows, output_counts

