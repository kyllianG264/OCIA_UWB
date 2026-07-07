import csv
import os
from typing import Dict, List


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


def rows_to_tracking_frames(rows: List[Dict[str, object]]):
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
                "on_terrain": str(row["on_terrain"]).strip() not in {"0", "false", "False", ""},
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
