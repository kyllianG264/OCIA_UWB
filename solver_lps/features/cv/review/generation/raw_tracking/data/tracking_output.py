from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

from solver_lps.session_assets import ASSETS_DIR, DEFAULT_SET, DEFAULT_SPORT, SessionAssets

MODELS_DIR = SessionAssets().models_dir
POSE_MODELS_DIR = SessionAssets().pose_models_dir


def sport_assets_dir(sport: str = DEFAULT_SPORT) -> Path:
    return SessionAssets(sport=sport).sport_dir


def sport_ground_dir(sport: str = DEFAULT_SPORT) -> Path:
    return SessionAssets(sport=sport).ground_dir


def set_dir(sport: str = DEFAULT_SPORT, asset_set: str = DEFAULT_SET) -> Path:
    return SessionAssets(sport=sport, asset_set=asset_set).set_dir


def set_input_dir(sport: str = DEFAULT_SPORT, asset_set: str = DEFAULT_SET) -> Path:
    return SessionAssets(sport=sport, asset_set=asset_set).input_dir


def set_output_dir(sport: str = DEFAULT_SPORT, asset_set: str = DEFAULT_SET) -> Path:
    return SessionAssets(sport=sport, asset_set=asset_set).output_dir


def set_analysis_dir(sport: str = DEFAULT_SPORT, asset_set: str = DEFAULT_SET) -> Path:
    return SessionAssets(sport=sport, asset_set=asset_set).analysis_dir


_DEFAULT_ASSETS = SessionAssets()
DEFAULT_TERRAIN_IMAGE = _DEFAULT_ASSETS.terrain_path
DEFAULT_CALIBRATION_PATH = _DEFAULT_ASSETS.calibration_path
DEFAULT_LEFT_VIDEO = _DEFAULT_ASSETS.input_dir / "left_video.mp4"
DEFAULT_RIGHT_VIDEO = _DEFAULT_ASSETS.input_dir / "right_video.mp4"
DEFAULT_POSITIONS_RAW = _DEFAULT_ASSETS.cv_positions_raw_path
DEFAULT_TRACKING_COMPOSITE = _DEFAULT_ASSETS.output_dir / "left_undistorted.mp4"
DEFAULT_LEFT_UNDISTORTED_VIDEO = _DEFAULT_ASSETS.output_dir / "left_undistorted.mp4"
DEFAULT_RIGHT_UNDISTORTED_VIDEO = _DEFAULT_ASSETS.output_dir / "right_undistorted.mp4"
DEFAULT_RUN_METADATA = _DEFAULT_ASSETS.output_dir / "run_metadata.json"
DEFAULT_POSITIONS_MERGED = _DEFAULT_ASSETS.cv_positions_merged_path
DEFAULT_POSITIONS_STABLE_HUNGARIAN = _DEFAULT_ASSETS.analysis_dir / "positions_stable_hungarian.csv"
DEFAULT_POSE_MODEL = POSE_MODELS_DIR / "yolo11x-pose.pt"

DETECTIONS_COLUMNS = [
    "frame",
    "timestamp_s",
    "timestamp_unix",
    "cam",
    "detection_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "foot_x",
    "foot_y",
    "conf",
]

PROJECTED_POSITIONS_COLUMNS = [
    "frame",
    "timestamp_s",
    "timestamp_unix",
    "cam",
    "detection_id",
    "source_foot_x",
    "source_foot_y",
    "X",
    "Y",
    "on_terrain",
    "conf",
]

POSITIONS_RAW_COLUMNS = ["frame", "timestamp_s", "timestamp_unix", "player_id", "X", "Y", "cam", "on_terrain"]

TRACKING_ASSIGNMENTS_COLUMNS = [
    "frame",
    "timestamp_s",
    "timestamp_unix",
    "cam",
    "track_id",
    "detection_id",
    "source_foot_x",
    "source_foot_y",
    "X",
    "Y",
    "on_terrain",
]


def configure_utf8_stdout() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)


def hms_to_s(hms: str) -> float:
    parts = hms.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Format HH:MM:SS attendu, recu : {hms!r}")
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def infer_media_start_unix(path: str) -> tuple[int | None, str | None]:
    if not os.path.isfile(path):
        return None, None
    try:
        unix_ts = int(os.path.getctime(path))
    except OSError:
        return None, None
    return unix_ts, "file_creation_time"


def unix_to_iso8601(unix_ts: int | float | None) -> str | None:
    if unix_ts is None:
        return None
    return dt.datetime.fromtimestamp(unix_ts, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def require_file(path: str, label: str = "fichier") -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} introuvable : {path}")
    return path


def load_json(path: str) -> dict[str, Any]:
    require_file(path, "JSON")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def count_csv_rows(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def read_csv_rows(path: str):
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_csv_rows(path: str, columns: list[str], rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def build_detection_output_paths(output_dir: str) -> tuple[str, str]:
    return (
        os.path.join(output_dir, "detections.csv"),
        os.path.join(output_dir, "run_metadata.json"),
    )


def open_detection_csv(csv_path: str):
    csv_handle = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_handle)
    writer.writerow(DETECTIONS_COLUMNS)
    return csv_handle, writer


def write_detection_summary(
    meta_path: str,
    *,
    video_gauche: str,
    video_droite: str,
    source_g: dict,
    source_d: dict,
    model: str,
    device: str,
    start: str,
    end: str | None,
    frames_processed: int,
    avg_fps: float,
    csv_path: str,
    conf_gauche: float,
    conf_droite: float,
    imgsz_gauche: int,
    imgsz_droite: int,
    detection_counts: dict,
) -> None:
    summary = {
        "video_gauche": video_gauche,
        "video_droite": video_droite,
        "video_gauche_start_unix": source_g["start_unix"],
        "video_gauche_start_iso": unix_to_iso8601(source_g["start_unix"]),
        "video_gauche_start_source": source_g["start_unix_source"],
        "video_droite_start_unix": source_d["start_unix"],
        "video_droite_start_iso": unix_to_iso8601(source_d["start_unix"]),
        "video_droite_start_source": source_d["start_unix_source"],
        "model": model,
        "device": str(device),
        "start": start,
        "end": end,
        "frames_processed": frames_processed,
        "avg_fps": round(avg_fps, 3) if avg_fps > 0 else 0.0,
        "csv_path": csv_path,
        "conf_gauche": conf_gauche,
        "conf_droite": conf_droite,
        "imgsz_gauche": imgsz_gauche,
        "imgsz_droite": imgsz_droite,
        "detection_counts": detection_counts,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
