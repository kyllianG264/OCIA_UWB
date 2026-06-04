"""
Utilitaires partages pour les scripts NeptuVision.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import shutil
import sys
from typing import Any

import cv2
import numpy as np

from core.paths import PYTHON_CV_LOGS_DIR

CSV_CANDIDATES = [
    "positions_merged.csv",
    "positions_solved.csv",
    "positions_raw.csv",
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


def sync_file_to_python_cv_logs(source_path: str, target_name: str | None = None) -> str | None:
    if not source_path or not os.path.isfile(source_path):
        return None
    destination_dir = ensure_dir(str(PYTHON_CV_LOGS_DIR))
    destination_name = target_name or os.path.basename(source_path)
    destination_path = os.path.join(destination_dir, destination_name)
    shutil.copy2(source_path, destination_path)
    return destination_path


def require_file(path: str, label: str = "fichier") -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} introuvable : {path}")
    return path


def load_json(path: str) -> dict[str, Any]:
    require_file(path, "JSON")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_calibration(path: str) -> dict[str, Any]:
    data = load_json(path)
    try:
        h_g = np.array(data["cam_gauche"]["H"], dtype=float)
        h_d = np.array(data["cam_droite"]["H"], dtype=float)
        bounds = data["terrain_bounds"]
    except KeyError as exc:
        raise ValueError(f"Calibration invalide, cle manquante : {exc}") from exc

    if "gauche" in bounds and "droite" in bounds:
        bounds_g = bounds["gauche"]
        bounds_d = bounds["droite"]
    else:
        bounds_g = {
            "x_min": bounds.get("CX_L", 0),
            "x_max": bounds.get("CX_R", 9999),
            "y_min": bounds.get("CY_TOP", 0),
            "y_max": bounds.get("CY_MID", 9999),
        }
        bounds_d = {
            "x_min": bounds.get("CX_L", 0),
            "x_max": bounds.get("CX_R", 9999),
            "y_min": bounds.get("CY_MID", 0),
            "y_max": bounds.get("CY_BOT", 9999),
        }

    return {
        "raw": data,
        "H_g": h_g,
        "H_d": h_d,
        "bounds_g": bounds_g,
        "bounds_d": bounds_d,
    }


def resolve_positions_csv(base_dir: str) -> str:
    for name in CSV_CANDIDATES:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            return path
    expected = ", ".join(CSV_CANDIDATES)
    raise FileNotFoundError(f"Aucun CSV compatible trouve dans {base_dir}. Attendus : {expected}")


def infer_side(row: dict[str, Any]) -> str | None:
    for key in ("demi_terrain", "cam", "side", "team"):
        value = row.get(key)
        if not value:
            continue
        normalized = str(value).strip().lower()
        if normalized.startswith("g"):
            return "gauche"
        if normalized.startswith("d"):
            return "droite"

    player_id = str(row.get("player_id", "")).strip().upper()
    if player_id.startswith("G"):
        return "gauche"
    if player_id.startswith("D"):
        return "droite"
    return None


def count_csv_rows(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)
