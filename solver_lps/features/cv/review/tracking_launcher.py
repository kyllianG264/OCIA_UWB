"""Application services used by the CV tracking page."""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from solver_lps.features.cv.review.generation.raw_tracking.data.tracking_output import DEFAULT_POSE_MODEL
from solver_lps.session_assets import SessionAssets


REPO_ROOT = Path(__file__).resolve().parents[4]
PIPELINE_SCRIPT = REPO_ROOT / "solver_lps" / "features" / "cv" / "review" / "generation" / "raw_tracking" / "orchestration" / "pipeline.py"
CALIBRATION_APP_SCRIPT = REPO_ROOT / "solver_lps" / "features" / "cv" / "review" / "generation" / "calibration" / "presentation" / "calibration_app.py"


def _first_existing_path(*candidates):
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path.resolve())
    return str(Path(candidates[0]).resolve())


def default_model_path():
    assets = SessionAssets()
    return _first_existing_path(
        DEFAULT_POSE_MODEL,
        assets.pose_models_dir / "yolo11x-pose.pt",
        assets.pose_models_dir / "yolo11n-pose.pt",
    )


def session_defaults(sport, asset_set):
    assets = SessionAssets(sport=sport, asset_set=asset_set)
    return {
        "left_video": assets.input_dir / "left_video.mp4",
        "right_video": assets.input_dir / "right_video.mp4",
        "calibration": assets.calibration_path,
        "output_root": assets.output_dir,
        "analysis_root": assets.analysis_dir,
    }


def python_has_module(module_name):
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def validate_launch(fields):
    values = {field["key"]: field["value"] for field in fields}
    missing = []
    for key in ("left_video", "right_video", "calibration", "model"):
        path = values.get(key, "").strip()
        if not path or not Path(path).is_file():
            missing.append(f"{key} introuvable : {path}")
    if missing:
        return False, missing
    if not python_has_module("ultralytics"):
        return False, [
            "Module manquant dans cet environnement Python : ultralytics",
            "Installe-le dans solver_lps\\.solver_env puis relance F5.",
        ]
    return True, []


def _reader_thread(stream, output_queue):
    try:
        for line in iter(stream.readline, ""):
            output_queue.put(line.rstrip())
    finally:
        stream.close()


def launch_tracking(fields):
    values = {field["key"]: field["value"] for field in fields}
    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--video_gauche", values["left_video"],
        "--video_droite", values["right_video"],
        "--calibration", values["calibration"],
        "--model", values["model"],
        "--output_root", values["output_root"],
        "--analysis_root", values["analysis_root"],
        "--conf", values["conf"],
        "--imgsz", values["imgsz"],
        "--device", values["device"],
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    output_queue = queue.Queue()
    threading.Thread(target=_reader_thread, args=(process.stdout, output_queue), daemon=True).start()
    return process, output_queue


def _same_file_signature(source_path, target_path):
    try:
        source_stat = source_path.stat()
        target_stat = target_path.stat()
    except OSError:
        return False
    return source_stat.st_size == target_stat.st_size and source_stat.st_mtime_ns == target_stat.st_mtime_ns


def _copy_video(source_path, target_path):
    if source_path.resolve() == target_path.resolve():
        return target_path.resolve(), None
    if target_path.exists() and _same_file_signature(source_path, target_path):
        return target_path.resolve(), None
    try:
        shutil.copy2(source_path, target_path)
        return target_path.resolve(), None
    except OSError as exc:
        stem = f"{target_path.stem}__{source_path.stem}"
        fallback = target_path.with_name(f"{stem}{target_path.suffix}")
        counter = 2
        while fallback.exists() and not _same_file_signature(source_path, fallback):
            fallback = target_path.with_name(f"{stem}_{counter}{target_path.suffix}")
            counter += 1
        if not fallback.exists():
            shutil.copy2(source_path, fallback)
        return fallback.resolve(), f"Impossible d'ecraser {target_path.name} ({exc}). Utilisation de {fallback.name}."


def sync_selected_videos(fields):
    values = {field["key"]: field["value"] for field in fields}
    assets = SessionAssets(values["sport"].strip(), values["asset_set"].strip())
    assets.ensure_directories()
    warnings = []
    for key, filename in (("left_video", "left_video.mp4"), ("right_video", "right_video.mp4")):
        synced, warning = _copy_video(Path(values[key]), assets.input_dir / filename)
        values[key] = str(synced)
        if warning:
            warnings.append(warning)
    values["output_root"] = str(assets.output_dir.resolve())
    values["analysis_root"] = str(assets.analysis_dir.resolve())
    values["calibration"] = values["calibration"] or str(assets.calibration_path.resolve())
    for field in fields:
        if field["key"] in values:
            field["value"] = values[field["key"]]
    return warnings


def launch_calibration(fields):
    values = {field["key"]: field["value"] for field in fields}
    assets = SessionAssets(values.get("sport", "basket").strip() or "basket", values.get("asset_set", "set1").strip() or "set1")
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(REPO_ROOT),
        SOLVER_SPORT=assets.sport,
        SOLVER_ASSET_SET=assets.asset_set,
        SOLVER_LEFT_VIDEO=values.get("left_video", "").strip(),
        SOLVER_RIGHT_VIDEO=values.get("right_video", "").strip(),
        SOLVER_CALIBRATION_OUTPUT_DIR=str(assets.ground_output_dir.resolve()),
    )
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(CALIBRATION_APP_SCRIPT)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
