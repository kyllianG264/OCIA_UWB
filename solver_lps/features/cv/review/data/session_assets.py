"""Backward-compatible CV path helpers backed by the neutral session contract."""

from __future__ import annotations

import json
from pathlib import Path

from solver_lps.session_assets import (
    ASSETS_DIR,
    DEFAULT_SET,
    DEFAULT_SPORT,
    SOLVER_ROOT,
    SessionAssets,
    session_assets,
)


def sport_dir(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).sport_dir


def ground_dir(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).ground_dir


def gray_zone_dir(sport=DEFAULT_SPORT):
    return ground_dir(sport) / "gray_zone"


def set_dir(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return SessionAssets(sport=sport, asset_set=asset_set).set_dir


def set_input_dir(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return SessionAssets(sport=sport, asset_set=asset_set).input_dir


def set_output_dir(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return SessionAssets(sport=sport, asset_set=asset_set).output_dir


def set_analysis_dir(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return SessionAssets(sport=sport, asset_set=asset_set).analysis_dir


def ground_terrain_path(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).terrain_path


def ground_calibration_path(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).calibration_path


def cv_review_calibration_path(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).calibration_path


def ground_anchors_layout_path(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).anchors_layout_path


def gray_zone_input_dir(sport=DEFAULT_SPORT):
    return gray_zone_dir(sport) / "input"


def gray_zone_outputs_dir(sport=DEFAULT_SPORT):
    return gray_zone_dir(sport) / "output"


def gray_zone_analysis_dir(sport=DEFAULT_SPORT):
    return gray_zone_dir(sport) / "analysis"


def default_cv_positions_raw_path(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return SessionAssets(sport=sport, asset_set=asset_set).cv_positions_raw_path


def default_cv_positions_merged_path(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return SessionAssets(sport=sport, asset_set=asset_set).cv_positions_merged_path


def default_cv_video_path(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return default_cv_left_video_path(sport, asset_set)


def default_cv_left_video_path(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    output_dir = SessionAssets(sport=sport, asset_set=asset_set).output_dir
    tracked_path = output_dir / "left_undistorted_tracked.mp4"
    return tracked_path if tracked_path.exists() else output_dir / "left_undistorted.mp4"


def default_cv_right_video_path(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    output_dir = SessionAssets(sport=sport, asset_set=asset_set).output_dir
    tracked_path = output_dir / "right_undistorted_tracked.mp4"
    return tracked_path if tracked_path.exists() else output_dir / "right_undistorted.mp4"


def default_cv_metadata_path(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return SessionAssets(sport=sport, asset_set=asset_set).output_dir / "run_metadata.json"


GROUND_TERRAIN_PATH = ground_terrain_path()
GROUND_CALIBRATION_PATH = ground_calibration_path()
CV_REVIEW_CALIBRATION_PATH = cv_review_calibration_path()
GROUND_ANCHORS_LAYOUT_PATH = ground_anchors_layout_path()

DEFAULT_CV_POSITIONS_RAW_PATH = default_cv_positions_raw_path()
DEFAULT_CV_POSITIONS_MERGED_PATH = default_cv_positions_merged_path()
DEFAULT_CV_VIDEO_PATH = default_cv_video_path()
DEFAULT_CV_LEFT_VIDEO_PATH = default_cv_left_video_path()
DEFAULT_CV_RIGHT_VIDEO_PATH = default_cv_right_video_path()
DEFAULT_CV_METADATA_PATH = default_cv_metadata_path()


def first_existing_path(*candidates):
    """Keep the legacy string return type used by review consumers."""
    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            return str(candidate)
    for candidate in candidates:
        if candidate is not None:
            return str(candidate)
    return None


def load_ground_anchor_layouts(sport=DEFAULT_SPORT):
    layout_path = ground_anchors_layout_path(sport)
    if not layout_path.exists():
        return {}
    try:
        with layout_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {}
    layouts = payload.get("layouts", {})
    return layouts if isinstance(layouts, dict) else {}
