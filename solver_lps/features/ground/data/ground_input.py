from pathlib import Path

from solver_lps.session_assets import DEFAULT_SPORT, SessionAssets


def sport_dir(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).sport_dir


def ground_dir(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).ground_dir


def ground_input_dir(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).ground_input_dir


def ground_output_dir(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).ground_output_dir


def ground_terrain_path(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).terrain_path


def ground_calibration_path(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).calibration_path


def ground_output_calibration_path(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).ground_output_dir / "calibration.json"


def cv_review_calibration_path(sport=DEFAULT_SPORT):
    return ground_calibration_path(sport)


def ground_anchors_layout_path(sport=DEFAULT_SPORT):
    return SessionAssets(sport=sport).anchors_layout_path


def first_existing_path(*candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    for candidate in candidates:
        if candidate is not None:
            return Path(candidate)
    return None
