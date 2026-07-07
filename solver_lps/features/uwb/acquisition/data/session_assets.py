"""UWB path helpers backed by the feature-neutral session contract."""

from solver_lps.session_assets import (
    ASSETS_DIR,
    DEFAULT_SET,
    DEFAULT_SPORT,
    SOLVER_ROOT,
    SessionAssets,
    session_assets,
)


def sport_dir(sport=DEFAULT_SPORT):
    return session_assets(sport).sport_dir


def set_dir(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return session_assets(sport, asset_set).set_dir


def uwb_input_dir(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return session_assets(sport, asset_set).uwb_input_dir


def uwb_output_dir(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return session_assets(sport, asset_set).uwb_output_dir


def default_uwb_raw_path(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return session_assets(sport, asset_set).uwb_raw_path


def default_uwb_review_log_path(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return default_uwb_raw_path(sport, asset_set)


def default_uwb_tag_review_path(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return session_assets(sport, asset_set).uwb_tag_review_path


def default_uwb_merged_path(mode, sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return session_assets(sport, asset_set).uwb_positions_path(mode)


DEFAULT_UWB_REVIEW_LOG_PATH = default_uwb_review_log_path()
DEFAULT_UWB_TAG_REVIEW_PATH = default_uwb_tag_review_path()


__all__ = [
    "ASSETS_DIR",
    "DEFAULT_SET",
    "DEFAULT_SPORT",
    "DEFAULT_UWB_REVIEW_LOG_PATH",
    "DEFAULT_UWB_TAG_REVIEW_PATH",
    "SOLVER_ROOT",
    "SessionAssets",
    "default_uwb_merged_path",
    "default_uwb_raw_path",
    "default_uwb_review_log_path",
    "default_uwb_tag_review_path",
    "session_assets",
    "set_dir",
    "sport_dir",
    "uwb_input_dir",
    "uwb_output_dir",
]
