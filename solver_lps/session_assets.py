"""Canonical, feature-neutral paths for one solver session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SOLVER_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = SOLVER_ROOT / "assets"
DEFAULT_SPORT = "basket"
DEFAULT_SET = "set1"


def _safe_segment(value: str, *, label: str) -> str:
    segment = str(value or "").strip()
    if not segment or segment in {".", ".."} or Path(segment).name != segment:
        raise ValueError(f"Invalid {label}: {value!r}")
    return segment


def first_existing_path(*candidates):
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


@dataclass(frozen=True)
class SessionAssets:
    sport: str = DEFAULT_SPORT
    asset_set: str = DEFAULT_SET

    def __post_init__(self):
        object.__setattr__(self, "sport", _safe_segment(self.sport, label="sport"))
        object.__setattr__(self, "asset_set", _safe_segment(self.asset_set, label="asset set"))

    @property
    def sport_dir(self):
        return ASSETS_DIR / self.sport

    @property
    def set_dir(self):
        return self.sport_dir / self.asset_set

    @property
    def input_dir(self):
        return self.set_dir / "input"

    @property
    def output_dir(self):
        return self.set_dir / "output"

    @property
    def analysis_dir(self):
        return self.set_dir / "analysis"

    @property
    def uwb_input_dir(self):
        return self.set_dir / "uwb" / "input"

    @property
    def uwb_output_dir(self):
        return self.set_dir / "uwb" / "output"

    @property
    def ground_dir(self):
        return self.sport_dir / "ground"

    @property
    def ground_input_dir(self):
        return self.ground_dir / "input"

    @property
    def ground_output_dir(self):
        return self.ground_dir / "output"

    @property
    def gray_zone_dir(self):
        return self.ground_dir / "gray_zone"

    @property
    def gray_zone_input_dir(self):
        return self.gray_zone_dir / "input"

    @property
    def gray_zone_output_dir(self):
        return self.gray_zone_dir / "output"

    @property
    def gray_zone_analysis_dir(self):
        return self.gray_zone_dir / "analysis"

    @property
    def terrain_path(self):
        return self.ground_dir / "terrain.png"

    @property
    def models_dir(self):
        return ASSETS_DIR / "models"

    @property
    def pose_models_dir(self):
        return self.models_dir / "pose"

    @property
    def calibration_path(self):
        return first_existing_path(
            self.ground_output_dir / "calibration.json",
            self.ground_dir / "calibration.json",
        ) or self.ground_output_dir / "calibration.json"

    @property
    def anchors_layout_path(self):
        return first_existing_path(
            self.ground_dir / "anchors_layout.json",
            self.ground_input_dir / "anchors_layout.json",
        ) or self.ground_dir / "anchors_layout.json"

    @property
    def cv_positions_raw_path(self):
        return self.output_dir / "positions_raw.csv"

    @property
    def cv_positions_merged_path(self):
        return self.analysis_dir / "positions_merged.csv"

    @property
    def uwb_raw_path(self):
        return self.uwb_input_dir / "uwb_raw.csv"

    @property
    def uwb_tag_review_path(self):
        return self.uwb_input_dir / "uwb_tag_review.csv"

    @property
    def uwb_positions_merged_path(self):
        return self.uwb_output_dir / "positions_merged.csv"

    def uwb_positions_path(self, mode="two_d"):
        normalized = str(mode or "two_d").strip().lower()
        if normalized == "two_d":
            return self.uwb_positions_merged_path
        if normalized not in {"three_d", "three_d_to_2d"}:
            raise ValueError(f"Unsupported UWB calculation mode: {mode!r}")
        return self.uwb_output_dir / f"positions_merged_{normalized}.csv"

    def ensure_directories(self):
        for directory in (
            self.input_dir,
            self.output_dir,
            self.analysis_dir,
            self.uwb_input_dir,
            self.uwb_output_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self.set_dir


def session_assets(sport=DEFAULT_SPORT, asset_set=DEFAULT_SET):
    return SessionAssets(sport=sport, asset_set=asset_set)
