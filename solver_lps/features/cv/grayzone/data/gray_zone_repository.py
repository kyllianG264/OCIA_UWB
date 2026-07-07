from dataclasses import dataclass
import json
from pathlib import Path

from solver_lps.session_assets import DEFAULT_SPORT, SessionAssets


@dataclass(frozen=True)
class GrayZoneAssetRepository:
    terrain_path: str
    calibration_path: str
    input_dir: str
    outputs_dir: str
    analysis_dir: str

    @property
    def raw_positions_path(self):
        return str(Path(self.input_dir) / "positions_raw.csv")

    @property
    def metadata_path(self):
        return str(Path(self.analysis_dir) / "gray_zone_metadata.json")


def default_gray_zone_repository(sport: str = DEFAULT_SPORT) -> GrayZoneAssetRepository:
    assets = SessionAssets(sport=sport)
    return GrayZoneAssetRepository(
        terrain_path=str(assets.terrain_path),
        calibration_path=str(assets.calibration_path),
        input_dir=str(assets.gray_zone_input_dir),
        outputs_dir=str(assets.gray_zone_output_dir),
        analysis_dir=str(assets.gray_zone_analysis_dir),
    )


def load_gray_zone_overlay(sport: str = DEFAULT_SPORT):
    """Load the sport-level gray-zone polygons as presentation-ready cells."""
    repository = default_gray_zone_repository(sport)
    metadata_path = Path(repository.metadata_path)
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    zone_cells = []
    for polygon in payload.get("polygons") or []:
        if len(polygon) < 3:
            continue
        x_values = [float(point[0]) for point in polygon]
        y_values = [float(point[1]) for point in polygon]
        left, right = min(x_values), max(x_values)
        top, bottom = min(y_values), max(y_values)
        zone_cells.append(
            {
                "center": ((left + right) / 2.0, (top + bottom) / 2.0),
                "width": right - left,
                "height": bottom - top,
                "gray_score": 1.0,
            }
        )
    disappearance_cells = list(payload.get("disappearance_cells") or [])
    disappearance_points = list(payload.get("disappearance_points") or [])
    handoff_points = list(payload.get("handoff_points") or [])
    handoff_rejected_points = list(payload.get("handoff_rejected_points") or [])
    handoff_ellipse = payload.get("handoff_ellipse")
    if not zone_cells and not disappearance_cells and not disappearance_points and not handoff_points and not handoff_rejected_points and not handoff_ellipse:
        return None
    return {
        "cells": disappearance_cells,
        "disappearance_points": disappearance_points,
        "event_points": handoff_points,
        "rejected_event_points": handoff_rejected_points,
        "ellipse": handoff_ellipse,
        "zone_cells": zone_cells,
        "layer_color": "green",
        "metadata_path": str(metadata_path),
    }
