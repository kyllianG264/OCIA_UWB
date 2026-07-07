from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class CameraZone:
    camera: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    margin: float = 0.0

    def contains(self, x: float, y: float, *, margin: Optional[float] = None) -> bool:
        band = self.margin if margin is None else float(margin)
        return (
            self.x_min - band <= float(x) <= self.x_max + band
            and self.y_min - band <= float(y) <= self.y_max + band
        )


def load_camera_zones(calibration_geometry: Dict[str, object], *, margin: float = 0.0) -> Dict[str, CameraZone]:
    terrain_bounds = calibration_geometry.get("terrain_bounds", {}) or {}
    zones: Dict[str, CameraZone] = {}
    for camera_name, bounds in terrain_bounds.items():
        if not isinstance(bounds, dict):
            continue
        try:
            zones[str(camera_name).strip().lower()] = CameraZone(
                camera=str(camera_name).strip().lower(),
                x_min=float(bounds["x_min"]),
                x_max=float(bounds["x_max"]),
                y_min=float(bounds["y_min"]),
                y_max=float(bounds["y_max"]),
                margin=float(margin),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return zones
