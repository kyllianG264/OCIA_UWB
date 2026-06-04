"""
Reglages de performance partages pour le tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.paths import POSE_MODELS_DIR

FPS_TARGET_CHOICES = (10, 15, 20, 25)
AGGRESSIVENESS_CHOICES = ("low", "medium", "high")


@dataclass(frozen=True)
class PerformanceSettings:
    model: str
    conf: float
    imgsz: int
    separate_camera_models: bool


def _round_imgsz(value: int) -> int:
    value = max(640, min(1792, value))
    return max(640, int(round(value / 32) * 32))


def resolve_performance_settings(target_fps: int, aggressiveness: str) -> PerformanceSettings:
    base_imgsz_by_fps = {
        10: 1536,
        15: 1408,
        20: 1280,
        25: 960,
    }
    imgsz_delta_by_aggressiveness = {
        "low": 192,
        "medium": 0,
        "high": -192,
    }
    conf_by_aggressiveness = {
        "low": 0.03,
        "medium": 0.05,
        "high": 0.08,
    }

    base_imgsz = base_imgsz_by_fps.get(target_fps, 1408)
    imgsz = _round_imgsz(base_imgsz + imgsz_delta_by_aggressiveness.get(aggressiveness, 0))
    conf = conf_by_aggressiveness.get(aggressiveness, 0.05)

    return PerformanceSettings(
        model=str(Path(POSE_MODELS_DIR) / "yolo11x-pose.pt"),
        conf=conf,
        imgsz=imgsz,
        separate_camera_models=True,
    )
