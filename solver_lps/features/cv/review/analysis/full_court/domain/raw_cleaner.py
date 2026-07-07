from __future__ import annotations

from typing import Iterable, List

from ..data.csv_raw_reader import Detection
from .camera_zones import CameraZone


def keep_useful_detections(
    detections: Iterable[Detection],
    *,
    zone: CameraZone | None,
    strong_margin: float = 0.0,
) -> List[Detection]:
    kept: List[Detection] = []
    for detection in detections:
        if not detection.on_terrain:
            continue
        if zone is not None and not zone.contains(detection.x, detection.y, margin=strong_margin):
            continue
        kept.append(detection)
    return kept
