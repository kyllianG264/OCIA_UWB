from __future__ import annotations

from typing import Dict, Iterable, List

from ..data.csv_raw_reader import Detection
from .camera_zones import CameraZone


def detection_score(detection: Detection, zone: CameraZone | None) -> float:
    score = 0.0
    if detection.on_terrain:
        score += 10.0
    if zone is not None and zone.contains(detection.x, detection.y):
        score += 5.0
        center_x = (zone.x_min + zone.x_max) / 2.0
        center_y = (zone.y_min + zone.y_max) / 2.0
        score -= abs(float(detection.x) - center_x) * 0.01
        score -= abs(float(detection.y) - center_y) * 0.01
    return score


def select_best_camera_detections(
    detections: Iterable[Detection],
    *,
    camera_zones: Dict[str, CameraZone],
    expected_players: int,
) -> List[Detection]:
    scored = sorted(
        detections,
        key=lambda detection: (
            -detection_score(detection, camera_zones.get(str(detection.cam).strip().lower())),
            str(detection.cam),
            str(detection.raw_player_id),
        ),
    )
    return scored[: max(0, int(expected_players))]
