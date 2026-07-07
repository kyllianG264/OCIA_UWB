from __future__ import annotations

from typing import Dict, List

from ..data.raw_input import Detection
from .observation_dedup import dedupe_owned_detections as dedupe_observation_owned_detections


def filter_duplicate_observations(
    detections: List[Detection],
    active_tracks,
    lost_tracks,
    current_frame: int,
    args,
    stats: Dict[str, int],
    spawn_guard=None,
) -> List[Detection]:
    return dedupe_observation_owned_detections(
        detections,
        active_tracks,
        lost_tracks,
        current_frame,
        args,
        stats,
        spawn_guard=spawn_guard,
    )
