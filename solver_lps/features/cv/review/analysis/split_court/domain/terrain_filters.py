from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..data.raw_input import Detection
from .merger_filters import merge_camera_duplicates as merge_tracking_detections
from .terrain_rules import normalize_detection_for_court


def filter_terrain_observation(detection: Detection, bounds, split_y: Optional[float]) -> Optional[Detection]:
    return normalize_detection_for_court(detection, bounds, split_y)


def filter_terrain_observations(frame_detections: List[Detection], args, stats: Dict[str, int]) -> Tuple[int, Optional[float], List[Detection]]:
    current_frame = frame_detections[0].frame if frame_detections else 0
    current_timestamp_s = frame_detections[0].timestamp_s if frame_detections else None
    usable_detections: List[Detection] = []

    for detection in frame_detections:
        normalized = filter_terrain_observation(detection, args.court_bounds, args.split_y)
        if normalized is None:
            stats["detections_rejected"] += 1
            continue
        usable_detections.append(normalized)

    stats["detections_used"] += len(usable_detections)
    if not usable_detections:
        return current_frame, current_timestamp_s, []

    detections = merge_tracking_detections(usable_detections, args.merge_distance, args.split_y)
    stats["detections_merged"] += sum(item.merged_count - 1 for item in detections)
    return current_frame, current_timestamp_s, detections
