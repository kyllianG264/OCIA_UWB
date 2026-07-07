from typing import List, Optional

from ..data.raw_input import Detection
from .terrain_rules import euclidean, is_near_split


def merge_camera_duplicates(detections: List[Detection], merge_distance: float, split_y: Optional[float]) -> List[Detection]:
    merged: List[Detection] = []
    used = set()
    for index, detection in enumerate(detections):
        if index in used:
            continue
        cluster = [detection]
        used.add(index)
        for other_index in range(index + 1, len(detections)):
            if other_index in used:
                continue
            other = detections[other_index]
            if detection.cam == other.cam:
                continue
            if detection.on_terrain != other.on_terrain:
                continue
            if detection.on_terrain and not (
                is_near_split(detection.y, split_y, 70.0)
                and is_near_split(other.y, split_y, 70.0)
            ):
                continue
            if euclidean((detection.x, detection.y), (other.x, other.y)) > merge_distance:
                continue
            cluster.append(other)
            used.add(other_index)
        if len(cluster) == 1:
            merged.append(detection)
            continue
        avg_x = sum(item.x for item in cluster) / len(cluster)
        avg_y = sum(item.y for item in cluster) / len(cluster)
        merged.append(
            Detection(
                frame=detection.frame,
                timestamp_s=detection.timestamp_s,
                timestamp_unix=detection.timestamp_unix,
                raw_player_id=detection.raw_player_id,
                x=avg_x,
                y=avg_y,
                cam="+".join(sorted({item.cam for item in cluster if item.cam})),
                on_terrain=detection.on_terrain,
                source_ids=sorted({sid for item in cluster for sid in item.source_ids}),
                source_cams=sorted({cam for item in cluster for cam in item.source_cams}),
                merged_count=len(cluster),
                status="merged",
                spawn_allowed=any(item.spawn_allowed for item in cluster),
            )
        )
    return merged
