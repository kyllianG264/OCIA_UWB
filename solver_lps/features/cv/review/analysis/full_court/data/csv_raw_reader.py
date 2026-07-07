import csv
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RawDetection:
    frame: int
    timestamp_s: float
    timestamp_unix: str
    raw_player_id: str
    x: float
    y: float
    cam: str
    on_terrain: bool
    source_ids: List[str] = field(default_factory=list)
    source_cams: List[str] = field(default_factory=list)
    merged_count: int = 1
    status: str = "raw"
    primary_cam: str = ""


Detection = RawDetection


def read_raw_detections(csv_path: str) -> List[RawDetection]:
    detections: List[RawDetection] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            detections.append(
                RawDetection(
                    frame=int(float(row["frame"])),
                    timestamp_s=float(row.get("timestamp_s") or 0.0),
                    timestamp_unix=str(row.get("timestamp_unix") or ""),
                    raw_player_id=str(row.get("player_id") or "").strip(),
                    x=float(row["X"]),
                    y=float(row["Y"]),
                    cam=str(row.get("cam") or row.get("demi_terrain") or "").strip().lower(),
                    on_terrain=str(row.get("on_terrain", "1")).strip() not in {"0", "false", "False", ""},
                )
            )
    return detections


def load_frames(csv_path: str) -> List[List[RawDetection]]:
    grouped: Dict[int, List[RawDetection]] = {}
    for detection in read_raw_detections(csv_path):
        grouped.setdefault(detection.frame, []).append(detection)
    return [grouped[frame] for frame in sorted(grouped)]
