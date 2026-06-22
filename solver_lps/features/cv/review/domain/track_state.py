from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TrackState:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    last_frame: int
    last_timestamp_s: float
    hits: int = 1
    misses: int = 0
    age: int = 1
    confirmed: bool = False
    last_raw_player_id: str = ""
    last_cam: str = ""
    last_on_terrain: bool = True
    confidence: float = 0.2
    display_pos: Optional[Tuple[float, float]] = None

    def track_label(self) -> str:
        return f"T{self.track_id:03d}"
