from __future__ import annotations

from typing import Optional


def filter_continuity_track(track, target_frame: int, target_timestamp_s: Optional[float], args) -> None:
    from .tracking_pipeline import advance_track_without_detection

    advance_track_without_detection(track, target_frame, target_timestamp_s, args)
