from __future__ import annotations

from typing import Optional

from ..data.raw_input import Detection


def filter_reattach_candidate(track, detection: Detection, predicted_state, frame_gap: int, args) -> Optional[float]:
    from .tracking_pipeline import match_cost

    return match_cost(track, detection, predicted_state, frame_gap, args, reattach=True)
