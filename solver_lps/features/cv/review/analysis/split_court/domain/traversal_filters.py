from __future__ import annotations

from typing import Optional

from .terrain_rules import allow_camera_handoff


def filter_traversal_transition(
    previous_cam: str,
    new_cam: str,
    prev_y: float,
    new_y: float,
    split_y: Optional[float],
    *,
    frame_gap: int,
    prev_on_terrain: bool,
    new_on_terrain: bool,
) -> bool:
    return allow_camera_handoff(
        previous_cam,
        new_cam,
        prev_y,
        new_y,
        split_y,
        frame_gap=frame_gap,
        prev_on_terrain=prev_on_terrain,
        new_on_terrain=new_on_terrain,
    )
