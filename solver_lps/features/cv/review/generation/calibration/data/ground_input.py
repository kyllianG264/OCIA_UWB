from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from solver_lps.features.cv.review.data.session_assets import (
    DEFAULT_SPORT,
    SessionAssets,
)


@dataclass(frozen=True)
class GroundInput:
    ground_dir: Path
    video_input_dir: Path
    ground_output_dir: Path
    default_left_video_path: Path
    default_right_video_path: Path
    default_output_dir: Path
    terrain_path: str
    terrain_rgb: np.ndarray
    terrain_width: int
    terrain_height: int


def load_ground_input(
    sport: str = DEFAULT_SPORT,
) -> GroundInput:
    assets = SessionAssets(sport=sport)
    ground_root = assets.ground_dir
    video_input_dir = assets.ground_input_dir
    if not video_input_dir.exists():
        raise FileNotFoundError(f"Video input directory introuvable : {video_input_dir}")
    output_dir = assets.ground_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    terrain_path = str(assets.terrain_path)
    if not os.path.isfile(terrain_path):
        raise FileNotFoundError(f"Missing court image: {terrain_path}")

    terrain_pil = Image.open(terrain_path).convert("RGB")
    terrain_rgb = np.array(terrain_pil)
    terrain_width, terrain_height = terrain_pil.size
    return GroundInput(
        ground_dir=ground_root,
        video_input_dir=video_input_dir,
        ground_output_dir=output_dir,
        default_left_video_path=video_input_dir / "left_video.mp4",
        default_right_video_path=video_input_dir / "right_video.mp4",
        default_output_dir=output_dir,
        terrain_path=terrain_path,
        terrain_rgb=terrain_rgb,
        terrain_width=terrain_width,
        terrain_height=terrain_height,
    )
