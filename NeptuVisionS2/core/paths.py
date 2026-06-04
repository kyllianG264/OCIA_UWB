"""
Chemins centralises du projet.
"""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT_DIR.parent
ASSETS_DIR = ROOT_DIR / "assets"
DOCS_DIR = ROOT_DIR / "docs"
MODELS_DIR = ASSETS_DIR / "models"
POSE_MODELS_DIR = MODELS_DIR / "pose"
TERRAIN_DIR = ASSETS_DIR / "terrain"
VIDEOS_DIR = ASSETS_DIR / "videos"
SAMPLE_VIDEOS_DIR = VIDEOS_DIR / "samples"
PYTHON_DIR = REPO_ROOT / "python"
PYTHON_CV_LOGS_DIR = PYTHON_DIR / "CV Logs"

DEFAULT_TERRAIN_IMAGE = TERRAIN_DIR / "terrain-de-basket.jpg"
DEFAULT_POSE_MODEL = POSE_MODELS_DIR / "yolo11x-pose.pt"
