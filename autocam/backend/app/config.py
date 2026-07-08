"""
config.py
Gestion centralisée des paramètres du POC AutoCam.
Tous les réglages sont persistés dans data/config.json.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = DATA_DIR / "config.json"

DEFAULTS: dict = {
    "pose_backend": "detector_yolo_pose",
    "detector_pose": {
        "detector_model_name": "yolo11n.pt",      # repérage initial + tracking (classe person)
        "pose_model_name": "yolo11n-pose.pt",     # extraction des keypoints biomécaniques
        "detector_conf_threshold": 0.35,
        "pose_conf_threshold": 0.5,
        "max_frames": 300,
        "center_crop_ratio": 1.0,
        "min_body_height_ratio": 0.0,
        "min_hip_velocity": 0.015,
        "device": "auto",
    },
    "yolo": {
        "model_name": "yolov8n-pose.pt",  # nano=rapide, s=équilibré, m=précis
        "conf_threshold": 0.5,
        "max_frames": 300,
        "center_crop_ratio": 1.0,
        "min_body_height_ratio": 0.0,
        "min_hip_velocity": 0.015,
        "device": "auto",                 # "auto", "cpu", "cuda" ou "mps"
    },
    "features": {
        "visibility_threshold": 0.4,      # Seuil de visibilité des landmarks clés
        "flight_ankle_y_threshold": 0.6,  # Seuil Y (normalisé) pour "phase aérienne"
        "min_valid_frames": 5,            # Frames valides minimum par clip
    },
    "clustering": {
        "algorithm": "kmeans_auto",
        "n_init": 20,
        "max_iter": 500,
        "random_state": 42,
        "max_auto_clusters": 8,
        "min_silhouette_score": 0.0,
        "manual_distance_factor": 1.5,
        "min_clusters": 0,
    },
    "display": {
        "cols_per_row": 3,
    },
    # Objectif identité (caméra de profil) : poids UNIFORMES sur les features
    # vivantes (anthropométrie + bio du saut, complémentaires et de force
    # comparable — mesuré via 1-NN). Seules les largeurs L/R, invalides de
    # profil, restent à 0. Voir feature_extractor.py.
    "feature_weights": {
        "shoulder_width_ratio": 0.0,   # ❌ vue côté : L/R épaules superposées
        "hip_shoulder_ratio":   0.0,   # ❌ vue côté : ratio instable
        "lateral_oscillation":  0.0,   # ❌ profondeur, bruité de profil
        "hip_height_max":       1.0,
        "trunk_angle_mean":     1.0,
        "trunk_angle_std":      1.0,
        "flight_ratio":         1.0,
        "stride_oscillation":   1.0,
        "approach_speed_norm":  1.0,
        "crural_index":         1.0,
        "brachial_index":       1.0,
        "leg_torso_ratio":      1.0,
        "arm_torso_ratio":      1.0,
        "leg_arm_ratio":        1.0,
    },
}


def load_config() -> dict:
    """Charge la config depuis JSON, avec merge automatique sur DEFAULTS."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Merge section par section : les clés manquantes viennent des defaults
            merged: dict = {
                "pose_backend": saved.get("pose_backend", DEFAULTS["pose_backend"]),
            }
            if merged["pose_backend"] == "sam3_yolo":
                merged["pose_backend"] = "detector_yolo_pose"
            for section in DEFAULTS:
                if section == "pose_backend":
                    continue
                merged[section] = {**DEFAULTS[section], **saved.get(section, {})}
            return merged
        except (json.JSONDecodeError, KeyError):
            pass
    return {section: (dict(values) if isinstance(values, dict) else values) for section, values in DEFAULTS.items()}


def save_config(config: dict) -> None:
    """Sauvegarde la config dans data/config.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def reset_config() -> dict:
    """Réinitialise aux valeurs par défaut et sauvegarde."""
    config = {section: dict(values) for section, values in DEFAULTS.items()}
    save_config(config)
    return config
