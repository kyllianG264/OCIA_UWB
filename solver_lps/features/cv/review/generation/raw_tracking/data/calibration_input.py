from __future__ import annotations

from typing import Any

import numpy as np

from solver_lps.features.cv.review.generation.raw_tracking.data.tracking_output import (
    load_json,
)


def load_calibration(path: str) -> dict[str, Any]:
    data = load_json(path)
    try:
        h_g = np.array(data["cam_gauche"]["H"], dtype=float)
        h_d = np.array(data["cam_droite"]["H"], dtype=float)
        bounds = data["terrain_bounds"]
    except KeyError as exc:
        raise ValueError(f"Calibration invalide, cle manquante : {exc}") from exc

    if "gauche" in bounds and "droite" in bounds:
        bounds_g = bounds["gauche"]
        bounds_d = bounds["droite"]
    else:
        bounds_g = {
            "x_min": bounds.get("CX_L", 0),
            "x_max": bounds.get("CX_R", 9999),
            "y_min": bounds.get("CY_TOP", 0),
            "y_max": bounds.get("CY_MID", 9999),
        }
        bounds_d = {
            "x_min": bounds.get("CX_L", 0),
            "x_max": bounds.get("CX_R", 9999),
            "y_min": bounds.get("CY_MID", 0),
            "y_max": bounds.get("CY_BOT", 9999),
        }

    view_g = data.get("cam_gauche", {}).get("undistort_view") or data.get("undistort_view_g") or {"scale": 1.0, "offset_x": 0.0, "offset_y": 0.0}
    view_d = data.get("cam_droite", {}).get("undistort_view") or data.get("undistort_view_d") or {"scale": 1.0, "offset_x": 0.0, "offset_y": 0.0}

    return {
        "raw": data,
        "H_g": h_g,
        "H_d": h_d,
        "bounds_g": bounds_g,
        "bounds_d": bounds_d,
        "distortion_g": data.get("cam_gauche", {}).get("distortion"),
        "distortion_d": data.get("cam_droite", {}).get("distortion"),
        "undistort_view_g": view_g,
        "undistort_view_d": view_d,
    }
