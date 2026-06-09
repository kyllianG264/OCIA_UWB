import json
import os


def load_terrain_calibration(calibration_path):
    if not calibration_path or not os.path.exists(calibration_path):
        return None
    try:
        with open(calibration_path, "r", encoding="utf-8") as fh:
            calibration = json.load(fh)
    except Exception:
        return None

    terrain_bounds = calibration.get("terrain_bounds") or {}
    has_explicit_split_y = "split_y" in calibration
    explicit_split_y = calibration.get("split_y")
    split_axis = calibration.get("split_axis")
    x_values = []
    y_values = []
    halves = []
    for bounds in terrain_bounds.values():
        if not isinstance(bounds, dict):
            continue
        try:
            x_values.extend([float(bounds["x_min"]), float(bounds["x_max"])])
            y_values.extend([float(bounds["y_min"]), float(bounds["y_max"])])
            halves.append(
                {
                    "x_min": float(bounds["x_min"]),
                    "x_max": float(bounds["x_max"]),
                    "y_min": float(bounds["y_min"]),
                    "y_max": float(bounds["y_max"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not x_values or not y_values:
        return None

    y_values.sort()
    if split_axis == "x":
        split_y = None
    elif has_explicit_split_y:
        try:
            split_y = float(explicit_split_y)
        except (TypeError, ValueError):
            split_y = None
    else:
        split_y = None
    if split_y is None:
        if len(halves) >= 2:
            sorted_halves = sorted(halves, key=lambda item: (item["y_min"] + item["y_max"]) / 2.0)
            upper_half = sorted_halves[0]
            lower_half = sorted_halves[-1]
            split_y = (upper_half["y_max"] + lower_half["y_min"]) / 2.0
        else:
            split_y = (min(y_values) + max(y_values)) / 2.0
    return {
        "split_y": split_y,
        "bounds": (min(x_values), max(x_values), min(y_values), max(y_values)),
        "halves": halves,
        "terrain_png_size": calibration.get("terrain_png_size"),
        "terrain_image_path": calibration.get("terrain_image_path"),
    }


def load_calibration_geometry(calibration_path):
    calibration = load_terrain_calibration(calibration_path)
    if calibration is None:
        return {"split_y": None, "bounds": None}
    return {
        "split_y": calibration.get("split_y"),
        "bounds": calibration.get("bounds"),
    }
