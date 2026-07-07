import json
import os
import warnings


class CalibrationWarning(UserWarning):
    """A calibration could not be loaded without changing legacy return values."""


def _warn(message):
    warnings.warn(message, CalibrationWarning, stacklevel=2)


def load_terrain_calibration(calibration_path):
    if not calibration_path:
        _warn("Calibration path is empty")
        return None
    if not os.path.exists(calibration_path):
        _warn(f"Calibration file does not exist: {calibration_path}")
        return None
    try:
        with open(calibration_path, "r", encoding="utf-8") as fh:
            calibration = json.load(fh)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _warn(f"Cannot read calibration {calibration_path}: {exc}")
        return None
    if not isinstance(calibration, dict):
        _warn(f"Calibration root must be an object: {calibration_path}")
        return None

    terrain_bounds = calibration.get("terrain_bounds") or {}
    if not isinstance(terrain_bounds, dict):
        _warn(f"terrain_bounds must be an object in calibration {calibration_path}")
        return None
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
        _warn(f"Calibration has no valid terrain bounds: {calibration_path}")
        return None

    y_values.sort()
    if split_axis == "x":
        # The legacy field is consumed strictly as a Y coordinate downstream.
        split_y = None
    elif has_explicit_split_y:
        try:
            split_y = float(explicit_split_y)
        except (TypeError, ValueError):
            _warn(f"Invalid split_y {explicit_split_y!r} in calibration {calibration_path}")
            split_y = None
    else:
        split_y = None
    if split_y is None and split_axis != "x":
        if len(halves) >= 2:
            sorted_halves = sorted(halves, key=lambda item: (item["y_min"] + item["y_max"]) / 2.0)
            upper_half = sorted_halves[0]
            lower_half = sorted_halves[-1]
            split_y = (upper_half["y_max"] + lower_half["y_min"]) / 2.0
        else:
            split_y = (min(y_values) + max(y_values)) / 2.0
    if split_axis not in {None, "x", "y"}:
        _warn(f"Unsupported split_axis {split_axis!r} in calibration {calibration_path}")
    terrain_image_path = calibration.get("terrain_image_path")
    if terrain_image_path and not os.path.isabs(terrain_image_path):
        terrain_image_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(calibration_path)), terrain_image_path)
        )
    return {
        "split_y": split_y,
        "bounds": (min(x_values), max(x_values), min(y_values), max(y_values)),
        "halves": halves,
        "terrain_bounds": terrain_bounds,
        "split_axis": split_axis,
        "terrain_png_size": calibration.get("terrain_png_size"),
        "terrain_image_path": terrain_image_path,
    }


def load_calibration_geometry(calibration_path):
    calibration = load_terrain_calibration(calibration_path)
    if calibration is None:
        return {"split_y": None, "bounds": None, "terrain_bounds": {}, "split_axis": None}
    return {
        "split_y": calibration.get("split_y"),
        "bounds": calibration.get("bounds"),
        "terrain_bounds": calibration.get("terrain_bounds", {}),
        "split_axis": calibration.get("split_axis"),
    }


def load_ground_calibration(calibration_path):
    return load_terrain_calibration(calibration_path)


def load_ground_geometry(calibration_path):
    return load_calibration_geometry(calibration_path)
