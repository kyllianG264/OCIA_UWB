import json
import os
from collections import deque

# =========================================================
# SCENE / WINDOW
# =========================================================
WIDTH, HEIGHT = 1300, 950
FPS = 60

BG_COLOR = (12, 14, 20)
GRID_COLOR = (50, 56, 68)
BOUNDARY_COLOR = (90, 100, 118)
HUD_TEXT = (235, 235, 235)
HUD_SUB = (190, 195, 205)

# =========================================================
# WORLD GEOMETRY (cm)
# =========================================================
CENTER_X = 1250
CENTER_Y = 900

LEFT_X = 550
RIGHT_X = 1950
TOP_Y = 350
BOTTOM_Y = 1450

TOP_MID_Y = 100
BOTTOM_MID_Y = 1800

WORLD_MIN_X = 200
WORLD_MAX_X = 2300
WORLD_MIN_Y = 0
WORLD_MAX_Y = 1900

# =========================================================
# ANCHOR HEIGHTS (cm)
# =========================================================
ANCHOR_Z_LOW = 1500
ANCHOR_Z_HIGH = 2400
ANCHOR_Z_MID = 1050
ANCHOR_Z_MID_HIGH = 2100

# =========================================================
# TAG MOTION / HEIGHT
# =========================================================
TAG_BASE_Z_CM = 20.0
JUMP_EXTRA_HEIGHT_CM = 100.0
AUTO_JUMP_PROB_PER_SEC = 0.35
JUMP_DURATION_MIN = 0.70
JUMP_DURATION_MAX = 1.10

# =========================================================
# UWB NOISE MODEL
# =========================================================
NOISE_RATIO = 0.005
SPIKE_PROB = 0.00
SPIKE_AMPLITUDE = 0.0

DEFAULT_ALPHA_DIST = 0.16
DEFAULT_ALPHA_POS = 0.12
DEFAULT_PRECISION_TOLERANCE_CM = 10.0
DEFAULT_NOISE_RATIO = NOISE_RATIO
DEFAULT_SPIKE_PROB = SPIKE_PROB
DEFAULT_SPIKE_AMPLITUDE = SPIKE_AMPLITUDE

# =========================================================
# CAMERA
# =========================================================
FOV_DEG = 60
NEAR_PLANE = 5.0
CAMERA_YAW = -0.55
CAMERA_PITCH = 0.55
CAMERA_DISTANCE = 3700.0

# =========================================================
# TRAILS / HISTORY
# =========================================================
TRAIL_MAX_POINTS = 200
REAL_HISTORY_MAXLEN = 1200

CACHE_FILE = os.path.join(os.path.dirname(__file__), "settings_cache.json")


def _default_layouts():
    return {
        "4": {
            "anchors": {
                "1": [LEFT_X, TOP_Y, ANCHOR_Z_LOW],
                "2": [RIGHT_X, TOP_Y, ANCHOR_Z_LOW],
                "3": [RIGHT_X, BOTTOM_Y, ANCHOR_Z_LOW],
                "4": [LEFT_X, BOTTOM_Y, ANCHOR_Z_HIGH],
            },
            "name": "4 ancres : carre avec A4 plus haute",
        },
        "5": {
            "anchors": {
                "1": [LEFT_X, TOP_Y, ANCHOR_Z_LOW],
                "2": [RIGHT_X, TOP_Y, ANCHOR_Z_LOW],
                "3": [RIGHT_X, BOTTOM_Y, ANCHOR_Z_LOW],
                "4": [LEFT_X, BOTTOM_Y, ANCHOR_Z_HIGH],
                "5": [CENTER_X, TOP_MID_Y, ANCHOR_Z_MID],
            },
            "name": "5 ancres : carre + pointe avant",
        },
        "6": {
            "anchors": {
                "1": [LEFT_X, TOP_Y, ANCHOR_Z_LOW],
                "2": [RIGHT_X, TOP_Y, ANCHOR_Z_LOW],
                "3": [RIGHT_X, BOTTOM_Y, ANCHOR_Z_LOW],
                "4": [LEFT_X, BOTTOM_Y, ANCHOR_Z_HIGH],
                "5": [CENTER_X, TOP_MID_Y, ANCHOR_Z_MID],
                "6": [CENTER_X, BOTTOM_MID_Y, ANCHOR_Z_MID_HIGH],
            },
            "name": "6 ancres : carre + pointes avant/arriere",
        },
    }


def get_default_settings():
    return {
        "active_anchor_count": 6,
        "alpha_dist": DEFAULT_ALPHA_DIST,
        "alpha_pos": DEFAULT_ALPHA_POS,
        "precision_tolerance_cm": DEFAULT_PRECISION_TOLERANCE_CM,
        "noise_ratio": DEFAULT_NOISE_RATIO,
        "spike_prob": DEFAULT_SPIKE_PROB,
        "spike_amplitude": DEFAULT_SPIKE_AMPLITUDE,
        "layouts": _default_layouts(),
    }


def _sanitize_settings(raw):
    defaults = get_default_settings()
    if not isinstance(raw, dict):
        return defaults

    settings = defaults
    settings["active_anchor_count"] = int(raw.get("active_anchor_count", defaults["active_anchor_count"]))
    settings["active_anchor_count"] = max(4, min(6, settings["active_anchor_count"]))
    settings["alpha_dist"] = float(raw.get("alpha_dist", defaults["alpha_dist"]))
    settings["alpha_pos"] = float(raw.get("alpha_pos", defaults["alpha_pos"]))
    settings["precision_tolerance_cm"] = float(raw.get("precision_tolerance_cm", defaults["precision_tolerance_cm"]))
    settings["noise_ratio"] = float(raw.get("noise_ratio", defaults["noise_ratio"]))
    settings["spike_prob"] = float(raw.get("spike_prob", defaults["spike_prob"]))
    settings["spike_amplitude"] = float(raw.get("spike_amplitude", defaults["spike_amplitude"]))

    settings["alpha_dist"] = max(0.01, min(0.95, settings["alpha_dist"]))
    settings["alpha_pos"] = max(0.01, min(0.95, settings["alpha_pos"]))
    settings["precision_tolerance_cm"] = max(1.0, min(1000.0, settings["precision_tolerance_cm"]))
    settings["noise_ratio"] = max(0.0, min(0.2, settings["noise_ratio"]))
    settings["spike_prob"] = max(0.0, min(1.0, settings["spike_prob"]))
    settings["spike_amplitude"] = max(0.0, min(1000.0, settings["spike_amplitude"]))

    layouts = defaults["layouts"]
    raw_layouts = raw.get("layouts", {})
    if isinstance(raw_layouts, dict):
        for key, default_layout in layouts.items():
            layout = raw_layouts.get(key, {})
            if isinstance(layout, dict):
                name = layout.get("name", default_layout["name"])
                anchors = {}
                raw_anchors = layout.get("anchors", {})
                if isinstance(raw_anchors, dict):
                    for anchor_id, default_pos in default_layout["anchors"].items():
                        values = raw_anchors.get(anchor_id, default_pos)
                        if isinstance(values, (list, tuple)) and len(values) == 3:
                            anchors[anchor_id] = [float(values[0]), float(values[1]), float(values[2])]
                        else:
                            anchors[anchor_id] = list(default_pos)
                else:
                    anchors = {anchor_id: list(pos) for anchor_id, pos in default_layout["anchors"].items()}
                layouts[key] = {"name": str(name), "anchors": anchors}
    settings["layouts"] = layouts
    return settings


def load_settings():
    if not os.path.exists(CACHE_FILE):
        return get_default_settings()
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError, TypeError):
        return get_default_settings()
    return _sanitize_settings(raw)


def save_settings(settings):
    payload = _sanitize_settings(settings)
    with open(CACHE_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def get_anchor_layout(settings, n):
    layout = settings["layouts"][str(n)]
    anchors = {
        int(anchor_id): tuple(pos)
        for anchor_id, pos in layout["anchors"].items()
    }
    return anchors, layout["name"]


def set_active_anchor_count(settings, count):
    settings["active_anchor_count"] = max(4, min(6, int(count)))


def adjust_setting(settings, key, delta):
    if key == "alpha_dist":
        settings[key] = max(0.01, min(0.95, settings[key] + delta))
    elif key == "alpha_pos":
        settings[key] = max(0.01, min(0.95, settings[key] + delta))
    elif key == "precision_tolerance_cm":
        settings[key] = max(1.0, min(1000.0, settings[key] + delta))
    elif key == "noise_ratio":
        settings[key] = max(0.0, min(0.2, settings[key] + delta))
    elif key == "spike_prob":
        settings[key] = max(0.0, min(1.0, settings[key] + delta))
    elif key == "spike_amplitude":
        settings[key] = max(0.0, min(1000.0, settings[key] + delta))


def adjust_anchor(settings, anchor_count, anchor_id, axis_index, delta):
    anchor = settings["layouts"][str(anchor_count)]["anchors"][str(anchor_id)]
    anchor[axis_index] += delta


def reset_stats():
    return {
        "err3d_sum": 0.0,
        "errxy_sum": 0.0,
        "errz_sum": 0.0,
        "prec_sum": 0.0,
        "count": 0,
        "err3d_max": 0.0,
        "errz_max": 0.0,
        "traj_err_xy_sum": 0.0,
        "traj_err_3d_sum": 0.0,
        "traj_prec_sum": 0.0,
        "traj_count": 0,
        "traj_err_xy_max": 0.0,
        "traj_err_3d_max": 0.0,
        "real_path_len": 0.0,
        "est_path_len": 0.0,
        "delay_est_sum": 0.0,
        "delay_measured_sum": 0.0,
        "delay_count": 0,
        "delay_measured_count": 0,
        "delay_measured_max": 0.0,
        "last_real": None,
        "last_est": None,
    }


def create_state(settings=None):
    if settings is None:
        settings = load_settings()
    return {
        "t": 0.0,
        "settings": settings,
        "stats": reset_stats(),
        "dist_smooth": {},
        "tag_est_smooth": None,
        "trail_real": deque(maxlen=TRAIL_MAX_POINTS),
        "trail_est": deque(maxlen=TRAIL_MAX_POINTS),
        "real_history": deque(maxlen=REAL_HISTORY_MAXLEN),
        "jump_active": False,
        "jump_timer": 0.0,
        "jump_duration": 0.0,
        "jump_extra_height": 0.0,
        "ui": {
            "hud_expanded": True,
            "selected_control": 0,
            "cache_dirty": False,
            "cache_message": "Cache charge",
        },
    }


def full_reset(state, anchor_count):
    state["t"] = 0.0
    refresh_runtime_state(state, anchor_count, clear_real_trail=True)

    state["jump_active"] = False
    state["jump_timer"] = 0.0
    state["jump_duration"] = 0.0
    state["jump_extra_height"] = 0.0


def refresh_runtime_state(state, anchor_count, clear_real_trail=False):
    state["stats"] = reset_stats()
    state["tag_est_smooth"] = None
    if clear_real_trail:
        state["trail_real"].clear()
    state["trail_est"].clear()
    state["real_history"].clear()

    anchors, _ = get_anchor_layout(state["settings"], anchor_count)
    state["dist_smooth"] = {aid: None for aid in anchors.keys()}


anchor_colors = {
    1: (90, 185, 255),
    2: (120, 255, 140),
    3: (255, 175, 70),
    4: (195, 130, 255),
    5: (255, 105, 145),
    6: (95, 230, 230),
}
