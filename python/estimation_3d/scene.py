import json
import os
from collections import deque

from player_analytics import create_player_analytics, reset_player_analytics


WIDTH = 1520
HEIGHT = 920
FPS = 60

BG_COLOR = (12, 14, 24)
GRID_COLOR = (48, 62, 76)
BOUNDARY_COLOR = (110, 130, 150)
HUD_TEXT = (245, 245, 245)
HUD_SUB = (180, 196, 210)

CENTER_X = 1250.0
CENTER_Y = 900.0
LEFT_X = -250.0
RIGHT_X = 2750.0
TOP_Y = -150.0
BOTTOM_Y = 1950.0
TOP_MID_Y = -280.0
BOTTOM_MID_Y = 2080.0

WORLD_MIN_X = -450.0
WORLD_MAX_X = 2950.0
WORLD_MIN_Y = -350.0
WORLD_MAX_Y = 2150.0

ANCHOR_Z_LOW = 600.0
ANCHOR_Z_HIGH = 600.0
ANCHOR_Z_MID = 600.0
ANCHOR_Z_MID_HIGH = 600.0

TAG_BASE_Z_CM = 95.0
JUMP_EXTRA_HEIGHT_CM = 70.0

NOISE_RATIO = 0.012
SPIKE_PROB = 0.02
SPIKE_AMPLITUDE = 110.0

DEFAULT_ALPHA_DIST = 0.12
DEFAULT_ALPHA_POS = 0.08
DEFAULT_PRECISION_TOLERANCE_CM = 90.0
DEFAULT_NOISE_RATIO = NOISE_RATIO
DEFAULT_SPIKE_PROB = SPIKE_PROB
DEFAULT_SPIKE_AMPLITUDE = SPIKE_AMPLITUDE

FOV_DEG = 55.0
NEAR_PLANE = 20.0
CAMERA_YAW = 0.65
CAMERA_PITCH = 0.72
CAMERA_DISTANCE = 3300.0

TRAIL_MAX_POINTS = 220
REAL_HISTORY_MAXLEN = 480

CACHE_FILE = os.path.join(os.path.dirname(__file__), "settings_cache.json")
SETTINGS_VERSION = 2


def _default_layouts():
    return {
        "2": {
            "name": "2 ancres : ligne",
            "anchors": {
                "1": [LEFT_X, CENTER_Y, ANCHOR_Z_LOW],
                "2": [RIGHT_X, CENTER_Y, ANCHOR_Z_HIGH],
            },
        },
        "3": {
            "name": "3 ancres : triangle",
            "anchors": {
                "1": [LEFT_X, BOTTOM_Y, ANCHOR_Z_LOW],
                "2": [CENTER_X, TOP_Y, ANCHOR_Z_HIGH],
                "3": [RIGHT_X, BOTTOM_Y, ANCHOR_Z_MID],
            },
        },
        "4": {
            "name": "4 ancres : carre 3D",
            "anchors": {
                "1": [LEFT_X, TOP_Y, ANCHOR_Z_LOW],
                "2": [RIGHT_X, TOP_Y, ANCHOR_Z_HIGH],
                "3": [RIGHT_X, BOTTOM_Y, ANCHOR_Z_LOW],
                "4": [LEFT_X, BOTTOM_Y, ANCHOR_Z_HIGH],
            },
        },
        "5": {
            "name": "5 ancres : carre + haut",
            "anchors": {
                "1": [LEFT_X, TOP_Y, ANCHOR_Z_LOW],
                "2": [RIGHT_X, TOP_Y, ANCHOR_Z_HIGH],
                "3": [RIGHT_X, BOTTOM_Y, ANCHOR_Z_LOW],
                "4": [LEFT_X, BOTTOM_Y, ANCHOR_Z_HIGH],
                "5": [CENTER_X, TOP_MID_Y, ANCHOR_Z_MID_HIGH],
            },
        },
        "6": {
            "name": "6 ancres : carre + haut/bas",
            "anchors": {
                "1": [LEFT_X, TOP_Y, ANCHOR_Z_LOW],
                "2": [RIGHT_X, TOP_Y, ANCHOR_Z_HIGH],
                "3": [RIGHT_X, BOTTOM_Y, ANCHOR_Z_LOW],
                "4": [LEFT_X, BOTTOM_Y, ANCHOR_Z_HIGH],
                "5": [CENTER_X, TOP_MID_Y, ANCHOR_Z_MID_HIGH],
                "6": [CENTER_X, BOTTOM_MID_Y, ANCHOR_Z_MID],
            },
        },
    }


def get_default_settings():
    return {
        "settings_version": SETTINGS_VERSION,
        "active_anchor_count": 4,
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

    settings = get_default_settings()
    same_layout_version = int(raw.get("settings_version", 0)) == SETTINGS_VERSION
    settings["active_anchor_count"] = max(2, min(6, int(raw.get("active_anchor_count", defaults["active_anchor_count"]))))
    settings["alpha_dist"] = max(0.01, min(0.95, float(raw.get("alpha_dist", defaults["alpha_dist"]))))
    settings["alpha_pos"] = max(0.01, min(0.95, float(raw.get("alpha_pos", defaults["alpha_pos"]))))
    settings["precision_tolerance_cm"] = max(5.0, min(2000.0, float(raw.get("precision_tolerance_cm", defaults["precision_tolerance_cm"]))))
    settings["noise_ratio"] = max(0.0, min(0.25, float(raw.get("noise_ratio", defaults["noise_ratio"]))))
    settings["spike_prob"] = max(0.0, min(1.0, float(raw.get("spike_prob", defaults["spike_prob"]))))
    settings["spike_amplitude"] = max(0.0, min(3000.0, float(raw.get("spike_amplitude", defaults["spike_amplitude"]))))

    raw_layouts = raw.get("layouts", {}) if same_layout_version else {}
    for key, default_layout in defaults["layouts"].items():
        layout = raw_layouts.get(key, {}) if isinstance(raw_layouts, dict) else {}
        name = str(layout.get("name", default_layout["name"])) if isinstance(layout, dict) else default_layout["name"]
        anchors = {}
        raw_anchors = layout.get("anchors", {}) if isinstance(layout, dict) else {}
        for anchor_id, default_pos in default_layout["anchors"].items():
            values = raw_anchors.get(anchor_id, default_pos) if isinstance(raw_anchors, dict) else default_pos
            if isinstance(values, (list, tuple)) and len(values) == 3:
                pos = [float(values[0]), float(values[1]), float(values[2])]
            else:
                pos = list(default_pos)
            anchors[anchor_id] = pos
        settings["layouts"][key] = {"name": name, "anchors": anchors}
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
    anchors = {int(anchor_id): tuple(pos) for anchor_id, pos in layout["anchors"].items()}
    return anchors, layout["name"]


def set_active_anchor_count(settings, count):
    settings["active_anchor_count"] = max(2, min(6, int(count)))


def adjust_setting(settings, key, delta):
    limits = {
        "alpha_dist": (0.01, 0.95),
        "alpha_pos": (0.01, 0.95),
        "precision_tolerance_cm": (5.0, 2000.0),
        "noise_ratio": (0.0, 0.25),
        "spike_prob": (0.0, 1.0),
        "spike_amplitude": (0.0, 3000.0),
    }
    lo, hi = limits[key]
    settings[key] = max(lo, min(hi, settings[key] + delta))


def adjust_anchor(settings, anchor_count, anchor_id, axis_index, delta):
    anchor = settings["layouts"][str(anchor_count)]["anchors"][str(anchor_id)]
    anchor[axis_index] += delta


def reset_stats():
    return {
        "err3d_sum": 0.0,
        "errxy_sum": 0.0,
        "errz_sum": 0.0,
        "precision_sum": 0.0,
        "delay_est_sum": 0.0,
        "delay_measured_sum": 0.0,
        "count": 0,
        "max_measured_delay": 0.0,
        "traj_err3d_sum": 0.0,
        "traj_errxy_sum": 0.0,
        "traj_errz_sum": 0.0,
        "traj_precision_sum": 0.0,
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
        "delay_dist_s": 0.0,
        "delay_pos_s": 0.0,
        "trail_real": deque(maxlen=TRAIL_MAX_POINTS),
        "trail_est": deque(maxlen=TRAIL_MAX_POINTS),
        "real_history": deque(maxlen=REAL_HISTORY_MAXLEN),
        "player_analytics": create_player_analytics(CENTER_X, CENTER_Y),
        "jump": {"active": False, "elapsed": 0.0, "duration": 0.0, "height": 0.0},
        "ui": {"selected_control": 0, "hud_expanded": True, "hud_visible": True, "cache_dirty": False, "cache_message": "", "display_cache": {}},
    }


def full_reset(state, anchor_count):
    refresh_runtime_state(state, anchor_count, clear_real_trail=True)


def refresh_runtime_state(state, anchor_count, clear_real_trail=False):
    state["stats"] = reset_stats()
    state["tag_est_smooth"] = None
    state["delay_dist_s"] = 0.0
    state["delay_pos_s"] = 0.0
    reset_player_analytics(state["player_analytics"])
    state["trail_est"].clear()
    if clear_real_trail:
        state["trail_real"].clear()
        state["real_history"].clear()
        state["t"] = 0.0
    anchors, _ = get_anchor_layout(state["settings"], anchor_count)
    state["dist_smooth"] = {aid: None for aid in anchors.keys()}


anchor_colors = {1: (80, 180, 255), 2: (120, 255, 120), 3: (255, 170, 60), 4: (180, 120, 255), 5: (255, 110, 150), 6: (90, 220, 220)}
