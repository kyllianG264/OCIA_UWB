import json
import os

from solver_lps.features.players.domain.player_registry import bind_selected_player_analytics, create_player_registry, reset_player_registry
from solver_lps.session_assets import DEFAULT_SPORT, SessionAssets


WIDTH, HEIGHT = 1400, 900
FPS = 60
SCALE = 0.2

CAMERA_X = 1250
CAMERA_Y = 900

CENTER_X = 1250
CENTER_Y = 900

LEFT_X = -250
RIGHT_X = 2750
TOP_Y = -150
BOTTOM_Y = 1950
TOP_MID_Y = -280
BOTTOM_MID_Y = 2080

TRI_TOP = (1250, -220)
TRI_BL = (-180, 1880)
TRI_BR = (2680, 1880)
LINE_LEFT = (-320, 900)
LINE_RIGHT = (2820, 900)

DEFAULT_ALPHA_DIST = 0.12
DEFAULT_ALPHA_POS = 0.10
DEFAULT_PRECISION_TOLERANCE_CM = 100.0
DEFAULT_NOISE_STD = 18.0
DEFAULT_SPIKE_PROB = 0.025
DEFAULT_SPIKE_AMPLITUDE = 120.0

CACHE_DIR = os.path.dirname(__file__)
SETTINGS_VERSION = 2


def _cache_file(sport: str = DEFAULT_SPORT) -> str:
    safe_sport = str(sport or DEFAULT_SPORT).strip() or DEFAULT_SPORT
    return os.path.join(CACHE_DIR, f"settings_cache_{safe_sport}.json")


def _fallback_layouts():
    return {
        "2": {"name": "2 ancres : ligne", "anchors": {"1": list(LINE_LEFT), "2": list(LINE_RIGHT)}},
        "3": {"name": "3 ancres : triangle", "anchors": {"1": list(TRI_BL), "2": list(TRI_TOP), "3": list(TRI_BR)}},
        "4": {"name": "4 ancres : carre", "anchors": {"1": [LEFT_X, TOP_Y], "2": [RIGHT_X, TOP_Y], "3": [RIGHT_X, BOTTOM_Y], "4": [LEFT_X, BOTTOM_Y]}},
        "5": {"name": "5 ancres : carre + pointe haut", "anchors": {"1": [LEFT_X, TOP_Y], "2": [RIGHT_X, TOP_Y], "3": [RIGHT_X, BOTTOM_Y], "4": [LEFT_X, BOTTOM_Y], "5": [CENTER_X, TOP_MID_Y]}},
        "6": {"name": "6 ancres : carre + haut/bas", "anchors": {"1": [LEFT_X, TOP_Y], "2": [RIGHT_X, TOP_Y], "3": [RIGHT_X, BOTTOM_Y], "4": [LEFT_X, BOTTOM_Y], "5": [CENTER_X, TOP_MID_Y], "6": [CENTER_X, BOTTOM_MID_Y]}},
    }


def _default_layouts(sport: str = DEFAULT_SPORT):
    layout_path = SessionAssets(sport=sport).anchors_layout_path
    try:
        with layout_path.open("r", encoding="utf-8") as handle:
            asset_layouts = json.load(handle).get("layouts", {})
    except (OSError, TypeError, ValueError):
        asset_layouts = {}
    return asset_layouts or _fallback_layouts()


def get_default_settings(sport: str = DEFAULT_SPORT):
    return {
        "settings_version": SETTINGS_VERSION,
        "sport": str(sport or DEFAULT_SPORT),
        "active_anchor_count": 4,
        "alpha_dist": DEFAULT_ALPHA_DIST,
        "alpha_pos": DEFAULT_ALPHA_POS,
        "precision_tolerance_cm": DEFAULT_PRECISION_TOLERANCE_CM,
        "noise_std": DEFAULT_NOISE_STD,
        "spike_prob": DEFAULT_SPIKE_PROB,
        "spike_amplitude": DEFAULT_SPIKE_AMPLITUDE,
        "layouts": _default_layouts(sport),
    }


def _sanitize_settings(raw, sport: str = DEFAULT_SPORT):
    defaults = get_default_settings(sport)
    if not isinstance(raw, dict):
        return defaults
    settings = get_default_settings(sport)
    same_layout_version = int(raw.get("settings_version", 0)) == SETTINGS_VERSION
    settings["active_anchor_count"] = max(2, min(6, int(raw.get("active_anchor_count", defaults["active_anchor_count"]))))
    settings["alpha_dist"] = max(0.01, min(0.95, float(raw.get("alpha_dist", defaults["alpha_dist"]))))
    settings["alpha_pos"] = max(0.01, min(0.95, float(raw.get("alpha_pos", defaults["alpha_pos"]))))
    settings["precision_tolerance_cm"] = max(1.0, min(2000.0, float(raw.get("precision_tolerance_cm", defaults["precision_tolerance_cm"]))))
    settings["noise_std"] = max(0.0, min(1000.0, float(raw.get("noise_std", defaults["noise_std"]))))
    settings["spike_prob"] = max(0.0, min(1.0, float(raw.get("spike_prob", defaults["spike_prob"]))))
    settings["spike_amplitude"] = max(0.0, min(2000.0, float(raw.get("spike_amplitude", defaults["spike_amplitude"]))))
    raw_layouts = raw.get("layouts", {}) if same_layout_version else {}
    for key, default_layout in defaults["layouts"].items():
        layout = raw_layouts.get(key, {}) if isinstance(raw_layouts, dict) else {}
        raw_anchors = layout.get("anchors", {}) if isinstance(layout, dict) else {}
        anchors = {}
        for anchor_id, default_pos in default_layout["anchors"].items():
            values = raw_anchors.get(anchor_id, default_pos) if isinstance(raw_anchors, dict) else default_pos
            anchors[anchor_id] = [float(values[0]), float(values[1])] if isinstance(values, (list, tuple)) and len(values) == 2 else list(default_pos)
        settings["layouts"][key] = {
            "name": str(layout.get("name", default_layout["name"])) if isinstance(layout, dict) else default_layout["name"],
            "anchors": anchors,
        }
    return settings


def load_settings(sport: str = DEFAULT_SPORT):
    cache_file = _cache_file(sport)
    if not os.path.exists(cache_file):
        return get_default_settings(sport)
    try:
        with open(cache_file, "r", encoding="utf-8") as fh:
            return _sanitize_settings(json.load(fh), sport)
    except (OSError, ValueError, TypeError):
        return get_default_settings(sport)


def save_settings(settings, sport: str = DEFAULT_SPORT):
    cache_file = _cache_file(sport)
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump(_sanitize_settings(settings, sport), fh, indent=2)


def get_anchor_layout(settings, count):
    layout = settings["layouts"][str(count)]
    anchors = {int(anchor_id): tuple(pos) for anchor_id, pos in layout["anchors"].items()}
    return anchors, layout["name"]


def set_active_anchor_count(settings, count):
    settings["active_anchor_count"] = max(2, min(6, int(count)))


def reset_stats():
    return {"raw_sum": 0.0, "smooth_sum": 0.0, "raw_precision_sum": 0.0, "smooth_precision_sum": 0.0, "count": 0, "raw_max": 0.0, "smooth_max": 0.0}


def create_state(settings=None, sport: str = DEFAULT_SPORT):
    if settings is None:
        settings = load_settings(sport)
    state = {
        "t": 0.0,
        "settings": settings,
        "stats": reset_stats(),
        "dist_smooth": {},
        "tag_est_smooth": None,
        "player_registry": create_player_registry(CENTER_X, CENTER_Y),
        "ui": {"source_status": "", "hud_visible": True, "display_cache": {}},
    }
    bind_selected_player_analytics(state)
    return state


def full_reset(state, anchor_count):
    state["t"] = 0.0
    state["stats"] = reset_stats()
    state["tag_est_smooth"] = None
    reset_player_registry(state["player_registry"])
    bind_selected_player_analytics(state)
    anchors, _ = get_anchor_layout(state["settings"], anchor_count)
    state["dist_smooth"] = {aid: None for aid in anchors.keys()}


anchor_colors = {1: (80, 180, 255), 2: (120, 255, 120), 3: (255, 170, 60), 4: (180, 120, 255), 5: (255, 110, 150), 6: (90, 220, 220)}
