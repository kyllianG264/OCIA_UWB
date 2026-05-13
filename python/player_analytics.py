from court_geometry import COURT_LENGTH_CM, COURT_WIDTH_CM, court_bounds


HEATMAP_COLS = 112
HEATMAP_ROWS = 60
JUMP_COUNT_THRESHOLD_CM = 12.0
PLAYER_SELECT_RADIUS_PX = 22


def create_player_analytics(center_x, center_y):
    left, right, top, bottom = court_bounds(center_x, center_y)
    return {
        "name": "Joueur 1",
        "source_label": "Estimation UWB",
        "card_visible": False,
        "center_x": center_x,
        "center_y": center_y,
        "bounds": (left, right, top, bottom),
        "heat_cols": HEATMAP_COLS,
        "heat_rows": HEATMAP_ROWS,
        "heatmap": [[0 for _ in range(HEATMAP_COLS)] for _ in range(HEATMAP_ROWS)],
        "heat_version": 0,
        "heatmap_snapshot": None,
        "heatmap_snapshot_version": None,
        "smoothed_heatmap_cache": None,
        "smoothed_heatmap_cache_key": None,
        "samples": 0,
        "time_s": 0.0,
        "distance_cm": 0.0,
        "speed_sum_cm_s": 0.0,
        "max_speed_cm_s": 0.0,
        "jump_count": 0,
        "jump_active": False,
        "max_jump_cm": 0.0,
        "last_pos": None,
        "last_height_cm": None,
        "recent_positions": [],
    }


def reset_player_analytics(analytics):
    visible = analytics.get("card_visible", False)
    center_x = analytics["center_x"]
    center_y = analytics["center_y"]
    fresh = create_player_analytics(center_x, center_y)
    fresh["card_visible"] = visible
    analytics.clear()
    analytics.update(fresh)


def ensure_player_analytics_defaults(analytics):
    center_x = analytics.get("center_x", 1250.0)
    center_y = analytics.get("center_y", 900.0)
    fresh = create_player_analytics(center_x, center_y)
    for key, value in fresh.items():
        analytics.setdefault(key, value)
    return analytics


def update_player_analytics(analytics, t, pos_xy, height_cm, jump_extra_cm, dt):
    ensure_player_analytics_defaults(analytics)
    if pos_xy is None:
        return

    analytics["samples"] += 1
    analytics["time_s"] = max(float(analytics.get("time_s") or 0.0), float(t or 0.0))
    if jump_extra_cm is not None:
        analytics["max_jump_cm"] = max(analytics["max_jump_cm"], jump_extra_cm)
    analytics["recent_positions"].append((pos_xy[0], pos_xy[1], height_cm))
    _accumulate_heat(analytics, pos_xy[0], pos_xy[1])

    last_pos = analytics["last_pos"]
    if last_pos is not None and dt > 0.0:
        dx = pos_xy[0] - last_pos[0]
        dy = pos_xy[1] - last_pos[1]
        distance_cm = (dx * dx + dy * dy) ** 0.5
        speed_cm_s = distance_cm / dt
        analytics["distance_cm"] += distance_cm
        analytics["speed_sum_cm_s"] += speed_cm_s
        analytics["max_speed_cm_s"] = max(analytics["max_speed_cm_s"], speed_cm_s)

    if jump_extra_cm is None:
        analytics["jump_active"] = False
    else:
        airborne = jump_extra_cm >= JUMP_COUNT_THRESHOLD_CM
        if airborne and not analytics["jump_active"]:
            analytics["jump_count"] += 1
        analytics["jump_active"] = airborne
    analytics["last_pos"] = pos_xy
    analytics["last_height_cm"] = height_cm


def average_speed_cm_s(analytics):
    ensure_player_analytics_defaults(analytics)
    if analytics["time_s"] <= 0.0:
        return 0.0
    return analytics["distance_cm"] / analytics["time_s"]


def average_speed_kmh(analytics):
    return average_speed_cm_s(analytics) * 0.036


def max_speed_kmh(analytics):
    ensure_player_analytics_defaults(analytics)
    return analytics["max_speed_cm_s"] * 0.036


def total_distance_m(analytics):
    ensure_player_analytics_defaults(analytics)
    return analytics["distance_cm"] / 100.0


def max_heat_value(analytics):
    ensure_player_analytics_defaults(analytics)
    return max((max(row) for row in analytics["heatmap"]), default=0)


def build_smoothed_heatmap(analytics, radius=3):
    ensure_player_analytics_defaults(analytics)
    source = analytics.get("heatmap_snapshot") or analytics["heatmap"]
    source_version = analytics.get("heatmap_snapshot_version")
    if source_version is None:
        source_version = analytics["heat_version"]
    cache_key = (source_version, radius)
    if analytics.get("smoothed_heatmap_cache_key") == cache_key and analytics.get("smoothed_heatmap_cache") is not None:
        return analytics["smoothed_heatmap_cache"]
    rows = analytics["heat_rows"]
    cols = analytics["heat_cols"]
    smoothed = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            total = 0.0
            weight_sum = 0.0
            for dy in range(-radius, radius + 1):
                sy = row + dy
                if sy < 0 or sy >= rows:
                    continue
                for dx in range(-radius, radius + 1):
                    sx = col + dx
                    if sx < 0 or sx >= cols:
                        continue
                    distance2 = dx * dx + dy * dy
                    weight = 1.0 / (1.0 + distance2)
                    total += source[sy][sx] * weight
                    weight_sum += weight
            smoothed[row][col] = total / max(weight_sum, 1e-6)
    analytics["smoothed_heatmap_cache"] = smoothed
    analytics["smoothed_heatmap_cache_key"] = cache_key
    return smoothed


def capture_heatmap_snapshot(analytics):
    ensure_player_analytics_defaults(analytics)
    analytics["heatmap_snapshot"] = [row[:] for row in analytics["heatmap"]]
    analytics["heatmap_snapshot_version"] = analytics["heat_version"]
    analytics["smoothed_heatmap_cache"] = None
    analytics["smoothed_heatmap_cache_key"] = None


def heat_color(value, max_value):
    ratio = 0.0 if max_value <= 0 else max(0.0, min(1.0, value / max_value))
    if ratio <= 0.5:
        local = ratio / 0.5
        r = int(40 + 215 * local)
        g = int(170 + 55 * (1.0 - local))
        b = 30
    else:
        local = (ratio - 0.5) / 0.5
        r = 255
        g = int(170 * (1.0 - local))
        b = 30 * (1.0 - local)
    alpha = 56 + int(110 * ratio)
    return r, g, b, alpha


def toggle_player_card(analytics):
    ensure_player_analytics_defaults(analytics)
    analytics["card_visible"] = not analytics["card_visible"]


def player_hit_test(screen_pos, player_screen_pos, radius_px=PLAYER_SELECT_RADIUS_PX):
    if player_screen_pos is None:
        return False
    dx = screen_pos[0] - player_screen_pos[0]
    dy = screen_pos[1] - player_screen_pos[1]
    return dx * dx + dy * dy <= radius_px * radius_px


def _accumulate_heat(analytics, x_cm, y_cm):
    left, right, top, bottom = analytics["bounds"]
    if x_cm < left or x_cm > right or y_cm < top or y_cm > bottom:
        return
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    col = min(analytics["heat_cols"] - 1, max(0, int((x_cm - left) / width * analytics["heat_cols"])))
    row = min(analytics["heat_rows"] - 1, max(0, int((y_cm - top) / height * analytics["heat_rows"])))
    analytics["heatmap"][row][col] += 1
    analytics["heat_version"] += 1
