import pygame

import os

from solver_lps.features.players.domain.player_analytics import average_speed_kmh, build_smoothed_heatmap, ensure_player_analytics_defaults, heat_color, max_speed_kmh, total_distance_m
from solver_lps.features.ground.domain.projection import fit_bounds_to_rect
from ...domain.scene import CAMERA_X, CAMERA_Y, HEIGHT, SCALE, WIDTH, anchor_colors


ANCHOR_CEILING_HEIGHT_CM = 600
HEATMAP_SURFACE_CACHE = {"version": None, "size": None, "surface": None}
CURRENT_LAYOUT = None
COURT_IMAGE_CACHE = {"path": None, "surface": None}
COORD_TRANSFORM = "swap_xy_flip_y"

SIDE_PANEL_W = 360
PADDING = 18
GAP = 16
DROPDOWN_OPTION_HEIGHT = 30


def _layout(screen):
    screen_w, screen_h = screen.get_size()
    side_panel_w = max(360, min(480, int(screen_w * 0.26)))
    padding = max(18, int(screen_h * 0.02))
    gap = max(16, int(screen_w * 0.012))
    court_rect = pygame.Rect(
        padding,
        padding,
        screen_w - side_panel_w - gap - (padding * 2),
        screen_h - (padding * 2),
    )
    side_rect = pygame.Rect(court_rect.right + gap, padding, side_panel_w, screen_h - (padding * 2))
    video_rect = pygame.Rect(side_rect.x + 12, side_rect.y + 42, side_rect.width - 24, max(280, int(side_rect.height * 0.44)))
    playback_button_rect = pygame.Rect(side_rect.x + 12, video_rect.bottom + 8, 34, 34)
    playback_bar_rect = pygame.Rect(playback_button_rect.right + 10, video_rect.bottom + 15, side_rect.width - 68, 18)
    heatmap_button_rect = pygame.Rect(side_rect.x + 12, side_rect.bottom - 44, side_rect.width - 24, 34)
    info_rect = pygame.Rect(
        side_rect.x + 12,
        playback_button_rect.bottom + 12,
        side_rect.width - 24,
        max(120, heatmap_button_rect.y - (playback_button_rect.bottom + 20)),
    )
    dropdown_rect = pygame.Rect(court_rect.x + 14, court_rect.y + 168, 220, 34)
    return {
        "court_rect": court_rect,
        "side_rect": side_rect,
        "video_rect": video_rect,
        "info_rect": info_rect,
        "heatmap_button_rect": heatmap_button_rect,
        "playback_button_rect": playback_button_rect,
        "playback_bar_rect": playback_bar_rect,
        "dropdown_rect": dropdown_rect,
    }


def _layout_or_default():
    global CURRENT_LAYOUT
    if CURRENT_LAYOUT is not None:
        return CURRENT_LAYOUT
    return {
        "court_rect": pygame.Rect(PADDING, PADDING, WIDTH - SIDE_PANEL_W - GAP - (PADDING * 2), HEIGHT - (PADDING * 2)),
        "side_rect": pygame.Rect(WIDTH - SIDE_PANEL_W - PADDING, PADDING, SIDE_PANEL_W, HEIGHT - (PADDING * 2)),
        "video_rect": pygame.Rect(WIDTH - SIDE_PANEL_W - PADDING + 12, PADDING + 42, SIDE_PANEL_W - 24, 420),
        "info_rect": pygame.Rect(WIDTH - SIDE_PANEL_W - PADDING + 12, PADDING + 42 + 420 + 58, SIDE_PANEL_W - 24, 250),
        "heatmap_button_rect": pygame.Rect(WIDTH - SIDE_PANEL_W - PADDING + 12, HEIGHT - PADDING - 44, SIDE_PANEL_W - 24, 34),
        "playback_button_rect": pygame.Rect(WIDTH - SIDE_PANEL_W - PADDING + 12, PADDING + 42 + 420 + 8, 34, 34),
        "playback_bar_rect": pygame.Rect(WIDTH - SIDE_PANEL_W - PADDING + 56, PADDING + 42 + 420 + 15, SIDE_PANEL_W - 68, 18),
        "dropdown_rect": pygame.Rect(32, 188, 220, 34),
    }


def _view_bounds(state):
    view = state.get("view") or {}
    bounds = view.get("bounds")
    if bounds is None:
        bounds = (CAMERA_X - 1500.0, CAMERA_X + 1500.0, CAMERA_Y - 1050.0, CAMERA_Y + 1050.0)
    return bounds


def _court_mapping(state):
    layout = state.get("_layout_cache")
    court_rect = layout["court_rect"]
    inner = pygame.Rect(court_rect.x + 4, court_rect.y + 4, court_rect.width - 8, court_rect.height - 8)
    view = state.get("view") or {}
    terrain_path = view.get("terrain_image_path")
    terrain_surface = _load_court_image_surface(terrain_path)
    if terrain_surface is not None:
        terrain_rect = _fit_rect(terrain_surface.get_size(), inner)
    else:
        terrain_rect = inner
    left, right, top, bottom = _view_bounds(state)
    projection = fit_bounds_to_rect((left, right, top, bottom), terrain_rect)
    return {
        "bounds": (left, right, top, bottom),
        "scale": projection.scale,
        "offset_x": projection.offset_x,
        "offset_y": projection.offset_y,
        "projection": projection,
        "terrain_rect": terrain_rect,
        "terrain_surface": terrain_surface,
        "positions_rect": terrain_rect,
        "heatmap_rect": terrain_rect,
        "anchors_rect": terrain_rect,
    }


def world_to_screen(state, x_cm, y_cm):
    screen_xy, _, _, _ = _point_to_screen(state, x_cm, y_cm)
    return screen_xy


def draw_text(screen, text, x, y, font, color=(255, 255, 255)):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))


def heatmap_button_rect(screen=None):
    if screen is not None:
        return _layout(screen)["heatmap_button_rect"].copy()
    return _layout_or_default()["heatmap_button_rect"].copy()


def player_dropdown_rect(screen=None):
    if screen is not None:
        return _layout(screen)["dropdown_rect"].copy()
    return _layout_or_default()["dropdown_rect"].copy()


def playback_button_rect(screen=None):
    if screen is not None:
        return _layout(screen)["playback_button_rect"].copy()
    return _layout_or_default()["playback_button_rect"].copy()


def playback_bar_rect(screen=None):
    if screen is not None:
        return _layout(screen)["playback_bar_rect"].copy()
    return _layout_or_default()["playback_bar_rect"].copy()


def playback_ratio(solution):
    playback = solution.get("uwb_playback") or solution.get("cv_playback")
    if not playback:
        return None
    duration_s = float(playback.get("duration_s", 0.0) or 0.0)
    if duration_s <= 0.0:
        return 0.0
    position_s = float(playback.get("position_s", 0.0) or 0.0)
    return max(0.0, min(1.0, position_s / duration_s))


def player_dropdown_options(solution):
    options = [("Tous", None)]
    for player_id in solution.get("all_player_ids", []):
        if all(option_value != player_id for _, option_value in options):
            options.append((player_id, player_id))
    return options


def player_dropdown_option_rects(solution):
    rects = []
    base = player_dropdown_rect()
    for index, option in enumerate(player_dropdown_options(solution)):
        rect = pygame.Rect(base.x, base.bottom + 4 + index * DROPDOWN_OPTION_HEIGHT, base.width, DROPDOWN_OPTION_HEIGHT)
        rects.append((option, rect))
    return rects


def _fmt(value, suffix=" cm"):
    if value is None:
        return "--"
    return f"{value:.1f}{suffix}"


def _smooth_display_value(state, key, value, alpha=0.14):
    ui = state.setdefault("ui", {})
    cache = ui.setdefault("display_cache", {})
    if value is None:
        cache[key] = None
        return None
    old = cache.get(key)
    smoothed = value if old is None else old + (value - old) * alpha
    cache[key] = smoothed
    return smoothed


def _video_surface_from_frame(frame_rgb):
    if frame_rgb is None:
        return None
    height, width = frame_rgb.shape[:2]
    surface = pygame.image.frombuffer(frame_rgb.tobytes(), (width, height), "RGB")
    return surface.convert()


def _extract_camera_column(frame_surface):
    frame_w, frame_h = frame_surface.get_size()
    camera_w = max(1, frame_w - frame_h)
    return frame_surface.subsurface((0, 0, camera_w, frame_h)).copy()


def _fit_rect(source_size, target_rect):
    source_w, source_h = source_size
    if source_w <= 0 or source_h <= 0:
        return target_rect.copy()
    scale = min(target_rect.width / source_w, target_rect.height / source_h)
    draw_w = max(1, int(source_w * scale))
    draw_h = max(1, int(source_h * scale))
    return pygame.Rect(
        target_rect.x + (target_rect.width - draw_w) // 2,
        target_rect.y + (target_rect.height - draw_h) // 2,
        draw_w,
        draw_h,
    )


def _load_court_image_surface(image_path):
    if not image_path:
        return None
    if not os.path.exists(image_path):
        return None
    if COURT_IMAGE_CACHE["path"] == image_path and COURT_IMAGE_CACHE["surface"] is not None:
        return COURT_IMAGE_CACHE["surface"]
    try:
        surface = pygame.image.load(image_path).convert()
    except (pygame.error, OSError):
        try:
            import cv2
            import numpy as np

            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("cv2.imread returned None")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            surface = pygame.image.frombuffer(np.ascontiguousarray(image_rgb).tobytes(), (width, height), "RGB").convert()
        except Exception:
            COURT_IMAGE_CACHE["path"] = image_path
            COURT_IMAGE_CACHE["surface"] = None
            return None
    COURT_IMAGE_CACHE["path"] = image_path
    COURT_IMAGE_CACHE["surface"] = surface
    return surface


def _terrain_debug_text(state, mapping):
    view = state.get("view") or {}
    terrain_path = view.get("terrain_image_path")
    bg_label = "terrain-de-basket.jpg introuvable" if terrain_path and mapping.get("terrain_surface") is None else os.path.basename(terrain_path or "")
    rect = mapping.get("terrain_rect") or pygame.Rect(0, 0, 0, 0)
    rect_text = f"terrain_rect=({rect.x},{rect.y},{rect.width},{rect.height})"
    transform = view.get("coord_transform", "identity")
    return bg_label, rect_text, transform


def _normalized_point(bounds, x_cm, y_cm):
    left, right, top, bottom = bounds
    width = max(float(right) - float(left), 1.0)
    height = max(float(bottom) - float(top), 1.0)
    u = (float(x_cm) - float(left)) / width
    v = (float(y_cm) - float(top)) / height
    return u, v


def _apply_normalized_transform(u, v, transform):
    if transform == "flip_x":
        return 1.0 - u, v
    if transform == "flip_y":
        return u, 1.0 - v
    if transform == "flip_x_flip_y":
        return 1.0 - u, 1.0 - v
    if transform == "swap_xy":
        return v, u
    if transform == "swap_xy_flip_x":
        return 1.0 - v, u
    if transform == "swap_xy_flip_y":
        return v, 1.0 - u
    if transform == "swap_xy_flip_x_flip_y":
        return 1.0 - v, 1.0 - u
    return u, v


def _screen_from_normalized(rect, u, v):
    draw_w = max(rect.width - 1, 0)
    draw_h = max(rect.height - 1, 0)
    px = rect.x + int(round(u * draw_w))
    py = rect.y + int(round(v * draw_h))
    return px, py


def _point_to_screen(state, x_cm, y_cm, transform=None):
    mapping = _court_mapping(state)
    rect = mapping["terrain_rect"]
    bounds = mapping["bounds"]
    transform = transform or state.get("view", {}).get("coord_transform", COORD_TRANSFORM)
    norm_before = _normalized_point(bounds, x_cm, y_cm)
    norm_after = _apply_normalized_transform(norm_before[0], norm_before[1], transform)
    screen_xy = _screen_from_normalized(rect, norm_after[0], norm_after[1])
    out_of_rect = not rect.collidepoint(screen_xy)
    return screen_xy, norm_before, norm_after, out_of_rect


def draw_grid(screen, state):
    screen.fill((16, 20, 24))
    mapping = _court_mapping(state)
    view = state.get("view") or {}
    terrain_rect = mapping["terrain_rect"]

    terrain_surface = _load_court_image_surface(view.get("terrain_image_path"))
    if view.get("terrain_image_path") and terrain_surface is None:
        warning = pygame.Surface(terrain_rect.size, pygame.SRCALPHA)
        warning.fill((60, 18, 18))
        screen.blit(warning, terrain_rect.topleft)
        err_font = pygame.font.SysFont(None, max(24, terrain_rect.height // 18))
        label = err_font.render("terrain-de-basket.jpg introuvable", True, (255, 170, 170))
        screen.blit(label, (terrain_rect.x + 20, terrain_rect.y + 20))
    elif terrain_surface is not None:
        scaled = pygame.transform.smoothscale(terrain_surface, terrain_rect.size)
        screen.blit(scaled, terrain_rect.topleft)
    else:
        warning = pygame.Surface(terrain_rect.size, pygame.SRCALPHA)
        warning.fill((60, 18, 18))
        screen.blit(warning, terrain_rect.topleft)
        err_font = pygame.font.SysFont(None, max(24, terrain_rect.height // 18))
        label = err_font.render("terrain-de-basket.jpg introuvable", True, (255, 170, 170))
        screen.blit(label, (terrain_rect.x + 20, terrain_rect.y + 20))


def draw_heatmap(screen, analytics):
    ensure_player_analytics_defaults(analytics)
    if analytics.get("heatmap_snapshot_version") is None:
        return
    state = analytics["render_state"]
    rect = state.get("_layout_cache", {}).get("terrain_rect")
    if rect is None:
        return
    if rect.width <= 0 or rect.height <= 0:
        return
    transform = state.get("view", {}).get("coord_transform", COORD_TRANSFORM)
    cache_key = (analytics["heatmap_snapshot_version"], rect.size, transform)
    if HEATMAP_SURFACE_CACHE["version"] != cache_key:
        grid = build_smoothed_heatmap(analytics, radius=3)
        max_value = max((max(row) for row in grid), default=0.0)
        heat_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        heat_surface.fill((0, 0, 0, 0))
        cols = max(1, int(analytics["heat_cols"]))
        rows = max(1, int(analytics["heat_rows"]))
        cell_w = max(1, int(round(rect.width / cols)))
        cell_h = max(1, int(round(rect.height / rows)))
        for row_index, row in enumerate(grid):
            for col_index, value in enumerate(row):
                if value <= 0:
                    continue
                u = (col_index + 0.5) / cols
                v = (row_index + 0.5) / rows
                u2, v2 = _apply_normalized_transform(u, v, transform)
                px = int(round(u2 * max(rect.width - 1, 0)))
                py = int(round(v2 * max(rect.height - 1, 0)))
                color = heat_color(value, max_value)
                cell_rect = pygame.Rect(px - cell_w // 2, py - cell_h // 2, cell_w, cell_h)
                pygame.draw.rect(heat_surface, color, cell_rect)
        HEATMAP_SURFACE_CACHE["surface"] = heat_surface
        HEATMAP_SURFACE_CACHE["version"] = cache_key
    screen.blit(HEATMAP_SURFACE_CACHE["surface"], rect)


def draw_camera_panel(screen, small_font, solution):
    layout = _layout(screen)
    side_rect = layout["side_rect"]
    video_rect = layout["video_rect"]
    pygame.draw.rect(screen, (8, 10, 14), side_rect, border_radius=10)
    pygame.draw.rect(screen, (90, 160, 215), side_rect, 2, border_radius=10)
    draw_text(screen, "Cameras", side_rect.x + 12, side_rect.y + 10, small_font, (235, 245, 255))

    frame_surface = _video_surface_from_frame(solution.get("cv_video_frame"))
    if frame_surface is None:
        draw_text(screen, "Aucune video CV disponible", video_rect.x + 10, video_rect.y + 10, small_font, (190, 200, 215))
        return

    camera_surface = _extract_camera_column(frame_surface)
    fitted = _fit_rect(camera_surface.get_size(), video_rect)
    scaled = pygame.transform.smoothscale(camera_surface, fitted.size)
    screen.blit(scaled, fitted)

    playback = solution.get("uwb_playback") or solution.get("cv_playback")
    if not playback:
        return

    button_rect = layout["playback_button_rect"]
    bar_rect = layout["playback_bar_rect"]
    paused = bool(playback.get("paused"))
    ratio = playback_ratio(solution) or 0.0

    pygame.draw.rect(screen, (18, 24, 32), button_rect, border_radius=8)
    pygame.draw.rect(screen, (100, 180, 255), button_rect, 2, border_radius=8)
    icon_color = (235, 245, 255)
    if paused:
        pygame.draw.polygon(
            screen,
            icon_color,
            [
                (button_rect.x + 12, button_rect.y + 8),
                (button_rect.x + 12, button_rect.bottom - 8),
                (button_rect.right - 10, button_rect.centery),
            ],
        )
    else:
        pygame.draw.rect(screen, icon_color, (button_rect.x + 10, button_rect.y + 8, 5, button_rect.height - 16), border_radius=2)
        pygame.draw.rect(screen, icon_color, (button_rect.right - 15, button_rect.y + 8, 5, button_rect.height - 16), border_radius=2)

    pygame.draw.rect(screen, (18, 24, 32), bar_rect, border_radius=9)
    pygame.draw.rect(screen, (60, 90, 120), bar_rect, 1, border_radius=9)
    fill_rect = pygame.Rect(bar_rect.x + 1, bar_rect.y + 1, max(0, int((bar_rect.width - 2) * ratio)), bar_rect.height - 2)
    if fill_rect.width > 0:
        pygame.draw.rect(screen, (100, 180, 255), fill_rect, border_radius=8)
    knob_x = bar_rect.x + int(ratio * bar_rect.width)
    pygame.draw.circle(screen, (235, 245, 255), (knob_x, bar_rect.centery), 7)
    pygame.draw.circle(screen, (100, 180, 255), (knob_x, bar_rect.centery), 7, 2)


def draw_info_panel(screen, font, small_font, active_ids, state, solution, layout_name, active_anchor_count, tag_real):
    info_rect = state["_layout_cache"]["info_rect"]
    pygame.draw.rect(screen, (12, 16, 22), info_rect, border_radius=8)
    pygame.draw.rect(screen, (60, 90, 120), info_rect, 1, border_radius=8)

    y = info_rect.y + 10
    draw_text(screen, "Vision / Solver", info_rect.x + 10, y, small_font, (255, 210, 120))
    y += 28

    playback = solution.get("uwb_playback") or solution.get("cv_playback")
    if playback:
        state_label = "pause" if playback.get("paused") else "lecture"
        frame_index = int(playback.get("frame_index", 0)) + 1
        frame_count = int(playback.get("frame_count", 0))
        position_s = float(playback.get("position_s", 0.0) or 0.0)
        duration_s = float(playback.get("duration_s", 0.0) or 0.0)
        draw_text(screen, f"Review {state_label} | frame {frame_index}/{frame_count}", info_rect.x + 10, y, small_font, (255, 230, 120))
        y += 24
        draw_text(screen, f"t={position_s:.1f}s / {duration_s:.1f}s", info_rect.x + 10, y, small_font, (200, 210, 225))
        y += 24
        draw_text(screen, "Space ou clic | barre souris | <- -> | PgUp/PgDn", info_rect.x + 10, y, small_font, (160, 175, 195))
        y += 28

    if solution.get("cv_status"):
        draw_text(screen, f"Vision : {solution['cv_status']}", info_rect.x + 10, y, small_font, (180, 235, 255))
        y += 24
        draw_text(screen, f"Joueurs visibles : {len(solution.get('cv_positions', []))}", info_rect.x + 10, y, small_font, (180, 235, 255))
        y += 24
        expected_count = state.get("ui", {}).get("expected_player_count")
        if expected_count is not None:
            draw_text(screen, f"Joueurs attendus : {int(expected_count)}  (- / +)", info_rect.x + 10, y, small_font, (170, 225, 255))
            y += 24
        if solution.get("selected_player_id"):
            draw_text(screen, f"Joueur suivi : {solution['selected_player_id']}", info_rect.x + 10, y, small_font, (255, 230, 120))
            y += 24
            if not solution.get("selected_player_visible", False):
                draw_text(screen, "Temporairement hors champ", info_rect.x + 10, y, small_font, (255, 180, 120))
                y += 24

    for aid in active_ids:
        raw = _smooth_display_value(state, f"a{aid}_raw", solution["dist_raw"].get(aid))
        smooth = _smooth_display_value(state, f"a{aid}_smooth", solution["dist_smooth"].get(aid))
        draw_text(screen, f"A{aid} brut={_fmt(raw)} | lisse={_fmt(smooth)}", info_rect.x + 10, y, small_font, anchor_colors[aid])
        y += 22

    if active_anchor_count >= 3 and tag_real is not None:
        y += 8
        draw_text(screen, f"Erreur brute : {_fmt(_smooth_display_value(state, 'raw_err', solution['raw_err']))}", info_rect.x + 10, y, small_font, (190, 190, 190))
        y += 22
        draw_text(screen, f"Erreur lissee : {_fmt(_smooth_display_value(state, 'smooth_err', solution['smooth_err']))}", info_rect.x + 10, y, small_font)


def draw_heatmap_button(screen, font):
    rect = _layout(screen)["heatmap_button_rect"]
    pygame.draw.rect(screen, (20, 24, 30), rect, border_radius=8)
    pygame.draw.rect(screen, (255, 210, 80), rect, 2, border_radius=8)
    draw_text(screen, "Generer heatmap", rect.x + 16, rect.y + 7, font, (255, 240, 190))


def draw_player_dropdown(screen, font, solution, ui_state):
    rect = _layout(screen)["dropdown_rect"]
    selection_mode = solution.get("selection_mode", "single")
    selected_player_id = solution.get("selected_player_id")
    label = "Tous" if selection_mode == "all" or not selected_player_id else selected_player_id
    pygame.draw.rect(screen, (20, 24, 30), rect, border_radius=8)
    pygame.draw.rect(screen, (120, 200, 255), rect, 2, border_radius=8)
    draw_text(screen, f"Affichage : {label}", rect.x + 12, rect.y + 6, font, (220, 245, 255))
    draw_text(screen, "v", rect.right - 18, rect.y + 6, font, (220, 245, 255))
    if not ui_state.get("player_dropdown_open"):
        return
    for (option_label, _), option_rect in player_dropdown_option_rects(solution):
        pygame.draw.rect(screen, (14, 18, 24), option_rect, border_radius=6)
        pygame.draw.rect(screen, (90, 160, 215), option_rect, 1, border_radius=6)
        draw_text(screen, option_label, option_rect.x + 10, option_rect.y + 4, font, (235, 245, 255))


def draw_player_card(screen, fonts, analytics):
    ensure_player_analytics_defaults(analytics)
    if not analytics["card_visible"]:
        return
    font, small_font = fonts
    card = pygame.Surface((340, 248), pygame.SRCALPHA)
    card.fill((8, 10, 14, 215))
    court_rect = _layout(screen)["court_rect"]
    screen.blit(card, (court_rect.right - 360, 18))
    x = court_rect.right - 340
    y = 34
    draw_text(screen, analytics["name"], x, y, font, (255, 255, 255))
    y += 34
    draw_text(screen, f"Base: {analytics.get('source_label', 'Estimation UWB')}", x, y, small_font, (255, 210, 120))
    y += 30
    draw_text(screen, f"Vitesse moyenne: {average_speed_kmh(analytics):.2f} km/h", x, y, small_font)
    y += 26
    draw_text(screen, f"Vitesse max: {max_speed_kmh(analytics):.2f} km/h", x, y, small_font)
    y += 26
    draw_text(screen, f"Distance totale: {total_distance_m(analytics):.1f} m", x, y, small_font)
    y += 26
    draw_text(screen, "Nombre de sauts: non observable en 2D", x, y, small_font)
    y += 26
    draw_text(screen, "Hauteur max saut: non observable", x, y, small_font)
    y += 26
    draw_text(screen, f"Echantillons: {analytics['samples']}", x, y, small_font)


def draw_scene(screen, fonts, active_anchors, active_ids, state, solution, tag_real, layout_name, active_anchor_count):
    global CURRENT_LAYOUT
    font, small_font = fonts
    state["player_analytics"]["render_state"] = state
    CURRENT_LAYOUT = _layout(screen)
    state["_layout_cache"] = CURRENT_LAYOUT
    draw_grid(screen, state)
    draw_heatmap(screen, state["player_analytics"])

    screen_w, screen_h = screen.get_size()
    overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
    mapping = _court_mapping(state)
    terrain_rect = mapping["terrain_rect"]
    transform = state.get("view", {}).get("coord_transform", COORD_TRANSFORM)
    for aid in active_ids:
        if aid not in solution["dist_raw"]:
            continue
        ax, ay = active_anchors[aid]
        (sx, sy), _, _, _ = _point_to_screen(state, ax, ay, transform)
        radius = max(1, int(solution["dist_raw"][aid] * mapping["scale"]))
        c = anchor_colors[aid]
        pygame.draw.circle(overlay, (c[0], c[1], c[2], 40), (sx, sy), radius)
    screen.blit(overlay, (0, 0))

    for aid, (ax, ay) in active_anchors.items():
        (sx, sy), _, _, _ = _point_to_screen(state, ax, ay, transform)
        c = anchor_colors[aid]
        halo = pygame.Surface((52, 52), pygame.SRCALPHA)
        pygame.draw.circle(halo, (c[0], c[1], c[2], 60), (26, 26), 24)
        screen.blit(halo, (sx - 26, sy - 26))
        pygame.draw.circle(screen, (245, 245, 245), (sx, sy), 14)
        pygame.draw.circle(screen, c, (sx, sy), 10)
        pygame.draw.circle(screen, (20, 24, 28), (sx, sy), 4)
        draw_text(screen, f"Portenta plafond {aid}", sx + 18, sy - 18, small_font, c)
        draw_text(screen, f"h={ANCHOR_CEILING_HEIGHT_CM/100:.1f} m", sx + 18, sy + 2, small_font, (235, 235, 235))

    if tag_real is not None:
        (tx, ty), _, _, _ = _point_to_screen(state, *tag_real, transform)
        pygame.draw.circle(screen, (255, 0, 0), (tx, ty), 11)
        draw_text(screen, "Tag reel", tx + 16, ty - 10, small_font, (255, 0, 0))

    selected_player_id = solution.get("selected_player_id")
    for player in solution.get("cv_positions", []):
        if selected_player_id and player["player_id"] != selected_player_id:
            continue
        before_x, before_y = player["x"], player["y"]
        (px, py), norm_before, norm_after, out_of_rect = _point_to_screen(state, before_x, before_y, transform)
        color = (255, 230, 120) if player.get("selected") else (80, 220, 255)
        pygame.draw.circle(screen, color, (px, py), 9 if player.get("selected") else 7)
        draw_text(screen, player["player_id"], px + 12, py - 14, small_font, color)
        if player["player_id"] == "T004":
            raw_x = player.get("raw_x", before_x)
            raw_y = player.get("raw_y", before_y)
            draw_text(screen, f"T004 raw=({raw_x:.1f},{raw_y:.1f})", px + 12, py + 10, small_font, (255, 220, 120))
            draw_text(screen, f"T004 norm_before=({norm_before[0]:.3f},{norm_before[1]:.3f})", px + 12, py + 34, small_font, (200, 210, 225))
            draw_text(screen, f"T004 norm_after=({norm_after[0]:.3f},{norm_after[1]:.3f})", px + 12, py + 58, small_font, (180, 255, 180))
            draw_text(screen, f"T004 screen=({px},{py}) OUT_OF_RECT={str(out_of_rect).lower()}", px + 12, py + 82, small_font, (255, 180, 120))

    for i, p in enumerate(solution["raw_intersections"]):
        (px, py), _, _, _ = _point_to_screen(state, *p, transform)
        pygame.draw.circle(screen, (255, 0, 255), (px, py), 7)
        draw_text(screen, f"Inter brute {i + 1}", px + 10, py - 10, small_font, (255, 120, 255))
    for i, p in enumerate(solution["smooth_intersections"]):
        (px, py), _, _, _ = _point_to_screen(state, *p, transform)
        pygame.draw.circle(screen, (255, 255, 255), (px, py), 7)
        draw_text(screen, f"Inter lissee {i + 1}", px + 10, py - 10, small_font, (255, 255, 255))

    if solution["tag_est_raw"] is not None:
        (ex, ey), _, _, _ = _point_to_screen(state, *solution["tag_est_raw"], transform)
        pygame.draw.circle(screen, (180, 180, 180), (ex, ey), 10)
        draw_text(screen, "Estimation brute", ex + 14, ey - 10, small_font, (180, 180, 180))
    if solution["tag_est_smooth"] is not None:
        (ex, ey), _, _, _ = _point_to_screen(state, *solution["tag_est_smooth"], transform)
        pygame.draw.circle(screen, (255, 210, 0), (ex, ey), 12)
        draw_text(screen, "Estimation lissee", ex + 14, ey - 10, small_font, (255, 210, 0))

    draw_text(screen, "UWB Estimation 2D", 30, 25, font)
    draw_text(screen, f"Ancres actives : {active_anchor_count} | {layout_name}", 30, 59, small_font, (220, 220, 220))
    draw_text(screen, "2..6 changer | Tab joueur | H hide | R reset | ESC quitter", 30, 85, small_font, (220, 220, 220))
    source_color = (120, 255, 210) if solution["source"] == "realtime" else (220, 220, 220)
    draw_text(screen, f"Source : {solution['status']}", 30, 111, small_font, source_color)
    bg_label, rect_text, debug_transform = _terrain_debug_text(state, mapping)
    draw_text(screen, f"BG={bg_label}", 30, 137, small_font, (255, 220, 120))
    draw_text(screen, rect_text, 30, 161, small_font, (200, 210, 225))
    draw_text(screen, f"positions_rect=({terrain_rect.x},{terrain_rect.y},{terrain_rect.width},{terrain_rect.height})", 30, 185, small_font, (200, 210, 225))
    draw_text(screen, f"heatmap_rect=({terrain_rect.x},{terrain_rect.y},{terrain_rect.width},{terrain_rect.height})", 30, 209, small_font, (200, 210, 225))
    draw_text(screen, f"anchors_rect=({terrain_rect.x},{terrain_rect.y},{terrain_rect.width},{terrain_rect.height})", 30, 233, small_font, (200, 210, 225))
    draw_text(screen, f"transform={debug_transform}", 30, 257, small_font, (200, 210, 225))
    draw_text(screen, "draw_order=terrain>heatmap>positions>anchors", 30, 281, small_font, (200, 210, 225))

    draw_player_dropdown(screen, small_font, solution, state.get("ui", {}))
    draw_camera_panel(screen, small_font, solution)
    draw_info_panel(screen, font, small_font, active_ids, state, solution, layout_name, active_anchor_count, tag_real)
    draw_heatmap_button(screen, small_font)
    draw_player_card(screen, fonts, state["player_analytics"])

