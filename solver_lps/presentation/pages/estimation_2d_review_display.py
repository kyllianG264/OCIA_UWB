import pygame

import os
import math

from solver_lps.features.players.domain.player_analytics import average_speed_kmh, build_smoothed_heatmap, ensure_player_analytics_defaults, heat_color, max_speed_kmh, total_distance_m
from solver_lps.features.ground.domain.court_geometry import court_bounds
from solver_lps.features.ground.domain.projection import fit_bounds_to_rect, project_court_cm_to_terrain_pixels
from solver_lps.presentation.pages.estimation_2d_review_scene import CAMERA_X, CAMERA_Y, HEIGHT, SCALE, WIDTH, anchor_colors


ANCHOR_CEILING_HEIGHT_CM = 600
HEATMAP_SURFACE_CACHE = {"version": None, "size": None, "surface": None}
GRAY_HEATMAP_SURFACE_CACHE = {"version": None, "size": None, "surface": None}
CURRENT_LAYOUT = None
COURT_IMAGE_CACHE = {"path": None, "surface": None}
COORD_TRANSFORM = "flip_x_flip_y"
CAMERA_PANEL_FIT_LOG_CACHE = {}

SIDE_PANEL_W = 360
PADDING = 18
GAP = 16
DROPDOWN_OPTION_HEIGHT = 30
LEFT_INFO_W = 300
PLAYER_STATS_H = 240
CV_VIEW_MODE_BUTTON_W = 92
CV_VIEW_MODE_BUTTON_H = 30


def _layout(screen, *, show_cv=True, show_uwb=True):
    screen_w, screen_h = screen.get_size()
    side_panel_w = max(460, min(620, int(screen_w * 0.32))) if show_cv else 0
    show_left_panel = show_cv or show_uwb
    left_panel_w = max(260, min(340, int(screen_w * 0.18))) if show_left_panel else 0
    padding = max(18, int(screen_h * 0.02))
    gap = max(16, int(screen_w * 0.012))
    left_gap = gap if show_left_panel else 0
    right_gap = gap if show_cv else 0
    court_rect = pygame.Rect(
        padding + left_panel_w + left_gap,
        padding,
        screen_w - left_panel_w - side_panel_w - left_gap - right_gap - (padding * 2),
        screen_h - (padding * 2),
    )
    left_info_rect = pygame.Rect(padding, padding, left_panel_w, screen_h - (padding * 2))
    side_rect = pygame.Rect(court_rect.right + right_gap, padding, side_panel_w, screen_h - (padding * 2))
    info_h = 0
    if show_uwb:
        info_h = max(240, min(420, int(screen_h * 0.44)))
    info_rect = pygame.Rect(left_info_rect.x, left_info_rect.y, left_info_rect.width, info_h)
    heatmap_button_rect = pygame.Rect(
        left_info_rect.x + 12,
        left_info_rect.y + 118,
        max(0, left_info_rect.width - 24),
        34,
    )
    stats_y = left_info_rect.y if not show_uwb else max(info_rect.bottom + 12, heatmap_button_rect.bottom + 12)
    stats_h = left_info_rect.bottom - stats_y
    player_stats_rect = pygame.Rect(left_info_rect.x, stats_y, left_info_rect.width, max(0, stats_h))
    camera_top = side_rect.y + 42
    camera_bottom = side_rect.bottom - 92
    camera_rect = pygame.Rect(
        side_rect.x + 8,
        camera_top,
        side_rect.width - 16,
        max(0, camera_bottom - camera_top),
    )
    half_video_gap = 0
    video_h = min(
        max(1, int(camera_rect.width * 9 / 16)),
        max(1, (camera_rect.height - half_video_gap) // 2),
    ) if show_cv else 0
    video_stack_h = (video_h * 2) + half_video_gap
    video_y = camera_rect.y + max(0, (camera_rect.height - video_stack_h) // 2)
    top_video_rect = pygame.Rect(camera_rect.x, video_y, camera_rect.width, video_h)
    bottom_video_rect = pygame.Rect(camera_rect.x, top_video_rect.bottom + half_video_gap, camera_rect.width, video_h)
    playback_button_rect = pygame.Rect(camera_rect.x, bottom_video_rect.bottom + 10, 34, 34)
    playback_bar_rect = pygame.Rect(playback_button_rect.right + 10, playback_button_rect.y + 7, camera_rect.width - 44, 18)
    dropdown_rect = pygame.Rect(court_rect.centerx - 110, court_rect.y + 14, 220, 34)
    cv_view_mode_rect = pygame.Rect(
        side_rect.right - ((CV_VIEW_MODE_BUTTON_W * 2) + 18),
        side_rect.y + 8,
        (CV_VIEW_MODE_BUTTON_W * 2) + 10,
        CV_VIEW_MODE_BUTTON_H,
    )
    return {
        "court_rect": court_rect,
        "left_info_rect": left_info_rect,
        "side_rect": side_rect,
        "camera_rect": camera_rect,
        "top_video_rect": top_video_rect,
        "bottom_video_rect": bottom_video_rect,
        "player_stats_rect": player_stats_rect,
        "info_rect": info_rect,
        "heatmap_button_rect": heatmap_button_rect,
        "playback_button_rect": playback_button_rect,
        "playback_bar_rect": playback_bar_rect,
        "dropdown_rect": dropdown_rect,
        "cv_view_mode_rect": cv_view_mode_rect,
    }


def _layout_or_default():
    global CURRENT_LAYOUT
    if CURRENT_LAYOUT is not None:
        return CURRENT_LAYOUT
    default_side_panel_w = 460
    default_camera_x = WIDTH - default_side_panel_w - PADDING + 8
    default_camera_y = PADDING + 42
    default_camera_w = default_side_panel_w - 16
    default_video_h = int(default_camera_w * 9 / 16)
    default_bottom_y = default_camera_y + default_video_h
    default_heatmap_rect = pygame.Rect(PADDING + 12, PADDING + 118, LEFT_INFO_W - 24, 34)
    return {
        "court_rect": pygame.Rect(PADDING + LEFT_INFO_W + GAP, PADDING, WIDTH - LEFT_INFO_W - default_side_panel_w - (GAP * 2) - (PADDING * 2), HEIGHT - (PADDING * 2)),
        "left_info_rect": pygame.Rect(PADDING, PADDING, LEFT_INFO_W, HEIGHT - (PADDING * 2)),
        "side_rect": pygame.Rect(WIDTH - default_side_panel_w - PADDING, PADDING, default_side_panel_w, HEIGHT - (PADDING * 2)),
        "camera_rect": pygame.Rect(default_camera_x, PADDING + 42, default_camera_w, 470),
        "top_video_rect": pygame.Rect(default_camera_x, default_camera_y, default_camera_w, default_video_h),
        "bottom_video_rect": pygame.Rect(default_camera_x, default_bottom_y, default_camera_w, default_video_h),
        "player_stats_rect": pygame.Rect(PADDING, default_heatmap_rect.bottom + 12, LEFT_INFO_W, HEIGHT - PADDING - default_heatmap_rect.bottom - 12),
        "info_rect": pygame.Rect(PADDING, PADDING, LEFT_INFO_W, 308),
        "heatmap_button_rect": default_heatmap_rect,
        "playback_button_rect": pygame.Rect(default_camera_x, default_bottom_y + default_video_h + 10, 34, 34),
        "playback_bar_rect": pygame.Rect(default_camera_x + 44, default_bottom_y + default_video_h + 17, default_camera_w - 44, 18),
        "dropdown_rect": pygame.Rect(PADDING + LEFT_INFO_W + GAP + 180, PADDING + 14, 220, 34),
        "cv_view_mode_rect": pygame.Rect(
            WIDTH - default_side_panel_w - PADDING + default_side_panel_w - 24 - ((CV_VIEW_MODE_BUTTON_W * 2) + 18),
            PADDING + 8,
            (CV_VIEW_MODE_BUTTON_W * 2) + 10,
            CV_VIEW_MODE_BUTTON_H,
        ),
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
    if CURRENT_LAYOUT is not None:
        return CURRENT_LAYOUT["heatmap_button_rect"].copy()
    if screen is not None:
        return _layout(screen)["heatmap_button_rect"].copy()
    return _layout_or_default()["heatmap_button_rect"].copy()


def player_dropdown_rect(screen=None):
    if CURRENT_LAYOUT is not None:
        return CURRENT_LAYOUT["dropdown_rect"].copy()
    if screen is not None:
        return _layout(screen)["dropdown_rect"].copy()
    return _layout_or_default()["dropdown_rect"].copy()


def playback_button_rect(screen=None):
    if CURRENT_LAYOUT is not None:
        return CURRENT_LAYOUT["playback_button_rect"].copy()
    if screen is not None:
        return _layout(screen)["playback_button_rect"].copy()
    return _layout_or_default()["playback_button_rect"].copy()


def playback_bar_rect(screen=None):
    if CURRENT_LAYOUT is not None:
        return CURRENT_LAYOUT["playback_bar_rect"].copy()
    if screen is not None:
        return _layout(screen)["playback_bar_rect"].copy()
    return _layout_or_default()["playback_bar_rect"].copy()


def cv_view_mode_button_rects(screen=None):
    if CURRENT_LAYOUT is not None:
        base = CURRENT_LAYOUT["cv_view_mode_rect"].copy()
    elif screen is not None:
        base = _layout(screen)["cv_view_mode_rect"].copy()
    else:
        base = _layout_or_default()["cv_view_mode_rect"].copy()
    raw_rect = pygame.Rect(base.x, base.y, CV_VIEW_MODE_BUTTON_W, base.height)
    merged_rect = pygame.Rect(raw_rect.right + 10, base.y, CV_VIEW_MODE_BUTTON_W, base.height)
    return {"raw": raw_rect, "merged": merged_rect}


def playback_ratio(solution):
    playback = solution.get("uwb_playback") or solution.get("cv_playback")
    if not playback:
        return None
    duration_s = float(playback.get("duration_s", 0.0) or 0.0)
    if duration_s <= 0.0:
        return 0.0
    position_s = float(playback.get("position_s", 0.0) or 0.0)
    return max(0.0, min(1.0, position_s / duration_s))


def _has_cv_mode(solution):
    return solution.get("review_mode") in {"cv", "both"}


def _has_uwb_mode(solution):
    return solution.get("review_mode") in {"uwb", "both"}


def _has_player_mode(solution):
    return solution.get("review_mode") in {"cv", "uwb", "both"}


def _has_uwb_diagnostics(solution):
    return _has_uwb_mode(solution) and not solution.get("uwb_merged_review", False)


def _uwb_point_for_view(solution, state, point):
    if str(solution.get("sport", "")).strip().lower() != "basket":
        return point
    bounds = (state.get("view") or {}).get("bounds")
    if bounds is None:
        return point
    return project_court_cm_to_terrain_pixels(
        point[0],
        point[1],
        court_bounds(1250.0, 900.0),
        bounds,
    )


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
    surface = pygame.image.frombuffer(frame_rgb.data, (width, height), "RGB")
    return surface.convert()


def _extract_camera_column(frame_surface):
    frame_w, frame_h = frame_surface.get_size()
    camera_w = max(1, frame_w - frame_h)
    return frame_surface.subsurface((0, 0, camera_w, frame_h)).copy()


def _split_camera_stack(camera_surface):
    cam_w, cam_h = camera_surface.get_size()
    half_h = max(1, cam_h // 2)
    top_half = camera_surface.subsurface((0, 0, cam_w, half_h)).copy()
    bottom_half = camera_surface.subsurface((0, cam_h - half_h, cam_w, half_h)).copy()
    mirrored_top = pygame.transform.flip(top_half, True, True)
    return bottom_half, mirrored_top


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


def render_frame_contain(screen, frame_surface, panel_rect, camera_id="camera"):
    frame_w, frame_h = frame_surface.get_size()
    panel_w, panel_h = panel_rect.size
    pygame.draw.rect(screen, (0, 0, 0), panel_rect)
    if frame_w <= 0 or frame_h <= 0 or panel_w <= 0 or panel_h <= 0:
        return pygame.Rect(panel_rect.x, panel_rect.y, 0, 0)

    scale = min(panel_w / frame_w, panel_h / frame_h)
    draw_w = max(1, int(frame_w * scale))
    draw_h = max(1, int(frame_h * scale))
    draw_x = panel_rect.x + int((panel_w - draw_w) / 2)
    draw_y = panel_rect.y + int((panel_h - draw_h) / 2)
    draw_rect = pygame.Rect(draw_x, draw_y, draw_w, draw_h)

    log_key = (camera_id, frame_w, frame_h, panel_w, panel_h, draw_w, draw_h, draw_x - panel_rect.x, draw_y - panel_rect.y)
    if CAMERA_PANEL_FIT_LOG_CACHE.get(camera_id) != log_key:
        CAMERA_PANEL_FIT_LOG_CACHE[camera_id] = log_key
        print(
            f"[CAMERA PANEL FIT] camera_id={camera_id} "
            f"frame={frame_w}x{frame_h} panel={panel_w}x{panel_h} "
            f"draw={draw_w}x{draw_h} offset=({draw_x - panel_rect.x},{draw_y - panel_rect.y})",
            flush=True,
        )

    scaled = pygame.transform.smoothscale(frame_surface, (draw_w, draw_h))
    clip_rect = screen.get_clip()
    screen.set_clip(panel_rect)
    screen.blit(scaled, draw_rect)
    screen.set_clip(clip_rect)
    return draw_rect


def _draw_camera_feed(screen, frame_surface, target_rect, camera_id, missing_label, small_font):
    if frame_surface is None:
        pygame.draw.rect(screen, (0, 0, 0), target_rect)
        draw_text(screen, missing_label, target_rect.x + 10, target_rect.y + 10, small_font, (190, 200, 215))
        return
    render_frame_contain(screen, frame_surface, target_rect, camera_id)


def _double_mirror_surface(surface):
    if surface is None:
        return None
    return pygame.transform.flip(surface, True, True)


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
    bg_label = "terrain introuvable" if terrain_path and mapping.get("terrain_surface") is None else os.path.basename(terrain_path or "")
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
    view = state.get("view") or {}
    transform = transform or view.get("coord_transform", COORD_TRANSFORM)
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
        label = err_font.render("terrain introuvable", True, (255, 170, 170))
        screen.blit(label, (terrain_rect.x + 20, terrain_rect.y + 20))
    elif terrain_surface is not None:
        scaled = pygame.transform.smoothscale(terrain_surface, terrain_rect.size)
        screen.blit(scaled, terrain_rect.topleft)
    else:
        warning = pygame.Surface(terrain_rect.size, pygame.SRCALPHA)
        warning.fill((60, 18, 18))
        screen.blit(warning, terrain_rect.topleft)
        err_font = pygame.font.SysFont(None, max(24, terrain_rect.height // 18))
        label = err_font.render("terrain introuvable", True, (255, 170, 170))
        screen.blit(label, (terrain_rect.x + 20, terrain_rect.y + 20))


def draw_midline(screen, state):
    view = state.get("view") or {}
    midline_points = view.get("midline_points") or []
    if len(midline_points) != 2:
        return
    start = world_to_screen(state, midline_points[0][0], midline_points[0][1])
    end = world_to_screen(state, midline_points[1][0], midline_points[1][1])
    pygame.draw.line(screen, (240, 240, 240), start, end, 2)


def draw_heatmap(screen, analytics):
    ensure_player_analytics_defaults(analytics)
    if analytics.get("heatmap_snapshot_version") is None:
        return
    state = analytics["render_state"]
    rect = _court_mapping(state)["terrain_rect"]
    if rect.width <= 0 or rect.height <= 0:
        return
    view = state.get("view") or {}
    transform = view.get("coord_transform", COORD_TRANSFORM)
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


def _draw_gray_heatmap_overlay(screen, state, gray_heatmap):
    event_points = list(gray_heatmap.get("event_points") or [])
    show_disappearances = state.get("ui", {}).get("show_gray_disappearances", True)
    disappearance_points = list(gray_heatmap.get("disappearance_points") or []) if show_disappearances else []
    rejected_event_points = list(gray_heatmap.get("rejected_event_points") or []) if show_disappearances else []
    if event_points or disappearance_points or rejected_event_points:
        terrain_rect = _court_mapping(state)["terrain_rect"]
        if terrain_rect.width <= 0 or terrain_rect.height <= 0:
            return False
        cache_key = (
            terrain_rect.size,
            len(event_points),
            len(disappearance_points),
            len(rejected_event_points),
            sum(int(point.get("frame", 0)) for point in event_points),
            sum(int(point.get("frame", 0)) for point in disappearance_points),
            sum(int(point.get("frame", 0)) for point in rejected_event_points),
            show_disappearances,
        )
        if GRAY_HEATMAP_SURFACE_CACHE["version"] != cache_key:
            surface = pygame.Surface(terrain_rect.size, pygame.SRCALPHA)
            glow = pygame.Surface(terrain_rect.size, pygame.SRCALPHA)
            terrain_center = terrain_rect.center

            def local_scene_point(point):
                scene_point = point.get("scene_point")
                if not scene_point:
                    return None
                screen_point = world_to_screen(state, scene_point[0], scene_point[1])
                return screen_point[0] - terrain_rect.x, screen_point[1] - terrain_rect.y

            def draw_frontier_lines(points):
                grouped_points = {}
                for point in points:
                    local_point = local_scene_point(point)
                    if local_point is None:
                        continue
                    grouped_points.setdefault(point.get("frontier_band") or "unknown", []).append(local_point)

                def clamp_local(point):
                    return (
                        max(0, min(terrain_rect.width - 1, int(round(point[0])))),
                        max(0, min(terrain_rect.height - 1, int(round(point[1])))),
                    )

                def extend_curve(local_points):
                    ordered_by_x = sorted(local_points, key=lambda item: item[0])
                    if len(ordered_by_x) < 4:
                        return ordered_by_x
                    extension_step = 18
                    extension_count = 4

                    def side_points(anchor_index, neighbor_index, direction):
                        anchor = ordered_by_x[anchor_index]
                        neighbor = ordered_by_x[neighbor_index]
                        dx = anchor[0] - neighbor[0]
                        dy = anchor[1] - neighbor[1]
                        norm = max(1.0, math.hypot(dx, dy))
                        ux = (dx / norm) * direction
                        uy = (dy / norm) * direction
                        return [
                            clamp_local(
                                (
                                    anchor[0] + ux * extension_step * index,
                                    anchor[1] + uy * extension_step * index,
                                )
                            )
                            for index in range(extension_count, 0, -1)
                        ]

                    left_extra = side_points(0, min(3, len(ordered_by_x) - 1), 1.0)
                    right_extra = list(reversed(side_points(-1, max(-4, -len(ordered_by_x)), 1.0)))
                    return left_extra + ordered_by_x + right_extra

                top_points = sorted(grouped_points.get("top") or [], key=lambda item: item[0])
                bottom_points = sorted(grouped_points.get("bottom") or [], key=lambda item: item[0])
                if len(top_points) >= 2 and len(bottom_points) >= 2:
                    top_points = extend_curve(top_points)
                    bottom_points = extend_curve(bottom_points)
                    grouped_points["top"] = top_points
                    grouped_points["bottom"] = bottom_points
                    polygon = top_points + list(reversed(bottom_points))
                    pygame.draw.polygon(glow, (70, 235, 110, 34), polygon)
                    pygame.draw.polygon(surface, (70, 235, 110, 58), polygon)

            def draw_points(points, color, radius, alpha, outline_alpha):
                for point in points:
                    local_point = local_scene_point(point)
                    if local_point is None:
                        continue
                    local_x, local_y = local_point
                    pygame.draw.circle(glow, (*color, max(22, alpha // 4)), (local_x, local_y), radius + 8)
                    pygame.draw.circle(surface, (*color, alpha), (local_x, local_y), radius)
                    pygame.draw.circle(surface, (245, 245, 245, outline_alpha), (local_x, local_y), radius + 1, 1)

            draw_frontier_lines(event_points)
            if show_disappearances:
                draw_points(rejected_event_points, (255, 130, 55), 3, 90, 75)
                draw_points(disappearance_points, (255, 185, 55), 4, 155, 120)
            merged = pygame.Surface(terrain_rect.size, pygame.SRCALPHA)
            merged.blit(glow, (0, 0))
            merged.blit(surface, (0, 0))
            GRAY_HEATMAP_SURFACE_CACHE["surface"] = merged
            GRAY_HEATMAP_SURFACE_CACHE["version"] = cache_key
        screen.blit(GRAY_HEATMAP_SURFACE_CACHE["surface"], terrain_rect.topleft)
        return True

    cells = list(gray_heatmap.get("cells") or [])
    if not cells:
        return False
    terrain_rect = _court_mapping(state)["terrain_rect"]
    if terrain_rect.width <= 0 or terrain_rect.height <= 0:
        return False
    cache_key = (
        terrain_rect.size,
        len(cells),
        round(sum(float(cell.get("gray_score", 0.0)) for cell in cells), 6),
        gray_heatmap.get("layer_color"),
    )
    if GRAY_HEATMAP_SURFACE_CACHE["version"] != cache_key:
        mask_w = max(72, terrain_rect.width // 5)
        mask_h = max(120, terrain_rect.height // 5)
        heat_surface = pygame.Surface((mask_w, mask_h), pygame.SRCALPHA)
        max_score = max((float(cell.get("gray_score", 0.0)) for cell in cells), default=0.0)
        for cell in cells:
            center = cell.get("center")
            if not center:
                continue
            screen_center = world_to_screen(state, center[0], center[1])
            local_x = screen_center[0] - terrain_rect.x
            local_y = screen_center[1] - terrain_rect.y
            px = int(round((local_x / max(terrain_rect.width, 1)) * mask_w))
            py = int(round((local_y / max(terrain_rect.height, 1)) * mask_h))
            radius_x = max(
                2,
                int(round(abs(world_to_screen(state, center[0] + (cell.get("width", 0.0) / 2.0), center[1])[0] - screen_center[0]) / max(terrain_rect.width, 1) * mask_w)),
            )
            radius_y = max(
                2,
                int(round(abs(world_to_screen(state, center[0], center[1] + (cell.get("height", 0.0) / 2.0))[1] - screen_center[1]) / max(terrain_rect.height, 1) * mask_h)),
            )
            radius = max(3, int(max(radius_x, radius_y) * 1.5))
            score = float(cell.get("gray_score", 0.0))
            if gray_heatmap.get("layer_color") == "green":
                intensity = 0.0 if max_score <= 0.0 else min(1.0, score / max_score)
                color = (40, 110 + int(120 * intensity), 80, 65 + int(105 * intensity))
            else:
                color = heat_color(score, max_score)
            pygame.draw.circle(heat_surface, color, (px, py), radius)
        GRAY_HEATMAP_SURFACE_CACHE["surface"] = pygame.transform.smoothscale(heat_surface, terrain_rect.size)
        GRAY_HEATMAP_SURFACE_CACHE["version"] = cache_key
    screen.blit(GRAY_HEATMAP_SURFACE_CACHE["surface"], terrain_rect.topleft)
    return True


def _draw_handoff_ellipse(screen, state, ellipse):
    if not ellipse or not ellipse.get("center"):
        return
    center_x, center_y = ellipse["center"]
    radius_x = max(1.0, float(ellipse.get("radius_x", 0.0)))
    radius_y = max(1.0, float(ellipse.get("radius_y", 0.0)))
    center = world_to_screen(state, center_x, center_y)
    right = world_to_screen(state, center_x + radius_x, center_y)
    bottom = world_to_screen(state, center_x, center_y + radius_y)
    screen_radius_x = max(4, abs(right[0] - center[0]))
    screen_radius_y = max(4, abs(bottom[1] - center[1]))
    local = pygame.Surface((screen_radius_x * 2 + 8, screen_radius_y * 2 + 8), pygame.SRCALPHA)
    local_rect = local.get_rect().inflate(-6, -6)
    pygame.draw.ellipse(local, (45, 220, 105, 48), local_rect)
    pygame.draw.ellipse(local, (80, 255, 135, 210), local_rect, 3)
    rotated = pygame.transform.rotate(local, -float(ellipse.get("angle_degrees", 0.0)))
    screen.blit(rotated, rotated.get_rect(center=center))


def draw_gray_zone_overlay(screen, state):
    if not state.get("ui", {}).get("show_gray_zone", True):
        return
    view = state.get("view") or {}
    gray_heatmap = (
        view.get("gray_heatmap") or {}
        if state.get("ui", {}).get("show_gray_observations", True)
        else {}
    )
    cells = (
        list(view.get("gray_zone_cells") or [])
        if state.get("ui", {}).get("show_gray_theory", True)
        else []
    )
    if not cells:
        _draw_gray_heatmap_overlay(screen, state, gray_heatmap)
        return
    terrain_rect = _court_mapping(state)["terrain_rect"]
    if terrain_rect.width <= 0 or terrain_rect.height <= 0:
        return
    mask_w = max(48, terrain_rect.width // 7)
    mask_h = max(96, terrain_rect.height // 7)
    mask_surface = pygame.Surface((mask_w, mask_h), pygame.SRCALPHA)
    fill_surface = pygame.Surface((mask_w, mask_h), pygame.SRCALPHA)
    for cell in cells:
        center = cell.get("center")
        if not center:
            continue
        screen_center = world_to_screen(state, center[0], center[1])
        local_x = screen_center[0] - terrain_rect.x
        local_y = screen_center[1] - terrain_rect.y
        mask_x = int(round((local_x / max(terrain_rect.width, 1)) * mask_w))
        mask_y = int(round((local_y / max(terrain_rect.height, 1)) * mask_h))
        screen_width = max(
            18,
            int(abs(world_to_screen(state, center[0] + (cell.get("width", 0.0) / 2.0), center[1])[0] - screen_center[0]) * 2.8),
        )
        screen_height = max(
            18,
            int(abs(world_to_screen(state, center[0], center[1] + (cell.get("height", 0.0) / 2.0))[1] - screen_center[1]) * 2.8),
        )
        radius = max(
            3,
            int(
                max(
                    (screen_width / max(terrain_rect.width, 1)) * mask_w,
                    (screen_height / max(terrain_rect.height, 1)) * mask_h,
                )
                / 2.0
            ),
        )
        pygame.draw.circle(mask_surface, (160, 166, 176, 46), (mask_x, mask_y), radius + 2)
        pygame.draw.circle(fill_surface, (136, 142, 152, 110), (mask_x, mask_y), max(2, radius - 1))

    soft_glow = pygame.transform.smoothscale(mask_surface, terrain_rect.size)
    soft_fill = pygame.transform.smoothscale(fill_surface, terrain_rect.size)
    screen.blit(soft_glow, terrain_rect.topleft)
    screen.blit(soft_fill, terrain_rect.topleft)
    _draw_gray_heatmap_overlay(screen, state, gray_heatmap)


def _draw_view_mode_buttons(screen, small_font, solution, prefix):
    active_view_mode = solution.get(f"{prefix}_view_mode", "raw")
    raw_available = bool(solution.get(f"{prefix}_raw_available", False))
    merged_available = bool(solution.get(f"{prefix}_merged_available", False))
    for mode, rect in cv_view_mode_button_rects(screen).items():
        is_available = raw_available if mode == "raw" else merged_available
        is_active = mode == active_view_mode
        if not is_available:
            fill_color = (14, 18, 24)
            border_color = (52, 60, 72)
            text_color = (120, 128, 140)
        else:
            fill_color = (100, 180, 255) if is_active else (18, 24, 32)
            border_color = (130, 210, 255) if is_active else (60, 90, 120)
            text_color = (10, 16, 22) if is_active else (210, 225, 240)
        pygame.draw.rect(screen, fill_color, rect, border_radius=8)
        pygame.draw.rect(screen, border_color, rect, 2, border_radius=8)
        label = ("Raw" if raw_available else "Raw off") if mode == "raw" else ("Merged" if merged_available else "Merged off")
        draw_text(screen, label, rect.x + 22, rect.y + 4, small_font, text_color)


def draw_camera_panel(screen, small_font, solution):
    if not _has_cv_mode(solution):
        return
    layout = CURRENT_LAYOUT if CURRENT_LAYOUT is not None else _layout(screen)
    side_rect = layout["side_rect"]
    top_video_rect = layout["top_video_rect"]
    bottom_video_rect = layout["bottom_video_rect"]
    pygame.draw.rect(screen, (8, 10, 14), side_rect, border_radius=10)
    pygame.draw.rect(screen, (90, 160, 215), side_rect, 2, border_radius=10)
    draw_text(screen, "Cameras", side_rect.x + 12, side_rect.y + 10, small_font, (235, 245, 255))
    _draw_view_mode_buttons(screen, small_font, solution, "cv")

    top_frame_surface = _video_surface_from_frame(solution.get("cv_left_video_frame"))
    bottom_frame_surface = _video_surface_from_frame(solution.get("cv_right_video_frame"))
    sport = str(solution.get("sport", "") or "").strip().lower()
    if top_frame_surface is not None or bottom_frame_surface is not None:
        if sport == "basket":
            _draw_camera_feed(screen, bottom_frame_surface, top_video_rect, "basket_right_on_top", "Video droite indisponible", small_font)
            _draw_camera_feed(screen, _double_mirror_surface(top_frame_surface), bottom_video_rect, "basket_left_on_bottom_double_mirror", "Video gauche indisponible", small_font)
        else:
            _draw_camera_feed(screen, bottom_frame_surface, bottom_video_rect, "right_on_bottom", "Video droite indisponible", small_font)
            _draw_camera_feed(screen, pygame.transform.flip(top_frame_surface, True, False) if top_frame_surface is not None else None, top_video_rect, "left_on_top_mirror_x", "Video gauche indisponible", small_font)
    else:
        frame_surface = _video_surface_from_frame(solution.get("cv_video_frame"))
        if frame_surface is None:
            draw_text(screen, "Aucune video CV disponible", top_video_rect.x + 10, top_video_rect.y + 10, small_font, (190, 200, 215))
            return
        camera_surface = _extract_camera_column(frame_surface)
        top_feed, bottom_feed = _split_camera_stack(camera_surface)
        _draw_camera_feed(screen, top_feed, top_video_rect, "legacy_left", "Video gauche indisponible", small_font)
        _draw_camera_feed(screen, bottom_feed, bottom_video_rect, "legacy_right", "Video droite indisponible", small_font)
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
    if not _has_uwb_mode(solution):
        return
    info_rect = state["_layout_cache"]["info_rect"]
    pygame.draw.rect(screen, (12, 16, 22), info_rect, border_radius=8)
    pygame.draw.rect(screen, (60, 90, 120), info_rect, 1, border_radius=8)
    if solution.get("review_mode") == "uwb":
        _draw_view_mode_buttons(screen, small_font, solution, "uwb")

    y = info_rect.y + 10
    mode_label = solution.get("review_mode", "both").upper()
    draw_text(screen, f"Vision / Solver ({mode_label})", info_rect.x + 10, y, small_font, (255, 210, 120))
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
        draw_text(screen, "Space/clic | barre souris | <- -> | PgUp/PgDn | M raw/merged", info_rect.x + 10, y, small_font, (160, 175, 195))
        y += 28
        gray_heatmap = (state.get("view") or {}).get("gray_heatmap")
        label = "Heatmap zone grise" if gray_heatmap else "Zone grise"
        draw_text(screen, f"{label} : {'on' if state.get('ui', {}).get('show_gray_zone', True) else 'off'} (G)", info_rect.x + 10, y, small_font, (185, 195, 210))
        y += 24

    if solution.get("uwb_merged_review", False):
        draw_text(screen, "Source positions : merged UWB via players", info_rect.x + 10, y, small_font, (180, 235, 255))
        return

    draw_text(screen, f"Ancres actives : {active_anchor_count}", info_rect.x + 10, y, small_font, (220, 220, 220))
    y += 24
    draw_text(screen, layout_name, info_rect.x + 10, y, small_font, (170, 200, 230))
    y += 28
    for aid in active_ids:
        raw = _smooth_display_value(state, f"a{aid}_raw", solution["dist_raw"].get(aid))
        smooth = _smooth_display_value(state, f"a{aid}_smooth", solution["dist_smooth"].get(aid))
        draw_text(screen, f"A{aid} brut={_fmt(raw)}", info_rect.x + 10, y, small_font, anchor_colors[aid])
        y += 20
        draw_text(screen, f"A{aid} lisse={_fmt(smooth)}", info_rect.x + 10, y, small_font, (225, 232, 240))
        y += 24

    if _has_uwb_mode(solution) and active_anchor_count >= 3 and tag_real is not None:
        y += 8
        draw_text(screen, f"Erreur brute : {_fmt(_smooth_display_value(state, 'raw_err', solution['raw_err']))}", info_rect.x + 10, y, small_font, (190, 190, 190))
        y += 22
        draw_text(screen, f"Erreur lissee : {_fmt(_smooth_display_value(state, 'smooth_err', solution['smooth_err']))}", info_rect.x + 10, y, small_font)


def draw_heatmap_button(screen, font):
    rect = heatmap_button_rect(screen)
    if rect.width <= 0 or rect.height <= 0:
        return
    pygame.draw.rect(screen, (20, 24, 30), rect, border_radius=8)
    pygame.draw.rect(screen, (255, 210, 80), rect, 2, border_radius=8)
    draw_text(screen, "Generer heatmap", rect.x + 16, rect.y + 7, font, (255, 240, 190))


def draw_player_dropdown(screen, font, solution, ui_state):
    if not _has_player_mode(solution):
        return
    rect = player_dropdown_rect(screen)
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
    font, small_font = fonts
    rect = CURRENT_LAYOUT["player_stats_rect"] if CURRENT_LAYOUT is not None else _layout(screen)["player_stats_rect"]
    if rect.width <= 0 or rect.height <= 0:
        return
    pygame.draw.rect(screen, (12, 16, 22), rect, border_radius=10)
    pygame.draw.rect(screen, (255, 210, 80), rect, 1, border_radius=10)
    x = rect.x + 14
    y = rect.y + 12
    draw_text(screen, "Joueur selectionne", x, y, small_font, (255, 210, 120))
    y += 28
    if not analytics["card_visible"]:
        draw_text(screen, "Aucun joueur selectionne", x, y, small_font, (180, 190, 205))
        return
    draw_text(screen, analytics["name"], x, y, font, (255, 255, 255))
    y += 34
    draw_text(screen, f"Base: {analytics.get('source_label', 'Estimation UWB')}", x, y, small_font, (255, 210, 120))
    y += 28
    draw_text(screen, f"Vitesse moyenne: {average_speed_kmh(analytics):.2f} km/h", x, y, small_font)
    y += 24
    draw_text(screen, f"Vitesse max: {max_speed_kmh(analytics):.2f} km/h", x, y, small_font)
    y += 24
    draw_text(screen, f"Distance totale: {total_distance_m(analytics):.1f} m", x, y, small_font)
    y += 24
    draw_text(screen, f"Echantillons: {analytics['samples']}", x, y, small_font)


def draw_scene(screen, fonts, active_anchors, active_ids, state, solution, tag_real, layout_name, active_anchor_count):
    global CURRENT_LAYOUT
    font, small_font = fonts
    state["player_analytics"]["render_state"] = state
    CURRENT_LAYOUT = _layout(
        screen,
        show_cv=_has_cv_mode(solution),
        show_uwb=_has_uwb_mode(solution),
    )
    state["_layout_cache"] = CURRENT_LAYOUT
    draw_grid(screen, state)
    draw_midline(screen, state)
    draw_gray_zone_overlay(screen, state)
    draw_heatmap(screen, state["player_analytics"])

    mapping = _court_mapping(state)
    terrain_rect = mapping["terrain_rect"]
    view = state.get("view") or {}
    transform = view.get("coord_transform", COORD_TRANSFORM)
    if _has_uwb_diagnostics(solution):
        screen_w, screen_h = screen.get_size()
        overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        for aid in active_ids:
            if aid not in solution["dist_raw"]:
                continue
            ax, ay = _uwb_point_for_view(solution, state, active_anchors[aid])
            (sx, sy), _, _, _ = _point_to_screen(state, ax, ay, transform)
            terrain_height = max(float(mapping["bounds"][3]) - float(mapping["bounds"][2]), 1.0)
            radius = max(1, int(solution["dist_raw"][aid] * (terrain_height / 2800.0) * mapping["scale"]))
            c = anchor_colors[aid]
            pygame.draw.circle(overlay, (c[0], c[1], c[2], 40), (sx, sy), radius)
        screen.blit(overlay, (0, 0))

        for aid, (ax, ay) in active_anchors.items():
            ax, ay = _uwb_point_for_view(solution, state, (ax, ay))
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
            display_tag = _uwb_point_for_view(solution, state, tag_real)
            (tx, ty), _, _, _ = _point_to_screen(state, *display_tag, transform)
            pygame.draw.circle(screen, (255, 0, 0), (tx, ty), 11)
            draw_text(screen, "Tag reel", tx + 16, ty - 10, small_font, (255, 0, 0))

    selected_player_id = solution.get("selected_player_id")
    for player in solution.get("cv_positions", []):
        if selected_player_id and player["player_id"] != selected_player_id:
            continue
        before_x, before_y = player["x"], player["y"]
        (px, py), _, _, _ = _point_to_screen(state, before_x, before_y, transform)
        color = (255, 230, 120) if player.get("selected") else (80, 220, 255)
        pygame.draw.circle(screen, color, (px, py), 9 if player.get("selected") else 7)
        draw_text(screen, player["player_id"], px + 12, py - 14, small_font, color)

    if _has_uwb_diagnostics(solution):
        for i, p in enumerate(solution["raw_intersections"]):
            display_point = _uwb_point_for_view(solution, state, p)
            (px, py), _, _, _ = _point_to_screen(state, *display_point, transform)
            pygame.draw.circle(screen, (255, 0, 255), (px, py), 7)
            draw_text(screen, f"Inter brute {i + 1}", px + 10, py - 10, small_font, (255, 120, 255))
        for i, p in enumerate(solution["smooth_intersections"]):
            display_point = _uwb_point_for_view(solution, state, p)
            (px, py), _, _, _ = _point_to_screen(state, *display_point, transform)
            pygame.draw.circle(screen, (255, 255, 255), (px, py), 7)
            draw_text(screen, f"Inter lissee {i + 1}", px + 10, py - 10, small_font, (255, 255, 255))

        if solution["tag_est_raw"] is not None:
            display_point = _uwb_point_for_view(solution, state, solution["tag_est_raw"])
            (ex, ey), _, _, _ = _point_to_screen(state, *display_point, transform)
            pygame.draw.circle(screen, (180, 180, 180), (ex, ey), 10)
            draw_text(screen, "Estimation brute", ex + 14, ey - 10, small_font, (180, 180, 180))
        if solution["tag_est_smooth"] is not None:
            display_point = _uwb_point_for_view(solution, state, solution["tag_est_smooth"])
            (ex, ey), _, _, _ = _point_to_screen(state, *display_point, transform)
            pygame.draw.circle(screen, (255, 210, 0), (ex, ey), 12)
            draw_text(screen, "Estimation lissee", ex + 14, ey - 10, small_font, (255, 210, 0))

    title = {
        "cv": "CV Review 2D",
        "uwb": "UWB Review 2D",
        "both": "CV + UWB Review 2D",
    }.get(solution.get("review_mode", "both"), "Review 2D")
    title_x = terrain_rect.x + 16
    title_y = terrain_rect.y + 16
    draw_text(screen, title, title_x, title_y, font)
    source_color = (120, 255, 210) if solution["source"] == "realtime" else (220, 220, 220)
    draw_text(screen, f"Source : {solution['status']}", title_x, title_y + 34, small_font, source_color)
    if _has_player_mode(solution):
        expected_count = state.get("ui", {}).get("expected_player_count")
        cv_caption = f"Joueurs visibles : {len(solution.get('cv_positions', []))}"
        if expected_count is not None:
            cv_caption += f" | attendus : {int(expected_count)}"
        draw_text(screen, cv_caption, title_x, title_y + 58, small_font, (180, 235, 255))

    draw_player_dropdown(screen, small_font, solution, state.get("ui", {}))
    draw_camera_panel(screen, small_font, solution)
    draw_info_panel(screen, font, small_font, active_ids, state, solution, layout_name, active_anchor_count, tag_real)
    if _has_player_mode(solution):
        draw_player_card(screen, fonts, state["player_analytics"])
        draw_heatmap_button(screen, small_font)

