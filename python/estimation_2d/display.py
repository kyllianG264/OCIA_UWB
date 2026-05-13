import pygame

from player_analytics import average_speed_kmh, build_smoothed_heatmap, ensure_player_analytics_defaults, heat_color, max_speed_kmh, total_distance_m
from court_geometry import draw_court_2d
from scene import CAMERA_X, CAMERA_Y, HEIGHT, SCALE, WIDTH, anchor_colors


ANCHOR_CEILING_HEIGHT_CM = 600
HEATMAP_SURFACE_CACHE = {"version": None, "size": None, "surface": None}
HEATMAP_BUTTON_RECT = pygame.Rect(WIDTH - 240, HEIGHT - 64, 210, 40)


def world_to_screen(x_cm, y_cm):
    sx = WIDTH // 2 + int((x_cm - CAMERA_X) * SCALE)
    sy = HEIGHT // 2 + int((y_cm - CAMERA_Y) * SCALE)
    return sx, sy


def draw_text(screen, text, x, y, font, color=(255, 255, 255)):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))


def heatmap_button_rect():
    return HEATMAP_BUTTON_RECT.copy()


def draw_grid(screen):
    draw_court_2d(screen, world_to_screen, CAMERA_X, CAMERA_Y)


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


def draw_heatmap(screen, analytics):
    ensure_player_analytics_defaults(analytics)
    if analytics.get("heatmap_snapshot_version") is None:
        return
    left, right, top, bottom = analytics["bounds"]
    sx0, sy0 = world_to_screen(left, top)
    sx1, sy1 = world_to_screen(right, bottom)
    rect = pygame.Rect(min(sx0, sx1), min(sy0, sy1), abs(sx1 - sx0), abs(sy1 - sy0))
    if rect.width <= 0 or rect.height <= 0:
        return
    cache_key = (analytics["heatmap_snapshot_version"], rect.size)
    if HEATMAP_SURFACE_CACHE["version"] != cache_key:
        grid = build_smoothed_heatmap(analytics, radius=3)
        max_value = max((max(row) for row in grid), default=0.0)
        heat_surface = pygame.Surface((analytics["heat_cols"], analytics["heat_rows"]), pygame.SRCALPHA)
        for row_index, row in enumerate(grid):
            for col_index, value in enumerate(row):
                heat_surface.set_at((col_index, row_index), heat_color(value, max_value))
        HEATMAP_SURFACE_CACHE["surface"] = pygame.transform.smoothscale(heat_surface, rect.size)
        HEATMAP_SURFACE_CACHE["version"] = cache_key
    screen.blit(HEATMAP_SURFACE_CACHE["surface"], rect)


def draw_heatmap_button(screen, font):
    rect = heatmap_button_rect()
    pygame.draw.rect(screen, (20, 24, 30), rect, border_radius=8)
    pygame.draw.rect(screen, (255, 210, 80), rect, 2, border_radius=8)
    draw_text(screen, "Generer heatmap", rect.x + 20, rect.y + 8, font, (255, 240, 190))


def draw_player_card(screen, fonts, analytics):
    ensure_player_analytics_defaults(analytics)
    if not analytics["card_visible"]:
        return
    font, small_font = fonts
    card = pygame.Surface((340, 248), pygame.SRCALPHA)
    card.fill((8, 10, 14, 215))
    screen.blit(card, (WIDTH - 360, 18))
    x = WIDTH - 340
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
    y += 26
    draw_text(screen, "Clic sur le tag reel: ouvrir / fermer", x, y, small_font, (180, 180, 180))


def draw_scene(screen, fonts, active_anchors, active_ids, state, solution, tag_real, layout_name, active_anchor_count):
    font, small_font = fonts
    draw_grid(screen)
    draw_heatmap(screen, state["player_analytics"])
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for aid in active_ids:
        if aid not in solution["dist_raw"]:
            continue
        ax, ay = active_anchors[aid]
        sx, sy = world_to_screen(ax, ay)
        radius = max(1, int(solution["dist_raw"][aid] * SCALE))
        c = anchor_colors[aid]
        pygame.draw.circle(overlay, (c[0], c[1], c[2], 40), (sx, sy), radius)
    screen.blit(overlay, (0, 0))

    for aid, (ax, ay) in active_anchors.items():
        sx, sy = world_to_screen(ax, ay)
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
        tx, ty = world_to_screen(*tag_real)
        pygame.draw.circle(screen, (255, 0, 0), (tx, ty), 11)
        draw_text(screen, "Tag reel", tx + 16, ty - 10, small_font, (255, 0, 0))

    for i, p in enumerate(solution["raw_intersections"]):
        px, py = world_to_screen(*p)
        pygame.draw.circle(screen, (255, 0, 255), (px, py), 7)
        draw_text(screen, f"Inter brute {i + 1}", px + 10, py - 10, small_font, (255, 120, 255))
    for i, p in enumerate(solution["smooth_intersections"]):
        px, py = world_to_screen(*p)
        pygame.draw.circle(screen, (255, 255, 255), (px, py), 7)
        draw_text(screen, f"Inter lissee {i + 1}", px + 10, py - 10, small_font, (255, 255, 255))

    if solution["tag_est_raw"] is not None:
        ex, ey = world_to_screen(*solution["tag_est_raw"])
        pygame.draw.circle(screen, (180, 180, 180), (ex, ey), 10)
        draw_text(screen, "Estimation brute", ex + 14, ey - 10, small_font, (180, 180, 180))
    if solution["tag_est_smooth"] is not None:
        ex, ey = world_to_screen(*solution["tag_est_smooth"])
        pygame.draw.circle(screen, (255, 210, 0), (ex, ey), 12)
        draw_text(screen, "Estimation lissee", ex + 14, ey - 10, small_font, (255, 210, 0))

    draw_hud(screen, font, small_font, active_ids, state, solution, layout_name, active_anchor_count, tag_real)
    draw_heatmap_button(screen, small_font)
    draw_player_card(screen, fonts, state["player_analytics"])


def draw_hud(screen, font, small_font, active_ids, state, solution, layout_name, active_anchor_count, tag_real):
    if not state.get("ui", {}).get("hud_visible", True):
        return
    hud_bg = pygame.Surface((650, 560), pygame.SRCALPHA)
    hud_bg.fill((0, 0, 0, 150))
    screen.blit(hud_bg, (15, 15))
    y = 25
    draw_text(screen, "UWB Estimation 2D", 30, y, font)
    y += 34
    draw_text(screen, f"Ancres actives : {active_anchor_count} | {layout_name}", 30, y, small_font, (220, 220, 220))
    y += 26
    draw_text(screen, "2..6 changer | H hide | R reset | ESC quitter", 30, y, small_font, (220, 220, 220))
    y += 26
    source_color = (120, 255, 210) if solution["source"] == "udp" else (220, 220, 220)
    draw_text(screen, f"Source : {solution['status']}", 30, y, small_font, source_color)
    y += 30
    draw_text(screen, "Bouton heatmap : capture snapshot | Clic tag : fiche joueur", 30, y, small_font, (255, 210, 120))
    y += 30
    for aid in active_ids:
        raw = _smooth_display_value(state, f"a{aid}_raw", solution["dist_raw"].get(aid))
        smooth = _smooth_display_value(state, f"a{aid}_smooth", solution["dist_smooth"].get(aid))
        draw_text(screen, f"A{aid} brut={_fmt(raw)} | lisse={_fmt(smooth)}", 30, y, small_font, anchor_colors[aid])
        y += 25
    y += 8
    if active_anchor_count == 2:
        draw_text(screen, "Mode 2 ancres : position 2D ambigue.", 30, y, small_font, (255, 180, 120))
        y += 24
        draw_text(screen, "Les deux intersections possibles sont affichees.", 30, y, small_font, (255, 180, 120))
        return
    if not solution["valid"]:
        draw_text(screen, "En attente de toutes les distances actives.", 30, y, small_font, (255, 180, 120))
        return
    if tag_real is None:
        if solution["tag_est_smooth"] is None:
            draw_text(screen, "Pas encore assez de distances pour estimer.", 30, y, small_font, (255, 180, 120))
            return
        x, y_est = solution["tag_est_smooth"]
        draw_text(screen, f"Position estimee : X={x:.1f} cm | Y={y_est:.1f} cm", 30, y, small_font, (255, 255, 255))
        return
    draw_text(screen, f"Erreur brute instantanee : {_fmt(_smooth_display_value(state, 'raw_err', solution['raw_err']))}", 30, y, small_font, (180, 180, 180))
    y += 25
    draw_text(screen, f"Erreur lissee instantanee : {_fmt(_smooth_display_value(state, 'smooth_err', solution['smooth_err']))}", 30, y, small_font)
    y += 25
    draw_text(screen, f"Erreur brute moyenne : {_fmt(_smooth_display_value(state, 'avg_raw', solution['avg_raw']))}", 30, y, small_font, (180, 180, 180))
    y += 25
    draw_text(screen, f"Erreur lissee moyenne : {_fmt(_smooth_display_value(state, 'avg_smooth', solution['avg_smooth']))}", 30, y, small_font)
    y += 25
    draw_text(screen, f"Erreur max brute : {_smooth_display_value(state, 'raw_max', state['stats']['raw_max']):.1f} cm", 30, y, small_font, (180, 180, 180))
    y += 25
    draw_text(screen, f"Erreur max lissee : {_smooth_display_value(state, 'smooth_max', state['stats']['smooth_max']):.1f} cm", 30, y, small_font)
    y += 35
    draw_text(screen, f"Precision brute : {_fmt(_smooth_display_value(state, 'raw_precision', solution['raw_precision']), ' %')}", 30, y, small_font, (180, 180, 180))
    y += 25
    draw_text(screen, f"Precision lissee : {_fmt(_smooth_display_value(state, 'smooth_precision', solution['smooth_precision']), ' %')}", 30, y, small_font)
