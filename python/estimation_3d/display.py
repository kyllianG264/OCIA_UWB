import math

import pygame

from court_geometry import court_bounds, draw_court_3d
from player_analytics import average_speed_kmh, build_smoothed_heatmap, ensure_player_analytics_defaults, heat_color, max_speed_kmh, total_distance_m
from scene import BG_COLOR, CAMERA_DISTANCE, CAMERA_PITCH, CAMERA_YAW, CENTER_X, CENTER_Y, HEIGHT, HUD_SUB, HUD_TEXT, NEAR_PLANE, WIDTH, anchor_colors

HEATMAP_BUTTON_RECT = pygame.Rect(WIDTH - 240, HEIGHT - 64, 210, 40)

def clamp(v, a, b):
    return max(a, min(b, v))


def lerp(a, b, t):
    return a + (b - a) * t


def vec_sub(a, b):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def length(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    l = length(v)
    if l == 0.0:
        return 0.0, 0.0, 0.0
    return v[0] / l, v[1] / l, v[2] / l


class OrbitCamera:
    def __init__(self):
        self.target = (CENTER_X, CENTER_Y, 0.0)
        self.yaw = CAMERA_YAW
        self.pitch = CAMERA_PITCH
        self.distance = CAMERA_DISTANCE
        self.follow_tag = True

    def update_target(self, point):
        if self.follow_tag:
            self.target = (point[0], point[1], 0.0)

    def get_position(self):
        cos_pitch = math.cos(self.pitch)
        return (
            self.target[0] + self.distance * math.cos(self.yaw) * cos_pitch,
            self.target[1] + self.distance * math.sin(self.yaw) * cos_pitch,
            self.target[2] + self.distance * math.sin(self.pitch),
        )

    def get_basis(self):
        position = self.get_position()
        forward = normalize(vec_sub(self.target, position))
        right = normalize(cross(forward, (0.0, 0.0, 1.0)))
        up = normalize(cross(right, forward))
        return position, right, up, forward

    def project(self, point3, focal):
        position, right, up, forward = self.get_basis()
        rel = vec_sub(point3, position)
        x_cam = dot(rel, right)
        y_cam = dot(rel, up)
        z_cam = dot(rel, forward)
        if z_cam <= NEAR_PLANE:
            return None
        sx = WIDTH // 2 + int(focal * x_cam / z_cam)
        sy = HEIGHT // 2 - int(focal * y_cam / z_cam)
        return sx, sy, z_cam


def draw_text(surface, text, x, y, font, color):
    surf = font.render(text, True, color)
    surface.blit(surf, (x, y))


def heatmap_button_rect():
    return HEATMAP_BUTTON_RECT.copy()


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


def draw_line_3d(surface, camera, p0, p1, color, focal, width=1):
    s0 = camera.project(p0, focal)
    s1 = camera.project(p1, focal)
    if s0 is None or s1 is None:
        return
    pygame.draw.line(surface, color, (s0[0], s0[1]), (s1[0], s1[1]), int(width))


def draw_circle_3d(surface, camera, point3, radius_world_cm, color, focal, width=0):
    pr = camera.project(point3, focal)
    if pr is None:
        return
    px_radius = max(2, int(focal * radius_world_cm / max(pr[2], 1.0)))
    pygame.draw.circle(surface, color, (pr[0], pr[1]), px_radius, width)


def draw_shadow(surface, camera, point3, focal, alpha=70):
    shadow_p = (point3[0], point3[1], 0.0)
    pr = camera.project(shadow_p, focal)
    if pr is None:
        return
    r = max(6, int(focal * 22.0 / max(pr[2], 1.0)))
    shadow = pygame.Surface((r * 4, r * 2), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (0, 0, 0, alpha), shadow.get_rect())
    surface.blit(shadow, shadow.get_rect(center=(pr[0], pr[1])))


def draw_ground_grid(surface, camera, focal):
    draw_court_3d(surface, camera, focal, CENTER_X, CENTER_Y)


def draw_anchor(surface, camera, pos3, anchor_id, color, focal, font):
    floor_p = (pos3[0], pos3[1], 0.0)
    draw_line_3d(surface, camera, floor_p, pos3, color, focal, 2)
    draw_circle_3d(surface, camera, pos3, 28.0, color, focal)
    p = camera.project(pos3, focal)
    if p is not None:
        draw_text(surface, f"A{anchor_id}", p[0] + 10, p[1] - 8, font, color)


def draw_trail(surface, camera, trail, color, focal, width):
    if len(trail) < 2:
        return
    pts2 = []
    for pt in trail:
        pr = camera.project(pt, focal)
        if pr is not None:
            pts2.append((pr[0], pr[1]))
    if len(pts2) >= 2:
        pygame.draw.lines(surface, color, False, pts2, width)


def draw_heatmap_3d(surface, camera, focal, analytics):
    ensure_player_analytics_defaults(analytics)
    if analytics.get("heatmap_snapshot_version") is None:
        return
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    left, right, top, bottom = analytics["bounds"]
    grid = build_smoothed_heatmap(analytics, radius=3)
    max_value = max((max(row) for row in grid), default=0.0)
    cols = analytics["heat_cols"]
    rows = analytics["heat_rows"]
    cell_w = (right - left) / cols
    cell_h = (bottom - top) / rows
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            x0 = left + col_index * cell_w
            y0 = top + row_index * cell_h
            points = [
                (x0, y0, 1.0),
                (x0 + cell_w, y0, 1.0),
                (x0 + cell_w, y0 + cell_h, 1.0),
                (x0, y0 + cell_h, 1.0),
            ]
            projected = []
            for point in points:
                pr = camera.project(point, focal)
                if pr is None:
                    projected = []
                    break
                projected.append((pr[0], pr[1]))
            if len(projected) == 4:
                pygame.draw.polygon(overlay, heat_color(value, max_value), projected)
    surface.blit(overlay, (0, 0))


def draw_player_card(screen, fonts, analytics):
    ensure_player_analytics_defaults(analytics)
    if not analytics["card_visible"]:
        return
    font, small_font = fonts
    card = pygame.Surface((360, 256), pygame.SRCALPHA)
    card.fill((5, 8, 14, 220))
    screen.blit(card, (WIDTH - 380, 18))
    x = WIDTH - 360
    y = 34
    draw_text(screen, analytics["name"], x, y, font, HUD_TEXT)
    y += 34
    draw_text(screen, f"Base: {analytics.get('source_label', 'Estimation UWB')}", x, y, small_font, (255, 210, 120))
    y += 30
    draw_text(screen, f"Vitesse moyenne: {average_speed_kmh(analytics):.2f} km/h", x, y, small_font, HUD_TEXT)
    y += 26
    draw_text(screen, f"Vitesse max: {max_speed_kmh(analytics):.2f} km/h", x, y, small_font, HUD_TEXT)
    y += 26
    draw_text(screen, f"Distance totale: {total_distance_m(analytics):.1f} m", x, y, small_font, HUD_TEXT)
    y += 26
    draw_text(screen, f"Nombre de sauts: {analytics['jump_count']}", x, y, small_font, HUD_TEXT)
    y += 26
    draw_text(screen, f"Hauteur max saut: {analytics['max_jump_cm']:.1f} cm", x, y, small_font, HUD_TEXT)
    y += 26
    draw_text(screen, f"Echantillons: {analytics['samples']}", x, y, small_font, HUD_SUB)


def draw_heatmap_button(screen, font):
    rect = heatmap_button_rect()
    pygame.draw.rect(screen, (20, 24, 30), rect, border_radius=8)
    pygame.draw.rect(screen, (255, 210, 80), rect, 2, border_radius=8)
    draw_text(screen, "Generer heatmap", rect.x + 20, rect.y + 8, font, (255, 240, 190))


def control_value_text(state, control):
    settings = state["settings"]
    if control["kind"] == "setting":
        value = settings[control["key"]]
        if control["key"].startswith("alpha") or control["key"].endswith("prob") or control["key"].endswith("ratio"):
            return f"{value:.3f}"
        return f"{value:.1f}"
    anchor = settings["layouts"][str(control["anchor_count"])]["anchors"][str(control["anchor_id"])]
    return f"{anchor[control['axis_index']]:.1f} cm"


def draw_scene(screen, camera, focal, fonts, active_anchors, active_ids, state, solution, tag_real, jump_extra, show_rays, show_trails, layout_name, active_anchor_count, control_items):
    font, small_font = fonts
    screen.fill(BG_COLOR)
    sky = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for yy in range(HEIGHT):
        k = yy / max(HEIGHT - 1, 1)
        c = (int(lerp(12, 32, k)), int(lerp(16, 42, k)), int(lerp(28, 58, k)), 255)
        pygame.draw.line(sky, c, (0, yy), (WIDTH, yy))
    screen.blit(sky, (0, 0))
    draw_ground_grid(screen, camera, focal)
    draw_heatmap_3d(screen, camera, focal, state["player_analytics"])
    if show_trails:
        draw_trail(screen, camera, state["trail_real"], (255, 80, 80), focal, 2)
        draw_trail(screen, camera, state["trail_est"], (255, 220, 40), focal, 2)
    for aid in active_ids:
        draw_anchor(screen, camera, active_anchors[aid], aid, anchor_colors[aid], focal, small_font)
    if show_rays:
        for aid in active_ids:
            if tag_real is not None:
                draw_line_3d(screen, camera, active_anchors[aid], tag_real, anchor_colors[aid], focal, 1)
    delayed_real = solution["delayed_real"]
    if delayed_real is not None:
        draw_shadow(screen, camera, delayed_real, focal, alpha=45)
        draw_circle_3d(screen, camera, delayed_real, 20.0, (180, 180, 180), focal, 2)
    if tag_real is not None:
        draw_shadow(screen, camera, tag_real, focal, alpha=90)
        draw_circle_3d(screen, camera, tag_real, 24.0, (255, 70, 70), focal)
    if solution["tag_est_raw"] is not None:
        draw_shadow(screen, camera, solution["tag_est_raw"], focal, alpha=55)
        draw_circle_3d(screen, camera, solution["tag_est_raw"], 20.0, (180, 180, 180), focal, 2)
    if solution["tag_est_smooth"] is not None:
        draw_shadow(screen, camera, solution["tag_est_smooth"], focal, alpha=80)
        draw_circle_3d(screen, camera, solution["tag_est_smooth"], 24.0, (255, 220, 40), focal)
    draw_hud(screen, font, small_font, active_anchor_count, layout_name, active_anchors, active_ids, solution, jump_extra, state, control_items)
    draw_heatmap_button(screen, small_font)
    draw_player_card(screen, fonts, state["player_analytics"])


def draw_hud(screen, font, small_font, active_anchor_count, layout_name, active_anchors, active_ids, solution, jump_extra, state, control_items):
    ui = state["ui"]
    if not ui.get("hud_visible", True):
        return
    panel_width = 510 if ui["hud_expanded"] else 350
    panel_height = 620 if ui["hud_expanded"] else 330
    hud = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    hud.fill((0, 0, 0, 155))
    screen.blit(hud, (18, 18))
    y = 28
    draw_text(screen, "UWB Estimation 3D", 34, y, font, HUD_TEXT)
    y += 34
    draw_text(screen, f"Ancres: {active_anchor_count} | {layout_name}", 34, y, small_font, HUD_SUB)
    y += 24
    draw_text(screen, "R reset | T trails | L rays | F follow | H hud | P stats", 34, y, small_font, HUD_SUB)
    y += 24
    draw_text(screen, "Fleches: edition | Shift: pas rapide | Molette: zoom", 34, y, small_font, HUD_SUB)
    y += 28
    draw_text(screen, "Bouton heatmap : capture snapshot | P/clic point estime : fiche", 34, y, small_font, (255, 210, 120))
    y += 26
    jump_color = (255, 200, 110) if jump_extra > 1.0 else HUD_SUB
    draw_text(screen, f"Source: {solution['status']}", 34, y, small_font, HUD_TEXT)
    y += 24
    draw_text(screen, f"Jump extra: {_smooth_display_value(state, 'jump_extra', jump_extra):.1f} cm", 34, y, small_font, jump_color)
    y += 24
    if solution["tag_est_smooth"] is not None:
        x = _smooth_display_value(state, "tag_est_x", solution["tag_est_smooth"][0])
        y_est = _smooth_display_value(state, "tag_est_y", solution["tag_est_smooth"][1])
        z = _smooth_display_value(state, "tag_est_z", solution["tag_est_smooth"][2])
        draw_text(screen, f"Estime lisse: X={x:.1f} Y={y_est:.1f} Z={z:.1f}", 34, y, small_font, HUD_TEXT)
        y += 24
    if solution["err3d"] is not None:
        draw_text(screen, f"Err 3D instantanee: {_smooth_display_value(state, 'err3d', solution['err3d']):.1f} cm", 34, y, small_font, HUD_TEXT)
        y += 22
        draw_text(screen, f"Err XY: {_smooth_display_value(state, 'errxy', solution['errxy']):.1f} cm | Err Z: {_smooth_display_value(state, 'errz', solution['errz']):.1f} cm", 34, y, small_font, HUD_SUB)
        y += 22
        draw_text(screen, f"Precision: {_smooth_display_value(state, 'precision', solution['precision']):.1f} %", 34, y, small_font, HUD_TEXT)
        y += 22
        draw_text(screen, f"Delay filtre: {_smooth_display_value(state, 'delay_total_ms', solution['delay_total_s'] * 1000.0):.0f} ms", 34, y, small_font, HUD_SUB)
        y += 22
        avg_delay_ms = (solution["avg_measured_delay"] or 0.0) * 1000.0
        draw_text(screen, f"Delay mesure: {_smooth_display_value(state, 'avg_measured_delay_ms', avg_delay_ms):.0f} ms moy", 34, y, small_font, HUD_SUB)
        y += 26
    if not ui["hud_expanded"]:
        return
    box = pygame.Rect(30, y, panel_width - 60, 210)
    pygame.draw.rect(screen, (255, 255, 255, 20), box, border_radius=8)
    pygame.draw.rect(screen, (255, 255, 255), box, 1, border_radius=8)
    visible_count = 7
    selected = max(0, min(ui["selected_control"], max(len(control_items) - 1, 0)))
    start_index = max(0, selected - visible_count // 2)
    end_index = min(len(control_items), start_index + visible_count)
    item_y = y + 10
    for idx in range(start_index, end_index):
        control = control_items[idx]
        row = pygame.Rect(40, item_y - 3, panel_width - 80, 24)
        label_color = HUD_TEXT if idx == selected else HUD_SUB
        if idx == selected:
            pygame.draw.rect(screen, (255, 220, 40), row, 1, border_radius=5)
        draw_text(screen, f"{control['label']}: {control_value_text(state, control)}", 48, item_y, small_font, label_color)
        item_y += 28
    info_y = y + 226
    draw_text(screen, "Ancres actives:", 34, info_y, small_font, HUD_TEXT)
    anchor_y = info_y + 24
    for aid in active_ids:
        x, y_anchor, z = active_anchors[aid]
        d_raw = _smooth_display_value(state, f"a{aid}_raw", solution["dist_raw"].get(aid))
        d_sm = _smooth_display_value(state, f"a{aid}_smooth", solution["dist_smooth"].get(aid))
        draw_text(screen, f"A{aid} [{x:.0f}, {y_anchor:.0f}, {z:.0f}] brut={d_raw if d_raw is not None else '--'} lisse={d_sm if d_sm is not None else '--'}", 34, anchor_y, small_font, anchor_colors[aid])
        anchor_y += 22
