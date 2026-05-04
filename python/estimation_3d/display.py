import math

import pygame

from scene import (
    BG_COLOR,
    BOUNDARY_COLOR,
    CAMERA_DISTANCE,
    CAMERA_PITCH,
    CAMERA_YAW,
    CENTER_X,
    CENTER_Y,
    GRID_COLOR,
    HEIGHT,
    HUD_SUB,
    HUD_TEXT,
    NEAR_PLANE,
    WIDTH,
    WORLD_MAX_X,
    WORLD_MAX_Y,
    WORLD_MIN_X,
    WORLD_MIN_Y,
    anchor_colors,
)


def clamp(v, a, b):
    return max(a, min(b, v))


def lerp(a, b, t):
    return a + (b - a) * t


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    l = length(v)
    if l < 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / l, v[1] / l, v[2] / l)


class OrbitCamera:
    def __init__(self):
        self.yaw = CAMERA_YAW
        self.pitch = CAMERA_PITCH
        self.distance = CAMERA_DISTANCE
        self.target = [CENTER_X, CENTER_Y, 550.0]
        self.follow_tag = False

    def update_target(self, tag_real):
        if self.follow_tag:
            self.target[0] = lerp(self.target[0], tag_real[0], 0.07)
            self.target[1] = lerp(self.target[1], tag_real[1], 0.07)
            self.target[2] = lerp(self.target[2], 450.0, 0.07)
        else:
            self.target[0] = lerp(self.target[0], CENTER_X, 0.08)
            self.target[1] = lerp(self.target[1], CENTER_Y, 0.08)
            self.target[2] = lerp(self.target[2], 550.0, 0.08)

    def get_position(self):
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        offset = (
            self.distance * cp * cy,
            self.distance * cp * sy,
            self.distance * sp,
        )
        return (
            self.target[0] + offset[0],
            self.target[1] + offset[1],
            self.target[2] + offset[2],
        )

    def get_basis(self):
        cam_pos = self.get_position()
        target = tuple(self.target)
        forward = normalize(vec_sub(target, cam_pos))
        world_up = (0.0, 0.0, 1.0)
        right = normalize(cross(forward, world_up))
        if length(right) < 1e-8:
            right = (1.0, 0.0, 0.0)
        up = normalize(cross(right, forward))
        return cam_pos, right, up, forward

    def project(self, point3, focal):
        cam_pos, right, up, forward = self.get_basis()
        rel = vec_sub(point3, cam_pos)
        x_cam = dot(rel, right)
        y_cam = dot(rel, up)
        z_cam = dot(rel, forward)
        if z_cam <= NEAR_PLANE:
            return None
        sx = WIDTH / 2 + x_cam * focal / z_cam
        sy = HEIGHT / 2 - y_cam * focal / z_cam
        scale = focal / z_cam
        return (sx, sy, z_cam, scale)


def draw_text(surface, text, x, y, font, color=HUD_TEXT):
    surf = font.render(text, True, color)
    surface.blit(surf, (x, y))


def draw_line_3d(surface, camera, p0, p1, color, focal, width=1):
    s0 = camera.project(p0, focal)
    s1 = camera.project(p1, focal)
    if s0 is None or s1 is None:
        return
    pygame.draw.line(surface, color, (int(s0[0]), int(s0[1])), (int(s1[0]), int(s1[1])), width)


def draw_circle_3d(surface, camera, point3, radius_world_cm, color, focal, width=0):
    pr = camera.project(point3, focal)
    if pr is None:
        return
    px_radius = max(2, int(radius_world_cm * pr[3]))
    pygame.draw.circle(surface, color, (int(pr[0]), int(pr[1])), px_radius, width)


def draw_shadow(surface, camera, point3, focal, alpha=80):
    shadow_p = (point3[0], point3[1], 0.0)
    pr = camera.project(shadow_p, focal)
    if pr is None:
        return
    r = max(4, int(45 * pr[3]))
    shadow = pygame.Surface((r * 4, r * 2), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (0, 0, 0, alpha), shadow.get_rect())
    surface.blit(shadow, (int(pr[0] - shadow.get_width() / 2), int(pr[1] - shadow.get_height() / 2)))


def draw_ground_grid(surface, camera, focal):
    for x in range(WORLD_MIN_X, WORLD_MAX_X + 1, 200):
        draw_line_3d(surface, camera, (x, WORLD_MIN_Y, 0.0), (x, WORLD_MAX_Y, 0.0), GRID_COLOR, focal, 1)
    for y in range(WORLD_MIN_Y, WORLD_MAX_Y + 1, 200):
        draw_line_3d(surface, camera, (WORLD_MIN_X, y, 0.0), (WORLD_MAX_X, y, 0.0), GRID_COLOR, focal, 1)
    corners = [
        (WORLD_MIN_X, WORLD_MIN_Y, 0.0),
        (WORLD_MAX_X, WORLD_MIN_Y, 0.0),
        (WORLD_MAX_X, WORLD_MAX_Y, 0.0),
        (WORLD_MIN_X, WORLD_MAX_Y, 0.0),
    ]
    for i in range(4):
        draw_line_3d(surface, camera, corners[i], corners[(i + 1) % 4], BOUNDARY_COLOR, focal, 3)


def draw_anchor(surface, camera, pos3, anchor_id, color, focal, font):
    floor_p = (pos3[0], pos3[1], 0.0)
    draw_line_3d(surface, camera, floor_p, pos3, (140, 145, 155), focal, 4)
    draw_circle_3d(surface, camera, floor_p, 18, (95, 95, 105), focal, 0)
    draw_circle_3d(surface, camera, pos3, 28, color, focal, 0)
    draw_circle_3d(surface, camera, pos3, 18, (255, 220, 120), focal, 0)
    p = camera.project(pos3, focal)
    if p is not None:
        draw_text(surface, f"P{anchor_id}", int(p[0] + 10), int(p[1] - 10), font, color)


def draw_trail(surface, camera, trail, color, focal, width=2):
    if len(trail) < 2:
        return
    pts2 = []
    for pt in trail:
        pr = camera.project(pt, focal)
        if pr is not None:
            pts2.append((int(pr[0]), int(pr[1])))
    if len(pts2) >= 2:
        pygame.draw.lines(surface, color, False, pts2, width)


def control_value_text(state, control):
    settings = state["settings"]
    if control["kind"] == "setting":
        value = settings[control["key"]]
        if control["key"] == "noise_ratio":
            return f"{value * 100.0:.1f} %"
        if control["key"] == "spike_prob":
            return f"{value * 100.0:.1f} %"
        if control["key"].startswith("alpha"):
            return f"{value:.2f}"
        return f"{value:.1f} cm"

    anchor = settings["layouts"][str(control["anchor_count"])]["anchors"][str(control["anchor_id"])]
    value = anchor[control["axis_index"]]
    return f"{value:.1f} cm"


def draw_scene(screen, camera, focal, fonts, active_anchors, active_ids, state, solution, tag_real, jump_extra, show_rays, show_trails, layout_name, active_anchor_count, control_items):
    font, small_font = fonts

    screen.fill(BG_COLOR)
    sky = pygame.Surface((WIDTH, HEIGHT))
    for yy in range(HEIGHT):
        k = yy / HEIGHT
        c = int(22 + 18 * (1 - k))
        pygame.draw.line(sky, (c, c + 2, c + 8), (0, yy), (WIDTH, yy))
    screen.blit(sky, (0, 0))

    draw_ground_grid(screen, camera, focal)
    draw_line_3d(screen, camera, (CENTER_X - 50, CENTER_Y, 0), (CENTER_X + 50, CENTER_Y, 0), (115, 125, 145), focal, 2)
    draw_line_3d(screen, camera, (CENTER_X, CENTER_Y - 50, 0), (CENTER_X, CENTER_Y + 50, 0), (115, 125, 145), focal, 2)

    if show_trails:
        draw_trail(screen, camera, state["trail_real"], (255, 70, 70), focal, 3)
        draw_trail(screen, camera, state["trail_est"], (235, 235, 255), focal, 2)

    tag_est_smooth = solution["tag_est_smooth"]
    tag_est_raw = solution["tag_est_raw"]
    delayed_real = solution["delayed_real"]

    draw_shadow(screen, camera, tag_real, focal, alpha=85)
    if tag_est_smooth is not None:
        draw_shadow(screen, camera, tag_est_smooth, focal, alpha=50)

    for aid in active_ids:
        draw_anchor(screen, camera, active_anchors[aid], aid, anchor_colors[aid], focal, small_font)

    if show_rays and tag_est_smooth is not None:
        for aid in active_ids:
            draw_line_3d(screen, camera, active_anchors[aid], tag_est_smooth, (115, 135, 150), focal, 1)

    draw_circle_3d(screen, camera, tag_real, 34, (255, 70, 70), focal, 0)
    draw_circle_3d(screen, camera, tag_real, 18, (255, 200, 200), focal, 0)

    if tag_est_raw is not None:
        draw_circle_3d(screen, camera, tag_est_raw, 18, (130, 130, 130), focal, 0)

    if tag_est_smooth is not None:
        draw_circle_3d(screen, camera, tag_est_smooth, 28, (255, 255, 255), focal, 0)
        draw_circle_3d(screen, camera, tag_est_smooth, 15, (165, 220, 255), focal, 0)
        draw_line_3d(screen, camera, tag_real, tag_est_smooth, (255, 225, 80), focal, 2)

    if delayed_real is not None and tag_est_smooth is not None:
        draw_line_3d(screen, camera, delayed_real, tag_est_smooth, (120, 255, 210), focal, 2)
        draw_circle_3d(screen, camera, delayed_real, 16, (90, 240, 190), focal, 0)

    p_real = camera.project(tag_real, focal)
    if p_real is not None:
        draw_text(screen, "Tag reel", int(p_real[0] + 12), int(p_real[1] - 18), small_font, (255, 100, 100))

    if tag_est_smooth is not None:
        p_est = camera.project(tag_est_smooth, focal)
        if p_est is not None:
            draw_text(screen, "Estimation 3D", int(p_est[0] + 12), int(p_est[1] - 18), small_font, (245, 245, 255))

    if delayed_real is not None:
        p_del = camera.project(delayed_real, focal)
        if p_del is not None:
            draw_text(screen, "Reel compense", int(p_del[0] + 12), int(p_del[1] - 18), small_font, (120, 255, 210))

    draw_hud(screen, font, small_font, active_anchor_count, layout_name, active_anchors, active_ids, solution, tag_real, jump_extra, state, control_items)


def draw_hud(screen, font, small_font, active_anchor_count, layout_name, active_anchors, active_ids, solution, tag_real, jump_extra, state, control_items):
    ui = state["ui"]
    panel_width = 820 if ui["hud_expanded"] else 360
    panel_height = 880 if ui["hud_expanded"] else 92

    hud = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    hud.fill((0, 0, 0, 155))
    screen.blit(hud, (16, 16))

    y = 28
    draw_text(screen, "UWB 3D - controle live", 30, y, font)
    y += 34
    draw_text(screen, f"Ancres actives : {active_anchor_count} | {layout_name}", 30, y, small_font, HUD_SUB)
    y += 24
    draw_text(screen, "H replie le HUD | Fleches haut/bas selection | gauche/droite modif | Shift accelere", 30, y, small_font, HUD_SUB)

    if not ui["hud_expanded"]:
        y += 26
        draw_text(screen, f"Retard moyen observe : {solution['avg_measured_delay']:.3f} s" if solution["avg_measured_delay"] is not None else "Retard moyen observe : --", 30, y, small_font, (120, 255, 210))
        y += 22
        draw_text(screen, ui["cache_message"], 30, y, small_font, (180, 255, 180))
        return

    y += 30
    draw_text(screen, f"Cache local : {ui['cache_message']}", 30, y, small_font, (180, 255, 180))
    y += 24
    draw_text(screen, f"Retard filtre distance : {solution['delay_dist']:.3f} s", 30, y, small_font, (255, 220, 150))
    y += 22
    draw_text(screen, f"Retard filtre position : {solution['delay_pos']:.3f} s", 30, y, small_font, (255, 220, 150))
    y += 22
    draw_text(screen, f"Retard total estime : {solution['delay_total']:.3f} s", 30, y, small_font, (255, 240, 170))
    y += 22
    draw_text(screen, f"Retard moyen estime : {solution['avg_delay_est']:.3f} s" if solution["avg_delay_est"] is not None else "Retard moyen estime : --", 30, y, small_font, (255, 240, 170))
    y += 22
    draw_text(screen, f"Retard observe instantane : {solution['measured_delay']:.3f} s" if solution["measured_delay"] is not None else "Retard observe instantane : --", 30, y, small_font, (120, 255, 210))
    y += 22
    draw_text(screen, f"Retard moyen observe : {solution['avg_measured_delay']:.3f} s" if solution["avg_measured_delay"] is not None else "Retard moyen observe : --", 30, y, small_font, (120, 255, 210))
    y += 22
    draw_text(screen, f"Retard max observe : {solution['max_measured_delay']:.3f} s" if solution["max_measured_delay"] is not None else "Retard max observe : --", 30, y, small_font, (120, 255, 210))
    y += 28

    jump_color = (255, 130, 130) if state["jump_active"] else (180, 255, 180)
    draw_text(screen, f"Hauteur reelle : {tag_real[2]:.1f} cm", 30, y, small_font, jump_color)
    y += 22

    tag_est_smooth = solution["tag_est_smooth"]
    if tag_est_smooth is not None:
        draw_text(screen, f"Hauteur estimee : {tag_est_smooth[2]:.1f} cm", 30, y, small_font, (245, 245, 255))
    else:
        draw_text(screen, "Hauteur estimee : --", 30, y, small_font, (245, 245, 255))
    y += 22

    draw_text(screen, f"Extra saut actuel : {jump_extra:.1f} cm | Saut : {'OUI' if state['jump_active'] else 'NON'}", 30, y, small_font, HUD_SUB)
    y += 28

    if solution["err3d"] is not None:
        draw_text(screen, f"Erreur 3D instantanee : {solution['err3d']:.1f} cm", 30, y, small_font)
        y += 22
        draw_text(screen, f"Erreur XY instantanee : {solution['errxy']:.1f} cm", 30, y, small_font)
        y += 22
        draw_text(screen, f"Erreur Z instantanee : {solution['errz']:.1f} cm", 30, y, small_font)
        y += 22
        draw_text(screen, f"Precision instantanee : {solution['precision']:.1f} %", 30, y, small_font)
        y += 22
        draw_text(screen, f"Erreur 3D moyenne : {solution['avg_err3d']:.1f} cm | max : {state['stats']['err3d_max']:.1f} cm", 30, y, small_font, HUD_SUB)
        y += 22
        draw_text(screen, f"Erreur XY moyenne : {solution['avg_errxy']:.1f} cm", 30, y, small_font, HUD_SUB)
        y += 22
        draw_text(screen, f"Erreur Z moyenne : {solution['avg_errz']:.1f} cm | max Z : {state['stats']['errz_max']:.1f} cm", 30, y, small_font, HUD_SUB)
        y += 22
        draw_text(screen, f"Precision moyenne : {solution['avg_precision']:.1f} %", 30, y, small_font, HUD_SUB)
        y += 28

    if solution["traj_err_xy"] is not None:
        draw_text(screen, f"Erreur trajectoire XY compensee : {solution['traj_err_xy']:.1f} cm", 30, y, small_font, (120, 255, 210))
        y += 22
        draw_text(screen, f"Erreur trajectoire 3D compensee : {solution['traj_err_3d']:.1f} cm", 30, y, small_font, (120, 255, 210))
        y += 22
        draw_text(screen, f"Precision trajectoire : {solution['traj_precision']:.1f} %", 30, y, small_font, (120, 255, 210))
        y += 22
        draw_text(screen, f"Erreur trajet XY moyenne : {solution['avg_traj_err_xy']:.1f} cm", 30, y, small_font, HUD_SUB)
        y += 22
        draw_text(screen, f"Erreur trajet 3D moyenne : {solution['avg_traj_err_3d']:.1f} cm", 30, y, small_font, HUD_SUB)
        y += 22
        if solution["path_ratio"] is not None:
            draw_text(screen, f"Fidelite longueur de trajectoire : {solution['path_ratio']:.1f} %", 30, y, small_font, HUD_SUB)
            y += 28

    draw_text(screen, "Edition detaillee", 430, 140, font)
    draw_text(screen, "Utilise les fleches du clavier pour modifier les valeurs", 430, 172, small_font, HUD_SUB)
    draw_text(screen, "Les changements sont enregistres dans settings_cache.json", 430, 194, small_font, HUD_SUB)

    box = pygame.Rect(430, 228, 370, 587)
    pygame.draw.rect(screen, (18, 24, 32), box, border_radius=10)
    pygame.draw.rect(screen, (70, 95, 120), box, 1, border_radius=10)

    selected = ui["selected_control"]
    visible_count = 20
    start_index = max(0, selected - visible_count // 2)
    end_index = min(len(control_items), start_index + visible_count)
    start_index = max(0, end_index - visible_count)

    item_y = 243
    for idx in range(start_index, end_index):
        control = control_items[idx]
        row = pygame.Rect(440, item_y - 4, 350, 30)
        if idx == selected:
            pygame.draw.rect(screen, (68, 112, 156), row, border_radius=8)
            pygame.draw.rect(screen, (165, 210, 255), row, 1, border_radius=8)
        label_color = (245, 248, 255) if idx == selected else HUD_SUB
        draw_text(screen, control["label"], 452, item_y, small_font, label_color)
        draw_text(screen, control_value_text(state, control), 665, item_y, small_font, anchor_colors.get(control.get("anchor_id", 0), (210, 230, 255)))
        item_y += 28

    info_y = 826
    draw_text(screen, "4/5/6 change le layout | R reset les stats | F suit le tag", 30, info_y, small_font, HUD_SUB)

    anchor_y = min(y, 680)
    anchor_y += 20
    draw_text(screen, f"Bruit relatif={state['settings']['noise_ratio'] * 100.0:.1f} % | spike prob={state['settings']['spike_prob'] * 100.0:.1f} % | spike amp={state['settings']['spike_amplitude']:.1f} cm", 30, anchor_y, small_font, (255, 180, 130))
    anchor_y += 24
    for aid in active_ids:
        x, y_anchor, z = active_anchors[aid]
        d_raw = solution["dist_raw"][aid]
        d_sm = solution["dist_smooth"][aid]
        draw_text(screen, f"A{aid} pos=({x:.0f}, {y_anchor:.0f}, {z:.0f}) cm | brut={d_raw:.1f} | lisse={d_sm:.1f}", 30, anchor_y, small_font, anchor_colors[aid])
        anchor_y += 20
