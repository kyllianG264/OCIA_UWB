import pygame
import math
import random

# =========================
# CONFIG FENÊTRE
# =========================
WIDTH, HEIGHT = 1400, 900
FPS = 60
SCALE = 0.2

CAMERA_X = 1250
CAMERA_Y = 900

# =========================
# GÉOMÉTRIE DE LA ZONE
# =========================
CENTER_X = 1250
CENTER_Y = 900

LEFT_X = 550
RIGHT_X = 1950
TOP_Y = 350
BOTTOM_Y = 1450

TOP_MID_Y = 100
BOTTOM_MID_Y = 1800

# Triangle
TRI_TOP = (1250, 300)
TRI_BL = (650, 1400)
TRI_BR = (1850, 1400)

# 2 ancres
LINE_LEFT = (500, 900)
LINE_RIGHT = (2000, 900)

# =========================
# BRUIT / SAUTS / FILTRES
# =========================
NOISE_STD = 18
JUMP_PROB = 0.025
JUMP_AMPLITUDE = 120

ALPHA_DIST = 0.12
ALPHA_POS = 0.10

# =========================
# PRÉCISION
# =========================
PRECISION_TOLERANCE_CM = 100.0
# 0 cm erreur = 100%
# 100 cm erreur = 0%

# =========================
# OUTILS
# =========================
def world_to_screen(x_cm, y_cm):
    sx = WIDTH // 2 + int((x_cm - CAMERA_X) * SCALE)
    sy = HEIGHT // 2 + int((y_cm - CAMERA_Y) * SCALE)
    return sx, sy

def draw_text(screen, text, x, y, font, color=(255, 255, 255)):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def smooth_value(old_val, new_val, alpha):
    if old_val is None:
        return new_val
    return alpha * new_val + (1 - alpha) * old_val

def smooth_point(old_point, new_point, alpha):
    if new_point is None:
        return old_point
    if old_point is None:
        return new_point
    x = alpha * new_point[0] + (1 - alpha) * old_point[0]
    y = alpha * new_point[1] + (1 - alpha) * old_point[1]
    return (x, y)

def noisy_distance(true_d):
    d = true_d + random.gauss(0, NOISE_STD)
    if random.random() < JUMP_PROB:
        d += random.uniform(-JUMP_AMPLITUDE, JUMP_AMPLITUDE)
    return max(1, d)

def circle_intersections(x0, y0, r0, x1, y1, r1):
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy)

    if d > r0 + r1:
        return []
    if d < abs(r0 - r1):
        return []
    if d == 0 and r0 == r1:
        return []

    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h_sq = r0 * r0 - a * a
    if h_sq < 0:
        h_sq = 0
    h = math.sqrt(h_sq)

    xm = x0 + a * dx / d
    ym = y0 + a * dy / d

    rx = -dy * (h / d)
    ry = dx * (h / d)

    p1 = (xm + rx, ym + ry)
    p2 = (xm - rx, ym - ry)
    return [p1, p2]

def estimate_position_least_squares(anchor_positions, distances):
    ids = sorted(anchor_positions.keys())
    if len(ids) < 3:
        return None

    ref_id = ids[0]
    x1, y1 = anchor_positions[ref_id]
    r1 = distances[ref_id]

    rows = []
    vals = []

    for aid in ids[1:]:
        xi, yi = anchor_positions[aid]
        ri = distances[aid]

        a = 2 * (xi - x1)
        b = 2 * (yi - y1)
        c = (r1 * r1 - ri * ri) - (x1 * x1 - xi * xi) - (y1 * y1 - yi * yi)

        rows.append((a, b))
        vals.append(c)

    s_aa = sum(a * a for a, b in rows)
    s_ab = sum(a * b for a, b in rows)
    s_bb = sum(b * b for a, b in rows)
    s_ac = sum(a * c for (a, b), c in zip(rows, vals))
    s_bc = sum(b * c for (a, b), c in zip(rows, vals))

    det = s_aa * s_bb - s_ab * s_ab
    if abs(det) < 1e-9:
        return None

    x = (s_ac * s_bb - s_ab * s_bc) / det
    y = (s_aa * s_bc - s_ab * s_ac) / det
    return (x, y)

def draw_grid(screen):
    step = 200

    left = CAMERA_X - (WIDTH / 2) / SCALE
    right = CAMERA_X + (WIDTH / 2) / SCALE
    top = CAMERA_Y - (HEIGHT / 2) / SCALE
    bottom = CAMERA_Y + (HEIGHT / 2) / SCALE

    grid_color = (40, 40, 55)

    for x in range(int(left // step) * step, int(right) + step, step):
        sx1, sy1 = world_to_screen(x, top)
        sx2, sy2 = world_to_screen(x, bottom)
        pygame.draw.line(screen, grid_color, (sx1, sy1), (sx2, sy2), 1)

    for y in range(int(top // step) * step, int(bottom) + step, step):
        sx1, sy1 = world_to_screen(left, y)
        sx2, sy2 = world_to_screen(right, y)
        pygame.draw.line(screen, grid_color, (sx1, sy1), (sx2, sy2), 1)

def error_to_precision_percent(err_cm, tolerance_cm=PRECISION_TOLERANCE_CM):
    if err_cm is None:
        return None
    p = 100.0 * (1.0 - err_cm / tolerance_cm)
    return max(0.0, min(100.0, p))

def reset_stats():
    return {
        "raw_sum": 0.0,
        "smooth_sum": 0.0,
        "raw_precision_sum": 0.0,
        "smooth_precision_sum": 0.0,
        "count": 0,
        "raw_max": 0.0,
        "smooth_max": 0.0,
    }

def get_anchor_layout(n):
    if n == 2:
        return {
            1: LINE_LEFT,
            2: LINE_RIGHT,
        }, "2 ancres : ligne"

    if n == 3:
        return {
            1: TRI_BL,
            2: TRI_TOP,
            3: TRI_BR,
        }, "3 ancres : triangle"

    if n == 4:
        return {
            1: (LEFT_X, TOP_Y),
            2: (RIGHT_X, TOP_Y),
            3: (RIGHT_X, BOTTOM_Y),
            4: (LEFT_X, BOTTOM_Y),
        }, "4 ancres : carré"

    if n == 5:
        return {
            1: (LEFT_X, TOP_Y),
            2: (RIGHT_X, TOP_Y),
            3: (RIGHT_X, BOTTOM_Y),
            4: (LEFT_X, BOTTOM_Y),
            5: (CENTER_X, TOP_MID_Y),
        }, "5 ancres : carré + pointe haut"

    return {
        1: (LEFT_X, TOP_Y),
        2: (RIGHT_X, TOP_Y),
        3: (RIGHT_X, BOTTOM_Y),
        4: (LEFT_X, BOTTOM_Y),
        5: (CENTER_X, TOP_MID_Y),
        6: (CENTER_X, BOTTOM_MID_Y),
    }, "6 ancres : carré + haut/bas"

def full_reset(anchor_count):
    global t, stats, dist_smooth, tag_est_smooth
    t = 0.0
    stats = reset_stats()
    tag_est_smooth = None
    active_anchors_tmp, _ = get_anchor_layout(anchor_count)
    dist_smooth = {aid: None for aid in active_anchors_tmp.keys()}

# =========================
# INIT
# =========================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("UWB - Test précision de 2 à 6 ancres")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 28)
small_font = pygame.font.SysFont("Arial", 22)

running = True
t = 0.0

active_anchor_count = 4
dist_smooth = {}
tag_est_smooth = None
stats = reset_stats()

anchor_colors = {
    1: (80, 180, 255),
    2: (120, 255, 120),
    3: (255, 170, 60),
    4: (180, 120, 255),
    5: (255, 110, 150),
    6: (90, 220, 220),
}

full_reset(active_anchor_count)

while running:
    dt = clock.tick(FPS) / 1000.0
    t += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_r:
                full_reset(active_anchor_count)

            elif event.key in [pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6]:
                new_count = int(event.unicode)
                if new_count != active_anchor_count:
                    active_anchor_count = new_count
                    full_reset(active_anchor_count)

    active_anchors, layout_name = get_anchor_layout(active_anchor_count)
    active_ids = sorted(active_anchors.keys())

    if set(dist_smooth.keys()) != set(active_ids):
        dist_smooth = {aid: None for aid in active_ids}

    # =========================
    # TAG RÉEL SIMULÉ
    # =========================
    tag_x = CENTER_X + 420 * math.cos(t * 0.10)
    tag_y = CENTER_Y + 280 * math.sin(t * 0.08)
    tag_real = (tag_x, tag_y)

    dist_raw = {}

    for aid, pos in active_anchors.items():
        d_true = distance(tag_real, pos)
        d_raw = noisy_distance(d_true)
        dist_raw[aid] = d_raw
        dist_smooth[aid] = smooth_value(dist_smooth[aid], d_raw, ALPHA_DIST)

    raw_intersections = []
    smooth_intersections = []

    if len(active_ids) >= 2:
        a1 = active_ids[0]
        a2 = active_ids[1]

        raw_intersections = circle_intersections(
            active_anchors[a1][0], active_anchors[a1][1], dist_raw[a1],
            active_anchors[a2][0], active_anchors[a2][1], dist_raw[a2]
        )

        smooth_intersections = circle_intersections(
            active_anchors[a1][0], active_anchors[a1][1], dist_smooth[a1],
            active_anchors[a2][0], active_anchors[a2][1], dist_smooth[a2]
        )

    tag_est_raw = estimate_position_least_squares(active_anchors, dist_raw)
    tag_est_from_smooth = estimate_position_least_squares(active_anchors, dist_smooth)
    tag_est_smooth = smooth_point(tag_est_smooth, tag_est_from_smooth, ALPHA_POS)

    raw_err = None
    smooth_err = None
    avg_raw = None
    avg_smooth = None

    raw_precision = None
    smooth_precision = None
    avg_raw_precision = None
    avg_smooth_precision = None

    if len(active_ids) >= 3 and tag_est_raw is not None and tag_est_smooth is not None:
        raw_err = distance(tag_real, tag_est_raw)
        smooth_err = distance(tag_real, tag_est_smooth)

        raw_precision = error_to_precision_percent(raw_err)
        smooth_precision = error_to_precision_percent(smooth_err)

        stats["raw_sum"] += raw_err
        stats["smooth_sum"] += smooth_err
        stats["raw_precision_sum"] += raw_precision
        stats["smooth_precision_sum"] += smooth_precision
        stats["count"] += 1
        stats["raw_max"] = max(stats["raw_max"], raw_err)
        stats["smooth_max"] = max(stats["smooth_max"], smooth_err)

        avg_raw = stats["raw_sum"] / stats["count"]
        avg_smooth = stats["smooth_sum"] / stats["count"]
        avg_raw_precision = stats["raw_precision_sum"] / stats["count"]
        avg_smooth_precision = stats["smooth_precision_sum"] / stats["count"]

    # =========================
    # DRAW
    # =========================
    screen.fill((12, 12, 20))
    draw_grid(screen)

    # Cercles bruts transparents
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for aid in active_ids:
        ax, ay = active_anchors[aid]
        sx, sy = world_to_screen(ax, ay)
        radius = max(1, int(dist_raw[aid] * SCALE))
        c = anchor_colors[aid]
        pygame.draw.circle(overlay, (c[0], c[1], c[2], 70), (sx, sy), radius, 2)
    screen.blit(overlay, (0, 0))

    # Cercles lissés
    for aid in active_ids:
        ax, ay = active_anchors[aid]
        sx, sy = world_to_screen(ax, ay)
        radius = max(1, int(dist_smooth[aid] * SCALE))
        pygame.draw.circle(screen, anchor_colors[aid], (sx, sy), radius, 2)

    # Ancres actives
    for aid, (ax, ay) in active_anchors.items():
        sx, sy = world_to_screen(ax, ay)
        pygame.draw.circle(screen, (255, 210, 0), (sx, sy), 12)
        draw_text(screen, f"Portenta {aid}", sx + 14, sy - 12, font)

    # Tag réel
    tx, ty = world_to_screen(tag_real[0], tag_real[1])
    pygame.draw.circle(screen, (255, 0, 0), (tx, ty), 11)
    draw_text(screen, "Tag réel", tx + 16, ty - 12, font, (255, 0, 0))

    # Intersections si 2 ancres
    if len(active_ids) == 2:
        for i, p in enumerate(raw_intersections):
            px, py = world_to_screen(p[0], p[1])
            pygame.draw.circle(screen, (255, 0, 255), (px, py), 7)
            draw_text(screen, f"Inter brute {i+1}", px + 12, py - 10, small_font, (255, 120, 255))

        for i, p in enumerate(smooth_intersections):
            px, py = world_to_screen(p[0], p[1])
            pygame.draw.circle(screen, (255, 255, 255), (px, py), 7)
            draw_text(screen, f"Inter lissée {i+1}", px + 12, py + 12, small_font, (255, 255, 255))

    # Estimation brute
    if len(active_ids) >= 3 and tag_est_raw is not None:
        ex, ey = world_to_screen(tag_est_raw[0], tag_est_raw[1])
        pygame.draw.circle(screen, (180, 180, 180), (ex, ey), 7)
        draw_text(screen, "Estimation brute", ex + 14, ey - 10, small_font, (180, 180, 180))

    # Estimation lissée
    if len(active_ids) >= 3 and tag_est_smooth is not None:
        sx, sy = world_to_screen(tag_est_smooth[0], tag_est_smooth[1])
        pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 10)
        draw_text(screen, "Estimation lissée", sx + 14, sy - 10, small_font, (255, 255, 255))

    # HUD fond
    hud_bg = pygame.Surface((560, 520), pygame.SRCALPHA)
    hud_bg.fill((0, 0, 0, 145))
    screen.blit(hud_bg, (15, 15))

    # HUD texte
    draw_text(screen, f"Ancres actives : {active_anchor_count}", 30, 25, font, (255, 255, 255))
    draw_text(screen, layout_name, 30, 60, small_font, (220, 220, 220))
    draw_text(screen, "Touches : 2..6 pour changer, R reset, ESC quitter", 30, 88, small_font, (220, 220, 220))
    draw_text(screen, f"Tolérance précision : {PRECISION_TOLERANCE_CM:.0f} cm", 30, 116, small_font, (220, 220, 220))

    y0 = 155
    for aid in active_ids:
        draw_text(
            screen,
            f"A{aid}  brut={dist_raw[aid]:.1f} cm   lissé={dist_smooth[aid]:.1f} cm",
            30,
            y0,
            small_font,
            anchor_colors[aid]
        )
        y0 += 28

    y0 += 10

    if active_anchor_count == 2:
        draw_text(screen, "Mode 2 ancres : position 2D ambiguë.", 30, y0, small_font, (255, 180, 120))
        y0 += 30
        draw_text(screen, "Pas de précision moyenne fiable.", 30, y0, small_font, (255, 180, 120))
        y0 += 30
        draw_text(screen, "Tu vois seulement les 2 intersections.", 30, y0, small_font, (255, 180, 120))
    else:
        raw_err_txt = "--" if raw_err is None else f"{raw_err:.1f} cm"
        smooth_err_txt = "--" if smooth_err is None else f"{smooth_err:.1f} cm"
        avg_raw_txt = "--" if avg_raw is None else f"{avg_raw:.1f} cm"
        avg_smooth_txt = "--" if avg_smooth is None else f"{avg_smooth:.1f} cm"

        raw_prec_txt = "--" if raw_precision is None else f"{raw_precision:.1f} %"
        smooth_prec_txt = "--" if smooth_precision is None else f"{smooth_precision:.1f} %"
        avg_raw_prec_txt = "--" if avg_raw_precision is None else f"{avg_raw_precision:.1f} %"
        avg_smooth_prec_txt = "--" if avg_smooth_precision is None else f"{avg_smooth_precision:.1f} %"

        draw_text(screen, f"Erreur instantanée brute   : {raw_err_txt}", 30, y0, small_font, (180, 180, 180))
        y0 += 28
        draw_text(screen, f"Erreur instantanée lissée  : {smooth_err_txt}", 30, y0, small_font, (255, 255, 255))
        y0 += 28
        draw_text(screen, f"Erreur moyenne brute       : {avg_raw_txt}", 30, y0, small_font, (180, 180, 180))
        y0 += 28
        draw_text(screen, f"Erreur moyenne lissée      : {avg_smooth_txt}", 30, y0, small_font, (255, 255, 255))
        y0 += 28
        draw_text(screen, f"Erreur max brute           : {stats['raw_max']:.1f} cm", 30, y0, small_font, (180, 180, 180))
        y0 += 28
        draw_text(screen, f"Erreur max lissée          : {stats['smooth_max']:.1f} cm", 30, y0, small_font, (255, 255, 255))
        y0 += 40

        draw_text(screen, f"Précision instantanée brute   : {raw_prec_txt}", 30, y0, small_font, (180, 180, 180))
        y0 += 28
        draw_text(screen, f"Précision instantanée lissée  : {smooth_prec_txt}", 30, y0, small_font, (255, 255, 255))
        y0 += 28
        draw_text(screen, f"Précision moyenne brute       : {avg_raw_prec_txt}", 30, y0, small_font, (180, 180, 180))
        y0 += 28
        draw_text(screen, f"Précision moyenne lissée      : {avg_smooth_prec_txt}", 30, y0, small_font, (255, 255, 255))

    pygame.display.flip()

pygame.quit()