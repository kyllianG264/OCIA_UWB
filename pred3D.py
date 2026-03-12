import pygame
import math
import random
from collections import deque

# =========================================================
# SCENE / WINDOW
# =========================================================
WIDTH, HEIGHT = 1600, 950
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
# IMPORTANT:
# For a real 3D solve, anchors must NOT all be coplanar.
# If all anchors have the same Z, the height of the tag
# becomes ambiguous from distances only.
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
# 0.5% mean scale -> use proportional sigma
# =========================================================
NOISE_RATIO = 0.005       # 0.5%
SPIKE_PROB = 0.010        # rare bad packet
SPIKE_AMPLITUDE = 40.0    # cm

ALPHA_DIST = 0.16
ALPHA_POS = 0.12

PRECISION_TOLERANCE_CM = 100.0

# =========================================================
# CAMERA
# =========================================================
FOV_DEG = 60
NEAR_PLANE = 5.0
CAMERA_YAW = -0.55
CAMERA_PITCH = 0.55
CAMERA_DISTANCE = 3700.0

# =========================================================
# TRAILS
# =========================================================
TRAIL_MAX_POINTS = 200

# =========================================================
# SMALL MATH TOOLS
# =========================================================
def clamp(v, a, b):
    return max(a, min(b, v))


def lerp(a, b, t):
    return a + (b - a) * t


def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


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


def distance_3d(a, b):
    return length(vec_sub(b, a))


def distance_xy(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def smooth_value(old_val, new_val, alpha):
    if old_val is None:
        return new_val
    return alpha * new_val + (1 - alpha) * old_val


def smooth_point3(old_point, new_point, alpha):
    if new_point is None:
        return old_point
    if old_point is None:
        return new_point
    return (
        alpha * new_point[0] + (1 - alpha) * old_point[0],
        alpha * new_point[1] + (1 - alpha) * old_point[1],
        alpha * new_point[2] + (1 - alpha) * old_point[2],
    )


def error_to_precision_percent(err_cm, tolerance_cm=PRECISION_TOLERANCE_CM):
    if err_cm is None:
        return None
    p = 100.0 * (1.0 - err_cm / tolerance_cm)
    return clamp(p, 0.0, 100.0)


def noisy_distance(true_d):
    sigma = max(1e-6, true_d * NOISE_RATIO)
    d = true_d + random.gauss(0, sigma)
    if random.random() < SPIKE_PROB:
        d += random.uniform(-SPIKE_AMPLITUDE, SPIKE_AMPLITUDE)
    return max(1.0, d)


def det3(m00, m01, m02, m10, m11, m12, m20, m21, m22):
    return (
        m00 * (m11 * m22 - m12 * m21)
        - m01 * (m10 * m22 - m12 * m20)
        + m02 * (m10 * m21 - m11 * m20)
    )


def solve_linear_3x3(M, V):
    m00, m01, m02 = M[0]
    m10, m11, m12 = M[1]
    m20, m21, m22 = M[2]
    v0, v1, v2 = V

    detM = det3(m00, m01, m02, m10, m11, m12, m20, m21, m22)
    if abs(detM) < 1e-9:
        return None

    dx = det3(v0, m01, m02, v1, m11, m12, v2, m21, m22) / detM
    dy = det3(m00, v0, m02, m10, v1, m12, m20, v2, m22) / detM
    dz = det3(m00, m01, v0, m10, m11, v1, m20, m21, v2) / detM
    return (dx, dy, dz)


def estimate_position_3d_least_squares(anchor_positions, distances):
    ids = sorted(anchor_positions.keys())
    if len(ids) < 4:
        return None

    ref_id = ids[0]
    x1, y1, z1 = anchor_positions[ref_id]
    r1 = distances[ref_id]

    rows = []
    vals = []

    for aid in ids[1:]:
        xi, yi, zi = anchor_positions[aid]
        ri = distances[aid]

        a = 2.0 * (xi - x1)
        b = 2.0 * (yi - y1)
        c = 2.0 * (zi - z1)
        d = (
            (r1 * r1 - ri * ri)
            - (x1 * x1 - xi * xi)
            - (y1 * y1 - yi * yi)
            - (z1 * z1 - zi * zi)
        )

        rows.append((a, b, c))
        vals.append(d)

    s_aa = sum(a * a for a, b, c in rows)
    s_ab = sum(a * b for a, b, c in rows)
    s_ac = sum(a * c for a, b, c in rows)
    s_bb = sum(b * b for a, b, c in rows)
    s_bc = sum(b * c for a, b, c in rows)
    s_cc = sum(c * c for a, b, c in rows)

    s_ad = sum(a * d for (a, b, c), d in zip(rows, vals))
    s_bd = sum(b * d for (a, b, c), d in zip(rows, vals))
    s_cd = sum(c * d for (a, b, c), d in zip(rows, vals))

    M = [
        [s_aa, s_ab, s_ac],
        [s_ab, s_bb, s_bc],
        [s_ac, s_bc, s_cc],
    ]
    V = [s_ad, s_bd, s_cd]

    return solve_linear_3x3(M, V)


def reset_stats():
    return {
        "err3d_sum": 0.0,
        "errxy_sum": 0.0,
        "errz_sum": 0.0,
        "prec_sum": 0.0,
        "count": 0,
        "err3d_max": 0.0,
        "errz_max": 0.0,
    }


def get_anchor_layout(n):
    if n == 4:
        return {
            1: (LEFT_X, TOP_Y, ANCHOR_Z_LOW),
            2: (RIGHT_X, TOP_Y, ANCHOR_Z_LOW),
            3: (RIGHT_X, BOTTOM_Y, ANCHOR_Z_LOW),
            4: (LEFT_X, BOTTOM_Y, ANCHOR_Z_HIGH),
        }, "4 ancres : carré avec A4 plus haute"

    if n == 5:
        return {
            1: (LEFT_X, TOP_Y, ANCHOR_Z_LOW),
            2: (RIGHT_X, TOP_Y, ANCHOR_Z_LOW),
            3: (RIGHT_X, BOTTOM_Y, ANCHOR_Z_LOW),
            4: (LEFT_X, BOTTOM_Y, ANCHOR_Z_HIGH),
            5: (CENTER_X, TOP_MID_Y, ANCHOR_Z_MID),
        }, "5 ancres : carré + pointe avant"

    return {
        1: (LEFT_X, TOP_Y, ANCHOR_Z_LOW),
        2: (RIGHT_X, TOP_Y, ANCHOR_Z_LOW),
        3: (RIGHT_X, BOTTOM_Y, ANCHOR_Z_LOW),
        4: (LEFT_X, BOTTOM_Y, ANCHOR_Z_HIGH),
        5: (CENTER_X, TOP_MID_Y, ANCHOR_Z_MID),
        6: (CENTER_X, BOTTOM_MID_Y, ANCHOR_Z_MID_HIGH),
    }, "6 ancres : carré + pointes avant/arrière"


def start_jump(state):
    state["jump_active"] = True
    state["jump_timer"] = 0.0
    state["jump_duration"] = random.uniform(JUMP_DURATION_MIN, JUMP_DURATION_MAX)
    state["jump_extra_height"] = JUMP_EXTRA_HEIGHT_CM * random.uniform(0.90, 1.10)


def update_jump(state, dt):
    if not state["jump_active"]:
        if random.random() < AUTO_JUMP_PROB_PER_SEC * dt:
            start_jump(state)
        return 0.0

    state["jump_timer"] += dt
    p = state["jump_timer"] / state["jump_duration"]

    if p >= 1.0:
        state["jump_active"] = False
        state["jump_timer"] = 0.0
        return 0.0

    return state["jump_extra_height"] * math.sin(math.pi * p)


# =========================================================
# CAMERA / PROJECTION
# =========================================================
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


# =========================================================
# DRAW HELPERS
# =========================================================
def draw_text(surface, text, x, y, font, color=HUD_TEXT):
    surf = font.render(text, True, color)
    surface.blit(surf, (x, y))


def draw_line_3d(surface, camera, p0, p1, color, focal, width=1):
    s0 = camera.project(p0, focal)
    s1 = camera.project(p1, focal)
    if s0 is None or s1 is None:
        return
    pygame.draw.line(
        surface,
        color,
        (int(s0[0]), int(s0[1])),
        (int(s1[0]), int(s1[1])),
        width,
    )


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
        p0 = (x, WORLD_MIN_Y, 0.0)
        p1 = (x, WORLD_MAX_Y, 0.0)
        draw_line_3d(surface, camera, p0, p1, GRID_COLOR, focal, 1)

    for y in range(WORLD_MIN_Y, WORLD_MAX_Y + 1, 200):
        p0 = (WORLD_MIN_X, y, 0.0)
        p1 = (WORLD_MAX_X, y, 0.0)
        draw_line_3d(surface, camera, p0, p1, GRID_COLOR, focal, 1)

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


def full_reset(state, anchor_count):
    state["t"] = 0.0
    state["stats"] = reset_stats()
    state["tag_est_smooth"] = None
    state["trail_real"].clear()
    state["trail_est"].clear()

    anchors, _ = get_anchor_layout(anchor_count)
    state["dist_smooth"] = {aid: None for aid in anchors.keys()}

    state["jump_active"] = False
    state["jump_timer"] = 0.0
    state["jump_duration"] = 0.0
    state["jump_extra_height"] = 0.0


# =========================================================
# INIT
# =========================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("UWB 3D - scene réaliste")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 24)
small_font = pygame.font.SysFont("Arial", 18)

focal = (WIDTH / 2) / math.tan(math.radians(FOV_DEG) / 2)

anchor_colors = {
    1: (90, 185, 255),
    2: (120, 255, 140),
    3: (255, 175, 70),
    4: (195, 130, 255),
    5: (255, 105, 145),
    6: (95, 230, 230),
}

camera = OrbitCamera()

state = {
    "t": 0.0,
    "stats": reset_stats(),
    "dist_smooth": {},
    "tag_est_smooth": None,
    "trail_real": deque(maxlen=TRAIL_MAX_POINTS),
    "trail_est": deque(maxlen=TRAIL_MAX_POINTS),
    "jump_active": False,
    "jump_timer": 0.0,
    "jump_duration": 0.0,
    "jump_extra_height": 0.0,
}

active_anchor_count = 6
show_rays = True
show_trails = True

mouse_orbit = False
last_mouse = (0, 0)

full_reset(state, active_anchor_count)

running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    state["t"] += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                full_reset(state, active_anchor_count)
            elif event.key == pygame.K_SPACE:
                start_jump(state)
            elif event.key == pygame.K_4:
                active_anchor_count = 4
                full_reset(state, active_anchor_count)
            elif event.key == pygame.K_5:
                active_anchor_count = 5
                full_reset(state, active_anchor_count)
            elif event.key == pygame.K_6:
                active_anchor_count = 6
                full_reset(state, active_anchor_count)
            elif event.key == pygame.K_t:
                show_trails = not show_trails
            elif event.key == pygame.K_l:
                show_rays = not show_rays
            elif event.key == pygame.K_f:
                camera.follow_tag = not camera.follow_tag

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_orbit = True
                last_mouse = event.pos
            elif event.button == 4:
                camera.distance = clamp(camera.distance - 180, 1400, 7000)
            elif event.button == 5:
                camera.distance = clamp(camera.distance + 180, 1400, 7000)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_orbit = False

        elif event.type == pygame.MOUSEMOTION and mouse_orbit:
            mx, my = event.pos
            dx = mx - last_mouse[0]
            dy = my - last_mouse[1]
            last_mouse = event.pos

            camera.yaw -= dx * 0.006
            camera.pitch += dy * 0.004
            camera.pitch = clamp(camera.pitch, 0.12, 1.25)

    active_anchors, layout_name = get_anchor_layout(active_anchor_count)
    active_ids = sorted(active_anchors.keys())

    if set(state["dist_smooth"].keys()) != set(active_ids):
        state["dist_smooth"] = {aid: None for aid in active_ids}

    # -----------------------------------------------------
    # TRUE TAG MOTION IN 3D
    # -----------------------------------------------------
    tag_x = CENTER_X + 420 * math.cos(state["t"] * 0.55)
    tag_y = CENTER_Y + 280 * math.sin(state["t"] * 0.44)

    jump_extra = update_jump(state, dt)
    tag_z = TAG_BASE_Z_CM + jump_extra

    tag_real = (tag_x, tag_y, tag_z)
    camera.update_target(tag_real)

    # -----------------------------------------------------
    # DISTANCES
    # -----------------------------------------------------
    dist_raw = {}
    dist_smooth = {}

    for aid, a_pos in active_anchors.items():
        d_true = distance_3d(tag_real, a_pos)
        d_raw = noisy_distance(d_true)
        dist_raw[aid] = d_raw
        state["dist_smooth"][aid] = smooth_value(state["dist_smooth"][aid], d_raw, ALPHA_DIST)
        dist_smooth[aid] = state["dist_smooth"][aid]

    # -----------------------------------------------------
    # 3D SOLVE
    # -----------------------------------------------------
    tag_est_raw = estimate_position_3d_least_squares(active_anchors, dist_raw)
    tag_est_from_smooth = estimate_position_3d_least_squares(active_anchors, dist_smooth)
    state["tag_est_smooth"] = smooth_point3(state["tag_est_smooth"], tag_est_from_smooth, ALPHA_POS)
    tag_est_smooth = state["tag_est_smooth"]

    err3d = None
    errxy = None
    errz = None
    avg_err3d = None
    avg_errxy = None
    avg_errz = None
    precision = None
    avg_precision = None

    if tag_est_smooth is not None:
        err3d = distance_3d(tag_real, tag_est_smooth)
        errxy = distance_xy(tag_real, tag_est_smooth)
        errz = abs(tag_real[2] - tag_est_smooth[2])
        precision = error_to_precision_percent(err3d)

        st = state["stats"]
        st["err3d_sum"] += err3d
        st["errxy_sum"] += errxy
        st["errz_sum"] += errz
        st["prec_sum"] += precision
        st["count"] += 1
        st["err3d_max"] = max(st["err3d_max"], err3d)
        st["errz_max"] = max(st["errz_max"], errz)

        avg_err3d = st["err3d_sum"] / st["count"]
        avg_errxy = st["errxy_sum"] / st["count"]
        avg_errz = st["errz_sum"] / st["count"]
        avg_precision = st["prec_sum"] / st["count"]

    state["trail_real"].append(tag_real)
    if tag_est_smooth is not None:
        state["trail_est"].append(tag_est_smooth)

    # -----------------------------------------------------
    # DRAW SCENE
    # -----------------------------------------------------
    screen.fill(BG_COLOR)

    # soft sky gradient
    sky = pygame.Surface((WIDTH, HEIGHT))
    for yy in range(HEIGHT):
        k = yy / HEIGHT
        c = int(22 + 18 * (1 - k))
        pygame.draw.line(sky, (c, c + 2, c + 8), (0, yy), (WIDTH, yy))
    screen.blit(sky, (0, 0))

    draw_ground_grid(screen, camera, focal)

    # area center marker
    draw_line_3d(screen, camera, (CENTER_X - 50, CENTER_Y, 0), (CENTER_X + 50, CENTER_Y, 0), (115, 125, 145), focal, 2)
    draw_line_3d(screen, camera, (CENTER_X, CENTER_Y - 50, 0), (CENTER_X, CENTER_Y + 50, 0), (115, 125, 145), focal, 2)

    if show_trails:
        draw_trail(screen, camera, state["trail_real"], (255, 70, 70), focal, 3)
        draw_trail(screen, camera, state["trail_est"], (235, 235, 255), focal, 2)

    # shadows
    draw_shadow(screen, camera, tag_real, focal, alpha=85)
    if tag_est_smooth is not None:
        draw_shadow(screen, camera, tag_est_smooth, focal, alpha=50)

    # anchors
    for aid in active_ids:
        draw_anchor(screen, camera, active_anchors[aid], aid, anchor_colors[aid], focal, small_font)

    # rays from estimate to anchors
    if show_rays and tag_est_smooth is not None:
        for aid in active_ids:
            draw_line_3d(screen, camera, active_anchors[aid], tag_est_smooth, (115, 135, 150), focal, 1)

    # real tag
    draw_circle_3d(screen, camera, tag_real, 34, (255, 70, 70), focal, 0)
    draw_circle_3d(screen, camera, tag_real, 18, (255, 200, 200), focal, 0)

    # estimated tag
    if tag_est_raw is not None:
        draw_circle_3d(screen, camera, tag_est_raw, 18, (130, 130, 130), focal, 0)

    if tag_est_smooth is not None:
        draw_circle_3d(screen, camera, tag_est_smooth, 28, (255, 255, 255), focal, 0)
        draw_circle_3d(screen, camera, tag_est_smooth, 15, (165, 220, 255), focal, 0)
        draw_line_3d(screen, camera, tag_real, tag_est_smooth, (255, 225, 80), focal, 2)

    # labels
    p_real = camera.project(tag_real, focal)
    if p_real is not None:
        draw_text(screen, "Stella réelle", int(p_real[0] + 12), int(p_real[1] - 18), small_font, (255, 100, 100))

    if tag_est_smooth is not None:
        p_est = camera.project(tag_est_smooth, focal)
        if p_est is not None:
            draw_text(screen, "Estimation 3D", int(p_est[0] + 12), int(p_est[1] - 18), small_font, (245, 245, 255))

    # -----------------------------------------------------
    # HUD
    # -----------------------------------------------------
    hud = pygame.Surface((660, 420), pygame.SRCALPHA)
    hud.fill((0, 0, 0, 145))
    screen.blit(hud, (16, 16))

    y = 28
    draw_text(screen, "UWB 3D - scène réaliste", 30, y, font)
    y += 34
    draw_text(screen, f"Ancres actives : {active_anchor_count}   |   {layout_name}", 30, y, small_font, HUD_SUB)
    y += 26
    draw_text(screen, "Contrôles : souris gauche orbite | molette zoom | 4/5/6 ancres | ESPACE saut | R reset | F follow | L rayons | T traces | ESC quitter", 30, y, small_font, HUD_SUB)
    y += 34

    draw_text(screen, f"Hauteur Stella réelle : {tag_real[2]:.1f} cm", 30, y, small_font, (255, 130, 130) if state["jump_active"] else (180, 255, 180))
    y += 24
    if tag_est_smooth is not None:
        draw_text(screen, f"Hauteur Stella estimée : {tag_est_smooth[2]:.1f} cm", 30, y, small_font, (245, 245, 255))
    else:
        draw_text(screen, "Hauteur Stella estimée : --", 30, y, small_font, (245, 245, 255))
    y += 24
    draw_text(screen, f"Extra saut actuel : {jump_extra:.1f} cm   |   Saut en cours : {'OUI' if state['jump_active'] else 'NON'}", 30, y, small_font, HUD_SUB)
    y += 32

    if err3d is not None:
        draw_text(screen, f"Erreur 3D instantanée : {err3d:.1f} cm", 30, y, small_font)
        y += 24
        draw_text(screen, f"Erreur XY instantanée : {errxy:.1f} cm", 30, y, small_font)
        y += 24
        draw_text(screen, f"Erreur Z instantanée : {errz:.1f} cm", 30, y, small_font)
        y += 24
        draw_text(screen, f"Précision instantanée : {precision:.1f} %", 30, y, small_font)
        y += 30
        draw_text(screen, f"Erreur 3D moyenne : {avg_err3d:.1f} cm   |   max : {state['stats']['err3d_max']:.1f} cm", 30, y, small_font, HUD_SUB)
        y += 24
        draw_text(screen, f"Erreur XY moyenne : {avg_errxy:.1f} cm", 30, y, small_font, HUD_SUB)
        y += 24
        draw_text(screen, f"Erreur Z moyenne : {avg_errz:.1f} cm   |   max Z : {state['stats']['errz_max']:.1f} cm", 30, y, small_font, HUD_SUB)
        y += 24
        draw_text(screen, f"Précision moyenne : {avg_precision:.1f} %", 30, y, small_font, HUD_SUB)
        y += 30
    else:
        draw_text(screen, "Pas assez d'information pour estimer la 3D.", 30, y, small_font, (255, 180, 130))
        y += 24
        draw_text(screen, "Il faut au moins 4 ancres non coplanaires.", 30, y, small_font, (255, 180, 130))
        y += 30

    for aid in active_ids:
        x, y_anchor, z = active_anchors[aid]
        d_raw = dist_raw[aid]
        d_sm = dist_smooth[aid]
        draw_text(
            screen,
            f"A{aid}  pos=({x:.0f}, {y_anchor:.0f}, {z:.0f}) cm   brut={d_raw:.1f} cm   lissé={d_sm:.1f} cm",
            30,
            y,
            small_font,
            anchor_colors[aid],
        )
        y += 22

    pygame.display.flip()

pygame.quit()
