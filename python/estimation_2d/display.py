import pygame

from scene import (
    CAMERA_X,
    CAMERA_Y,
    HEIGHT,
    SCALE,
    WIDTH,
    anchor_colors,
)


def world_to_screen(x_cm, y_cm):
    sx = WIDTH // 2 + int((x_cm - CAMERA_X) * SCALE)
    sy = HEIGHT // 2 + int((y_cm - CAMERA_Y) * SCALE)
    return sx, sy


def draw_text(screen, text, x, y, font, color=(255, 255, 255)):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))


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


def _fmt(value, suffix=" cm"):
    if value is None:
        return "--"
    return f"{value:.1f}{suffix}"


def draw_scene(screen, fonts, active_anchors, active_ids, state, solution, tag_real, layout_name, active_anchor_count):
    font, small_font = fonts
    screen.fill((12, 12, 20))
    draw_grid(screen)

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for aid in active_ids:
        if aid not in solution["dist_raw"]:
            continue
        ax, ay = active_anchors[aid]
        sx, sy = world_to_screen(ax, ay)
        radius = max(1, int(solution["dist_raw"][aid] * SCALE))
        c = anchor_colors[aid]
        pygame.draw.circle(overlay, (c[0], c[1], c[2], 70), (sx, sy), radius, 2)
    screen.blit(overlay, (0, 0))

    for aid in active_ids:
        if aid not in solution["dist_smooth"]:
            continue
        ax, ay = active_anchors[aid]
        sx, sy = world_to_screen(ax, ay)
        radius = max(1, int(solution["dist_smooth"][aid] * SCALE))
        pygame.draw.circle(screen, anchor_colors[aid], (sx, sy), radius, 2)

    for aid, (ax, ay) in active_anchors.items():
        sx, sy = world_to_screen(ax, ay)
        pygame.draw.circle(screen, (255, 210, 0), (sx, sy), 12)
        draw_text(screen, f"Portenta {aid}", sx + 14, sy - 12, font)

    if tag_real is not None:
        tx, ty = world_to_screen(tag_real[0], tag_real[1])
        pygame.draw.circle(screen, (255, 0, 0), (tx, ty), 11)
        draw_text(screen, "Tag reel", tx + 16, ty - 12, font, (255, 0, 0))

    if len(active_ids) == 2:
        for i, p in enumerate(solution["raw_intersections"]):
            px, py = world_to_screen(p[0], p[1])
            pygame.draw.circle(screen, (255, 0, 255), (px, py), 7)
            draw_text(screen, f"Inter brute {i + 1}", px + 12, py - 10, small_font, (255, 120, 255))
        for i, p in enumerate(solution["smooth_intersections"]):
            px, py = world_to_screen(p[0], p[1])
            pygame.draw.circle(screen, (255, 255, 255), (px, py), 7)
            draw_text(screen, f"Inter lissee {i + 1}", px + 12, py + 12, small_font)

    if solution["tag_est_raw"] is not None:
        ex, ey = world_to_screen(solution["tag_est_raw"][0], solution["tag_est_raw"][1])
        pygame.draw.circle(screen, (180, 180, 180), (ex, ey), 7)
        draw_text(screen, "Estimation brute", ex + 14, ey - 10, small_font, (180, 180, 180))

    if solution["tag_est_smooth"] is not None:
        ex, ey = world_to_screen(solution["tag_est_smooth"][0], solution["tag_est_smooth"][1])
        pygame.draw.circle(screen, (255, 255, 255), (ex, ey), 10)
        draw_text(screen, "Estimation lissee", ex + 14, ey - 10, small_font)

    draw_hud(screen, font, small_font, active_anchors, active_ids, state, solution, layout_name, active_anchor_count, tag_real)


def draw_hud(screen, font, small_font, active_anchors, active_ids, state, solution, layout_name, active_anchor_count, tag_real):
    hud_bg = pygame.Surface((650, 560), pygame.SRCALPHA)
    hud_bg.fill((0, 0, 0, 150))
    screen.blit(hud_bg, (15, 15))

    y = 25
    draw_text(screen, "UWB Estimation 2D", 30, y, font)
    y += 34
    draw_text(screen, f"Ancres actives : {active_anchor_count} | {layout_name}", 30, y, small_font, (220, 220, 220))
    y += 26
    draw_text(screen, "2..6 changer | R reset | ESC quitter", 30, y, small_font, (220, 220, 220))
    y += 26
    draw_text(screen, f"Source : {solution['status']}", 30, y, small_font, (120, 255, 210) if solution["source"] == "udp" else (220, 220, 220))
    y += 30

    for aid in active_ids:
        raw = solution["dist_raw"].get(aid)
        smooth = solution["dist_smooth"].get(aid)
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

    draw_text(screen, f"Erreur brute instantanee : {_fmt(solution['raw_err'])}", 30, y, small_font, (180, 180, 180))
    y += 25
    draw_text(screen, f"Erreur lissee instantanee : {_fmt(solution['smooth_err'])}", 30, y, small_font)
    y += 25
    draw_text(screen, f"Erreur brute moyenne : {_fmt(solution['avg_raw'])}", 30, y, small_font, (180, 180, 180))
    y += 25
    draw_text(screen, f"Erreur lissee moyenne : {_fmt(solution['avg_smooth'])}", 30, y, small_font)
    y += 25
    draw_text(screen, f"Erreur max brute : {state['stats']['raw_max']:.1f} cm", 30, y, small_font, (180, 180, 180))
    y += 25
    draw_text(screen, f"Erreur max lissee : {state['stats']['smooth_max']:.1f} cm", 30, y, small_font)
    y += 35
    draw_text(screen, f"Precision brute : {_fmt(solution['raw_precision'], ' %')}", 30, y, small_font, (180, 180, 180))
    y += 25
    draw_text(screen, f"Precision lissee : {_fmt(solution['smooth_precision'], ' %')}", 30, y, small_font)
