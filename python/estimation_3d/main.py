import math
import os
import sys

import pygame

sys.path.append(os.getcwd())

from display import OrbitCamera, clamp, draw_scene
from position_calcul import update_position_solution
from real_tag import compute_real_tag_position, start_jump
from scene import (
    FOV_DEG,
    FPS,
    HEIGHT,
    WIDTH,
    adjust_anchor,
    adjust_setting,
    create_state,
    full_reset,
    get_anchor_layout,
    load_settings,
    refresh_runtime_state,
    save_settings,
    set_active_anchor_count,
)
from uwb_sources import get_simulated_distances


def build_control_items(settings):
    count = settings["active_anchor_count"]
    layout = settings["layouts"][str(count)]

    items = [
        {"label": "Lissage distances", "kind": "setting", "key": "alpha_dist", "step": 0.01, "fast_step": 0.05},
        {"label": "Lissage position", "kind": "setting", "key": "alpha_pos", "step": 0.01, "fast_step": 0.05},
        {"label": "Tolerance precision", "kind": "setting", "key": "precision_tolerance_cm", "step": 1.0, "fast_step": 10.0},
        {"label": "Bruit relatif", "kind": "setting", "key": "noise_ratio", "step": 0.001, "fast_step": 0.005},
        {"label": "Probabilite spike", "kind": "setting", "key": "spike_prob", "step": 0.01, "fast_step": 0.05},
        {"label": "Amplitude spike", "kind": "setting", "key": "spike_amplitude", "step": 5.0, "fast_step": 25.0},
    ]

    axis_names = ("X", "Y", "Z")
    axis_steps = (10.0, 10.0, 5.0)
    axis_fast_steps = (50.0, 50.0, 25.0)
    for anchor_id in sorted(int(aid) for aid in layout["anchors"].keys()):
        for axis_index, axis_name in enumerate(axis_names):
            items.append({
                "label": f"Ancre {anchor_id} {axis_name}",
                "kind": "anchor",
                "anchor_count": count,
                "anchor_id": anchor_id,
                "axis_index": axis_index,
                "step": axis_steps[axis_index],
                "fast_step": axis_fast_steps[axis_index],
            })
    return items


def apply_control_delta(state, delta, fast=False):
    settings = state["settings"]
    controls = build_control_items(settings)
    if not controls:
        return False

    ui = state["ui"]
    ui["selected_control"] = max(0, min(ui["selected_control"], len(controls) - 1))
    control = controls[ui["selected_control"]]
    step = control["fast_step"] if fast else control["step"]
    signed_step = delta * step

    if control["kind"] == "setting":
        adjust_setting(settings, control["key"], signed_step)
    else:
        adjust_anchor(settings, control["anchor_count"], control["anchor_id"], control["axis_index"], signed_step)

    save_settings(settings)
    ui["cache_dirty"] = False
    ui["cache_message"] = "Cache mis a jour"
    return True


def move_control_selection(state, direction):
    controls = build_control_items(state["settings"])
    if not controls:
        state["ui"]["selected_control"] = 0
        return
    state["ui"]["selected_control"] = (state["ui"]["selected_control"] + direction) % len(controls)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("UWB 3D - architecture par features")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 24)
    small_font = pygame.font.SysFont("Arial", 18)
    focal = (WIDTH / 2) / math.tan(math.radians(FOV_DEG) / 2)

    camera = OrbitCamera()
    settings = load_settings()
    state = create_state(settings)

    active_anchor_count = settings["active_anchor_count"]
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
                mods = pygame.key.get_mods()
                fast = bool(mods & pygame.KMOD_SHIFT)

                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    full_reset(state, active_anchor_count)
                elif event.key == pygame.K_SPACE:
                    start_jump(state)
                elif event.key == pygame.K_4:
                    active_anchor_count = 4
                    set_active_anchor_count(state["settings"], active_anchor_count)
                    save_settings(state["settings"])
                    full_reset(state, active_anchor_count)
                elif event.key == pygame.K_5:
                    active_anchor_count = 5
                    set_active_anchor_count(state["settings"], active_anchor_count)
                    save_settings(state["settings"])
                    full_reset(state, active_anchor_count)
                elif event.key == pygame.K_6:
                    active_anchor_count = 6
                    set_active_anchor_count(state["settings"], active_anchor_count)
                    save_settings(state["settings"])
                    full_reset(state, active_anchor_count)
                elif event.key == pygame.K_t:
                    show_trails = not show_trails
                elif event.key == pygame.K_l:
                    show_rays = not show_rays
                elif event.key == pygame.K_f:
                    camera.follow_tag = not camera.follow_tag
                elif event.key == pygame.K_h:
                    state["ui"]["hud_expanded"] = not state["ui"]["hud_expanded"]
                elif event.key == pygame.K_UP:
                    move_control_selection(state, -1)
                elif event.key == pygame.K_DOWN:
                    move_control_selection(state, 1)
                elif event.key == pygame.K_LEFT:
                    if apply_control_delta(state, -1, fast=fast):
                        refresh_runtime_state(state, active_anchor_count)
                elif event.key == pygame.K_RIGHT:
                    if apply_control_delta(state, 1, fast=fast):
                        refresh_runtime_state(state, active_anchor_count)

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

        active_anchors, layout_name = get_anchor_layout(state["settings"], active_anchor_count)
        active_ids = sorted(active_anchors.keys())

        tag_real, jump_extra = compute_real_tag_position(state["t"], state, dt)
        state["real_history"].append((state["t"], tag_real))
        camera.update_target(tag_real)

        distance_packet = get_simulated_distances(tag_real, active_anchors, state["settings"])
        solution = update_position_solution(active_anchors, distance_packet, state, dt, tag_real)

        state["trail_real"].append(tag_real)
        if solution["tag_est_smooth"] is not None:
            state["trail_est"].append(solution["tag_est_smooth"])

        draw_scene(
            screen=screen,
            camera=camera,
            focal=focal,
            fonts=(font, small_font),
            active_anchors=active_anchors,
            active_ids=active_ids,
            state=state,
            solution=solution,
            tag_real=tag_real,
            jump_extra=jump_extra,
            show_rays=show_rays,
            show_trails=show_trails,
            layout_name=layout_name,
            active_anchor_count=active_anchor_count,
            control_items=build_control_items(state["settings"]),
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
