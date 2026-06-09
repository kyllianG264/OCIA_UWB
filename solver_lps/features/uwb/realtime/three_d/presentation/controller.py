import math

from solver_lps.features.players.domain.player_analytics import player_hit_test
from solver_lps.features.players.domain.player_registry import (
    DEFAULT_PLAYER_ID,
    bind_selected_player_analytics,
    capture_selected_player_heatmap,
    toggle_selected_player_card,
    update_registry_player,
)
from ..data.uwb_sources import DistanceSource
from ..domain.position_calcul import update_position_solution
from ..domain.real_tag import start_jump
from ..domain.scene import (
    FOV_DEG,
    FPS,
    TAG_BASE_Z_CM,
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
            items.append({"label": f"Ancre {anchor_id} {axis_name}", "kind": "anchor", "anchor_count": count, "anchor_id": anchor_id, "axis_index": axis_index, "step": axis_steps[axis_index], "fast_step": axis_fast_steps[axis_index]})
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
    ui["cache_dirty"] = True
    ui["cache_message"] = "Cache mis a jour"
    return True


def move_control_selection(state, direction):
    controls = build_control_items(state["settings"])
    if controls:
        state["ui"]["selected_control"] = (state["ui"]["selected_control"] + direction) % len(controls)


def run(args, pygame, screen, fonts, display_module):
    OrbitCamera = display_module.OrbitCamera
    clamp = display_module.clamp
    draw_scene = display_module.draw_scene
    heatmap_button_rect = display_module.heatmap_button_rect

    clock = pygame.time.Clock()
    font, small_font = fonts
    focal = (screen.get_width() / 2.0) / math.tan(math.radians(FOV_DEG) / 2.0)
    camera = OrbitCamera()
    settings = load_settings()
    state = create_state(settings)
    bind_selected_player_analytics(state)
    source = DistanceSource(bind_ip=args.ip, port=args.port)
    active_anchor_count = settings["active_anchor_count"]
    show_rays = True
    show_trails = True
    mouse_orbit = False
    last_mouse = (0, 0)
    mouse_down_pos = None
    running = True
    full_reset(state, active_anchor_count)

    while running:
        dt = clock.tick(FPS) / 1000.0
        state["t"] += dt
        for event in pygame.event.get():
            mods = pygame.key.get_mods()
            fast = bool(mods & pygame.KMOD_SHIFT)
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    full_reset(state, active_anchor_count)
                elif event.key in (pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                    active_anchor_count = int(event.unicode)
                    set_active_anchor_count(state["settings"], active_anchor_count)
                    save_settings(state["settings"])
                    full_reset(state, active_anchor_count)
                elif event.key == pygame.K_SPACE:
                    start_jump(state)
                elif event.key == pygame.K_t:
                    show_trails = not show_trails
                elif event.key == pygame.K_l:
                    show_rays = not show_rays
                elif event.key == pygame.K_f:
                    camera.follow_tag = not camera.follow_tag
                elif event.key == pygame.K_h:
                    state["ui"]["hud_visible"] = not state["ui"].get("hud_visible", True)
                elif event.key == pygame.K_p:
                    toggle_selected_player_card(state["player_registry"])
                    bind_selected_player_analytics(state)
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
                    if heatmap_button_rect().collidepoint(event.pos):
                        capture_selected_player_heatmap(state["player_registry"])
                        bind_selected_player_analytics(state)
                    else:
                        mouse_orbit = True
                        last_mouse = event.pos
                        mouse_down_pos = event.pos
                elif event.button == 4:
                    camera.distance = clamp(camera.distance - 180, 1400, 7000)
                elif event.button == 5:
                    camera.distance = clamp(camera.distance + 180, 1400, 7000)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if mouse_down_pos is not None:
                    dx = event.pos[0] - mouse_down_pos[0]
                    dy = event.pos[1] - mouse_down_pos[1]
                    if dx * dx + dy * dy <= 81:
                        estimated = state.get("tag_est_smooth")
                        projected = camera.project(estimated, focal) if estimated is not None else None
                        if projected is not None and player_hit_test(event.pos, (projected[0], projected[1]), radius_px=44):
                            toggle_selected_player_card(state["player_registry"])
                            bind_selected_player_analytics(state)
                mouse_orbit = False
                mouse_down_pos = None
            elif event.type == pygame.MOUSEMOTION and mouse_orbit:
                mx, my = event.pos
                dx = mx - last_mouse[0]
                dy = my - last_mouse[1]
                last_mouse = event.pos
                camera.yaw -= dx * 0.006
                camera.pitch = clamp(camera.pitch + dy * 0.004, 0.12, 1.25)

        active_anchors, layout_name = get_anchor_layout(state["settings"], active_anchor_count)
        active_ids = sorted(active_anchors.keys())
        distance_packet = source.get_distances(active_anchors)
        solution = update_position_solution(active_anchors, distance_packet, state, dt, None)
        if solution["tag_est_smooth"] is not None:
            estimated = solution["tag_est_smooth"]
            estimated_jump_extra = max(0.0, estimated[2] - TAG_BASE_Z_CM)
            update_registry_player(
                state["player_registry"],
                DEFAULT_PLAYER_ID,
                t=state["t"],
                pos_xy=(estimated[0], estimated[1]),
                height_cm=estimated[2],
                jump_extra_cm=estimated_jump_extra,
                dt=dt,
                name="Tag UWB",
                source_label="Estimation UWB 3D",
            )
            bind_selected_player_analytics(state)
            state["trail_est"].append(solution["tag_est_smooth"])
        draw_scene(screen, camera, focal, (font, small_font), active_anchors, active_ids, state, solution, None, 0.0, show_rays, show_trails, layout_name, active_anchor_count, build_control_items(state["settings"]))
        pygame.display.flip()
    source.close()
