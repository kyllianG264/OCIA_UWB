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
from ..domain.scene import FOV_DEG, FPS, TAG_BASE_Z_CM, create_state, full_reset, get_anchor_layout, load_settings, set_active_anchor_count, save_settings


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
    source = DistanceSource(uwb_tag_review_path=args.uwb_tag_log)
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
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    full_reset(state, active_anchor_count)
                elif event.key == pygame.K_SPACE:
                    source.toggle_review_pause()
                elif event.key == pygame.K_LEFT:
                    source.seek_review_frames(-15)
                elif event.key == pygame.K_RIGHT:
                    source.seek_review_frames(15)
                elif event.key == pygame.K_PAGEUP:
                    source.seek_review_relative(-5.0)
                elif event.key == pygame.K_PAGEDOWN:
                    source.seek_review_relative(5.0)
                elif event.key in (pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                    active_anchor_count = int(event.unicode)
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
                    state["ui"]["hud_visible"] = not state["ui"].get("hud_visible", True)
                elif event.key == pygame.K_p:
                    toggle_selected_player_card(state["player_registry"])
                    bind_selected_player_analytics(state)
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
        tag_real = distance_packet.get("tag_real")
        jump_extra = 0.0 if tag_real is None else max(0.0, tag_real[2] - TAG_BASE_Z_CM)
        if tag_real is not None:
            state["real_history"].append((state["t"], tag_real))
            camera.update_target(tag_real)
        solution = update_position_solution(active_anchors, distance_packet, state, dt, tag_real)
        if solution["tag_est_smooth"] is not None:
            estimated = solution["tag_est_smooth"]
            estimated_jump_extra = max(0.0, estimated[2] - TAG_BASE_Z_CM)
            update_registry_player(state["player_registry"], DEFAULT_PLAYER_ID, t=state["t"], pos_xy=(estimated[0], estimated[1]), height_cm=estimated[2], jump_extra_cm=estimated_jump_extra, dt=dt, name="Tag UWB", source_label="Estimation UWB 3D")
            bind_selected_player_analytics(state)
            state["trail_est"].append(solution["tag_est_smooth"])
        if tag_real is not None:
            state["trail_real"].append(tag_real)
        draw_scene(screen, camera, focal, (font, small_font), active_anchors, active_ids, state, solution, tag_real, jump_extra, show_rays, show_trails, layout_name, active_anchor_count, [])
        pygame.display.flip()
    source.close()
