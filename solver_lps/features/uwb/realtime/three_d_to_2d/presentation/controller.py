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
from ..domain.scene import FPS, create_state, full_reset, get_anchor_layout, load_settings, save_settings, set_active_anchor_count


def run(args, pygame, screen, fonts, display_module):
    draw_scene = display_module.draw_scene
    heatmap_button_rect = display_module.heatmap_button_rect
    world_to_screen = display_module.world_to_screen

    clock = pygame.time.Clock()
    settings = load_settings()
    state = create_state(settings)
    bind_selected_player_analytics(state)
    source = DistanceSource(bind_ip=args.ip, port=args.port)
    active_anchor_count = settings["active_anchor_count"]
    full_reset(state, active_anchor_count)
    running = True
    try:
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
                    elif event.key == pygame.K_h:
                        state["ui"]["hud_visible"] = not state["ui"].get("hud_visible", True)
                    elif event.key in (pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                        active_anchor_count = int(event.unicode)
                        set_active_anchor_count(settings, active_anchor_count)
                        save_settings(settings)
                        full_reset(state, active_anchor_count)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if heatmap_button_rect().collidepoint(event.pos):
                        capture_selected_player_heatmap(state["player_registry"])
                        bind_selected_player_analytics(state)
                    elif state.get("tag_est_smooth") is not None:
                        if player_hit_test(event.pos, world_to_screen(*state["tag_est_smooth"])):
                            toggle_selected_player_card(state["player_registry"])
                            bind_selected_player_analytics(state)
            active_anchors, layout_name = get_anchor_layout(settings, active_anchor_count)
            active_ids = sorted(active_anchors.keys())
            distance_packet = source.get_distances(None, None, active_anchors, settings)
            solution = update_position_solution(active_anchors, distance_packet, state, dt, None)
            if solution["tag_est_smooth"] is not None:
                update_registry_player(state["player_registry"], DEFAULT_PLAYER_ID, t=state["t"], pos_xy=solution["tag_est_smooth"], height_cm=None, jump_extra_cm=None, dt=dt, name="Tag UWB", source_label="Estimation UWB 3D -> 2D")
                bind_selected_player_analytics(state)
            draw_scene(screen, fonts, active_anchors, active_ids, state, solution, None, settings["tag_base_height_cm"], 0.0, layout_name, active_anchor_count)
            pygame.display.flip()
    finally:
        source.close()
