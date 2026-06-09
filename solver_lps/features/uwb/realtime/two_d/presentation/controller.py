from solver_lps.features.players.domain.player_analytics import player_hit_test
from solver_lps.features.players.domain.player_registry import (
    DEFAULT_PLAYER_ID,
    bind_selected_player_analytics,
    capture_selected_player_heatmap,
    select_player_profile,
    toggle_selected_player_card,
    update_registry_player,
)
from ..data.uwb_sources import DistanceSource
from ..domain.position_calcul import update_position_solution
from ..domain.scene import (
    FPS,
    create_state,
    full_reset,
    get_anchor_layout,
    load_settings,
    save_settings,
    set_active_anchor_count,
)


def run(args, pygame, screen, fonts, display_module):
    draw_scene = display_module.draw_scene
    heatmap_button_rect = display_module.heatmap_button_rect
    player_dropdown_option_rects = display_module.player_dropdown_option_rects
    player_dropdown_rect = display_module.player_dropdown_rect
    world_to_screen = display_module.world_to_screen

    clock = pygame.time.Clock()
    settings = load_settings()
    source = DistanceSource(
        bind_ip=args.ip,
        port=args.port,
        cv_log_path=args.cv_log,
        cv_calibration_path=args.cv_calibration,
        cv_video_path=args.cv_video,
        cv_player_id=args.cv_player,
        cv_expected_player_count=args.cv_expected_players,
    )
    state = create_state(settings)
    bind_selected_player_analytics(state)
    state["view"] = source.view_config
    state["player_analytics"]["source_label"] = source.source_label
    state["ui"]["player_dropdown_open"] = False
    state["ui"]["expected_player_count"] = args.cv_expected_players
    active_anchor_count = settings["active_anchor_count"]
    full_reset(state, active_anchor_count)
    last_solution = {"cv_positions": [], "all_player_ids": []}

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
                    elif event.key == pygame.K_TAB:
                        source.cycle_cv_player(-1 if (event.mod & pygame.KMOD_SHIFT) else 1)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        if state["ui"].get("expected_player_count") is not None:
                            count = max(1, int(state["ui"]["expected_player_count"]) - 1)
                            state["ui"]["expected_player_count"] = count
                            source.set_cv_expected_player_count(count)
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        if state["ui"].get("expected_player_count") is not None:
                            count = int(state["ui"]["expected_player_count"]) + 1
                            state["ui"]["expected_player_count"] = count
                            source.set_cv_expected_player_count(count)
                    elif event.key in (pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                        active_anchor_count = int(event.unicode)
                        set_active_anchor_count(settings, active_anchor_count)
                        save_settings(settings)
                        full_reset(state, active_anchor_count)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if player_dropdown_rect(screen).collidepoint(event.pos):
                        state["ui"]["player_dropdown_open"] = not state["ui"].get("player_dropdown_open", False)
                    elif state["ui"].get("player_dropdown_open", False):
                        handled_dropdown = False
                        for (_, player_id), option_rect in player_dropdown_option_rects(last_solution):
                            if option_rect.collidepoint(event.pos):
                                source.set_cv_player(player_id)
                                state["ui"]["player_dropdown_open"] = False
                                handled_dropdown = True
                                break
                        if not handled_dropdown:
                            state["ui"]["player_dropdown_open"] = False
                    elif heatmap_button_rect(screen).collidepoint(event.pos):
                        capture_selected_player_heatmap(state["player_registry"])
                        bind_selected_player_analytics(state)
                    else:
                        clicked_cv_player = None
                        for player in last_solution.get("cv_positions", []):
                            if player_hit_test(event.pos, world_to_screen(state, player["x"], player["y"])):
                                clicked_cv_player = player["player_id"]
                                break
                        if clicked_cv_player is not None:
                            source.set_cv_player(clicked_cv_player)
                            select_player_profile(state["player_registry"], clicked_cv_player, show_card=True)
                            bind_selected_player_analytics(state)
                        elif last_solution.get("tag_est_smooth") is not None:
                            if player_hit_test(event.pos, world_to_screen(state, *last_solution["tag_est_smooth"])):
                                select_player_profile(state["player_registry"], DEFAULT_PLAYER_ID)
                                toggle_selected_player_card(state["player_registry"])
                                bind_selected_player_analytics(state)

            active_anchors, layout_name = get_anchor_layout(settings, active_anchor_count)
            distance_packet = source.get_distances(None, active_anchors, settings)
            solution = update_position_solution(active_anchors, distance_packet, state, dt, tag_real=None)
            last_solution = solution

            for player in solution.get("cv_positions", []):
                update_registry_player(
                    state["player_registry"],
                    player["player_id"],
                    t=state["t"],
                    pos_xy=(player["x"], player["y"]),
                    height_cm=None,
                    jump_extra_cm=None,
                    dt=dt,
                    name=player["player_id"],
                    source_label="Tracking CV",
                )
            if solution["tag_est_smooth"] is not None:
                update_registry_player(
                    state["player_registry"],
                    DEFAULT_PLAYER_ID,
                    t=state["t"],
                    pos_xy=solution["tag_est_smooth"],
                    height_cm=None,
                    jump_extra_cm=None,
                    dt=dt,
                    name="Tag UWB",
                    source_label=source.source_label,
                )

            selected_player_id = solution.get("selected_player_id")
            if selected_player_id:
                select_player_profile(state["player_registry"], selected_player_id)
            elif solution["tag_est_smooth"] is not None and state["player_registry"].get("selected_player_id") == DEFAULT_PLAYER_ID:
                select_player_profile(state["player_registry"], DEFAULT_PLAYER_ID)
            bind_selected_player_analytics(state)
            draw_scene(screen, fonts, active_anchors, sorted(active_anchors.keys()), state, solution, None, layout_name, active_anchor_count)
            pygame.display.flip()
    finally:
        source.close()
