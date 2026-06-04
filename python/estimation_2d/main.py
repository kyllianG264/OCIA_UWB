import argparse

from player_analytics import capture_heatmap_snapshot, player_hit_test, toggle_player_card, update_player_analytics
from position_calcul import update_position_solution
from real_tag import compute_real_tag_position
from scene import FPS, HEIGHT, WIDTH, create_state, full_reset, get_anchor_layout, load_settings, save_settings, set_active_anchor_count
from uwb_sources import DistanceSource


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Estimation UWB 2D.")
    parser.add_argument("--source", choices=("simulation", "udp", "review"), default="simulation")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--uwb-log", default=None)
    parser.add_argument("--cv-log", default=None)
    parser.add_argument("--cv-calibration", default=None)
    parser.add_argument("--cv-video", default=None)
    parser.add_argument("--cv-player", default=None)
    parser.add_argument("--cv-expected-players", type=int, default=10)
    return parser.parse_args(argv)


def _playback_payload(solution):
    return solution.get("uwb_playback") or solution.get("cv_playback")


def _timeline_position_from_mouse(solution, bar_rect, mouse_pos):
    playback = _playback_payload(solution)
    if not playback:
        return None
    duration_s = float(playback.get("duration_s", 0.0) or 0.0)
    if duration_s <= 0.0:
        return 0.0
    if bar_rect.width <= 1:
        return 0.0
    ratio = (mouse_pos[0] - bar_rect.x) / float(bar_rect.width)
    ratio = max(0.0, min(1.0, ratio))
    return ratio * duration_s


def main(argv=None):
    args = parse_args(argv)
    import pygame
    from display import (
        draw_scene,
        heatmap_button_rect,
        playback_bar_rect,
        playback_button_rect,
        player_dropdown_option_rects,
        player_dropdown_rect,
        world_to_screen,
    )

    pygame.init()
    desktop_info = pygame.display.Info()
    screen = pygame.display.set_mode((desktop_info.current_w, desktop_info.current_h), pygame.NOFRAME)
    pygame.display.set_caption("UWB - Estimation 2D")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28)
    small_font = pygame.font.SysFont("Arial", 22)

    settings = load_settings()
    source = DistanceSource(
        mode=args.source,
        bind_ip=args.ip,
        port=args.port,
        uwb_review_log_path=args.uwb_log,
        cv_log_path=args.cv_log,
        cv_calibration_path=args.cv_calibration,
        cv_video_path=args.cv_video,
        cv_player_id=args.cv_player,
        cv_expected_player_count=args.cv_expected_players,
    )
    state = create_state(settings)
    state["player_analytics"]["source_label"] = source.source_label
    state["ui"]["player_dropdown_open"] = False
    state["ui"]["expected_player_count"] = args.cv_expected_players
    state["ui"]["playback_dragging"] = False
    state["ui"]["playback_drag_resume"] = False
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
                    elif event.key == pygame.K_SPACE and args.source == "review":
                        source.toggle_review_pause()
                    elif event.key == pygame.K_LEFT and args.source == "review":
                        source.seek_review_frames(-15)
                    elif event.key == pygame.K_RIGHT and args.source == "review":
                        source.seek_review_frames(15)
                    elif event.key == pygame.K_PAGEUP and args.source == "review":
                        source.seek_review_relative(-5.0)
                    elif event.key == pygame.K_PAGEDOWN and args.source == "review":
                        source.seek_review_relative(5.0)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        if state["ui"].get("expected_player_count") is not None:
                            state["ui"]["expected_player_count"] = max(1, int(state["ui"]["expected_player_count"]) - 1)
                            source.set_cv_expected_player_count(state["ui"]["expected_player_count"])
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        if state["ui"].get("expected_player_count") is not None:
                            state["ui"]["expected_player_count"] = int(state["ui"]["expected_player_count"]) + 1
                            source.set_cv_expected_player_count(state["ui"]["expected_player_count"])
                    elif event.key in (pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                        active_anchor_count = int(event.unicode)
                        set_active_anchor_count(settings, active_anchor_count)
                        save_settings(settings)
                        full_reset(state, active_anchor_count)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if args.source == "review" and playback_button_rect(screen).collidepoint(event.pos):
                        source.toggle_review_pause()
                    elif args.source == "review" and playback_bar_rect(screen).collidepoint(event.pos):
                        seek_position_s = _timeline_position_from_mouse(last_solution, playback_bar_rect(screen), event.pos)
                        if seek_position_s is not None:
                            playback = _playback_payload(last_solution) or {}
                            state["ui"]["playback_dragging"] = True
                            state["ui"]["playback_drag_resume"] = not bool(playback.get("paused", False))
                            source.set_review_video_scrubbing(True)
                            source.seek_review_absolute(seek_position_s, paused=True)
                    elif player_dropdown_rect(screen).collidepoint(event.pos):
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
                        capture_heatmap_snapshot(state["player_analytics"])
                    else:
                        clicked_cv_player = None
                        for player in last_solution.get("cv_positions", []):
                            player_screen_pos = world_to_screen(state, player["x"], player["y"])
                            if player_hit_test(event.pos, player_screen_pos):
                                clicked_cv_player = player["player_id"]
                                break
                        if clicked_cv_player is not None:
                            source.set_cv_player(clicked_cv_player)
                        elif source.uses_simulated_tag:
                            tag_real = compute_real_tag_position(state["t"])
                            if player_hit_test(event.pos, world_to_screen(state, *tag_real)):
                                toggle_player_card(state["player_analytics"])
                elif event.type == pygame.MOUSEMOTION and args.source == "review" and state["ui"].get("playback_dragging", False):
                    if not event.buttons[0]:
                        state["ui"]["playback_dragging"] = False
                        state["ui"]["playback_drag_resume"] = False
                        source.set_review_video_scrubbing(False)
                    else:
                        seek_position_s = _timeline_position_from_mouse(last_solution, playback_bar_rect(screen), event.pos)
                        if seek_position_s is not None:
                            source.seek_review_absolute(seek_position_s, paused=True)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and args.source == "review":
                    if state["ui"].get("playback_dragging", False):
                        seek_position_s = _timeline_position_from_mouse(last_solution, playback_bar_rect(screen), event.pos)
                        resume_playback = bool(state["ui"].get("playback_drag_resume", False))
                        state["ui"]["playback_dragging"] = False
                        state["ui"]["playback_drag_resume"] = False
                        source.set_review_video_scrubbing(False)
                        if seek_position_s is not None:
                            source.seek_review_absolute(seek_position_s, paused=not resume_playback)

            active_anchors, layout_name = get_anchor_layout(settings, active_anchor_count)
            tag_real = compute_real_tag_position(state["t"]) if source.uses_simulated_tag else None
            distance_packet = source.get_distances(tag_real, active_anchors, settings)
            solution = update_position_solution(active_anchors, distance_packet, state, dt, tag_real)
            last_solution = solution
            if solution.get("selection_mode") == "all":
                state["player_analytics"]["name"] = "Tous les joueurs"
            elif solution.get("selected_player_id"):
                state["player_analytics"]["name"] = solution["selected_player_id"]
            elif solution["tag_est_smooth"] is not None:
                state["player_analytics"]["name"] = "Tag UWB"
            if solution["tag_est_smooth"] is not None:
                update_player_analytics(state["player_analytics"], state["t"], solution["tag_est_smooth"], None, None, dt)
            draw_scene(screen, (font, small_font), active_anchors, sorted(active_anchors.keys()), state, solution, tag_real, layout_name, active_anchor_count)
            pygame.display.flip()
    finally:
        if args.source == "review":
            source.set_review_video_scrubbing(False)
        source.close()
        pygame.quit()


if __name__ == "__main__":
    main()
