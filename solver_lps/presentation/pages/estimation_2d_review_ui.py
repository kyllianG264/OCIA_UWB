import argparse

from solver_lps.features.players.domain.player_analytics import player_hit_test
from solver_lps.features.players.domain.player_registry import (
    DEFAULT_PLAYER_ID,
    bind_selected_player_analytics,
    capture_selected_player_heatmap,
    reset_player_registry,
    select_player_profile,
    set_registry_bounds,
    toggle_selected_player_card,
    update_registry_players_from_positions,
)
from solver_lps.features.uwb.calculus.domain.two_d.position_calcul import update_position_solution
from solver_lps.presentation.pages.estimation_2d_review_source import DistanceSource
from solver_lps.presentation.navigation import ANALYZE_GRAY_ZONE, REGENERATE_MERGED, RETURN_HOME
from solver_lps.presentation.pages.estimation_2d_review_scene import (
    FPS,
    create_state,
    full_reset,
    get_anchor_layout,
    load_settings,
    save_settings,
    set_active_anchor_count,
)


_PRESENTATION_PACKET_KEYS = (
    "review_mode",
    "cv_positions",
    "selected_player_id",
    "selection_mode",
    "all_player_ids",
    "selected_player_visible",
    "cv_status",
    "cv_video_frame",
    "cv_left_video_frame",
    "cv_right_video_frame",
    "cv_video_available",
    "cv_playback",
    "uwb_playback",
    "uwb_merged_review",
    "uwb_view_mode",
    "uwb_raw_available",
    "uwb_merged_available",
)


def compose_review_solution(solution, distance_packet):
    for key in _PRESENTATION_PACKET_KEYS:
        if key in distance_packet:
            solution[key] = distance_packet[key]
    return solution


def analytics_sample(state, source_name, playback):
    if not playback:
        return False, 0.0, 0.0
    frame_index = int(playback.get("frame_index", -1))
    position_s = float(playback.get("position_s", 0.0))
    samples = state["ui"].setdefault("analytics_samples", {})
    previous = samples.get(source_name)
    current = (frame_index, position_s)
    if previous == current:
        return False, position_s, 0.0
    if previous is not None and position_s < previous[1]:
        reset_player_registry(state["player_registry"])
        samples.clear()
        previous = None
    samples[source_name] = current
    dt = 0.0 if previous is None else max(0.0, position_s - previous[1])
    return True, position_s, dt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Estimation UWB 2D review UI.")
    parser.add_argument("--source", choices=("review",), default="review")
    parser.add_argument("--sport", default="basket")
    parser.add_argument("--asset-set", dest="asset_set", default="set1")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--uwb-log", default=None)
    parser.add_argument("--uwb-tag-log", default=None)
    parser.add_argument("--uwb-merged", default=None)
    parser.add_argument("--cv-log", default=None)
    parser.add_argument("--cv-calibration", default=None)
    parser.add_argument("--cv-video", default=None)
    parser.add_argument("--cv-player", default=None)
    parser.add_argument("--cv-expected-players", type=int, default=None)
    parser.add_argument("--review-data", choices=("uwb", "cv", "both"), default="cv")
    parser.add_argument("--solver-mode", choices=("2d", "3d", "3d_to_2d"), default="2d")
    return parser.parse_args(argv)


def _playback_payload(solution):
    return solution.get("uwb_playback") or solution.get("cv_playback")


def _has_cv_mode(solution):
    return solution.get("review_mode") in {"cv", "both"}


def _has_uwb_mode(solution):
    return solution.get("review_mode") in {"uwb", "both"}


def _has_player_mode(solution):
    return solution.get("review_mode") in {"cv", "uwb", "both"}


def _view_mode_prefix(solution):
    return "uwb" if solution.get("review_mode") == "uwb" else "cv"


def _resolve_registry_bounds(source, state):
    analytics_bounds = getattr(source, "analytics_bounds", None)
    if analytics_bounds is not None:
        return analytics_bounds
    view = state.get("view") or {}
    return view.get("bounds")


def _timeline_position_from_mouse(solution, bar_rect, mouse_pos):
    playback = _playback_payload(solution)
    if not playback:
        return None
    duration_s = float(playback.get("duration_s", 0.0) or 0.0)
    if duration_s <= 0.0 or bar_rect.width <= 1:
        return 0.0
    ratio = (mouse_pos[0] - bar_rect.x) / float(bar_rect.width)
    ratio = max(0.0, min(1.0, ratio))
    return ratio * duration_s


def run(args, pygame, screen, fonts):
    from . import estimation_2d_review_display as display

    draw_scene = display.draw_scene
    heatmap_button_rect = display.heatmap_button_rect
    cv_view_mode_button_rects = display.cv_view_mode_button_rects
    playback_bar_rect = display.playback_bar_rect
    playback_button_rect = display.playback_button_rect
    player_dropdown_option_rects = display.player_dropdown_option_rects
    player_dropdown_rect = display.player_dropdown_rect
    world_to_screen = display.world_to_screen

    clock = pygame.time.Clock()
    settings = load_settings(args.sport)
    source = DistanceSource(
        uwb_review_log_path=args.uwb_log,
        uwb_tag_review_path=args.uwb_tag_log,
        uwb_merged_path=args.uwb_merged,
        cv_log_path=args.cv_log,
        cv_calibration_path=args.cv_calibration,
        cv_video_path=args.cv_video,
        cv_player_id=args.cv_player,
        cv_expected_player_count=args.cv_expected_players,
        review_data_mode=args.review_data,
        sport=args.sport,
        asset_set=args.asset_set,
        solver_mode=args.solver_mode,
    )
    state = create_state(settings, args.sport)
    bind_selected_player_analytics(state)
    state["view"] = source.view_config
    set_registry_bounds(state["player_registry"], _resolve_registry_bounds(source, state))
    state["player_analytics"]["source_label"] = source.source_label
    state["ui"]["player_dropdown_open"] = False
    state["ui"]["expected_player_count"] = args.cv_expected_players
    state["ui"]["playback_dragging"] = False
    state["ui"]["playback_drag_resume"] = False
    state["ui"]["show_gray_zone"] = True
    state["ui"]["show_gray_theory"] = True
    state["ui"]["show_gray_observations"] = True
    state["ui"]["show_gray_disappearances"] = True
    active_anchor_count = settings["active_anchor_count"]
    full_reset(state, active_anchor_count)
    last_solution = {"cv_positions": [], "all_player_ids": [], "review_mode": args.review_data}
    current_tag_real = None
    back_rect = pygame.Rect(20, screen.get_height() - 58, 190, 40)
    regenerate_rect = pygame.Rect(220, screen.get_height() - 58, 230, 40)
    gray_zone_rect = pygame.Rect(460, screen.get_height() - 58, 250, 40)
    theory_rect = pygame.Rect(720, screen.get_height() - 58, 150, 40)
    observations_rect = pygame.Rect(880, screen.get_height() - 58, 190, 40)
    disappearances_rect = pygame.Rect(1080, screen.get_height() - 58, 190, 40)
    return_home = False
    regenerate_merged = False
    analyze_gray_zone = False

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
                        return_home = True
                        running = False
                    elif event.key == pygame.K_r:
                        full_reset(state, active_anchor_count)
                    elif event.key == pygame.K_h:
                        state["ui"]["hud_visible"] = not state["ui"].get("hud_visible", True)
                    elif event.key == pygame.K_TAB:
                        if _has_player_mode(last_solution):
                            source.cycle_cv_player(-1 if (event.mod & pygame.KMOD_SHIFT) else 1)
                    elif event.key == pygame.K_SPACE:
                        source.toggle_review_pause()
                    elif event.key == pygame.K_g:
                        state["ui"]["show_gray_zone"] = not state["ui"].get("show_gray_zone", True)
                    elif event.key == pygame.K_LEFT:
                        source.seek_review_frames(-15)
                    elif event.key == pygame.K_RIGHT:
                        source.seek_review_frames(15)
                    elif event.key == pygame.K_PAGEUP:
                        source.seek_review_relative(-5.0)
                    elif event.key == pygame.K_PAGEDOWN:
                        source.seek_review_relative(5.0)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        if _has_cv_mode(last_solution) and state["ui"].get("expected_player_count") is not None:
                            count = max(1, int(state["ui"]["expected_player_count"]) - 1)
                            state["ui"]["expected_player_count"] = count
                            source.set_cv_expected_player_count(count)
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        if _has_cv_mode(last_solution) and state["ui"].get("expected_player_count") is not None:
                            count = int(state["ui"]["expected_player_count"]) + 1
                            state["ui"]["expected_player_count"] = count
                            source.set_cv_expected_player_count(count)
                    elif event.key in (pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                        if _has_uwb_mode(last_solution):
                            active_anchor_count = int(event.unicode)
                            set_active_anchor_count(settings, active_anchor_count)
                            save_settings(settings, args.sport)
                            full_reset(state, active_anchor_count)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if back_rect.collidepoint(event.pos):
                        return_home = True
                        running = False
                    elif regenerate_rect.collidepoint(event.pos):
                        regenerate_merged = True
                        running = False
                    elif (
                        _has_cv_mode(last_solution)
                        and state["view"].get("gray_zone_input_available", False)
                        and gray_zone_rect.collidepoint(event.pos)
                    ):
                        analyze_gray_zone = True
                        running = False
                    elif _has_cv_mode(last_solution) and theory_rect.collidepoint(event.pos):
                        state["ui"]["show_gray_theory"] = not state["ui"].get("show_gray_theory", True)
                    elif _has_cv_mode(last_solution) and observations_rect.collidepoint(event.pos):
                        state["ui"]["show_gray_observations"] = not state["ui"].get("show_gray_observations", True)
                    elif _has_cv_mode(last_solution) and disappearances_rect.collidepoint(event.pos):
                        state["ui"]["show_gray_disappearances"] = not state["ui"].get("show_gray_disappearances", True)
                    elif playback_button_rect(screen).collidepoint(event.pos):
                        source.toggle_review_pause()
                    elif playback_bar_rect(screen).collidepoint(event.pos):
                        seek_position_s = _timeline_position_from_mouse(last_solution, playback_bar_rect(screen), event.pos)
                        if seek_position_s is not None:
                            playback = _playback_payload(last_solution) or {}
                            state["ui"]["playback_dragging"] = True
                            state["ui"]["playback_drag_resume"] = not bool(playback.get("paused", False))
                            source.set_review_video_scrubbing(True)
                            source.seek_review_absolute(seek_position_s, paused=True)
                    elif any(
                        rect.collidepoint(event.pos)
                        and last_solution.get(f"{_view_mode_prefix(last_solution)}_{mode}_available", False)
                        for mode, rect in cv_view_mode_button_rects(screen).items()
                    ):
                        mode_prefix = _view_mode_prefix(last_solution)
                        clicked_mode = next(
                            (
                                mode
                                for mode, rect in cv_view_mode_button_rects(screen).items()
                                if rect.collidepoint(event.pos)
                                and last_solution.get(f"{mode_prefix}_{mode}_available", False)
                            ),
                            None,
                        )
                        if clicked_mode is not None:
                            source.set_review_view_mode(clicked_mode)
                            continue
                    elif _has_player_mode(last_solution) and player_dropdown_rect(screen).collidepoint(event.pos):
                        state["ui"]["player_dropdown_open"] = not state["ui"].get("player_dropdown_open", False)
                    elif _has_player_mode(last_solution) and state["ui"].get("player_dropdown_open", False):
                        handled_dropdown = False
                        for (_, player_id), option_rect in player_dropdown_option_rects(last_solution):
                            if option_rect.collidepoint(event.pos):
                                source.set_cv_player(player_id)
                                state["ui"]["player_dropdown_open"] = False
                                handled_dropdown = True
                                break
                        if not handled_dropdown:
                            state["ui"]["player_dropdown_open"] = False
                    elif _has_player_mode(last_solution) and heatmap_button_rect(screen).collidepoint(event.pos):
                        capture_selected_player_heatmap(state["player_registry"])
                        bind_selected_player_analytics(state)
                    else:
                        clicked_cv_player = None
                        if _has_player_mode(last_solution):
                            for player in last_solution.get("cv_positions", []):
                                if player_hit_test(event.pos, world_to_screen(state, player["x"], player["y"])):
                                    clicked_cv_player = player["player_id"]
                                    break
                        if clicked_cv_player is not None:
                            source.set_cv_player(clicked_cv_player)
                            select_player_profile(state["player_registry"], clicked_cv_player, show_card=True)
                            bind_selected_player_analytics(state)
                        elif _has_uwb_mode(last_solution) and current_tag_real is not None:
                            if player_hit_test(event.pos, world_to_screen(state, *current_tag_real)):
                                select_player_profile(state["player_registry"], DEFAULT_PLAYER_ID)
                                toggle_selected_player_card(state["player_registry"])
                                bind_selected_player_analytics(state)
                elif event.type == pygame.MOUSEMOTION and state["ui"].get("playback_dragging", False):
                    if not event.buttons[0]:
                        state["ui"]["playback_dragging"] = False
                        state["ui"]["playback_drag_resume"] = False
                        source.set_review_video_scrubbing(False)
                    else:
                        seek_position_s = _timeline_position_from_mouse(last_solution, playback_bar_rect(screen), event.pos)
                        if seek_position_s is not None:
                            source.seek_review_absolute(seek_position_s, paused=True)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if state["ui"].get("playback_dragging", False):
                        seek_position_s = _timeline_position_from_mouse(last_solution, playback_bar_rect(screen), event.pos)
                        resume_playback = bool(state["ui"].get("playback_drag_resume", False))
                        state["ui"]["playback_dragging"] = False
                        state["ui"]["playback_drag_resume"] = False
                        source.set_review_video_scrubbing(False)
                        if seek_position_s is not None:
                            source.seek_review_absolute(seek_position_s, paused=not resume_playback)

            active_anchors, layout_name = get_anchor_layout(settings, active_anchor_count)
            distance_packet = source.get_distances(None, active_anchors, settings)
            review_tag_real = distance_packet.get("tag_real")
            tag_real = (review_tag_real[0], review_tag_real[1]) if review_tag_real is not None and len(review_tag_real) >= 2 else None
            current_tag_real = tag_real
            solution = update_position_solution(active_anchors, distance_packet, state, dt, tag_real)
            compose_review_solution(solution, distance_packet)
            solution["sport"] = args.sport
            solution["cv_view_mode"] = distance_packet.get("cv_view_mode", "merged")
            solution["cv_raw_available"] = distance_packet.get("cv_raw_available", False)
            solution["cv_merged_available"] = distance_packet.get("cv_merged_available", False)
            last_solution = solution
            set_registry_bounds(state["player_registry"], _resolve_registry_bounds(source, state))

            player_playback = solution.get("cv_playback") or solution.get("uwb_playback")
            player_source = "cv" if solution.get("cv_playback") else "uwb"
            player_sample, player_time, player_dt = analytics_sample(state, player_source, player_playback)
            if _has_player_mode(solution) and player_sample:
                update_registry_players_from_positions(
                    state["player_registry"],
                    solution.get("cv_positions", []),
                    t=player_time,
                    dt=player_dt,
                    source_label="Players merged",
                )

            selected_player_id = solution.get("selected_player_id")
            if _has_player_mode(solution) and selected_player_id:
                select_player_profile(
                    state["player_registry"],
                    selected_player_id,
                    show_card=solution.get("selection_mode", "single") != "all",
                )
            elif _has_uwb_mode(solution) and solution["tag_est_smooth"] is not None and state["player_registry"].get("selected_player_id") == DEFAULT_PLAYER_ID:
                select_player_profile(state["player_registry"], DEFAULT_PLAYER_ID)
            bind_selected_player_analytics(state)
            draw_scene(screen, fonts, active_anchors, sorted(active_anchors.keys()), state, solution, tag_real, layout_name, active_anchor_count)
            pygame.draw.rect(screen, (18, 30, 42), back_rect, border_radius=8)
            pygame.draw.rect(screen, (90, 185, 245), back_rect, 2, border_radius=8)
            back_label = fonts[1].render("Retour accueil", True, (225, 245, 255))
            screen.blit(back_label, back_label.get_rect(center=back_rect.center))
            pygame.draw.rect(screen, (42, 34, 16), regenerate_rect, border_radius=8)
            pygame.draw.rect(screen, (255, 196, 70), regenerate_rect, 2, border_radius=8)
            regenerate_label = fonts[1].render("Regenerer merged", True, (255, 225, 155))
            screen.blit(regenerate_label, regenerate_label.get_rect(center=regenerate_rect.center))
            if _has_cv_mode(solution):
                gray_available = state["view"].get("gray_zone_input_available", False)
                gray_border = (180, 185, 195) if gray_available else (70, 76, 86)
                gray_text = (235, 238, 242) if gray_available else (120, 126, 136)
                pygame.draw.rect(screen, (28, 31, 36), gray_zone_rect, border_radius=8)
                pygame.draw.rect(screen, gray_border, gray_zone_rect, 2, border_radius=8)
                label = "Analyser gray zone" if gray_available else "Raw gray zone absent"
                gray_label = fonts[1].render(label, True, gray_text)
                screen.blit(gray_label, gray_label.get_rect(center=gray_zone_rect.center))
                for rect, label, enabled, color in (
                    (theory_rect, "Theorie", state["ui"].get("show_gray_theory", True), (175, 180, 190)),
                    (
                        observations_rect,
                        "Observations",
                        state["ui"].get("show_gray_observations", True),
                        (70, 235, 110),
                    ),
                    (
                        disappearances_rect,
                        "Disparitions",
                        state["ui"].get("show_gray_disappearances", True),
                        (255, 185, 55),
                    ),
                ):
                    pygame.draw.rect(screen, (28, 31, 36), rect, border_radius=8)
                    pygame.draw.rect(screen, color if enabled else (70, 76, 86), rect, 2, border_radius=8)
                    toggle_label = fonts[1].render(f"{label} {'on' if enabled else 'off'}", True, color if enabled else (120, 126, 136))
                    screen.blit(toggle_label, toggle_label.get_rect(center=rect.center))
            pygame.display.flip()
    finally:
        source.set_review_video_scrubbing(False)
        source.close()
    if return_home:
        return RETURN_HOME
    if regenerate_merged:
        return REGENERATE_MERGED
    if analyze_gray_zone:
        return ANALYZE_GRAY_ZONE
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.cv_expected_players is None:
        args.cv_expected_players = 12 if str(args.sport or "").strip().lower() == "volley" else 10
    import pygame

    pygame.init()
    desktop_info = pygame.display.Info()
    screen = pygame.display.set_mode((desktop_info.current_w, desktop_info.current_h), pygame.NOFRAME)
    pygame.display.set_caption("Solver LPS - Estimation 2D")
    font = pygame.font.SysFont("Arial", 28)
    small_font = pygame.font.SysFont("Arial", 22)
    try:
        return run(args, pygame, screen, (font, small_font))
    finally:
        pygame.quit()
