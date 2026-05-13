import argparse

from player_analytics import capture_heatmap_snapshot, player_hit_test, toggle_player_card, update_player_analytics
from position_calcul import update_position_solution
from real_tag import compute_real_tag_position
from scene import FPS, HEIGHT, WIDTH, create_state, full_reset, get_anchor_layout, load_settings, save_settings, set_active_anchor_count
from uwb_sources import DistanceSource


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Estimation UWB 2D.")
    parser.add_argument("--source", choices=("simulation", "udp"), default="simulation")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import pygame
    from display import draw_scene, heatmap_button_rect, world_to_screen

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("UWB - Estimation 2D")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28)
    small_font = pygame.font.SysFont("Arial", 22)

    settings = load_settings()
    state = create_state(settings)
    source = DistanceSource(mode=args.source, bind_ip=args.ip, port=args.port)
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
                        capture_heatmap_snapshot(state["player_analytics"])
                    elif source.uses_simulated_tag:
                        tag_real = compute_real_tag_position(state["t"])
                        if player_hit_test(event.pos, world_to_screen(*tag_real)):
                            toggle_player_card(state["player_analytics"])
            active_anchors, layout_name = get_anchor_layout(settings, active_anchor_count)
            active_ids = sorted(active_anchors.keys())
            tag_real = compute_real_tag_position(state["t"]) if source.uses_simulated_tag else None
            distance_packet = source.get_distances(tag_real, active_anchors, settings)
            solution = update_position_solution(active_anchors, distance_packet, state, dt, tag_real)
            if solution["tag_est_smooth"] is not None:
                update_player_analytics(state["player_analytics"], state["t"], solution["tag_est_smooth"], None, None, dt)
            draw_scene(screen, (font, small_font), active_anchors, active_ids, state, solution, tag_real, layout_name, active_anchor_count)
            pygame.display.flip()
    finally:
        source.close()
        pygame.quit()


if __name__ == "__main__":
    main()
