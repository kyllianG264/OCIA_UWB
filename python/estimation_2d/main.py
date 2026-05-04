import argparse

import pygame

from display import draw_scene
from position_calcul import update_position_solution
from real_tag import compute_real_tag_position
from scene import (
    FPS,
    HEIGHT,
    WIDTH,
    create_state,
    full_reset,
    get_anchor_layout,
    load_settings,
    save_settings,
    set_active_anchor_count,
)
from uwb_sources import DistanceSource


def parse_args():
    parser = argparse.ArgumentParser(description="Estimation UWB 2D.")
    parser.add_argument("--source", choices=("simulation", "udp"), default="simulation")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    return parser.parse_args()


def main():
    args = parse_args()
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
                    elif event.key in (pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                        active_anchor_count = int(event.unicode)
                        set_active_anchor_count(settings, active_anchor_count)
                        save_settings(settings)
                        full_reset(state, active_anchor_count)

            active_anchors, layout_name = get_anchor_layout(settings, active_anchor_count)
            active_ids = sorted(active_anchors.keys())

            tag_real = compute_real_tag_position(state["t"]) if source.uses_simulated_tag else None
            distance_packet = source.get_distances(tag_real, active_anchors, settings)
            solution = update_position_solution(active_anchors, distance_packet, state, tag_real)

            draw_scene(
                screen=screen,
                fonts=(font, small_font),
                active_anchors=active_anchors,
                active_ids=active_ids,
                state=state,
                solution=solution,
                tag_real=tag_real,
                layout_name=layout_name,
                active_anchor_count=active_anchor_count,
            )
            pygame.display.flip()
    finally:
        source.close()
        pygame.quit()


if __name__ == "__main__":
    main()
