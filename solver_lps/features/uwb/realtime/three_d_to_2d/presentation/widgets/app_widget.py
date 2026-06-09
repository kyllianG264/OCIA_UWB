import argparse

from ..controller import run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Estimation 3D vers 2D.")
    parser.add_argument("--source", choices=("realtime",), default="realtime")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--uwb-tag-log", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import pygame
    from . import display

    pygame.init()
    screen = pygame.display.set_mode((display.WIDTH, display.HEIGHT))
    pygame.display.set_caption("Solver LPS - Estimation 3D vers 2D")
    font = pygame.font.SysFont("Arial", 28)
    small_font = pygame.font.SysFont("Arial", 22)
    try:
        run(args, pygame, screen, (font, small_font), display)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()

