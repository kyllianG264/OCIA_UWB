import argparse

from ..controller import run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Estimation UWB 3D.")
    parser.add_argument("--source", choices=("review",), default="review")
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
    pygame.display.set_caption("Solver LPS - Estimation 3D")
    font = pygame.font.SysFont("Arial", 28)
    small_font = pygame.font.SysFont("Arial", 20)
    try:
        run(args=args, pygame=pygame, screen=screen, fonts=(font, small_font), display_module=display)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()

