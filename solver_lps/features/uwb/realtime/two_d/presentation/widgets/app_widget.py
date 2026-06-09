import argparse

from ..controller import run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Estimation UWB 2D.")
    parser.add_argument("--source", choices=("realtime",), default="realtime")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--uwb-log", default=None)
    parser.add_argument("--uwb-tag-log", default=None)
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
    from . import display

    pygame.init()
    desktop_info = pygame.display.Info()
    screen = pygame.display.set_mode((desktop_info.current_w, desktop_info.current_h), pygame.NOFRAME)
    pygame.display.set_caption("Solver LPS - Estimation 2D")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28)
    small_font = pygame.font.SysFont("Arial", 22)
    try:
        run(args, pygame, screen, (font, small_font), display)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()

