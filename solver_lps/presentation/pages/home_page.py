import argparse
import importlib


PAGE_SPECS = {
    "home": {"label": "Accueil", "module": "solver_lps.presentation.pages.home_page", "needs_source": False},
    "udp_viewer": {"label": "UDP Viewer", "module": "solver_lps.presentation.pages.udp_viewer_page", "needs_source": True},
    "estimation_2d": {"label": "Estimation 2D", "module": "solver_lps.presentation.pages.estimation_2d_page", "needs_source": True},
    "estimation_3d": {"label": "Estimation 3D", "module": "solver_lps.presentation.pages.estimation_3d_page", "needs_source": True},
    "estimation_3d_to_2d": {
        "label": "Estimation 3D vers 2D",
        "module": "solver_lps.presentation.pages.estimation_3d_to_2d_page",
        "needs_source": True,
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Solver LPS main entrypoint.")
    parser.add_argument("--page", choices=tuple(PAGE_SPECS.keys()), default=None)
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--source", choices=("realtime", "review"), default=None)
    parser.add_argument("--uwb-log", default=None)
    parser.add_argument("--uwb-tag-log", default=None)
    parser.add_argument("--cv-log", default=None)
    parser.add_argument("--cv-calibration", default=None)
    parser.add_argument("--cv-video", default=None)
    parser.add_argument("--cv-player", default=None)
    parser.add_argument("--cv-expected-players", type=int, default=None)
    parser.add_argument("--review-data", choices=("uwb", "cv", "both"), default="both")
    parser.add_argument("--capture-output", default=None)
    return parser.parse_args(argv)


def _build_page_argv(page, args):
    page_argv = []
    if page in {"udp_viewer", "estimation_2d", "estimation_3d", "estimation_3d_to_2d"}:
        page_argv.extend(["--ip", args.ip, "--port", str(args.port)])
    if PAGE_SPECS[page]["needs_source"]:
        page_argv.extend(["--source", args.source or "realtime"])
    if page in {"estimation_2d", "estimation_3d", "estimation_3d_to_2d"}:
        if args.uwb_tag_log:
            page_argv.extend(["--uwb-tag-log", args.uwb_tag_log])
    if page == "estimation_2d":
        if args.uwb_log:
            page_argv.extend(["--uwb-log", args.uwb_log])
        if args.cv_log:
            page_argv.extend(["--cv-log", args.cv_log])
        if args.cv_calibration:
            page_argv.extend(["--cv-calibration", args.cv_calibration])
        if args.cv_video:
            page_argv.extend(["--cv-video", args.cv_video])
        if args.cv_player:
            page_argv.extend(["--cv-player", args.cv_player])
        if args.cv_expected_players is not None:
            page_argv.extend(["--cv-expected-players", str(args.cv_expected_players)])
        if args.review_data:
            page_argv.extend(["--review-data", args.review_data])
    if page == "udp_viewer":
        if args.capture_output:
            page_argv.extend(["--capture-output", args.capture_output])
        if args.uwb_log:
            page_argv.extend(["--uwb-log", args.uwb_log])
    return page_argv


def choose_page_pygame(default_source, default_review_data):
    import pygame

    selectable_pages = ["udp_viewer", "estimation_2d", "estimation_3d", "estimation_3d_to_2d"]
    pygame.init()
    desktop_info = pygame.display.Info()
    screen = pygame.display.set_mode((desktop_info.current_w, desktop_info.current_h), pygame.NOFRAME)
    pygame.display.set_caption("Solver LPS")
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("Arial", 34)
    item_font = pygame.font.SysFont("Arial", 28)
    help_font = pygame.font.SysFont("Arial", 22)

    selected = 0
    source = default_source or "realtime"
    source_cycle = ("realtime", "review")
    review_data = default_review_data or "both"
    review_data_cycle = ("uwb", "cv", "both")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit(0)
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(selectable_pages)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(selectable_pages)
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_TAB):
                    try:
                        source_index = source_cycle.index(source)
                    except ValueError:
                        source_index = 0
                    step = -1 if event.key == pygame.K_LEFT else 1
                    source = source_cycle[(source_index + step) % len(source_cycle)]
                elif event.key == pygame.K_c:
                    try:
                        review_data_index = review_data_cycle.index(review_data)
                    except ValueError:
                        review_data_index = 2
                    review_data = review_data_cycle[(review_data_index + 1) % len(review_data_cycle)]
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    pygame.quit()
                    return selectable_pages[selected], source, review_data

        screen.fill((14, 18, 28))
        header = title_font.render("Solver LPS", True, (245, 245, 245))
        screen.blit(header, (40, 32))
        subtitle = help_font.render("Choisis une page avec les fleches, puis Entree", True, (180, 200, 220))
        screen.blit(subtitle, (42, 78))
        source_text = help_font.render(f"Source pour les pages de calcul : {source}", True, (255, 210, 90))
        screen.blit(source_text, (42, 118))
        review_data_text = help_font.render(f"Review 2D : {review_data}", True, (140, 220, 255))
        screen.blit(review_data_text, (42, 148))

        y = 180
        for index, page in enumerate(selectable_pages):
            active = index == selected
            rect = pygame.Rect(36, y - 8, 780, 44)
            if active:
                pygame.draw.rect(screen, (255, 210, 60), rect, border_radius=8)
                color = (18, 18, 22)
            else:
                pygame.draw.rect(screen, (44, 54, 70), rect, 1, border_radius=8)
                color = (230, 230, 235)
            label = item_font.render(f"{index + 1}. {PAGE_SPECS[page]['label']}", True, color)
            screen.blit(label, (52, y))
            y += 58

        footer = help_font.render("Gauche/Droite/Tab : realtime/review | C : UWB/CV/les 2 | Echap : quitter", True, (160, 175, 195))
        screen.blit(footer, (42, 510))
        pygame.display.flip()
        clock.tick(60)


def main(argv=None):
    args = parse_args(argv)
    if args.page and args.page != "home":
        page = args.page
        source = args.source
    else:
        page, source, review_data = choose_page_pygame(args.source, args.review_data)
        args.review_data = review_data
    args.source = source
    module = importlib.import_module(PAGE_SPECS[page]["module"])
    return module.main(_build_page_argv(page, args))
