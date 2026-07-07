import argparse
import importlib

from solver_lps.session_assets import ASSETS_DIR, DEFAULT_SET, DEFAULT_SPORT, SessionAssets
from solver_lps.presentation.navigation import RETURN_HOME


PAGE_SPECS = {
    "home": {"label": "Accueil", "module": "solver_lps.presentation.pages.home_page", "needs_source": False},
    "cv_tracking": {"label": "CV Tracking", "module": "solver_lps.presentation.pages.cv_tracking_page", "needs_source": False},
    "udp_viewer": {"label": "UDP Reader", "module": "solver_lps.presentation.pages.udp_viewer_page", "needs_source": True},
    "estimation_2d": {"label": "Estimation 2D", "module": "solver_lps.presentation.pages.estimation_2d_page", "needs_source": True},
}

SOLVER_MODES = {
    "2d": "estimation_2d",
    "3d": "estimation_2d",
    "3d_to_2d": "estimation_2d",
}

def _list_sports():
    if not ASSETS_DIR.exists():
        return ["basket"]
    sports = [item.name for item in ASSETS_DIR.iterdir() if item.is_dir() and item.name != "models"]
    return sports or ["basket"]


def _list_sets(sport):
    sport_dir = ASSETS_DIR / str(sport)
    if not sport_dir.exists():
        return ["set1"]

    def _looks_like_set_dir(path):
        if not path.is_dir():
            return False
        if path.name in {"ground", "models"} or path.name.startswith("."):
            return False
        expected_children = ("input", "output", "analysis", "uwb")
        return any((path / child).exists() for child in expected_children)

    sets = [item.name for item in sport_dir.iterdir() if _looks_like_set_dir(item)]
    return sorted(sets) or ["set1"]


def _create_set_dirs(sport, set_name):
    assets = SessionAssets(sport=sport, asset_set=set_name)
    if not assets.sport_dir.is_dir():
        raise ValueError(f"Unknown sport: {assets.sport!r}")
    return assets.ensure_directories()


def _validate_session_selection(sport, asset_set):
    assets = SessionAssets(sport=sport, asset_set=asset_set)
    if not assets.sport_dir.is_dir():
        raise ValueError(f"Unknown sport: {assets.sport!r}")
    if not assets.set_dir.is_dir():
        raise ValueError(f"Unknown asset set for {assets.sport}: {assets.asset_set!r}")
    return assets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Solver LPS main entrypoint.")
    parser.add_argument("--page", choices=tuple(PAGE_SPECS.keys()), default=None)
    parser.add_argument("--sport", default="basket")
    parser.add_argument("--asset-set", dest="asset_set", default="set1")
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
    parser.add_argument("--review-data", choices=("uwb", "cv", "both"), default="cv")
    parser.add_argument("--solver-mode", choices=tuple(SOLVER_MODES), default="2d")
    parser.add_argument("--capture-output", default=None)
    return parser.parse_args(argv)


def _build_page_argv(page, args):
    page_argv = []
    if page in {"cv_tracking", "udp_viewer", "estimation_2d"}:
        page_argv.extend(["--sport", str(args.sport), "--asset-set", str(args.asset_set)])
    if page in {"udp_viewer", "estimation_2d"}:
        page_argv.extend(["--ip", args.ip, "--port", str(args.port)])
    if PAGE_SPECS[page]["needs_source"]:
        page_argv.extend(["--source", args.source or "realtime"])
    if page == "estimation_2d":
        page_argv.extend(["--solver-mode", args.solver_mode])
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


def _solver_page_for_mode(mode):
    return SOLVER_MODES.get(mode, "estimation_2d")


def choose_page_pygame(default_source, default_review_data, default_sport=DEFAULT_SPORT, default_asset_set=DEFAULT_SET):
    import pygame

    selectable_pages = ["cv_tracking", "udp_viewer", "solver"]
    sports = _list_sports()
    pygame.init()
    desktop_info = pygame.display.Info()
    screen = pygame.display.set_mode((desktop_info.current_w, desktop_info.current_h), pygame.NOFRAME)
    pygame.display.set_caption("Solver LPS")
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("Arial", 34)
    item_font = pygame.font.SysFont("Arial", 28)
    help_font = pygame.font.SysFont("Arial", 22)

    selected = 0
    source = default_source or "review"
    source_cycle = ("realtime", "review")
    review_data = default_review_data or "cv"
    solver_mode = "2d"
    solver_mode_cycle = tuple(SOLVER_MODES)
    sport = default_sport if default_sport in sports else sports[0]
    set_options = _list_sets(sport)
    set_index = set_options.index(default_asset_set) if default_asset_set in set_options else 0
    creating_set = False
    new_set_name = ""
    open_dropdown = None

    while True:
        create_button_rect = None
        sport_rect = pygame.Rect(screen.get_width() - 380, 28, 320, 44)
        set_rect = pygame.Rect(screen.get_width() - 380, 84, 320, 44)
        create_button_rect = pygame.Rect(screen.get_width() - 380, 140, 320, 40)
        source_rect = pygame.Rect(42, 108, 260, 40)
        cv_rect = pygame.Rect(320, 108, 110, 40)
        uwb_rect = pygame.Rect(440, 108, 110, 40)
        solver_mode_rect = pygame.Rect(580, 108, 220, 40)
        sport_option_rects = []
        set_option_rects = []
        source_option_rects = []
        solver_mode_option_rects = []
        if open_dropdown == "sport":
            for index, option in enumerate(sports):
                sport_option_rects.append((option, pygame.Rect(sport_rect.x, sport_rect.bottom + 6 + index * 40, sport_rect.width, 36)))
        elif open_dropdown == "set":
            for index, option in enumerate(set_options):
                set_option_rects.append((option, pygame.Rect(set_rect.x, set_rect.bottom + 6 + index * 40, set_rect.width, 36)))
        elif open_dropdown == "source":
            for index, option in enumerate(source_cycle):
                source_option_rects.append((option, pygame.Rect(source_rect.x, source_rect.bottom + 6 + index * 40, source_rect.width, 36)))
        elif open_dropdown == "solver_mode":
            for index, option in enumerate(solver_mode_cycle):
                solver_mode_option_rects.append((option, pygame.Rect(solver_mode_rect.x, solver_mode_rect.bottom + 6 + index * 40, solver_mode_rect.width, 36)))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if creating_set:
                    continue
                if source_rect.collidepoint(event.pos):
                    open_dropdown = None if open_dropdown == "source" else "source"
                    continue
                if solver_mode_rect.collidepoint(event.pos):
                    open_dropdown = None if open_dropdown == "solver_mode" else "solver_mode"
                    continue
                if cv_rect.collidepoint(event.pos):
                    review_data = "both" if review_data == "uwb" else "cv"
                    open_dropdown = None
                    continue
                if uwb_rect.collidepoint(event.pos):
                    review_data = "both" if review_data == "cv" else "uwb"
                    open_dropdown = None
                    continue
                if sport_rect.collidepoint(event.pos):
                    open_dropdown = None if open_dropdown == "sport" else "sport"
                    continue
                if set_rect.collidepoint(event.pos):
                    open_dropdown = None if open_dropdown == "set" else "set"
                    continue
                if open_dropdown == "source":
                    clicked_option = next((option for option, rect in source_option_rects if rect.collidepoint(event.pos)), None)
                    if clicked_option is not None:
                        source = clicked_option
                    open_dropdown = None
                    continue
                if open_dropdown == "solver_mode":
                    clicked_option = next((option for option, rect in solver_mode_option_rects if rect.collidepoint(event.pos)), None)
                    if clicked_option is not None:
                        solver_mode = clicked_option
                    open_dropdown = None
                    continue
                if open_dropdown == "sport":
                    clicked_option = next((option for option, rect in sport_option_rects if rect.collidepoint(event.pos)), None)
                    if clicked_option is not None:
                        sport = clicked_option
                        set_options = _list_sets(sport)
                        set_index = 0
                    open_dropdown = None
                    continue
                if open_dropdown == "set":
                    clicked_option = next((option for option, rect in set_option_rects if rect.collidepoint(event.pos)), None)
                    if clicked_option is not None and clicked_option in set_options:
                        set_index = set_options.index(clicked_option)
                    open_dropdown = None
                    continue
                if create_button_rect is not None and create_button_rect.collidepoint(event.pos):
                    creating_set = True
                    new_set_name = ""
                    open_dropdown = None
                else:
                    open_dropdown = None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit(0)
                if creating_set:
                    if event.key == pygame.K_RETURN:
                        candidate = new_set_name.strip()
                        if candidate:
                            _create_set_dirs(sport, candidate)
                            set_options = _list_sets(sport)
                            set_index = set_options.index(candidate) if candidate in set_options else 0
                        creating_set = False
                        new_set_name = ""
                    elif event.key == pygame.K_BACKSPACE:
                        new_set_name = new_set_name[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        creating_set = False
                        new_set_name = ""
                    elif event.unicode and event.unicode.isprintable():
                        new_set_name += event.unicode
                    continue
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
                    review_data = "both" if review_data == "uwb" else "cv"
                elif event.key == pygame.K_u:
                    review_data = "both" if review_data == "cv" else "uwb"
                elif event.key == pygame.K_m:
                    solver_mode = "2d"
                elif event.key == pygame.K_n:
                    creating_set = True
                    new_set_name = ""
                    open_dropdown = None
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    pygame.quit()
                    chosen_set = set_options[set_index] if set_options else "set1"
                    page = selectable_pages[selected]
                    if page == "solver":
                        page = _solver_page_for_mode(solver_mode)
                    return page, source, review_data, sport, chosen_set, solver_mode

        screen.fill((14, 18, 28))
        header = title_font.render("Solver LPS", True, (245, 245, 245))
        screen.blit(header, (40, 32))
        subtitle = help_font.render("Choisis une page puis Enter", True, (180, 200, 220))
        screen.blit(subtitle, (42, 78))
        pygame.draw.rect(screen, (18, 23, 31), source_rect, border_radius=10)
        pygame.draw.rect(screen, (120, 200, 255), source_rect, 2, border_radius=10)
        source_text = help_font.render(f"Mode : {source} v", True, (220, 245, 255))
        screen.blit(source_text, (source_rect.x + 14, source_rect.y + 9))
        cv_active = review_data in {"cv", "both"}
        uwb_active = review_data in {"uwb", "both"}
        for rect, label, active, color in (
            (cv_rect, "CV", cv_active, (140, 220, 255)),
            (uwb_rect, "UWB", uwb_active, (255, 210, 90)),
        ):
            pygame.draw.rect(screen, (18, 23, 31), rect, border_radius=10)
            pygame.draw.rect(screen, color if active else (76, 92, 112), rect, 2, border_radius=10)
            button_text = help_font.render(f"{label} {'x' if active else 'o'}", True, color if active else (210, 220, 230))
            screen.blit(button_text, (rect.x + 20, rect.y + 9))
        pygame.draw.rect(screen, (18, 23, 31), solver_mode_rect, border_radius=10)
        pygame.draw.rect(screen, (255, 170, 120), solver_mode_rect, 2, border_radius=10)
        solver_mode_text = help_font.render(f"Solver : {solver_mode.replace('_', ' ')} v", True, (255, 220, 190))
        screen.blit(solver_mode_text, (solver_mode_rect.x + 14, solver_mode_rect.y + 9))
        chosen_set = set_options[set_index] if set_options else "set1"
        pygame.draw.rect(screen, (18, 23, 31), sport_rect, border_radius=10)
        pygame.draw.rect(screen, (120, 200, 255), sport_rect, 2, border_radius=10)
        pygame.draw.rect(screen, (18, 23, 31), set_rect, border_radius=10)
        pygame.draw.rect(screen, (255, 210, 90), set_rect, 2, border_radius=10)
        pygame.draw.rect(screen, (18, 23, 31), create_button_rect, border_radius=10)
        pygame.draw.rect(screen, (140, 220, 255), create_button_rect, 2, border_radius=10)
        sport_text = help_font.render(f"Changer sport : {sport} v", True, (220, 245, 255))
        set_text = help_font.render(f"Set actuel : {chosen_set} v", True, (255, 235, 180))
        create_text = help_font.render("Creer un set", True, (220, 245, 255))
        screen.blit(sport_text, (sport_rect.x + 14, sport_rect.y + 10))
        screen.blit(set_text, (set_rect.x + 14, set_rect.y + 10))
        screen.blit(create_text, (create_button_rect.x + 14, create_button_rect.y + 8))
        for option, rect in sport_option_rects:
            active = option == sport
            pygame.draw.rect(screen, (18, 23, 31), rect, border_radius=8)
            pygame.draw.rect(screen, (120, 200, 255) if active else (76, 92, 112), rect, 2, border_radius=8)
            option_text = help_font.render(option, True, (220, 245, 255) if active else (210, 220, 230))
            screen.blit(option_text, (rect.x + 12, rect.y + 7))
        for option, rect in set_option_rects:
            active = option == chosen_set
            pygame.draw.rect(screen, (18, 23, 31), rect, border_radius=8)
            pygame.draw.rect(screen, (255, 210, 90) if active else (76, 92, 112), rect, 2, border_radius=8)
            option_text = help_font.render(option, True, (255, 235, 180) if active else (210, 220, 230))
            screen.blit(option_text, (rect.x + 12, rect.y + 7))

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
            label_text = "Solver" if page == "solver" else PAGE_SPECS[page]["label"]
            label = item_font.render(f"{index + 1}. {label_text}", True, color)
            screen.blit(label, (52, y))
            y += 58

        footer = help_font.render("Mode source, cases CV/UWB, solver 2D | Clique : sport/set | Echap : quitter", True, (160, 175, 195))
        screen.blit(footer, (42, 510))
        create_hint = help_font.render("Bouton en haut a droite : creer un set", True, (160, 175, 195))
        screen.blit(create_hint, (42, 540))
        if open_dropdown == "source":
            for option, rect in source_option_rects:
                active = option == source
                pygame.draw.rect(screen, (18, 23, 31), rect, border_radius=8)
                pygame.draw.rect(screen, (120, 200, 255) if active else (76, 92, 112), rect, 2, border_radius=8)
                option_text = help_font.render(option, True, (220, 245, 255) if active else (210, 220, 230))
                screen.blit(option_text, (rect.x + 12, rect.y + 7))
        elif open_dropdown == "solver_mode":
            for option, rect in solver_mode_option_rects:
                active = option == solver_mode
                pygame.draw.rect(screen, (18, 23, 31), rect, border_radius=8)
                pygame.draw.rect(screen, (255, 170, 120) if active else (76, 92, 112), rect, 2, border_radius=8)
                option_text = help_font.render(option, True, (255, 220, 190) if active else (210, 220, 230))
                screen.blit(option_text, (rect.x + 12, rect.y + 7))
        if creating_set:
            modal = pygame.Rect(screen.get_width() // 2 - 260, screen.get_height() // 2 - 70, 520, 140)
            pygame.draw.rect(screen, (18, 23, 31), modal, border_radius=12)
            pygame.draw.rect(screen, (255, 210, 90), modal, 2, border_radius=12)
            title = help_font.render(f"Nouveau set pour {sport}", True, (245, 245, 245))
            screen.blit(title, (modal.x + 18, modal.y + 18))
            prompt = help_font.render("Nom du set puis Entree", True, (180, 200, 220))
            screen.blit(prompt, (modal.x + 18, modal.y + 48))
            value_rect = pygame.Rect(modal.x + 18, modal.y + 80, modal.width - 36, 36)
            pygame.draw.rect(screen, (12, 16, 22), value_rect, border_radius=8)
            pygame.draw.rect(screen, (120, 200, 255), value_rect, 2, border_radius=8)
            value = help_font.render(new_set_name or "set2", True, (220, 245, 255))
            screen.blit(value, (value_rect.x + 10, value_rect.y + 6))
        pygame.display.flip()
        clock.tick(60)


def main(argv=None):
    args = parse_args(argv)
    while True:
        if args.page and args.page != "home":
            page = args.page
            source = args.source
        else:
            page, source, review_data, sport, asset_set, solver_mode = choose_page_pygame(
                args.source,
                args.review_data,
                args.sport,
                args.asset_set,
            )
            args.review_data = review_data
            args.sport = sport
            args.asset_set = asset_set
            args.page = page
            args.solver_mode = solver_mode
        args.source = source
        _validate_session_selection(args.sport, args.asset_set)
        module = importlib.import_module(PAGE_SPECS[page]["module"])
        result = module.main(_build_page_argv(page, args))
        if result != RETURN_HOME:
            return result
        args.page = None
