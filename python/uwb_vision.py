import argparse
import importlib.util
import inspect
import os
import sys
from contextlib import contextmanager


def _is_python_root(path):
    return os.path.isfile(os.path.join(path, "uwb_vision.py")) and os.path.isfile(
        os.path.join(path, "estimation_2d", "main.py")
    )


def _candidate_paths():
    candidates = []
    if "__file__" in globals():
        candidates.append(os.path.abspath(os.path.dirname(__file__)))
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0:
        candidates.append(os.path.abspath(os.path.dirname(argv0)))
        candidates.append(os.path.abspath(argv0))
    source_file = inspect.getsourcefile(lambda: 0)
    if source_file:
        candidates.append(os.path.abspath(os.path.dirname(source_file)))
    cwd = os.path.abspath(os.getcwd())
    candidates.append(cwd)
    candidates.append(os.path.join(cwd, "python"))
    return candidates


def _resolve_root():
    seen = set()
    for candidate in _candidate_paths():
        current = candidate
        while current and current not in seen:
            seen.add(current)
            if _is_python_root(current):
                return current
            child = os.path.join(current, "python")
            if _is_python_root(child):
                return child
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    return os.path.abspath(os.path.join(os.getcwd(), "python"))


ROOT = _resolve_root()
APP_SPECS = {
    "2d": {
        "label": "Estimation 2D",
        "path": os.path.join(ROOT, "estimation_2d", "main.py"),
        "needs_source": True,
    },
    "3d": {
        "label": "Estimation 3D",
        "path": os.path.join(ROOT, "estimation_3d", "main.py"),
        "needs_source": False,
    },
    "3d2d": {
        "label": "Estimation 3D vers 2D",
        "path": os.path.join(ROOT, "estimation_3d_to_2d", "main.py"),
        "needs_source": True,
    },
    "viewer": {
        "label": "Visualiseur UDP",
        "path": os.path.join(ROOT, "udp_distance_viewer.py"),
        "needs_source": False,
    },
    "logger": {
        "label": "Logger calibration",
        "path": os.path.join(ROOT, "calibration_logger.py"),
        "needs_source": False,
    },
}
MENU = tuple(APP_SPECS.keys())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Suite UWB unifiee.")
    parser.add_argument("--mode", choices=MENU, default=None)
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--source", choices=("simulation", "udp"), default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def ask_value(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or str(default)


def ask_float_or_none(prompt):
    raw = input(f"{prompt} [vide pour ignorer]: ").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("Nombre invalide.")
        return ask_float_or_none(prompt)


def ask_source(default="simulation"):
    while True:
        raw = ask_value("Source (simulation/udp)", default).lower()
        if raw in {"simulation", "udp"}:
            return raw
        print("Choix invalide.")


def choose_mode():
    print("Modes disponibles:")
    for index, mode in enumerate(MENU, start=1):
        print(f"  {index}. {APP_SPECS[mode]['label']} ({mode})")
    while True:
        raw = input("Selection: ").strip()
        try:
            index = int(raw)
        except ValueError:
            print("Selection invalide.")
            continue
        if 1 <= index <= len(MENU):
            return MENU[index - 1]
        print("Selection hors plage.")


def choose_mode_pygame(default_source):
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((860, 560))
    pygame.display.set_caption("OCIA UWB - Lanceur")
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("Arial", 34)
    item_font = pygame.font.SysFont("Arial", 28)
    help_font = pygame.font.SysFont("Arial", 22)

    selected = 0
    source = default_source or "simulation"
    modes = list(MENU)

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
                    selected = (selected - 1) % len(modes)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(modes)
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_TAB):
                    source = "udp" if source == "simulation" else "simulation"
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    pygame.quit()
                    return modes[selected], source

        screen.fill((14, 18, 28))
        header = title_font.render("OCIA UWB", True, (245, 245, 245))
        screen.blit(header, (40, 32))
        subtitle = help_font.render("Choisis un mode avec les fleches, puis Entree", True, (180, 200, 220))
        screen.blit(subtitle, (42, 78))

        source_text = help_font.render(f"Source pour 2D / 3D->2D : {source}", True, (255, 210, 90))
        screen.blit(source_text, (42, 118))

        y = 180
        for index, mode in enumerate(modes):
            active = index == selected
            rect = pygame.Rect(36, y - 8, 780, 44)
            if active:
                pygame.draw.rect(screen, (255, 210, 60), rect, border_radius=8)
                color = (18, 18, 22)
            else:
                pygame.draw.rect(screen, (44, 54, 70), rect, 1, border_radius=8)
                color = (230, 230, 235)
            label = item_font.render(f"{index + 1}. {APP_SPECS[mode]['label']}", True, color)
            screen.blit(label, (52, y))
            y += 58

        footer = help_font.render("Gauche/Droite ou Tab : changer source | Echap : quitter", True, (160, 175, 195))
        screen.blit(footer, (42, 510))
        pygame.display.flip()
        clock.tick(60)


def build_mode_argv(mode, args):
    mode_argv = []
    if mode in {"2d", "3d2d", "viewer", "logger"}:
        mode_argv.extend(["--ip", args.ip, "--port", str(args.port)])
    if APP_SPECS[mode]["needs_source"]:
        mode_argv.extend(["--source", args.source or ask_source()])
    if mode == "logger":
        duration = args.duration if args.duration is not None else ask_float_or_none("Duree en secondes")
        if duration is not None:
            mode_argv.extend(["--duration", str(duration)])
        if args.output:
            mode_argv.extend(["--output", args.output])
    return mode_argv


@contextmanager
def module_import_context(module_dir):
    previous_cwd = os.getcwd()
    original_sys_path = list(sys.path)
    os.chdir(module_dir)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = original_sys_path


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unload_mode_modules():
    for name in (
        "display",
        "main",
        "position_calcul",
        "real_tag",
        "scene",
        "uwb_sources",
        "player_analytics",
        "player_motion",
        "court_geometry",
    ):
        sys.modules.pop(name, None)


def run_mode(mode, mode_argv):
    spec = APP_SPECS[mode]
    module_dir = os.path.dirname(spec["path"])
    module_name = f"uwb_suite_{mode}"
    with module_import_context(module_dir):
        try:
            unload_mode_modules()
            module = load_module(module_name, spec["path"])
            return module.main(mode_argv)
        except ModuleNotFoundError as exc:
            if exc.name == "pygame" and mode in {"2d", "3d", "3d2d"}:
                raise SystemExit("Le mode graphique demande pygame. Installe-le puis relance le logiciel.") from exc
            raise


def main(argv=None):
    args = parse_args(argv)
    if args.mode:
        mode = args.mode
        source = args.source
    else:
        try:
            mode, source = choose_mode_pygame(args.source)
        except ModuleNotFoundError:
            mode = choose_mode()
            source = args.source
    args.source = source
    mode_argv = build_mode_argv(mode, args)
    print(f"Lancement: {APP_SPECS[mode]['label']}")
    return run_mode(mode, mode_argv)


if __name__ == "__main__":
    main()
