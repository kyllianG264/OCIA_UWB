import argparse
import queue
import textwrap
from pathlib import Path

from solver_lps.features.cv.review import tracking_launcher
from solver_lps.presentation.navigation import RETURN_HOME


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CV tracking launcher.")
    parser.add_argument("--sport", default="basket")
    parser.add_argument("--asset-set", dest="asset_set", default="set1")
    parser.add_argument("--left-video", default=None)
    parser.add_argument("--right-video", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--conf", default="0.05")
    parser.add_argument("--imgsz", default="1408")
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def _select_video_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilename(
            title="Choisir une video",
            filetypes=[
                ("Videos", "*.mp4 *.mov *.avi *.mkv"),
                ("Tous les fichiers", "*.*"),
            ],
        )
    finally:
        root.destroy()
    return selected or None


def _wrap_log_lines(lines, width):
    wrapped = []
    safe_width = max(40, int(width))
    for line in lines:
        text = line if line else ""
        chunks = textwrap.wrap(
            text,
            width=safe_width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped.extend(chunks or [""])
    return wrapped


def _build_fields(args):
    return [
        {"key": "sport", "label": "Sport", "value": str(args.sport)},
        {"key": "asset_set", "label": "Set", "value": str(args.asset_set)},
        {"key": "left_video", "label": "Video gauche", "value": str(Path(args.left_video).resolve())},
        {"key": "right_video", "label": "Video droite", "value": str(Path(args.right_video).resolve())},
        {"key": "calibration", "label": "Calibration", "value": str(Path(args.calibration).resolve())},
        {"key": "model", "label": "Modele pose", "value": str(Path(args.model).resolve())},
        {"key": "output_root", "label": "Dossier sortie", "value": str(Path(args.output_root).resolve())},
        {"key": "analysis_root", "label": "Dossier analyse", "value": str((Path(args.output_root).parent / "analysis").resolve())},
        {"key": "conf", "label": "Confidence", "value": str(args.conf)},
        {"key": "imgsz", "label": "Image Size", "value": str(args.imgsz)},
        {"key": "device", "label": "Device", "value": str(args.device)},
    ]


def main(argv=None):
    args = parse_args(argv)
    defaults = tracking_launcher.session_defaults(args.sport, args.asset_set)
    args.left_video = args.left_video or str(defaults["left_video"])
    args.right_video = args.right_video or str(defaults["right_video"])
    args.calibration = args.calibration or str(defaults["calibration"])
    args.model = args.model or tracking_launcher.default_model_path()
    args.output_root = args.output_root or str(defaults["output_root"])
    import pygame

    pygame.init()
    desktop_info = pygame.display.Info()
    screen = pygame.display.set_mode((desktop_info.current_w, desktop_info.current_h), pygame.NOFRAME)
    pygame.display.set_caption("Solver LPS - CV Tracking")
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("Arial", 34)
    label_font = pygame.font.SysFont("Arial", 24)
    mono_font = pygame.font.SysFont("Consolas", 20)

    fields = _build_fields(args)
    selected_index = 0
    editing = False
    process = None
    calibration_process = None
    output_queue = None
    logs = []
    status = "Pret a lancer le tracking CV."
    back_rect = pygame.Rect(screen.get_width() - 228, 22, 192, 46)
    return_home = False

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return_home = True
                        running = False
                    elif editing:
                        if event.key == pygame.K_RETURN:
                            editing = False
                        elif event.key == pygame.K_BACKSPACE:
                            fields[selected_index]["value"] = fields[selected_index]["value"][:-1]
                        elif event.key == pygame.K_TAB:
                            editing = False
                        elif event.unicode and event.unicode.isprintable():
                            fields[selected_index]["value"] += event.unicode
                    else:
                        if event.key == pygame.K_UP:
                            selected_index = (selected_index - 1) % len(fields)
                        elif event.key == pygame.K_DOWN:
                            selected_index = (selected_index + 1) % len(fields)
                        elif event.key == pygame.K_RETURN:
                            editing = True
                        elif event.key == pygame.K_F5:
                            if process is None:
                                logs = []
                                sync_warnings = tracking_launcher.sync_selected_videos(fields)
                                if sync_warnings:
                                    logs.extend(sync_warnings)
                                    logs = logs[-120:]
                                is_valid, launch_errors = tracking_launcher.validate_launch(fields)
                                if not is_valid:
                                    status = "Pre-verification echouee."
                                    logs.extend(launch_errors)
                                    logs = logs[-120:]
                                else:
                                    status = "Videos synchronisees, lancement du pipeline..."
                                    if sync_warnings:
                                        status = "Videos synchronisees avec fichier de secours, lancement du pipeline..."
                                    process, output_queue = tracking_launcher.launch_tracking(fields)
                        elif event.key == pygame.K_F6:
                            if calibration_process is None or calibration_process.poll() is not None:
                                if not tracking_launcher.python_has_module("streamlit"):
                                    status = "Impossible de lancer la calibration."
                                    logs.append("Module manquant dans cet environnement Python : streamlit")
                                    logs.append("Installe-le dans l'environnement utilise par CV Tracking puis relance F6.")
                                    logs = logs[-120:]
                                else:
                                    calibration_process = tracking_launcher.launch_calibration(fields)
                                    status = "Calibration Streamlit lancee."
                            else:
                                status = "Calibration Streamlit deja ouverte."
                        elif event.key == pygame.K_F2:
                            if fields[selected_index]["key"] in {"left_video", "right_video"}:
                                selected_file = _select_video_file()
                                if selected_file:
                                    fields[selected_index]["value"] = str(Path(selected_file).resolve())
                                    status = f"Video selectionnee pour {fields[selected_index]['label']}."
                                else:
                                    status = "Selection annulee ou explorateur indisponible."
                        elif event.key == pygame.K_r:
                            fields = _build_fields(args)
                            status = "Champs reinitialises."
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if back_rect.collidepoint(event.pos):
                        return_home = True
                        running = False

            if output_queue is not None:
                while True:
                    try:
                        line = output_queue.get_nowait()
                    except queue.Empty:
                        break
                    logs.append(line)
                    logs = logs[-120:]

            if process is not None:
                return_code = process.poll()
                if return_code is not None:
                    status = "Tracking termine avec succes." if return_code == 0 else f"Tracking echoue (code {return_code})."
                    process = None
                    output_queue = None

            screen.fill((12, 16, 22))
            title = title_font.render("CV Tracking", True, (245, 245, 245))
            screen.blit(title, (36, 24))
            pygame.draw.rect(screen, (25, 42, 60), back_rect, border_radius=9)
            pygame.draw.rect(screen, (120, 200, 255), back_rect, 2, border_radius=9)
            back_label = label_font.render("Retour accueil", True, (225, 245, 255))
            screen.blit(back_label, back_label.get_rect(center=back_rect.center))
            subtitle = label_font.render("Entree: editer | F2: choisir video | F5: lancer | F6: calib | R: reset | Echap: retour", True, (175, 190, 210))
            screen.blit(subtitle, (38, 70))

            y = 120
            for index, field in enumerate(fields):
                active = index == selected_index
                rect = pygame.Rect(34, y, screen.get_width() - 68, 58)
                border_color = (255, 210, 70) if active else (70, 86, 104)
                pygame.draw.rect(screen, (18, 23, 31), rect, border_radius=10)
                pygame.draw.rect(screen, border_color, rect, 2, border_radius=10)
                label = label_font.render(field["label"], True, (230, 235, 240))
                value_color = (255, 240, 170) if active and editing else (190, 215, 255)
                value = mono_font.render(field["value"], True, value_color)
                screen.blit(label, (rect.x + 14, rect.y + 8))
                screen.blit(value, (rect.x + 14, rect.y + 30))
                y += 70

            status_rect = pygame.Rect(34, y + 6, screen.get_width() - 68, 52)
            pygame.draw.rect(screen, (18, 23, 31), status_rect, border_radius=10)
            pygame.draw.rect(screen, (120, 200, 255), status_rect, 1, border_radius=10)
            status_label = label_font.render(status, True, (220, 245, 255))
            screen.blit(status_label, (status_rect.x + 14, status_rect.y + 14))

            logs_rect = pygame.Rect(34, y + 74, screen.get_width() - 68, max(180, screen.get_height() - (y + 108)))
            pygame.draw.rect(screen, (8, 11, 16), logs_rect, border_radius=10)
            pygame.draw.rect(screen, (70, 86, 104), logs_rect, 1, border_radius=10)
            logs_title = label_font.render("Sortie pipeline", True, (235, 245, 255))
            screen.blit(logs_title, (logs_rect.x + 14, logs_rect.y + 10))

            log_y = logs_rect.y + 42
            approx_chars = max(60, (logs_rect.width - 28) // 11)
            wrapped_logs = _wrap_log_lines(logs, approx_chars)
            if not logs:
                empty = mono_font.render("Aucune sortie pour le moment.", True, (150, 165, 185))
                screen.blit(empty, (logs_rect.x + 14, log_y))
            else:
                visible_line_count = max(1, (logs_rect.height - 54) // 22)
                for line in wrapped_logs[-visible_line_count:]:
                    rendered = mono_font.render(line, True, (210, 220, 230))
                    screen.blit(rendered, (logs_rect.x + 14, log_y))
                    log_y += 22

            pygame.display.flip()
            clock.tick(60)
    finally:
        if process is not None:
            process.terminate()
        if calibration_process is not None and calibration_process.poll() is None:
            calibration_process.terminate()
        pygame.quit()
    return RETURN_HOME if return_home else 0
