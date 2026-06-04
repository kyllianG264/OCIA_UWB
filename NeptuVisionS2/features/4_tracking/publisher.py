"""
Compatibilite legacy : ancien point d'entree monolithique.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
FEATURES_DIR = os.path.join(ROOT_DIR, "features")


def run_step(command: list[str], label: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"[{label}] {' '.join(command)}")
    print("=" * 72)
    completed = subprocess.run(command, cwd=ROOT_DIR)
    if completed.returncode != 0:
        raise SystemExit(f"Echec a l'etape {label} (code {completed.returncode}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibilite legacy vers detection -> projection -> tracking")
    parser.add_argument("--video_gauche", required=True)
    parser.add_argument("--video_droite", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", default="./resultats")
    parser.add_argument("--target_fps", default="15")
    parser.add_argument("--aggressiveness", default="medium")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", default=None)
    parser.add_argument("--imgsz", default=None)
    parser.add_argument("--conf_gauche", default=None)
    parser.add_argument("--conf_droite", default=None)
    parser.add_argument("--imgsz_gauche", default=None)
    parser.add_argument("--imgsz_droite", default=None)
    parser.add_argument("--start", default="00:00:00")
    parser.add_argument("--end", default=None)
    parser.add_argument("--mqtt_host", default="localhost")
    parser.add_argument("--mqtt_port", default="1883")
    parser.add_argument("--disable_bounds_filter", action="store_true")
    args = parser.parse_args()

    detection_dir = os.path.join(args.output, "detection")
    projection_dir = os.path.join(args.output, "projection")

    detection_cmd = [
        sys.executable,
        os.path.join(FEATURES_DIR, "2_detection", "2_detector.py"),
        "--video_gauche",
        args.video_gauche,
        "--video_droite",
        args.video_droite,
        "--output",
        detection_dir,
        "--target_fps",
        str(args.target_fps),
        "--aggressiveness",
        args.aggressiveness,
        "--device",
        args.device,
        "--start",
        args.start,
    ]
    if args.model:
        detection_cmd.extend(["--model", args.model])
    if args.conf:
        detection_cmd.extend(["--conf", str(args.conf)])
    if args.imgsz:
        detection_cmd.extend(["--imgsz", str(args.imgsz)])
    if args.conf_gauche:
        detection_cmd.extend(["--conf_gauche", str(args.conf_gauche)])
    if args.conf_droite:
        detection_cmd.extend(["--conf_droite", str(args.conf_droite)])
    if args.imgsz_gauche:
        detection_cmd.extend(["--imgsz_gauche", str(args.imgsz_gauche)])
    if args.imgsz_droite:
        detection_cmd.extend(["--imgsz_droite", str(args.imgsz_droite)])
    if args.end:
        detection_cmd.extend(["--end", args.end])

    projection_cmd = [
        sys.executable,
        os.path.join(FEATURES_DIR, "3_projection", "3_projector.py"),
        "--input_csv",
        os.path.join(detection_dir, "detections.csv"),
        "--calibration",
        args.calibration,
        "--output",
        projection_dir,
    ]
    if args.disable_bounds_filter:
        projection_cmd.append("--disable_bounds_filter")

    tracking_cmd = [
        sys.executable,
        os.path.join(FEATURES_DIR, "4_tracking", "4_tracker.py"),
        "--input_csv",
        os.path.join(projection_dir, "projected_positions.csv"),
        "--output",
        args.output,
        "--mqtt_host",
        args.mqtt_host,
        "--mqtt_port",
        str(args.mqtt_port),
    ]

    run_step(detection_cmd, "detection")
    run_step(projection_cmd, "projection")
    run_step(tracking_cmd, "tracking")


if __name__ == "__main__":
    main()
