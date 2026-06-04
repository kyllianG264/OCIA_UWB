"""
Orchestrateur CLI pour le pipeline principal NeptuVision.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from core.paths import DEFAULT_POSE_MODEL
from core.performance import AGGRESSIVENESS_CHOICES, FPS_TARGET_CHOICES, resolve_performance_settings
from core.utils import configure_utf8_stdout, ensure_dir

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
FEATURES_DIR = os.path.join(ROOT_DIR, "features")


def run_step(command: list[str], label: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"[{label}] {' '.join(command)}")
    print("=" * 72)
    completed = subprocess.run(command, cwd=ROOT_DIR)
    if completed.returncode != 0:
        raise SystemExit(f"Echec a l'etape {label} (code {completed.returncode}).")


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Orchestration du pipeline NeptuVision")
    parser.add_argument("--video_gauche", required=True)
    parser.add_argument("--video_droite", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output_root", default=os.path.join(ROOT_DIR, "NeptuVisionResults"))
    parser.add_argument("--target_fps", type=int, choices=FPS_TARGET_CHOICES, default=15)
    parser.add_argument("--aggressiveness", choices=AGGRESSIVENESS_CHOICES, default="medium")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf_gauche", type=float, default=None)
    parser.add_argument("--conf_droite", type=float, default=None)
    parser.add_argument("--imgsz_gauche", type=int, default=None)
    parser.add_argument("--imgsz_droite", type=int, default=None)
    parser.add_argument("--start", default="00:00:00")
    parser.add_argument("--end", default=None)
    parser.add_argument("--mqtt_host", default="localhost")
    parser.add_argument("--mqtt_port", type=int, default=1883)
    parser.add_argument("--max_track_distance", type=float, default=120.0)
    parser.add_argument("--max_idle_frames", type=int, default=15)
    parser.add_argument("--max_absence", type=int, default=10)
    parser.add_argument("--merge_dist", type=int, default=80)
    parser.add_argument("--max_jump_dist", type=int, default=120)
    parser.add_argument("--disable_bounds_filter", action="store_true")
    parser.add_argument("--skip_tracking_video", action="store_true")
    args = parser.parse_args()

    perf = resolve_performance_settings(args.target_fps, args.aggressiveness)
    args.model = args.model or perf.model or str(DEFAULT_POSE_MODEL)
    args.conf = args.conf if args.conf is not None else perf.conf
    args.imgsz = args.imgsz if args.imgsz is not None else perf.imgsz

    detection_dir = ensure_dir(os.path.join(args.output_root, "3_Detection"))
    projection_dir = ensure_dir(os.path.join(args.output_root, "4_Projection"))
    tracking_dir = ensure_dir(os.path.join(args.output_root, "5_Tracking"))
    solver_dir = ensure_dir(os.path.join(args.output_root, "6_Solver"))

    detection_cmd = [
        sys.executable,
        os.path.join(FEATURES_DIR, "2_detection", "2_detector.py"),
        "--video_gauche",
        args.video_gauche,
        "--video_droite",
        args.video_droite,
        "--output",
        detection_dir,
        "--model",
        args.model,
        "--target_fps",
        str(args.target_fps),
        "--aggressiveness",
        args.aggressiveness,
        "--device",
        args.device,
        "--conf",
        str(args.conf),
        "--imgsz",
        str(args.imgsz),
        "--start",
        args.start,
    ]
    if args.conf_gauche is not None:
        detection_cmd.extend(["--conf_gauche", str(args.conf_gauche)])
    if args.conf_droite is not None:
        detection_cmd.extend(["--conf_droite", str(args.conf_droite)])
    if args.imgsz_gauche is not None:
        detection_cmd.extend(["--imgsz_gauche", str(args.imgsz_gauche)])
    if args.imgsz_droite is not None:
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
        tracking_dir,
        "--mqtt_host",
        args.mqtt_host,
        "--mqtt_port",
        str(args.mqtt_port),
        "--max_track_distance",
        str(args.max_track_distance),
        "--max_idle_frames",
        str(args.max_idle_frames),
    ]

    render_cmd = [
        sys.executable,
        os.path.join(FEATURES_DIR, "4_tracking", "4_render_video.py"),
        "--video_gauche",
        args.video_gauche,
        "--video_droite",
        args.video_droite,
        "--detections_csv",
        os.path.join(detection_dir, "detections.csv"),
        "--assignments_csv",
        os.path.join(tracking_dir, "tracking_assignments.csv"),
        "--output",
        os.path.join(tracking_dir, "tracking_composite.mp4"),
        "--start",
        args.start,
    ]
    if args.end:
        render_cmd.extend(["--end", args.end])

    solver_cmd = [
        sys.executable,
        os.path.join(FEATURES_DIR, "5_fusion", "5_solver.py"),
        "--input_csv",
        os.path.join(tracking_dir, "positions_raw.csv"),
        "--output",
        solver_dir,
        "--max_absence",
        str(args.max_absence),
        "--merge_dist",
        str(args.merge_dist),
        "--max_jump_dist",
        str(args.max_jump_dist),
    ]

    run_step(detection_cmd, "detection")
    run_step(projection_cmd, "projection")
    run_step(tracking_cmd, "tracking")
    if not args.skip_tracking_video:
        run_step(render_cmd, "tracking_video")
    run_step(solver_cmd, "solver")

    print(f"\nPipeline termine. Resultats : {args.output_root}")


if __name__ == "__main__":
    main()
