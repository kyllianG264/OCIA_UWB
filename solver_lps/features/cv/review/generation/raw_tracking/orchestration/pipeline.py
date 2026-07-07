"""
Orchestrateur CLI pour le pipeline principal NeptuVision.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

from solver_lps.features.cv.review.generation.raw_tracking.data.tracking_output import (
    DEFAULT_LEFT_UNDISTORTED_VIDEO,
    DEFAULT_POSE_MODEL,
    DEFAULT_RIGHT_UNDISTORTED_VIDEO,
    configure_utf8_stdout,
    ensure_dir,
)
from solver_lps.features.cv.review.data.session_assets import DEFAULT_SET, DEFAULT_SPORT
from solver_lps.features.cv.review.generate_cv_positions import generate_cv_positions
from solver_lps.session_assets import SessionAssets


@dataclass(frozen=True)
class PipelineScriptPaths:
    script_dir: str
    raw_tracking_dir: str
    review_dir: str
    raw_tracking_orchestration_dir: str
    raw_tracking_data_dir: str


def resolve_pipeline_script_paths(current_file: str) -> PipelineScriptPaths:
    script_dir = os.path.dirname(os.path.abspath(current_file))
    raw_tracking_dir = os.path.dirname(script_dir)
    review_dir = os.path.dirname(os.path.dirname(raw_tracking_dir))
    raw_tracking_orchestration_dir = os.path.join(raw_tracking_dir, "orchestration")
    raw_tracking_data_dir = os.path.join(raw_tracking_dir, "data")
    return PipelineScriptPaths(
        script_dir=script_dir,
        raw_tracking_dir=raw_tracking_dir,
        review_dir=review_dir,
        raw_tracking_orchestration_dir=raw_tracking_orchestration_dir,
        raw_tracking_data_dir=raw_tracking_data_dir,
    )


def analysis_root_from_output_root(output_root: str) -> str:
    normalized_output = os.path.abspath(str(output_root))
    parent_dir = os.path.dirname(normalized_output)
    if os.path.basename(normalized_output).lower() == "output":
        return os.path.join(parent_dir, "analysis")
    return os.path.join(normalized_output, "analysis")


def default_undistorted_video_paths(output_root: str) -> tuple[str, str]:
    return (
        os.path.join(output_root, os.path.basename(str(DEFAULT_LEFT_UNDISTORTED_VIDEO))),
        os.path.join(output_root, os.path.basename(str(DEFAULT_RIGHT_UNDISTORTED_VIDEO))),
    )


def run_step(command: list[str], label: str, *, cwd: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"[{label}] {' '.join(command)}")
    print("=" * 72)
    completed = subprocess.run(command, cwd=cwd)
    if completed.returncode != 0:
        raise SystemExit(f"Echec a l'etape {label} (code {completed.returncode}).")


def build_detection_command(
    *,
    detector_script: str,
    video_gauche: str,
    video_droite: str,
    output_dir: str,
    model: str | None,
    device: str,
    conf: float | None,
    imgsz: int | None,
    conf_gauche: float | None,
    conf_droite: float | None,
    imgsz_gauche: int | None,
    imgsz_droite: int | None,
    start: str,
    end: str | None,
) -> list[str]:
    command = [
        sys.executable,
        detector_script,
        "--video_gauche",
        video_gauche,
        "--video_droite",
        video_droite,
        "--output",
        output_dir,
        "--device",
        device,
        "--start",
        start,
    ]
    if model:
        command.extend(["--model", model])
    if conf is not None:
        command.extend(["--conf", str(conf)])
    if imgsz is not None:
        command.extend(["--imgsz", str(imgsz)])
    if conf_gauche is not None:
        command.extend(["--conf_gauche", str(conf_gauche)])
    if conf_droite is not None:
        command.extend(["--conf_droite", str(conf_droite)])
    if imgsz_gauche is not None:
        command.extend(["--imgsz_gauche", str(imgsz_gauche)])
    if imgsz_droite is not None:
        command.extend(["--imgsz_droite", str(imgsz_droite)])
    if end:
        command.extend(["--end", end])
    return command


def build_projection_command(
    *,
    projection_script: str,
    input_csv: str,
    calibration: str,
    output_dir: str,
    disable_bounds_filter: bool,
) -> list[str]:
    command = [
        sys.executable,
        projection_script,
        "--input_csv",
        input_csv,
        "--calibration",
        calibration,
        "--output",
        output_dir,
    ]
    if disable_bounds_filter:
        command.append("--disable_bounds_filter")
    return command


def build_tracking_command(
    *,
    tracking_script: str,
    input_csv: str,
    output_dir: str,
    max_track_distance: float,
    max_idle_frames: int,
) -> list[str]:
    return [
        sys.executable,
        tracking_script,
        "--input_csv",
        input_csv,
        "--output",
        output_dir,
        "--max_track_distance",
        str(max_track_distance),
        "--max_idle_frames",
        str(max_idle_frames),
    ]


PIPELINE_SCRIPTS = resolve_pipeline_script_paths(__file__)


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Orchestration du pipeline NeptuVision")
    parser.add_argument("--video_gauche", required=True)
    parser.add_argument("--video_droite", required=True)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--sport", default=os.environ.get("SOLVER_SPORT", DEFAULT_SPORT))
    parser.add_argument("--asset-set", default=os.environ.get("SOLVER_ASSET_SET", DEFAULT_SET))
    parser.add_argument("--court-mode", choices=("auto", "split", "full"), default="auto")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--analysis_root", default=None)
    parser.add_argument("--model", default=str(DEFAULT_POSE_MODEL))
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=1408)
    parser.add_argument("--conf_gauche", type=float, default=None)
    parser.add_argument("--conf_droite", type=float, default=None)
    parser.add_argument("--imgsz_gauche", type=int, default=None)
    parser.add_argument("--imgsz_droite", type=int, default=None)
    parser.add_argument("--start", default="00:00:00")
    parser.add_argument("--end", default=None)
    parser.add_argument("--max_track_distance", type=float, default=120.0)
    parser.add_argument("--max_idle_frames", type=int, default=15)
    parser.add_argument("--disable_bounds_filter", action="store_true")
    parser.add_argument("--skip_tracking_video", action="store_true")
    args = parser.parse_args()

    selected_assets = SessionAssets(sport=args.sport, asset_set=args.asset_set)
    args.calibration = str(args.calibration or selected_assets.calibration_path)
    output_root = ensure_dir(args.output_root or str(selected_assets.output_dir))
    analysis_root = ensure_dir(args.analysis_root or analysis_root_from_output_root(output_root))

    with tempfile.TemporaryDirectory(prefix="solver_lps_cv_pipeline_") as temp_root:
        detection_dir = ensure_dir(os.path.join(temp_root, "detection"))
        projection_dir = ensure_dir(os.path.join(temp_root, "projection"))
        tracking_dir = output_root
        undistorted_left_video, undistorted_right_video = default_undistorted_video_paths(output_root)

        render_cmd = [
            sys.executable,
            os.path.join(PIPELINE_SCRIPTS.raw_tracking_data_dir, "render_video.py"),
            "--video_gauche",
            args.video_gauche,
            "--video_droite",
            args.video_droite,
            "--output_dir",
            output_root,
            "--calibration",
            args.calibration,
            "--start",
            args.start,
        ]
        if args.end:
            render_cmd.extend(["--end", args.end])

        detection_cmd = build_detection_command(
            detector_script=os.path.join(PIPELINE_SCRIPTS.raw_tracking_orchestration_dir, "detector.py"),
            video_gauche=undistorted_left_video,
            video_droite=undistorted_right_video,
            output_dir=detection_dir,
            model=args.model,
            device=args.device,
            conf=args.conf,
            imgsz=args.imgsz,
            conf_gauche=args.conf_gauche,
            conf_droite=args.conf_droite,
            imgsz_gauche=args.imgsz_gauche,
            imgsz_droite=args.imgsz_droite,
            start=args.start,
            end=args.end,
        )

        projection_cmd = build_projection_command(
            projection_script=os.path.join(PIPELINE_SCRIPTS.raw_tracking_orchestration_dir, "projection.py"),
            input_csv=os.path.join(detection_dir, "detections.csv"),
            calibration=args.calibration,
            output_dir=projection_dir,
            disable_bounds_filter=args.disable_bounds_filter,
        )

        tracking_cmd = build_tracking_command(
            tracking_script=os.path.join(PIPELINE_SCRIPTS.raw_tracking_orchestration_dir, "raw_tracking.py"),
            input_csv=os.path.join(projection_dir, "projected_positions.csv"),
            output_dir=tracking_dir,
            max_track_distance=args.max_track_distance,
            max_idle_frames=args.max_idle_frames,
        )

        tracked_render_cmd = [
            sys.executable,
            os.path.join(PIPELINE_SCRIPTS.raw_tracking_data_dir, "render_video.py"),
            "--video_gauche",
            args.video_gauche,
            "--video_droite",
            args.video_droite,
            "--output_dir",
            output_root,
            "--calibration",
            args.calibration,
            "--detections_csv",
            os.path.join(detection_dir, "detections.csv"),
            "--assignments_csv",
            os.path.join(output_root, "tracking_assignments.csv"),
            "--start",
            args.start,
        ]
        if args.end:
            tracked_render_cmd.extend(["--end", args.end])

        merged_output_path = os.path.join(analysis_root, "positions_merged.csv")
        run_step(render_cmd, "undistort_video", cwd=PIPELINE_SCRIPTS.review_dir)
        run_step(detection_cmd, "detection", cwd=PIPELINE_SCRIPTS.review_dir)
        run_step(projection_cmd, "projection", cwd=PIPELINE_SCRIPTS.review_dir)
        run_step(tracking_cmd, "tracking", cwd=PIPELINE_SCRIPTS.review_dir)

        generated_path = generate_cv_positions(
            SimpleNamespace(
                sport=args.sport,
                asset_set=args.asset_set,
                court_mode=args.court_mode,
                input_path=os.path.join(output_root, "positions_raw.csv"),
                output_path=merged_output_path,
                calibration_path=args.calibration,
            )
        )

        if not args.skip_tracking_video:
            run_step(tracked_render_cmd, "tracking_video", cwd=PIPELINE_SCRIPTS.review_dir)
        print(f"Positions CV generees: {generated_path}")

    print(f"\nPipeline termine. Output : {output_root}")
    print(f"Pipeline termine. Analysis : {analysis_root}")


if __name__ == "__main__":
    main()
