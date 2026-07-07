import argparse

from solver_lps.features.cv.review.generate_cv_positions import generate_cv_positions
from solver_lps.features.cv.grayzone.application.gray_zone_analysis import build_gray_zone
from solver_lps.features.cv.grayzone.data.gray_zone_repository import default_gray_zone_repository
from solver_lps.presentation.pages.estimation_2d_review_ui import main as review_ui_main
from solver_lps.features.uwb.acquisition.presentation.realtime.viewer_cli import (
    main as realtime_main,
)
from solver_lps.features.uwb.orchestration.review_mode import run_review
from solver_lps.presentation.loading_screen import run_with_square_progress
from solver_lps.presentation.navigation import ANALYZE_GRAY_ZONE, REGENERATE_MERGED
from solver_lps.session_assets import SessionAssets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Estimation 2D page entrypoint.")
    parser.add_argument("--sport", default="basket")
    parser.add_argument("--asset-set", dest="asset_set", default="set1")
    parser.add_argument("--source", choices=("realtime", "review"), default="realtime")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--uwb-log", default=None)
    parser.add_argument("--uwb-tag-log", default=None)
    parser.add_argument("--cv-log", default=None)
    parser.add_argument("--cv-calibration", default=None)
    parser.add_argument("--cv-video", default=None)
    parser.add_argument("--cv-player", default=None)
    parser.add_argument("--cv-expected-players", type=int, default=None)
    parser.add_argument("--review-data", choices=("uwb", "cv", "both"), default="cv")
    parser.add_argument("--solver-mode", choices=("2d", "3d", "3d_to_2d"), default="2d")
    return parser.parse_args(argv)


def _append_optional(args_list, flag, value):
    if value:
        args_list.extend([flag, str(value)])


def _append_optional_int(args_list, flag, value):
    if value is not None:
        args_list.extend([flag, str(value)])


def _build_common_args(args):
    return [
        "--source",
        args.source,
        "--sport",
        args.sport,
        "--asset-set",
        args.asset_set,
        "--ip",
        args.ip,
        "--port",
        str(args.port),
    ]


def _build_realtime_args(args):
    return ["--ip", args.ip, "--port", str(args.port)]


def _calculation_mode(solver_mode):
    return {"2d": "two_d", "3d": "three_d", "3d_to_2d": "three_d_to_2d"}[solver_mode]


def _build_review_tracking_args(args, uwb_merged_path=None):
    argv = _build_common_args(args)
    _append_optional(argv, "--uwb-log", args.uwb_log)
    _append_optional(argv, "--uwb-tag-log", args.uwb_tag_log)
    _append_optional(argv, "--cv-log", args.cv_log)
    _append_optional(argv, "--cv-calibration", args.cv_calibration)
    _append_optional(argv, "--cv-video", args.cv_video)
    _append_optional(argv, "--cv-player", args.cv_player)
    _append_optional_int(argv, "--cv-expected-players", args.cv_expected_players)
    _append_optional(argv, "--review-data", args.review_data)
    _append_optional(argv, "--uwb-merged", uwb_merged_path)
    _append_optional(argv, "--solver-mode", args.solver_mode)
    return argv


def _generate_review_merged(args, session, progress_callback):
    sources = [source for source in ("cv", "uwb") if args.review_data in {source, "both"}]
    total_units = max(1, len(sources) * 1000)
    uwb_merged_path = None

    for source_index, source in enumerate(sources):
        def stage_progress(completed, total, *, _index=source_index):
            ratio = 0.0 if total <= 0 else min(1.0, float(completed) / float(total))
            progress_callback(round((_index + ratio) * 1000), total_units)

        if source == "cv":
            generate_cv_positions(session, progress_callback=stage_progress)
        else:
            uwb_merged_path = run_review(
                calculation_mode=_calculation_mode(args.solver_mode),
                input_path=args.uwb_log,
                session=session,
                progress_callback=stage_progress,
            )
    return uwb_merged_path


def _analyze_gray_zone(sport, progress_callback):
    repository = default_gray_zone_repository(sport)
    progress_callback(0, 1)
    result = build_gray_zone(
        repository.raw_positions_path,
        repository.calibration_path,
        output_dir=repository.analysis_dir,
    )
    progress_callback(1, 1)
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.cv_expected_players is None:
        args.cv_expected_players = 12 if str(args.sport or "").strip().lower() == "volley" else 10
    if args.source != "review":
        return realtime_main(_build_realtime_args(args))

    session = SessionAssets(sport=args.sport, asset_set=args.asset_set)
    calculation_mode = _calculation_mode(args.solver_mode)
    existing_uwb_merged = session.uwb_positions_path(calculation_mode)
    uwb_merged_path = existing_uwb_merged if existing_uwb_merged.is_file() else None

    while True:
        result = review_ui_main(_build_review_tracking_args(args, uwb_merged_path))
        if result == ANALYZE_GRAY_ZONE:
            run_with_square_progress(
                lambda progress: _analyze_gray_zone(args.sport, progress),
                title="Preparation de la review",
                label=f"Analyse gray zone {args.sport}",
            )
            continue
        if result != REGENERATE_MERGED:
            return result

        selected_sources = " + ".join(
            source.upper() for source in ("cv", "uwb") if args.review_data in {source, "both"}
        )
        generated_uwb_path = run_with_square_progress(
            lambda progress: _generate_review_merged(args, session, progress),
            title="Preparation de la review",
            label=f"Calcul du merged {selected_sources}",
        )
        if generated_uwb_path is not None:
            uwb_merged_path = generated_uwb_path
