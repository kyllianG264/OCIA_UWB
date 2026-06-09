import argparse

from solver_lps.features.uwb.realtime.two_d.presentation.widgets.app_widget import (
    main as realtime_main,
)
from solver_lps.features.uwb.review.two_d.presentation.widgets.app_widget import (
    main as review_main,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Estimation 2D page entrypoint.")
    parser.add_argument("--source", choices=("realtime", "review"), default="realtime")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--uwb-log", default=None)
    parser.add_argument("--uwb-tag-log", default=None)
    parser.add_argument("--cv-log", default=None)
    parser.add_argument("--cv-calibration", default=None)
    parser.add_argument("--cv-video", default=None)
    parser.add_argument("--cv-player", default=None)
    parser.add_argument("--cv-expected-players", type=int, default=10)
    parser.add_argument("--review-data", choices=("uwb", "cv", "both"), default="both")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    page_argv = ["--source", args.source, "--ip", args.ip, "--port", str(args.port)]
    if args.uwb_log:
        page_argv.extend(["--uwb-log", args.uwb_log])
    if args.uwb_tag_log:
        page_argv.extend(["--uwb-tag-log", args.uwb_tag_log])
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
    return review_main(page_argv) if args.source == "review" else realtime_main(page_argv)
