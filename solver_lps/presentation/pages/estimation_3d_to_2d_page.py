import argparse

from solver_lps.features.uwb.realtime.three_d_to_2d.presentation.widgets.app_widget import (
    main as realtime_main,
)
from solver_lps.features.uwb.review.three_d_to_2d.presentation.widgets.app_widget import (
    main as review_main,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Estimation 3D vers 2D page entrypoint.")
    parser.add_argument("--source", choices=("realtime", "review"), default="realtime")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--uwb-tag-log", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    page_argv = ["--source", args.source, "--ip", args.ip, "--port", str(args.port)]
    if args.uwb_tag_log:
        page_argv.extend(["--uwb-tag-log", args.uwb_tag_log])
    return review_main(page_argv) if args.source == "review" else realtime_main(page_argv)
