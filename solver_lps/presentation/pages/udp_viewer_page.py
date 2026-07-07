import argparse

from solver_lps.features.uwb.acquisition.presentation.realtime.viewer_cli import (
    main as realtime_main,
)
from solver_lps.features.uwb.acquisition.presentation.review.viewer_cli import (
    main as review_main,
)
from solver_lps.features.uwb.acquisition.data.session_assets import default_uwb_raw_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="UDP viewer page entrypoint.")
    parser.add_argument("--source", choices=("realtime", "review"), default="realtime")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--uwb-log", default=None)
    parser.add_argument("--capture-output", default=None)
    parser.add_argument("--sport", default="basket")
    parser.add_argument("--asset-set", dest="asset_set", default="set1")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.source == "review":
        review_argv = []
        if args.uwb_log:
            review_argv.extend(["--uwb-log", args.uwb_log])
        return review_main(review_argv)

    realtime_argv = ["--ip", args.ip, "--port", str(args.port)]
    capture_output = args.capture_output or default_uwb_raw_path(args.sport, args.asset_set)
    realtime_argv.extend(["--capture-output", str(capture_output)])
    return realtime_main(realtime_argv)
