import argparse

from solver_lps.features.udp_viewer.realtime.presentation.widgets.viewer_widget import (
    main as realtime_main,
)
from solver_lps.features.udp_viewer.review.presentation.widgets.viewer_widget import (
    main as review_main,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="UDP viewer page entrypoint.")
    parser.add_argument("--source", choices=("realtime", "review"), default="realtime")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--uwb-log", default=None)
    parser.add_argument("--capture-output", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.source == "review":
        review_argv = []
        if args.uwb_log:
            review_argv.extend(["--uwb-log", args.uwb_log])
        return review_main(review_argv)

    realtime_argv = ["--ip", args.ip, "--port", str(args.port)]
    if args.capture_output:
        realtime_argv.extend(["--capture-output", args.capture_output])
    return realtime_main(realtime_argv)
