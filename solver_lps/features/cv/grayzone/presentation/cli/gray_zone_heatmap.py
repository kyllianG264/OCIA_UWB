import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from solver_lps.features.cv.grayzone.application.gray_zone_analysis import build_gray_zone
from solver_lps.features.cv.grayzone.data.gray_zone_repository import default_gray_zone_repository


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build camera blind zones from raw CV observations")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-csv", help="positions_raw.csv produced by CV")
    source.add_argument("--tracking-csv", help=argparse.SUPPRESS)
    parser.add_argument("--sport", default="basket")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--grid-step", type=float, default=8.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repository = default_gray_zone_repository(args.sport)
    raw_csv = args.raw_csv or args.tracking_csv
    gray_zone, exported = build_gray_zone(
        raw_csv,
        args.calibration or repository.calibration_path,
        output_dir=args.output_dir or repository.analysis_dir,
        grid_step=args.grid_step,
    )
    print(f"Observations raw: {gray_zone.metadata['raw_observation_count']}")
    print(f"Cameras calibrees: {gray_zone.metadata['camera_count']}")
    print(f"Metadata: {exported['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
