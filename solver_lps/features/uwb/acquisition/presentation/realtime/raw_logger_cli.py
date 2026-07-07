import argparse
import time

from solver_lps.features.uwb.acquisition.data.raw_output import (
    append_rows,
    build_frame_rows,
    load_last_frame_index,
)
from solver_lps.features.uwb.acquisition.data.udp_input import UdpInput
from solver_lps.features.uwb.acquisition.domain.udp_reader import UdpReader, parse_message_fields
from solver_lps.features.uwb.acquisition.data.session_assets import (
    DEFAULT_SET,
    DEFAULT_SPORT,
    default_uwb_raw_path,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Log UDP UWB distances to a normalized raw CSV.")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--max-age", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--sport", default=DEFAULT_SPORT)
    parser.add_argument("--asset-set", dest="asset_set", default=DEFAULT_SET)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_path = str(args.output or default_uwb_raw_path(args.sport, args.asset_set))
    reader = UdpReader(UdpInput(bind_ip=args.ip, port=args.port), max_age_s=args.max_age)
    frame_index = load_last_frame_index(output_path) + 1
    rows_written = 0
    deadline = None if args.duration is None else time.time() + float(args.duration)

    try:
        while deadline is None or time.time() < deadline:
            frame_rows = []
            for message, addr, parsed, received_at in reader.poll():
                if not parsed:
                    continue
                frame_rows.extend(
                    build_frame_rows(
                        frame_index=frame_index,
                        timestamp_s=received_at,
                        message=message,
                        addr=addr,
                        parsed=parsed,
                        fields=parse_message_fields(message),
                    )
                )
                frame_index += 1
            if frame_rows:
                append_rows(output_path, frame_rows)
                rows_written += len(frame_rows)
            time.sleep(0.05)
    finally:
        reader.close()

    print(f"{rows_written} lignes ecrites dans {output_path}")


if __name__ == "__main__":
    main()
