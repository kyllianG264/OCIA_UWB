import argparse
import csv
import json
import os
import time
from datetime import datetime

from uwb_udp import UdpDistanceReceiver, parse_message_fields


LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
CSV_COLUMNS = (
    "timestamp_iso",
    "timestamp_s",
    "source_ip",
    "source_port",
    "event",
    "anchor_id",
    "distance_cm",
    "raw_message",
    "fields_json",
)


def default_output_path():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"uwb_calibration_{stamp}.csv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Log UDP UWB distances to CSV.")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4210)
    parser.add_argument("--max-age", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def packet_event(message):
    fields = parse_message_fields(message)
    parsed = fields.get("event") or fields.get("type")
    if parsed:
        return parsed.strip().upper()
    return "DISTANCE"


def build_rows(message, addr, parsed):
    fields = parse_message_fields(message)
    now_s = time.time()
    base = {
        "timestamp_iso": datetime.fromtimestamp(now_s).isoformat(timespec="milliseconds"),
        "timestamp_s": now_s,
        "source_ip": addr[0],
        "source_port": addr[1],
        "event": packet_event(message),
        "raw_message": message,
        "fields_json": json.dumps(fields, ensure_ascii=True, sort_keys=True),
    }
    rows = []
    for anchor_id, distance_cm in sorted(parsed.items()):
        row = dict(base)
        row["anchor_id"] = anchor_id
        row["distance_cm"] = distance_cm
        rows.append(row)
    return rows


def main(argv=None):
    args = parse_args(argv)
    output_path = args.output or default_output_path()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    needs_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    receiver = UdpDistanceReceiver(bind_ip=args.ip, port=args.port, max_age_s=args.max_age)
    rows_written = 0
    deadline = None if args.duration is None else time.time() + float(args.duration)

    try:
        with open(output_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            if needs_header:
                writer.writeheader()
            while deadline is None or time.time() < deadline:
                for message, addr, parsed in receiver.poll():
                    for row in build_rows(message, addr, parsed):
                        writer.writerow(row)
                        rows_written += 1
                fh.flush()
                time.sleep(0.05)
    finally:
        receiver.close()

    print(f"{rows_written} lignes ecrites dans {output_path}")


if __name__ == "__main__":
    main()
