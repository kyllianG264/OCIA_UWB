"""Append-only normalized raw UWB CSV output."""

from __future__ import annotations

import csv
from pathlib import Path


FIELDNAMES = (
    "frame",
    "timestamp_s",
    "anchor_id",
    "distance_cm",
    "source_ip",
    "source_port",
    "message",
)


def load_last_frame_index(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return -1
    last_frame = -1
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                last_frame = max(last_frame, int(float(row.get("frame", -1))))
            except (TypeError, ValueError):
                continue
    return last_frame


def build_frame_rows(*, frame_index, timestamp_s, message, addr, parsed, fields=None):
    del fields  # Parsed metadata is intentionally not duplicated in the normalized schema.
    source_ip, source_port = (addr or ("", ""))[:2]
    return [
        {
            "frame": int(frame_index),
            "timestamp_s": float(timestamp_s),
            "anchor_id": int(anchor_id),
            "distance_cm": float(distance_cm),
            "source_ip": str(source_ip),
            "source_port": source_port,
            "message": str(message),
        }
        for anchor_id, distance_cm in sorted(parsed.items())
    ]


def append_rows(csv_path, rows):
    rows = list(rows)
    if not rows:
        return 0
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def delete_raw_file(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return False
    path.unlink()
    return True


class RawCaptureWriter:
    """Append normalized UDP packets while preserving frame continuity."""

    def __init__(self, csv_path):
        self.csv_path = str(csv_path)
        self.next_frame_index = load_last_frame_index(self.csv_path) + 1
        self.rows_written = 0

    def append_packet(self, *, timestamp_s, message, addr, parsed, fields=None):
        rows = build_frame_rows(
            frame_index=self.next_frame_index,
            timestamp_s=timestamp_s,
            message=message,
            addr=addr,
            parsed=parsed,
            fields=fields,
        )
        written = append_rows(self.csv_path, rows)
        if written:
            self.next_frame_index += 1
            self.rows_written += written
        return written
