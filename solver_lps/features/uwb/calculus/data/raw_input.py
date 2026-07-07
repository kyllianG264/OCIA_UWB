"""Normalized raw UWB input readers for position calculations."""

import csv
import os


def load_raw_rows(csv_path):
    rows = []
    if not csv_path or not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append(
                    {
                        "frame": int(float(row.get("frame") or 0)),
                        "timestamp_s": float(row.get("timestamp_s") or 0.0),
                        "anchor_id": int(row.get("anchor_id")),
                        "distance_cm": float(row.get("distance_cm")),
                    }
                )
            except (TypeError, ValueError):
                continue
    return rows


def load_raw_frames(csv_path):
    frames_by_index = {}
    for row in load_raw_rows(csv_path):
        frame_index = row["frame"]
        frame = frames_by_index.setdefault(
            frame_index,
            {"frame": frame_index, "timestamp_s": row["timestamp_s"], "distances": {}},
        )
        frame["distances"][row["anchor_id"]] = row["distance_cm"]
    frames = []
    for sequence_index, frame_index in enumerate(sorted(frames_by_index)):
        frame = dict(frames_by_index[frame_index])
        frame["sequence_index"] = sequence_index
        frames.append(frame)
    return frames


class IncrementalRawFrameReader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.last_frame_index = -1

    def read_new_frames(self):
        frames = load_raw_frames(self.csv_path)
        fresh_frames = [frame for frame in frames if int(frame.get("frame", -1)) > self.last_frame_index]
        if fresh_frames:
            self.last_frame_index = max(int(frame["frame"]) for frame in fresh_frames)
        return fresh_frames
