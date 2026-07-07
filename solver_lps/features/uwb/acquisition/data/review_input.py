"""Review UWB input source with simple playback controls."""

from __future__ import annotations

import bisect
import csv
import time
from pathlib import Path

from solver_lps.features.uwb.acquisition.data.session_assets import (
    DEFAULT_UWB_REVIEW_LOG_PATH,
)


def _load_review_frames(csv_path):
    frames_by_index = {}
    if not csv_path or not Path(csv_path).exists():
        return []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                frame_index = int(float(row.get("frame") or 0))
                timestamp_s = float(row.get("timestamp_s") or 0.0)
                anchor_id = int(row.get("anchor_id"))
                distance_cm = float(row.get("distance_cm"))
            except (TypeError, ValueError):
                continue
            frame = frames_by_index.setdefault(
                frame_index,
                {"frame": frame_index, "timestamp_s": timestamp_s, "distances": {}},
            )
            frame["distances"][anchor_id] = distance_cm
    return [
        {**frames_by_index[index], "sequence_index": sequence_index}
        for sequence_index, index in enumerate(sorted(frames_by_index))
    ]


class UwbReviewSource:
    def __init__(self, csv_path=DEFAULT_UWB_REVIEW_LOG_PATH):
        self.csv_path = str(csv_path)
        self.frames = _load_review_frames(self.csv_path)
        self.frame_index = 0
        self.playback_started_at = time.monotonic()
        self.playback_position_s = 0.0
        self.playback_paused = False

        if self.frames:
            first_timestamp = float(self.frames[0].get("timestamp_s", 0.0))
            self.normalized_timestamps = [
                max(0.0, float(frame.get("timestamp_s", 0.0)) - first_timestamp)
                for frame in self.frames
            ]
        else:
            self.normalized_timestamps = []

        positive_deltas = [
            self.normalized_timestamps[index] - self.normalized_timestamps[index - 1]
            for index in range(1, len(self.normalized_timestamps))
            if self.normalized_timestamps[index] > self.normalized_timestamps[index - 1]
        ]
        self.nominal_frame_period_s = min(positive_deltas) if positive_deltas else (1.0 / 25.0)
        self.playback_duration_s = (
            self.normalized_timestamps[-1] + self.nominal_frame_period_s if self.normalized_timestamps else 0.0
        )

    def set_playback(self, position_s=None, paused=None):
        if position_s is not None:
            self.playback_position_s = min(self.playback_duration_s, max(0.0, float(position_s)))
            self.playback_started_at = time.monotonic() - self.playback_position_s
        if paused is not None:
            self.playback_paused = bool(paused)
            if not self.playback_paused:
                self.playback_started_at = time.monotonic() - self.playback_position_s

    def toggle_pause(self):
        self.set_playback(paused=not self.playback_paused)
        return self.playback_paused

    def seek_relative(self, delta_s):
        self.set_playback(position_s=self.playback_position_s + float(delta_s), paused=True)
        return self.playback_position_s

    def seek_absolute(self, position_s, paused=True):
        self.set_playback(position_s=position_s, paused=paused)
        return self.playback_position_s

    def seek_frames(self, delta_frames):
        if not self.frames:
            return self.playback_position_s
        target_index = max(0, min(len(self.frames) - 1, self.resolve_index() + int(delta_frames)))
        position_s = self.normalized_timestamps[target_index] if self.normalized_timestamps else 0.0
        self.frame_index = target_index
        self.set_playback(position_s=position_s, paused=True)
        return self.playback_position_s

    def resolve_index(self):
        if not self.frames:
            return 0
        if self.playback_duration_s <= 0.0:
            return min(self.frame_index, len(self.frames) - 1)
        if self.playback_paused:
            elapsed_s = self.playback_position_s
        else:
            elapsed_s = time.monotonic() - self.playback_started_at
            if elapsed_s >= self.playback_duration_s:
                elapsed_s = self.playback_duration_s
                self.playback_paused = True
            self.playback_position_s = elapsed_s
        index = bisect.bisect_right(self.normalized_timestamps, elapsed_s) - 1
        self.frame_index = max(0, min(index, len(self.frames) - 1))
        return self.frame_index

    def get_packet(self, active_ids):
        if not self.frames:
            return {
                "raw": {},
                "valid": False,
                "source": "review",
                "status": "review: aucun log UWB charge",
                "uwb_playback": self.playback_state,
            }
        self.frame_index = self.resolve_index()
        frame = self.frames[self.frame_index]
        active_ids = [int(anchor_id) for anchor_id in active_ids]
        raw = {
            anchor_id: frame["distances"][anchor_id]
            for anchor_id in active_ids
            if anchor_id in frame["distances"]
        }
        return {
            "raw": raw,
            "valid": all(anchor_id in raw for anchor_id in active_ids),
            "source": "review",
            "status": (
                f"review UWB: {Path(self.csv_path).name} | "
                f"frame {self.frame_index + 1}/{len(self.frames)}"
            ),
            "uwb_playback": self.playback_state,
            "timestamp_s": frame["timestamp_s"],
            "frame": frame.get("frame"),
        }

    @property
    def playback_state(self):
        return {
            "paused": self.playback_paused,
            "position_s": self.playback_position_s,
            "duration_s": self.playback_duration_s,
            "frame_index": self.frame_index,
            "frame_count": len(self.frames),
        }


class UwbTagReviewSource:
    def __init__(self, csv_path):
        self.csv_path = str(csv_path)
        self.samples = self._load_samples(self.csv_path)
        self.timestamps = [sample["timestamp_s"] for sample in self.samples]

    def _load_samples(self, csv_path):
        if not csv_path or not Path(csv_path).exists():
            return []
        samples = []
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                timestamp_s = self._to_float(row.get("timestamp_s", row.get("t")), None)
                x_value = self._to_float(row.get("x_cm", row.get("x", row.get("X"))), None)
                y_value = self._to_float(row.get("y_cm", row.get("y", row.get("Y"))), None)
                z_value = self._to_float(row.get("z_cm", row.get("z", row.get("Z"))), None)
                if timestamp_s is None or x_value is None or y_value is None:
                    continue
                point = (x_value, y_value) if z_value is None else (x_value, y_value, z_value)
                samples.append({"timestamp_s": timestamp_s, "point": point})
        return sorted(samples, key=lambda sample: sample["timestamp_s"])

    def _to_float(self, value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def get_position_at(self, timestamp_s):
        if not self.samples:
            return None
        target_time = max(0.0, float(timestamp_s or 0.0))
        index = bisect.bisect_right(self.timestamps, target_time) - 1
        index = max(0, min(index, len(self.samples) - 1))
        return self.samples[index]["point"]
