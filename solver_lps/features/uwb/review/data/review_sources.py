import bisect
import csv
import os
import time


_APP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
DEFAULT_UWB_REVIEW_LOG_PATH = os.path.join(_APP_ROOT, "features", "uwb", "review", "data", "uwb_review.csv")
DEFAULT_UWB_TAG_REVIEW_PATH = os.path.join(_APP_ROOT, "features", "uwb", "review", "data", "uwb_tag_review.csv")


def load_uwb_review_frames(csv_path):
    frames_by_time = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                timestamp_s = float(row.get("timestamp_s") or 0.0)
                anchor_id = int(row.get("anchor_id"))
                distance_cm = float(row.get("distance_cm"))
            except (TypeError, ValueError):
                continue
            bucket = frames_by_time.setdefault(timestamp_s, {})
            bucket[anchor_id] = distance_cm
    frames = []
    for index, timestamp_s in enumerate(sorted(frames_by_time.keys())):
        frames.append(
            {
                "sequence_index": index,
                "timestamp_s": timestamp_s,
                "distances": dict(frames_by_time[timestamp_s]),
            }
        )
    return frames


def load_tag_review_samples(csv_path):
    samples = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                timestamp_s = float(row.get("timestamp_s") or 0.0)
                x_cm = float(row.get("x_cm"))
                y_cm = float(row.get("y_cm"))
                z_cm = float(row.get("z_cm") or row.get("height_cm") or 0.0)
            except (TypeError, ValueError):
                continue
            samples.append(
                {
                    "timestamp_s": timestamp_s,
                    "x_cm": x_cm,
                    "y_cm": y_cm,
                    "z_cm": z_cm,
                }
            )
    return sorted(samples, key=lambda item: item["timestamp_s"])


class _BasePlaybackSource:
    def __init__(self, timestamps):
        self.frame_index = 0
        self.playback_started_at = time.monotonic()
        self.playback_position_s = 0.0
        self.playback_paused = False
        self.normalized_timestamps = timestamps
        positive_deltas = [
            self.normalized_timestamps[index] - self.normalized_timestamps[index - 1]
            for index in range(1, len(self.normalized_timestamps))
            if self.normalized_timestamps[index] > self.normalized_timestamps[index - 1]
        ]
        self.nominal_frame_period_s = min(positive_deltas) if positive_deltas else (1.0 / 25.0)
        self.playback_duration_s = (
            self.normalized_timestamps[-1] + self.nominal_frame_period_s if self.normalized_timestamps else 0.0
        )

    @property
    def playback_state(self):
        return {
            "paused": self.playback_paused,
            "position_s": self.playback_position_s,
            "duration_s": self.playback_duration_s,
            "frame_index": self.frame_index,
            "frame_count": len(self.normalized_timestamps),
        }

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
        if not self.normalized_timestamps:
            return self.playback_position_s
        target_index = max(0, min(len(self.normalized_timestamps) - 1, self.frame_index + int(delta_frames)))
        position_s = self.normalized_timestamps[target_index] if self.normalized_timestamps else 0.0
        self.set_playback(position_s=position_s, paused=True)
        self.frame_index = target_index
        return self.playback_position_s

    def resolve_index(self):
        if not self.normalized_timestamps:
            return 0
        if self.playback_duration_s <= 0.0:
            return min(self.frame_index, len(self.normalized_timestamps) - 1)
        if self.playback_paused:
            elapsed_s = self.playback_position_s
        else:
            elapsed_s = time.monotonic() - self.playback_started_at
            if elapsed_s >= self.playback_duration_s:
                elapsed_s = self.playback_duration_s
                self.playback_paused = True
            self.playback_position_s = elapsed_s
        index = bisect.bisect_right(self.normalized_timestamps, elapsed_s) - 1
        return max(0, min(index, len(self.normalized_timestamps) - 1))


class UwbReviewSource(_BasePlaybackSource):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.frames = load_uwb_review_frames(csv_path) if csv_path and os.path.exists(csv_path) else []
        if self.frames:
            first_timestamp = self.frames[0]["timestamp_s"]
            normalized_timestamps = [max(0.0, frame["timestamp_s"] - first_timestamp) for frame in self.frames]
        else:
            normalized_timestamps = []
        super().__init__(normalized_timestamps)

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
        raw = {int(aid): frame["distances"][int(aid)] for aid in active_ids if int(aid) in frame["distances"]}
        return {
            "raw": raw,
            "valid": all(int(aid) in raw for aid in active_ids),
            "source": "review",
            "status": f"review UWB: {os.path.basename(self.csv_path)} | frame {self.frame_index + 1}/{len(self.frames)}",
            "uwb_playback": self.playback_state,
            "timestamp_s": frame["timestamp_s"],
        }


class UwbTagReviewSource:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.samples = load_tag_review_samples(csv_path) if csv_path and os.path.exists(csv_path) else []
        if self.samples:
            first_timestamp = self.samples[0]["timestamp_s"]
            self.normalized_timestamps = [max(0.0, item["timestamp_s"] - first_timestamp) for item in self.samples]
        else:
            self.normalized_timestamps = []

    def get_position_at(self, position_s):
        if not self.samples:
            return None
        if len(self.samples) == 1:
            sample = self.samples[0]
            return sample["x_cm"], sample["y_cm"], sample["z_cm"]
        index = bisect.bisect_right(self.normalized_timestamps, float(position_s)) - 1
        index = max(0, min(index, len(self.samples) - 1))
        if index >= len(self.samples) - 1:
            sample = self.samples[index]
            return sample["x_cm"], sample["y_cm"], sample["z_cm"]
        next_index = index + 1
        t0 = self.normalized_timestamps[index]
        t1 = self.normalized_timestamps[next_index]
        if t1 <= t0:
            sample = self.samples[index]
            return sample["x_cm"], sample["y_cm"], sample["z_cm"]
        ratio = max(0.0, min(1.0, (float(position_s) - t0) / (t1 - t0)))
        current = self.samples[index]
        following = self.samples[next_index]
        return (
            current["x_cm"] + (following["x_cm"] - current["x_cm"]) * ratio,
            current["y_cm"] + (following["y_cm"] - current["y_cm"]) * ratio,
            current["z_cm"] + (following["z_cm"] - current["z_cm"]) * ratio,
        )


class TagReviewPlaybackSource(_BasePlaybackSource):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.samples = load_tag_review_samples(csv_path) if csv_path and os.path.exists(csv_path) else []
        if self.samples:
            first_timestamp = self.samples[0]["timestamp_s"]
            normalized_timestamps = [max(0.0, item["timestamp_s"] - first_timestamp) for item in self.samples]
        else:
            normalized_timestamps = []
        super().__init__(normalized_timestamps)

    def get_position(self):
        if not self.samples:
            return None
        self.frame_index = self.resolve_index()
        position_s = self.playback_state["position_s"]
        helper = UwbTagReviewSource(self.csv_path)
        helper.samples = self.samples
        helper.normalized_timestamps = self.normalized_timestamps
        return helper.get_position_at(position_s)
