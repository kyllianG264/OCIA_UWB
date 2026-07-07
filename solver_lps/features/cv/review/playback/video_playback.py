from __future__ import annotations

import bisect
import os
import time

try:
    import cv2
except ImportError:
    cv2 = None


def first_existing_path(*candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def resolve_side_video_path(reference_csv_path, explicit_path, filename):
    candidates = _side_video_candidates(reference_csv_path, explicit_path, filename)
    return first_existing_path(*candidates)


def _case_insensitive_file(directory, filename):
    if not directory or not os.path.isdir(directory):
        return None
    expected = filename.lower()
    try:
        return next(
            (os.path.join(directory, item) for item in os.listdir(directory) if item.lower() == expected),
            None,
        )
    except OSError:
        return None


def _side_video_candidates(reference_csv_path, explicit_path, filename):
    tracked_filename = filename.replace(".mp4", "_tracked.mp4")
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    if reference_csv_path:
        reference_dir = os.path.dirname(reference_csv_path)
        session_dir = os.path.abspath(os.path.join(reference_dir, ".."))
        output_dir = os.path.join(session_dir, "output")
        input_dir = os.path.join(session_dir, "input")
        side_name = "left_video.mp4" if filename.startswith("left_") else "right_video.mp4"
        candidates.extend(
            [
                os.path.join(reference_dir, tracked_filename),
                os.path.join(reference_dir, filename),
                os.path.join(output_dir, tracked_filename),
                os.path.join(output_dir, filename),
                _case_insensitive_file(input_dir, side_name),
            ]
        )
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


class TrackedVideoPlayback:
    def __init__(self, reference_csv_path, explicit_video_path, frames):
        self.frames = frames or []
        self.frame_index = 0
        self.video_scrubbing = False
        self.playback_started_at = time.monotonic()
        self.playback_position_s = 0.0
        self.playback_paused = False
        self.left_video_candidates = _side_video_candidates(
            reference_csv_path, explicit_video_path, "left_undistorted.mp4"
        )
        self.right_video_candidates = _side_video_candidates(
            reference_csv_path, None, "right_undistorted.mp4"
        )
        self.left_video_path = first_existing_path(*self.left_video_candidates)
        self.right_video_path = first_existing_path(*self.right_video_candidates)
        self.left_video_capture = None
        self.right_video_capture = None
        self.video_capture = None
        self._last_video_sequence_index = None
        self._last_video_frame = None
        self._last_left_video_frame = None
        self._last_right_video_frame = None

        if self.frames:
            timestamps = getattr(self.frames, "timestamps", None)
            if timestamps is None:
                timestamps = [frame["timestamp_s"] for frame in self.frames]
            first_timestamp = timestamps[0]
            self.normalized_timestamps = [max(0.0, timestamp - first_timestamp) for timestamp in timestamps]
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
        self._open_captures()

    def _open_captures(self):
        if cv2 is None:
            return
        self.left_video_capture, self.left_video_path = self._open_first_capture(self.left_video_candidates)
        self.right_video_capture, self.right_video_path = self._open_first_capture(self.right_video_candidates)
        self.video_capture = self.left_video_capture or self.right_video_capture

    @staticmethod
    def _open_first_capture(candidates):
        for candidate in candidates:
            if not candidate or not os.path.isfile(candidate):
                continue
            capture = cv2.VideoCapture(candidate)
            if capture.isOpened():
                return capture, candidate
            capture.release()
        return None, first_existing_path(*candidates)

    @property
    def has_video(self):
        return self.left_video_capture is not None or self.right_video_capture is not None

    @property
    def playback_state(self):
        return {
            "paused": self.playback_paused,
            "position_s": self.playback_position_s,
            "duration_s": self.playback_duration_s,
            "frame_index": self.frame_index,
            "frame_count": len(self.frames),
        }

    @property
    def video_frames(self):
        return {
            "video_frame": self._last_video_frame,
            "left_video_frame": self._last_left_video_frame,
            "right_video_frame": self._last_right_video_frame,
            "video_available": self._last_video_frame is not None,
        }

    def resolve_playback_index(self):
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

    def _read_capture_frame(self, capture, target_index):
        if capture is None or cv2 is None:
            return None
        if self._last_video_sequence_index is not None and target_index == (self._last_video_sequence_index + 1):
            ok, frame = capture.read()
        else:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            ok, frame = capture.read()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read_video_frame(self, sequence_index, include_video=True):
        if not include_video or not self.has_video or cv2 is None:
            return self._last_video_frame
        if self._last_video_sequence_index == sequence_index and self._last_video_frame is not None:
            return self._last_video_frame
        if self.video_scrubbing and self._last_video_frame is not None:
            return self._last_video_frame
        target_index = max(0, int(sequence_index))
        left_rgb = self._read_capture_frame(self.left_video_capture, target_index)
        right_rgb = self._read_capture_frame(self.right_video_capture, target_index)
        self._last_video_sequence_index = target_index
        self._last_left_video_frame = left_rgb
        self._last_right_video_frame = right_rgb
        self._last_video_frame = left_rgb if left_rgb is not None else right_rgb
        return self._last_video_frame

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
        target_index = max(0, min(len(self.frames) - 1, self.frame_index + int(delta_frames)))
        position_s = self.normalized_timestamps[target_index] if self.normalized_timestamps else 0.0
        self.set_playback(position_s=position_s, paused=True)
        self.frame_index = target_index
        return self.playback_position_s

    def set_video_scrubbing(self, active):
        self.video_scrubbing = bool(active)

    def close(self):
        for capture in (self.left_video_capture, self.right_video_capture, self.video_capture):
            if capture is None:
                continue
            try:
                capture.release()
            except Exception:
                pass
