import bisect
import csv
import os
import random
import sys
import time


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from uwb_udp import UdpDistanceReceiver

try:
    from cv_sources import (
        DEFAULT_CV_CALIBRATION_PATH,
        DEFAULT_CV_LOG_PATH,
        DEFAULT_CV_VIDEO_PATH,
        CvLogSource,
        distance_2d,
    )
except ImportError:
    from estimation_2d.cv_sources import (
        DEFAULT_CV_CALIBRATION_PATH,
        DEFAULT_CV_LOG_PATH,
        DEFAULT_CV_VIDEO_PATH,
        CvLogSource,
        distance_2d,
    )


DEFAULT_UWB_REVIEW_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "uwb_review.csv")


def noisy_distance(true_d, settings):
    d = true_d + random.gauss(0.0, settings["noise_std"])
    if random.random() < settings["spike_prob"]:
        d += random.uniform(-settings["spike_amplitude"], settings["spike_amplitude"])
    return max(1.0, d)


def get_simulated_distances(tag_real, anchors, settings):
    raw = {}
    for aid, anchor_pos in anchors.items():
        raw[aid] = noisy_distance(distance_2d(tag_real, anchor_pos), settings)
    return {
        "raw": raw,
        "valid": True,
        "source": "simulation",
        "status": "simulation",
    }


def _load_uwb_review_frames(csv_path):
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


class UwbReviewSource:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.frames = _load_uwb_review_frames(csv_path) if csv_path and os.path.exists(csv_path) else []
        self.frame_index = 0
        self.playback_started_at = time.monotonic()
        self.playback_position_s = 0.0
        self.playback_paused = False
        if self.frames:
            first_timestamp = self.frames[0]["timestamp_s"]
            self.normalized_timestamps = [max(0.0, frame["timestamp_s"] - first_timestamp) for frame in self.frames]
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

    @property
    def playback_state(self):
        return {
            "paused": self.playback_paused,
            "position_s": self.playback_position_s,
            "duration_s": self.playback_duration_s,
            "frame_index": self.frame_index,
            "frame_count": len(self.frames),
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
        if not self.frames:
            return self.playback_position_s
        target_index = max(0, min(len(self.frames) - 1, self.frame_index + int(delta_frames)))
        position_s = self.normalized_timestamps[target_index] if self.normalized_timestamps else 0.0
        self.set_playback(position_s=position_s, paused=True)
        self.frame_index = target_index
        return self.playback_position_s

    def _resolve_index(self):
        if not self.frames:
            return 0
        if self.playback_duration_s <= 0.0:
            return min(self.frame_index, len(self.frames) - 1)
        if self.playback_paused:
            elapsed_s = self.playback_position_s
        else:
            elapsed_s = (time.monotonic() - self.playback_started_at) % self.playback_duration_s
            self.playback_position_s = elapsed_s
        index = bisect.bisect_right(self.normalized_timestamps, elapsed_s) - 1
        return max(0, min(index, len(self.frames) - 1))

    def get_packet(self, active_ids):
        if not self.frames:
            return {
                "raw": {},
                "valid": False,
                "source": "review",
                "status": "review: aucun log UWB charge",
                "uwb_playback": self.playback_state,
            }
        self.frame_index = self._resolve_index()
        frame = self.frames[self.frame_index]
        raw = {int(aid): frame["distances"][int(aid)] for aid in active_ids if int(aid) in frame["distances"]}
        return {
            "raw": raw,
            "valid": all(int(aid) in raw for aid in active_ids),
            "source": "review",
            "status": f"review UWB: {os.path.basename(self.csv_path)} | frame {self.frame_index + 1}/{len(self.frames)}",
            "uwb_playback": self.playback_state,
        }


class DistanceSource:
    def __init__(
        self,
        mode="simulation",
        bind_ip="0.0.0.0",
        port=4210,
        max_age_s=2.0,
        uwb_review_log_path=DEFAULT_UWB_REVIEW_LOG_PATH,
        cv_log_path=DEFAULT_CV_LOG_PATH,
        cv_calibration_path=DEFAULT_CV_CALIBRATION_PATH,
        cv_video_path=DEFAULT_CV_VIDEO_PATH,
        cv_player_id=None,
        cv_expected_player_count=None,
    ):
        self.mode = mode
        self.receiver = None
        self.uwb_review_source = None
        self.cv_source = None
        uwb_review_resolved = uwb_review_log_path or DEFAULT_UWB_REVIEW_LOG_PATH
        cv_log_resolved = cv_log_path or DEFAULT_CV_LOG_PATH
        cv_calibration_resolved = cv_calibration_path or DEFAULT_CV_CALIBRATION_PATH
        cv_video_resolved = cv_video_path or DEFAULT_CV_VIDEO_PATH
        if mode == "udp":
            self.receiver = UdpDistanceReceiver(bind_ip=bind_ip, port=port, max_age_s=max_age_s)
        if mode == "review" and os.path.exists(uwb_review_resolved):
            self.uwb_review_source = UwbReviewSource(uwb_review_resolved)
        if os.path.exists(cv_log_resolved):
            self.cv_source = CvLogSource(
                cv_log_resolved,
                calibration_path=cv_calibration_resolved,
                player_id=cv_player_id,
                video_path=cv_video_resolved,
                expected_player_count=cv_expected_player_count,
            )

    @property
    def uses_simulated_tag(self):
        return self.mode == "simulation"

    @property
    def source_label(self):
        if self.cv_source is not None and self.mode == "udp":
            return "Distances UWB + Vision"
        if self.cv_source is not None and self.mode == "review":
            return "Review UWB + Vision"
        if self.cv_source is not None and self.mode == "simulation":
            return "Simulation + Vision"
        if self.mode == "udp":
            return "Distances UWB"
        if self.mode == "review":
            return "Review UWB"
        return "Estimation UWB"

    @property
    def view_config(self):
        if self.cv_source is None:
            return None
        return self.cv_source.view_config

    @property
    def analytics_bounds(self):
        if self.cv_source is None:
            return None
        return self.cv_source.analytics_bounds

    def _merge_cv_overlay(self, packet):
        if self.cv_source is None:
            return packet
        cv_packet = self.cv_source.get_frame_packet(include_video=True)
        packet["cv_positions"] = list(cv_packet.get("cv_positions", []))
        packet["selected_player_id"] = cv_packet.get("selected_player_id")
        packet["selection_mode"] = cv_packet.get("selection_mode", "single")
        packet["all_player_ids"] = list(cv_packet.get("all_player_ids", []))
        packet["selected_player_visible"] = cv_packet.get("selected_player_visible", False)
        packet["primary_position"] = cv_packet.get("primary_position")
        packet["cv_status"] = cv_packet.get("status")
        packet["cv_video_frame"] = cv_packet.get("video_frame")
        packet["cv_video_available"] = cv_packet.get("video_available", False)
        packet["cv_playback"] = cv_packet.get("playback")
        return packet

    def set_cv_player(self, player_id):
        if self.cv_source is None:
            return None
        self.cv_source.set_selected_player(player_id)
        return self.cv_source.player_id

    def cycle_cv_player(self, step=1):
        if self.cv_source is None:
            return None
        return self.cv_source.cycle_selected_player(step=step)

    def set_cv_expected_player_count(self, count):
        if self.cv_source is None:
            return None
        return self.cv_source.set_expected_player_count(count)

    def toggle_review_pause(self):
        state = None
        if self.uwb_review_source is not None:
            state = self.uwb_review_source.toggle_pause()
            if self.cv_source is not None:
                self.cv_source.set_playback(position_s=self.uwb_review_source.playback_position_s, paused=state)
            return state
        if self.cv_source is not None and self.mode == "review":
            return self.cv_source.toggle_pause()
        return None

    def seek_review_relative(self, delta_s):
        position_s = None
        if self.uwb_review_source is not None:
            position_s = self.uwb_review_source.seek_relative(delta_s)
            if self.cv_source is not None:
                self.cv_source.set_playback(position_s=position_s, paused=True)
            return position_s
        if self.cv_source is not None and self.mode == "review":
            return self.cv_source.seek_relative(delta_s)
        return None

    def seek_review_absolute(self, position_s, paused=True):
        if self.uwb_review_source is not None:
            position_s = self.uwb_review_source.seek_absolute(position_s, paused=paused)
            if self.cv_source is not None:
                self.cv_source.set_playback(position_s=position_s, paused=paused)
            return position_s
        if self.cv_source is not None and self.mode == "review":
            return self.cv_source.seek_absolute(position_s, paused=paused)
        return None

    def seek_review_frames(self, delta_frames):
        position_s = None
        if self.uwb_review_source is not None:
            position_s = self.uwb_review_source.seek_frames(delta_frames)
            if self.cv_source is not None:
                self.cv_source.set_playback(position_s=position_s, paused=True)
            return position_s
        if self.cv_source is not None and self.mode == "review":
            return self.cv_source.seek_frames(delta_frames)
        return None

    def set_review_video_scrubbing(self, active):
        if self.cv_source is None:
            return None
        self.cv_source.set_video_scrubbing(active)
        return self.cv_source.video_scrubbing

    def get_distances(self, tag_real, anchors, settings):
        if self.mode == "review":
            active_ids = sorted(anchors.keys())
            if self.uwb_review_source is not None:
                packet = self.uwb_review_source.get_packet(active_ids)
                if self.cv_source is not None:
                    playback = packet.get("uwb_playback")
                    if playback is not None:
                        self.cv_source.set_playback(position_s=playback.get("position_s"), paused=playback.get("paused"))
                return self._merge_cv_overlay(packet)
            packet = {
                "raw": {},
                "valid": False,
                "source": "review",
                "status": "review: aucun log UWB charge",
                "uwb_playback": None,
            }
            return self._merge_cv_overlay(packet)

        if self.mode != "udp":
            return self._merge_cv_overlay(get_simulated_distances(tag_real, anchors, settings))

        active_ids = sorted(anchors.keys())
        raw = self.receiver.get_distances(active_ids)
        return self._merge_cv_overlay(
            {
                "raw": raw,
                "valid": all(aid in raw for aid in active_ids),
                "source": "udp",
                "status": self.receiver.get_status_text(active_ids),
            }
        )

    def close(self):
        if self.cv_source is not None and self.cv_source.video_capture is not None:
            self.cv_source.video_capture.release()
        if self.receiver is not None:
            self.receiver.close()
