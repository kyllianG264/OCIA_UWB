import os

from solver_lps.features.cv.review.data.cv_log_source import (
    DEFAULT_CV_CALIBRATION_PATH,
    DEFAULT_CV_LOG_PATH,
    DEFAULT_CV_VIDEO_PATH,
    CvLogSource,
)
from solver_lps.features.uwb.review.data.review_sources import (
    DEFAULT_UWB_REVIEW_LOG_PATH,
    DEFAULT_UWB_TAG_REVIEW_PATH,
    UwbReviewSource,
    UwbTagReviewSource,
)


class DistanceSource:
    def __init__(
        self,
        uwb_review_log_path=DEFAULT_UWB_REVIEW_LOG_PATH,
        uwb_tag_review_path=DEFAULT_UWB_TAG_REVIEW_PATH,
        cv_log_path=DEFAULT_CV_LOG_PATH,
        cv_calibration_path=DEFAULT_CV_CALIBRATION_PATH,
        cv_video_path=DEFAULT_CV_VIDEO_PATH,
        cv_player_id=None,
        cv_expected_player_count=None,
        review_data_mode="both",
        **_kwargs,
    ):
        review_data_mode = str(review_data_mode or "both").strip().lower()
        if review_data_mode not in {"uwb", "cv", "both"}:
            review_data_mode = "both"
        self.review_data_mode = review_data_mode
        self.use_uwb = review_data_mode in {"uwb", "both"}
        self.use_cv = review_data_mode in {"cv", "both"}
        uwb_review_resolved = uwb_review_log_path or DEFAULT_UWB_REVIEW_LOG_PATH
        uwb_tag_review_resolved = uwb_tag_review_path or DEFAULT_UWB_TAG_REVIEW_PATH
        cv_log_resolved = cv_log_path or DEFAULT_CV_LOG_PATH
        cv_calibration_resolved = cv_calibration_path or DEFAULT_CV_CALIBRATION_PATH
        cv_video_resolved = cv_video_path or DEFAULT_CV_VIDEO_PATH

        self.uwb_review_source = (
            UwbReviewSource(uwb_review_resolved)
            if self.use_uwb and os.path.exists(uwb_review_resolved)
            else None
        )
        self.uwb_tag_review_source = (
            UwbTagReviewSource(uwb_tag_review_resolved)
            if self.use_uwb and os.path.exists(uwb_tag_review_resolved)
            else None
        )
        self.cv_source = None
        if self.use_cv and os.path.exists(cv_log_resolved):
            self.cv_source = CvLogSource(
                cv_log_resolved,
                calibration_path=cv_calibration_resolved,
                player_id=cv_player_id,
                video_path=cv_video_resolved,
                expected_player_count=cv_expected_player_count,
            )

    @property
    def source_label(self):
        if self.uwb_review_source is not None and self.cv_source is not None:
            return "Review UWB + CV"
        if self.cv_source is not None:
            return "Review CV"
        return "Review UWB"

    @property
    def view_config(self):
        return None if self.cv_source is None else self.cv_source.view_config

    @property
    def analytics_bounds(self):
        return None if self.cv_source is None else self.cv_source.analytics_bounds

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
        if self.uwb_review_source is None:
            return self.cv_source.toggle_pause() if self.cv_source is not None else None
        state = self.uwb_review_source.toggle_pause()
        if self.cv_source is not None:
            self.cv_source.set_playback(position_s=self.uwb_review_source.playback_position_s, paused=state)
        return state

    def seek_review_relative(self, delta_s):
        if self.uwb_review_source is None:
            return self.cv_source.seek_relative(delta_s) if self.cv_source is not None else None
        position_s = self.uwb_review_source.seek_relative(delta_s)
        if self.cv_source is not None:
            self.cv_source.set_playback(position_s=position_s, paused=True)
        return position_s

    def seek_review_absolute(self, position_s, paused=True):
        if self.uwb_review_source is None:
            return self.cv_source.seek_absolute(position_s, paused=paused) if self.cv_source is not None else None
        position_s = self.uwb_review_source.seek_absolute(position_s, paused=paused)
        if self.cv_source is not None:
            self.cv_source.set_playback(position_s=position_s, paused=paused)
        return position_s

    def seek_review_frames(self, delta_frames):
        if self.uwb_review_source is None:
            return self.cv_source.seek_frames(delta_frames) if self.cv_source is not None else None
        position_s = self.uwb_review_source.seek_frames(delta_frames)
        if self.cv_source is not None:
            self.cv_source.set_playback(position_s=position_s, paused=True)
        return position_s

    def set_review_video_scrubbing(self, active):
        if self.cv_source is None:
            return None
        self.cv_source.set_video_scrubbing(active)
        return self.cv_source.video_scrubbing

    def get_distances(self, _tag_real, anchors, _settings):
        active_ids = sorted(anchors.keys())
        if self.uwb_review_source is not None:
            packet = self.uwb_review_source.get_packet(active_ids)
            if self.uwb_tag_review_source is not None:
                playback = packet.get("uwb_playback") or {}
                packet["tag_real"] = self.uwb_tag_review_source.get_position_at(playback.get("position_s", 0.0))
            if self.cv_source is not None:
                playback = packet.get("uwb_playback")
                if playback is not None:
                    self.cv_source.set_playback(position_s=playback.get("position_s"), paused=playback.get("paused"))
            return self._merge_cv_overlay(packet)

        packet = {
            "raw": {},
            "valid": False,
            "source": "review",
            "status": "review: aucun log UWB charge" if self.cv_source is None else "review CV",
            "uwb_playback": None,
        }
        if self.cv_source is not None:
            return self._merge_cv_overlay(packet)
        if self.uwb_tag_review_source is not None:
            packet["tag_real"] = self.uwb_tag_review_source.get_position_at(0.0)
        return self._merge_cv_overlay(packet)

    def close(self):
        if self.cv_source is not None and self.cv_source.video_capture is not None:
            self.cv_source.video_capture.release()
