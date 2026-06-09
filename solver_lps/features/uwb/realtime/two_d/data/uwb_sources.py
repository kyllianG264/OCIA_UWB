from solver_lps.features.cv.realtime.data.cv_realtime_source import CvRealtimeSource
from solver_lps.features.cv.review.data.cv_log_source import (
    DEFAULT_CV_CALIBRATION_PATH,
    DEFAULT_CV_LOG_PATH,
    DEFAULT_CV_VIDEO_PATH,
)
from solver_lps.features.uwb.realtime.data.udp_distance_receiver import UdpDistanceReceiver


class DistanceSource:
    def __init__(
        self,
        bind_ip="0.0.0.0",
        port=4210,
        max_age_s=2.0,
        cv_log_path=DEFAULT_CV_LOG_PATH,
        cv_calibration_path=DEFAULT_CV_CALIBRATION_PATH,
        cv_video_path=DEFAULT_CV_VIDEO_PATH,
        cv_player_id=None,
        cv_expected_player_count=None,
    ):
        self.receiver = UdpDistanceReceiver(bind_ip=bind_ip, port=port, max_age_s=max_age_s)
        self.cv_source = CvRealtimeSource(
            csv_path=cv_log_path,
            calibration_path=cv_calibration_path,
            video_path=cv_video_path,
            player_id=cv_player_id,
            expected_player_count=cv_expected_player_count,
        )

    @property
    def source_label(self):
        return "Realtime UWB + Vision" if self.cv_source is not None else "Realtime UWB"

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

    def get_distances(self, _tag_real, anchors, _settings):
        active_ids = sorted(anchors.keys())
        raw = self.receiver.get_distances(active_ids)
        packet = {
            "raw": raw,
            "valid": all(aid in raw for aid in active_ids),
            "source": "realtime",
            "status": self.receiver.get_status_text(active_ids).replace("udp", "realtime", 1),
        }
        return self._merge_cv_overlay(packet)

    def close(self):
        if self.cv_source is not None:
            self.cv_source.close()
        self.receiver.close()
