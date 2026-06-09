CV_ALL_PLAYERS = "__all__"


class CvRealtimeSource:
    def __init__(self, player_id=None, expected_player_count=None, **_kwargs):
        self.player_id = player_id or CV_ALL_PLAYERS
        self.expected_player_count = expected_player_count
        self.video_scrubbing = False

    @property
    def view_config(self):
        return None

    @property
    def analytics_bounds(self):
        return None

    @property
    def status(self):
        return "vision realtime: aucune source live branchee"

    def set_selected_player(self, player_id):
        self.player_id = player_id or CV_ALL_PLAYERS

    def cycle_selected_player(self, step=1):
        return self.player_id

    def set_expected_player_count(self, count):
        self.expected_player_count = count
        return count

    def set_video_scrubbing(self, active):
        self.video_scrubbing = bool(active)

    def get_frame_packet(self, include_video=True):
        return {
            "raw": {},
            "valid": False,
            "source": "vision",
            "status": self.status,
            "cv_positions": [],
            "selected_player_id": None if self.player_id == CV_ALL_PLAYERS else self.player_id,
            "selection_mode": "all" if self.player_id == CV_ALL_PLAYERS else "single",
            "all_player_ids": [],
            "selected_player_visible": False,
            "primary_position": None,
            "timestamp_s": 0.0,
            "video_frame": None,
            "video_available": False,
            "playback": None,
        }

    def close(self):
        return None
