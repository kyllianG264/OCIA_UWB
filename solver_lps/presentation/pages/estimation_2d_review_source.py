"""Assembly adapter for synchronized CV and UWB review sources."""

from __future__ import annotations

from pathlib import Path

from solver_lps.features.ground.domain.calibration import load_terrain_calibration
from solver_lps.features.ground.domain.court_geometry import court_bounds
from solver_lps.features.ground.domain.projection import project_court_cm_to_terrain_pixels
from solver_lps.features.cv.review.playback import TrackedVideoPlayback
from solver_lps.features.cv.grayzone.data.gray_zone_repository import (
    default_gray_zone_repository,
    load_gray_zone_overlay,
)
from solver_lps.features.players.data.merged_input import TrackedPlayersCsvSource
from solver_lps.features.uwb.acquisition.data.review_input import UwbReviewSource, UwbTagReviewSource
from solver_lps.session_assets import DEFAULT_SET, DEFAULT_SPORT, SessionAssets, first_existing_path

from .review_clock import ReviewClock


def _first_video_path(assets, explicit_path=None):
    return first_existing_path(
        explicit_path,
        assets.output_dir / "left_undistorted_tracked.mp4",
        assets.output_dir / "left_undistorted.mp4",
    )


class DistanceSource:
    def __init__(
        self,
        uwb_review_log_path=None,
        uwb_tag_review_path=None,
        uwb_merged_path=None,
        cv_log_path=None,
        cv_calibration_path=None,
        cv_video_path=None,
        cv_player_id=None,
        cv_expected_player_count=None,
        review_data_mode="both",
        sport=DEFAULT_SPORT,
        asset_set=DEFAULT_SET,
        solver_mode="2d",
        **_kwargs,
    ):
        self.assets = SessionAssets(sport=sport, asset_set=asset_set)
        self.sport = self.assets.sport
        self.asset_set = self.assets.asset_set
        normalized_review_mode = str(review_data_mode or "both").strip().lower()
        self.review_data_mode = normalized_review_mode if normalized_review_mode in {"uwb", "cv", "both"} else "both"
        self.use_uwb = self.review_data_mode in {"uwb", "both"}
        self.solver_mode = str(solver_mode or "2d")

        calibration_path = first_existing_path(cv_calibration_path, self.assets.calibration_path)
        self.ground_calibration = load_terrain_calibration(str(calibration_path))
        self.ground_bounds = (
            tuple(self.ground_calibration.get("bounds"))
            if self.ground_calibration and self.ground_calibration.get("bounds") is not None
            else court_bounds(1250.0, 900.0)
        )
        self.cv_calibration_resolved = str(calibration_path)
        self.cv_video_resolved = _first_video_path(self.assets, cv_video_path)
        self._cv_player_id = cv_player_id
        self._cv_expected_player_count = cv_expected_player_count

        explicit_cv_path = Path(cv_log_path) if cv_log_path else None
        self.cv_raw_log_path = explicit_cv_path if explicit_cv_path and explicit_cv_path.name == "positions_raw.csv" else self.assets.cv_positions_raw_path
        self.cv_merged_log_path = explicit_cv_path if explicit_cv_path and explicit_cv_path.name != "positions_raw.csv" else self.assets.cv_positions_merged_path
        self.cv_raw_available = self.cv_raw_log_path.exists()
        self.cv_merged_available = self.cv_merged_log_path.exists()
        self.cv_view_mode = "merged" if self.cv_merged_available else "raw"

        uwb_tag_path = Path(uwb_tag_review_path) if uwb_tag_review_path else self.assets.uwb_tag_review_path
        self.uwb_raw_path = Path(uwb_review_log_path) if uwb_review_log_path else self.assets.uwb_raw_path
        self.uwb_raw_available = self.use_uwb and self.uwb_raw_path.exists()
        self.uwb_review_source = UwbReviewSource(self.uwb_raw_path) if self.uwb_raw_available else None
        self.uwb_tag_review_source = UwbTagReviewSource(uwb_tag_path) if self.use_uwb and uwb_tag_path.exists() else None
        calculation_mode = {"2d": "two_d", "3d": "three_d", "3d_to_2d": "three_d_to_2d"}[self.solver_mode]
        self.uwb_merged_path = Path(uwb_merged_path) if uwb_merged_path else self.assets.uwb_positions_path(calculation_mode)
        self.uwb_player_source = (
            TrackedPlayersCsvSource(str(self.uwb_merged_path))
            if self.use_uwb and self.uwb_merged_path.exists()
            else None
        )
        self.uwb_merged_available = self.uwb_player_source is not None
        self.uwb_view_mode = "merged" if self.uwb_merged_available else "raw"
        self.cv_review_source = self._build_cv_review_source(self.cv_view_mode) if self.review_data_mode in {"cv", "both"} else None
        self.cv_video_playback = self._build_cv_video_playback()
        self.review_clock = ReviewClock(self._playback_sources())

    def _playback_sources(self):
        active_uwb_source = self.uwb_review_source if self.uwb_view_mode == "raw" else self.uwb_player_source
        return [source for source in (self.cv_review_source, active_uwb_source) if source is not None]

    @property
    def source_label(self):
        return {"cv": "Review CV", "uwb": "Review UWB"}.get(self.review_data_mode, "Review CV + UWB")

    @property
    def view_config(self):
        sport = getattr(self, "sport", DEFAULT_SPORT)
        repository = default_gray_zone_repository(sport)
        gray_overlay = load_gray_zone_overlay(sport)
        return {
            "mode": "world",
            "bounds": self.ground_bounds,
            "terrain_image_path": str(self.assets.terrain_path) if self.assets.terrain_path.exists() else None,
            "coord_transform": "flip_x_flip_y",
            "gray_heatmap": gray_overlay,
            "gray_zone_cells": [] if gray_overlay is None else gray_overlay.get("zone_cells", []),
            "gray_zone_input_available": Path(repository.raw_positions_path).is_file(),
        }

    @property
    def analytics_bounds(self):
        return self.ground_bounds

    def _cv_log_path_for_mode(self, mode):
        path = self.cv_raw_log_path if mode == "raw" else self.cv_merged_log_path
        return path if path.exists() else None

    def _build_cv_review_source(self, mode):
        csv_path = self._cv_log_path_for_mode(mode)
        if csv_path is None:
            return None
        expected_count = None if mode == "raw" else self._cv_expected_player_count
        return TrackedPlayersCsvSource(
            str(csv_path),
            player_id=self._cv_player_id,
            expected_player_count=expected_count,
        )

    def _build_cv_video_playback(self):
        if self.cv_review_source is None:
            return None
        return TrackedVideoPlayback(
            self.cv_review_source.csv_path,
            None if self.cv_video_resolved is None else str(self.cv_video_resolved),
            self.cv_review_source.frames,
        )

    def _clock_playback(self, playback):
        synchronized = dict(playback or {})
        synchronized.update(self.review_clock.state)
        return synchronized

    def _cv_packet(self):
        if self.cv_review_source is None:
            return None
        packet = self.cv_review_source.get_frame_packet()
        video_playback = getattr(self, "cv_video_playback", None)
        if video_playback is not None:
            video_playback.read_video_frame(self.cv_review_source.frame_index)
            packet.update(video_playback.video_frames)
        if self.cv_view_mode == "raw":
            for position in packet.get("player_positions", []):
                position["x"] = position.get("raw_x", position["x"])
                position["y"] = position.get("raw_y", position["y"])
        packet["playback"] = self._clock_playback(packet.get("playback"))
        return packet

    def _uwb_player_packet(self):
        if self.uwb_view_mode != "merged" or self.uwb_player_source is None:
            return None
        packet = self.uwb_player_source.get_frame_packet()
        if self.sport == "basket":
            world_bounds = court_bounds(1250.0, 900.0)
            for position in packet.get("player_positions", []):
                position["world_x_cm"] = position["x"]
                position["world_y_cm"] = position["y"]
                position["x"], position["y"] = project_court_cm_to_terrain_pixels(
                    position["x"],
                    position["y"],
                    world_bounds,
                    self.ground_bounds,
                )
                position["heatmap_x"] = position["x"]
                position["heatmap_y"] = position["y"]
        packet["playback"] = self._clock_playback(packet.get("playback"))
        return packet

    def _uwb_raw_packet(self, active_ids):
        source = self.uwb_review_source
        if self.uwb_view_mode != "raw" or source is None or not source.frames:
            return None
        frame_index = source.resolve_index()
        frame = source.frames[frame_index]
        distances = {
            anchor_id: frame["distances"][anchor_id]
            for anchor_id in active_ids
            if anchor_id in frame["distances"]
        }
        playback = dict(source.playback_state)
        playback["frame_index"] = frame_index
        return {
            "raw": distances,
            "valid": all(anchor_id in distances for anchor_id in active_ids),
            "source": "review",
            "status": f"review UWB raw: frame {frame_index + 1}/{len(source.frames)}",
            "uwb_playback": self._clock_playback(playback),
        }

    def _merge_cv_overlay(self, packet):
        cv_packet = self._cv_packet() or {}
        uwb_packet = self._uwb_player_packet() or {}
        player_positions = [
            *cv_packet.get("player_positions", []),
            *uwb_packet.get("player_positions", []),
        ]
        all_player_ids = list(dict.fromkeys([
            *cv_packet.get("all_player_ids", []),
            *uwb_packet.get("all_player_ids", []),
        ]))
        selected_packet = cv_packet if cv_packet.get("selected_player_id") is not None else uwb_packet
        packet.update(
            cv_positions=player_positions,
            selected_player_id=selected_packet.get("selected_player_id"),
            selection_mode=selected_packet.get("selection_mode", "single"),
            all_player_ids=all_player_ids,
            selected_player_visible=selected_packet.get("selected_player_visible", False),
            primary_position=selected_packet.get("primary_position"),
            cv_status=cv_packet.get("status"),
            cv_video_frame=cv_packet.get("video_frame"),
            cv_left_video_frame=cv_packet.get("left_video_frame"),
            cv_right_video_frame=cv_packet.get("right_video_frame"),
            cv_video_available=cv_packet.get("video_available", False),
            cv_playback=cv_packet.get("playback"),
            uwb_playback=uwb_packet.get("playback") or packet.get("uwb_playback"),
            uwb_merged_review=bool(uwb_packet),
            uwb_view_mode=self.uwb_view_mode,
            uwb_raw_available=self.uwb_raw_available,
            uwb_merged_available=self.uwb_merged_available,
            cv_view_mode=self.cv_view_mode,
            cv_raw_available=self.cv_raw_available,
            cv_merged_available=self.cv_merged_available,
            review_mode=self.review_data_mode,
        )
        if self.review_data_mode == "cv":
            packet.update(
                valid=cv_packet.get("valid", False),
                source=cv_packet.get("source", "players"),
                status=cv_packet.get("status", "players: aucun log charge"),
                uwb_playback=None,
            )
        elif self.review_data_mode == "uwb":
            if self.uwb_view_mode == "merged":
                packet.update(
                    valid=uwb_packet.get("valid", False),
                    source=uwb_packet.get("source", "players"),
                    status=uwb_packet.get("status", "players UWB: aucun merged charge"),
                )
        return packet

    def set_cv_player(self, player_id):
        self._cv_player_id = player_id
        for source in (self.cv_review_source, self.uwb_player_source):
            if source is not None:
                source.set_selected_player(player_id)

    def cycle_cv_player(self, step=1):
        source = self.cv_review_source or self.uwb_player_source
        if source is not None:
            selected = source.cycle_selected_player(step)
            self.set_cv_player(selected)
            return selected
        return None

    def set_cv_expected_player_count(self, count):
        self._cv_expected_player_count = count
        if self.cv_review_source is not None and self.cv_view_mode == "merged":
            return self.cv_review_source.set_expected_player_count(count)
        return None

    def set_cv_view_mode(self, mode):
        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode not in {"raw", "merged"} or self._cv_log_path_for_mode(normalized_mode) is None:
            return self.cv_view_mode
        previous_player_id = self._cv_player_id
        if self.cv_review_source is not None:
            previous_player_id = self.cv_review_source.player_id
        next_source = self._build_cv_review_source(normalized_mode)
        if next_source is None:
            return self.cv_view_mode
        if self.cv_review_source is not None:
            self.cv_review_source.close()
        if self.cv_video_playback is not None:
            self.cv_video_playback.close()
        self.cv_view_mode = normalized_mode
        self.cv_review_source = next_source
        self.cv_video_playback = self._build_cv_video_playback()
        if previous_player_id is not None:
            self.cv_review_source.set_selected_player(previous_player_id)
        self.review_clock.replace_sources(self._playback_sources())
        self.review_clock.seek_absolute(0.0, paused=False)
        return self.cv_view_mode

    def set_uwb_view_mode(self, mode):
        normalized_mode = str(mode or "").strip().lower()
        available = {
            "raw": self.uwb_raw_available,
            "merged": self.uwb_merged_available,
        }
        if normalized_mode not in available or not available[normalized_mode]:
            return self.uwb_view_mode
        self.uwb_view_mode = normalized_mode
        self.review_clock.replace_sources(self._playback_sources())
        self.review_clock.seek_absolute(0.0, paused=False)
        return self.uwb_view_mode

    def set_review_view_mode(self, mode):
        if self.review_data_mode == "uwb":
            return self.set_uwb_view_mode(mode)
        return self.set_cv_view_mode(mode)

    def toggle_review_pause(self):
        return self.review_clock.toggle_pause()

    def seek_review_relative(self, delta_s):
        return self.review_clock.seek_relative(delta_s)

    def seek_review_absolute(self, position_s, paused=True):
        return self.review_clock.seek_absolute(position_s, paused=paused)

    def seek_review_frames(self, delta_frames):
        return self.review_clock.seek_frames(delta_frames)

    def set_review_video_scrubbing(self, active):
        video_playback = getattr(self, "cv_video_playback", None)
        if video_playback is not None:
            video_playback.set_video_scrubbing(active)

    def get_distances(self, _tag_real, anchors, _settings):
        self.review_clock.sample()
        packet = self._uwb_raw_packet(sorted(anchors)) or {
            "raw": {},
            "valid": False,
            "source": "players",
            "status": "review merged",
        }
        if self.uwb_tag_review_source is not None:
            packet["tag_real"] = self.uwb_tag_review_source.get_position_at(self.review_clock.position_s)
        return self._merge_cv_overlay(packet)

    def close(self):
        if self.cv_review_source is not None:
            self.cv_review_source.close()
        if self.uwb_player_source is not None:
            self.uwb_player_source.close()
        video_playback = getattr(self, "cv_video_playback", None)
        if video_playback is not None:
            video_playback.close()
