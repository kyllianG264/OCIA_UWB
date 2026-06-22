import tempfile
import unittest
from pathlib import Path
from unittest import mock

from solver_lps.features.cv.review.data.cv_log_source import (
    CvLogSource,
    _detect_coordinate_space,
    _load_cv_frames,
    _resolve_related_path,
    _resolve_default_cv_log_path,
    _resolve_refreshable_cv_log_path,
    _resolve_terrain_image_path,
)
from solver_lps.features.ground.domain.court_geometry import court_bounds


class CvLogSourceMappingTests(unittest.TestCase):
    def test_map_position_to_scene_rotates_calibration_into_court_space(self):
        source = CvLogSource.__new__(CvLogSource)
        source.calibration = {"bounds": (0.0, 400.0, 0.0, 800.0), "terrain_png_size": [401, 801]}
        source.court_bounds = court_bounds(1250.0, 900.0)
        source.coordinate_space = "terrain_pixels"

        left, right, top, bottom = source.court_bounds

        self.assertEqual((left, bottom), source._map_position_to_scene(0.0, 0.0))
        self.assertEqual((right, bottom), source._map_position_to_scene(0.0, 800.0))
        self.assertEqual((left, top), source._map_position_to_scene(400.0, 0.0))
        self.assertEqual((right, top), source._map_position_to_scene(400.0, 800.0))
        self.assertEqual(((left + right) / 2.0, (top + bottom) / 2.0), source._map_position_to_scene(200.0, 400.0))

    def test_view_config_uses_world_bounds_with_named_jpg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = Path(temp_dir) / "terrain.png"
            jpg_path = Path(temp_dir) / "terrain-de-basket.jpg"
            png_path.write_text("png", encoding="utf-8")
            jpg_path.write_text("jpg", encoding="utf-8")

            source = CvLogSource.__new__(CvLogSource)
            source.calibration = {"bounds": (0.0, 400.0, 0.0, 800.0), "terrain_image_path": str(png_path), "split_y": 400.0}
            source.court_bounds = court_bounds(1250.0, 900.0)

            view = source.view_config

        self.assertEqual("world", view["mode"])
        self.assertEqual(source.court_bounds, view["bounds"])
        self.assertEqual(str(jpg_path), view["terrain_image_path"])
        self.assertIsNone(view["split_y"])

    def test_map_position_to_scene_keeps_solver_world_coordinates_unchanged(self):
        source = CvLogSource.__new__(CvLogSource)
        source.calibration = {"bounds": (0.0, 400.0, 0.0, 800.0), "terrain_png_size": [401, 801]}
        source.court_bounds = court_bounds(1250.0, 900.0)
        source.coordinate_space = "solver_world"

        self.assertEqual((1234.5, 678.0), source._map_position_to_scene(1234.5, 678.0))

    def test_map_position_to_scene_clamps_slightly_outside_terrain_pixels(self):
        source = CvLogSource.__new__(CvLogSource)
        source.calibration = {"bounds": (0.0, 400.0, 0.0, 800.0), "terrain_png_size": [401, 801]}
        source.court_bounds = court_bounds(1250.0, 900.0)
        source.coordinate_space = "terrain_pixels"

        left, right, top, bottom = source.court_bounds

        self.assertEqual((left, bottom), source._map_position_to_scene(-12.0, -20.0))
        self.assertEqual((right, top), source._map_position_to_scene(412.0, 840.0))

    def test_detect_coordinate_space_recognizes_terrain_pixels(self):
        calibration = {"bounds": (0.0, 396.0, 0.0, 735.0), "terrain_png_size": [399, 736]}
        stats = {"on_terrain_bounds": (0.0, 384.0, 0.0, 735.0)}

        self.assertEqual("terrain_pixels", _detect_coordinate_space(calibration, stats))

    def test_detect_coordinate_space_keeps_solver_world_values(self):
        calibration = {"bounds": (0.0, 396.0, 0.0, 735.0), "terrain_png_size": [399, 736]}
        stats = {"on_terrain_bounds": (-250.0, 2750.0, -150.0, 1950.0)}

        self.assertEqual("solver_world", _detect_coordinate_space(calibration, stats))

    def test_resolve_related_path_prefers_csv_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "positions_merged.csv"
            calibration_path = Path(temp_dir) / "calibration.json"
            csv_path.write_text("frame\n", encoding="utf-8")
            calibration_path.write_text("{}", encoding="utf-8")

            resolved = _resolve_related_path(str(csv_path), None, "calibration.json", "fallback.json")

        self.assertEqual(str(calibration_path), resolved)

    def test_resolve_terrain_image_path_prefers_named_jpg_in_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = Path(temp_dir) / "terrain.png"
            jpg_path = Path(temp_dir) / "terrain-de-basket.jpg"
            png_path.write_text("png", encoding="utf-8")
            jpg_path.write_text("jpg", encoding="utf-8")

            resolved = _resolve_terrain_image_path(str(png_path))

        self.assertEqual(str(jpg_path), resolved)

    def test_get_frame_packet_prefers_raw_positions_for_visual_alignment(self):
        source = CvLogSource.__new__(CvLogSource)
        source.calibration = {"bounds": (0.0, 400.0, 0.0, 800.0), "terrain_png_size": [401, 801]}
        source.coordinate_space = "terrain_pixels"
        source.court_bounds = court_bounds(1250.0, 900.0)
        source.frames = [
            {
                "frame": 0,
                "timestamp_s": 0.0,
                "sequence_index": 0,
                "players": [
                    {
                        "player_id": "P1",
                        "raw_player_id": "P1",
                        "source_player_ids": ["P1"],
                        "x": 250.0,
                        "y": 300.0,
                        "raw_x": 100.0,
                        "raw_y": 700.0,
                        "half": "gauche",
                        "on_terrain": True,
                    }
                ],
            }
        ]
        source.csv_path = "positions_merged.csv"
        source.all_player_ids = ["P1"]
        source.player_id = "P1"
        source.expected_player_count = None
        source.frame_index = 0
        source._last_smoothed_frame_index = None
        source._smoothed_scene_positions = {}
        source.video_capture = None
        source.playback_paused = True
        source.playback_position_s = 0.0
        source.playback_duration_s = 0.0
        source.normalized_timestamps = [0.0]
        source.video_scrubbing = False
        source._last_video_frame = None

        packet = source.get_frame_packet(include_video=False)
        expected = source._map_position_to_scene(100.0, 700.0)

        self.assertEqual(expected, (packet["cv_positions"][0]["x"], packet["cv_positions"][0]["y"]))
        self.assertEqual(expected, packet["primary_position"])

    def test_load_cv_frames_does_not_fall_back_to_raw_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "positions_raw.csv"
            csv_path.write_text("frame,timestamp_s,player_id,X,Y,cam,on_terrain\n1,0.0,7,10,20,left,1\n", encoding="utf-8")

            with mock.patch(
                "solver_lps.features.cv.review.data.cv_log_source.build_hungarian_kalman_cv_tracks",
                return_value=([], {"rows_out": 0}),
            ):
                frames = _load_cv_frames(str(csv_path), split_y=None, calibration_path=None, expected_player_count=10)

        self.assertEqual([], frames)

    def test_default_cv_log_path_prefers_raw_over_merged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            external_dir = Path(temp_dir) / "external"
            internal_dir = Path(temp_dir) / "internal"
            external_dir.mkdir()
            internal_dir.mkdir()
            external_raw = external_dir / "positions_raw.csv"
            external_merged = external_dir / "positions_merged.csv"
            internal_raw = internal_dir / "positions_raw.csv"
            internal_merged = internal_dir / "positions_merged.csv"
            external_merged.write_text("merged", encoding="utf-8")
            internal_raw.write_text("raw", encoding="utf-8")

            resolved = _resolve_default_cv_log_path(str(external_dir), str(internal_dir))

        self.assertEqual(str(internal_raw), resolved)

    def test_refreshable_cv_log_path_prefers_neighbor_raw(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            merged_path = Path(temp_dir) / "positions_merged.csv"
            raw_path = Path(temp_dir) / "positions_raw.csv"
            merged_path.write_text("merged", encoding="utf-8")
            raw_path.write_text("raw", encoding="utf-8")

            resolved = _resolve_refreshable_cv_log_path(str(merged_path))

        self.assertEqual(str(raw_path), resolved)

    def test_cv_log_source_switches_merged_input_to_raw_neighbor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            merged_path = Path(temp_dir) / "positions_merged.csv"
            raw_path = Path(temp_dir) / "positions_raw.csv"
            raw_path.write_text("frame,timestamp_s,player_id,X,Y,cam,on_terrain\n1,0.0,7,10,20,left,1\n", encoding="utf-8")
            merged_path.write_text("frame,timestamp_s,player_id,X,Y,cam,on_terrain\n1,0.0,7,10,20,left,1\n", encoding="utf-8")

            with mock.patch(
                "solver_lps.features.cv.review.data.cv_log_source._load_cv_frames",
                return_value=[],
            ) as mocked_load:
                source = CvLogSource(str(merged_path))

        mocked_load.assert_called_once()
        self.assertEqual(str(raw_path), source.csv_path)


if __name__ == "__main__":
    unittest.main()
