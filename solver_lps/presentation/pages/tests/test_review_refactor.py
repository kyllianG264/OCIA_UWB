import tempfile
import unittest
import csv
from pathlib import Path
from unittest.mock import Mock, patch

from solver_lps.features.uwb.calculus.domain.two_d.position_calcul import (
    update_position_solution as uwb_update_position_solution,
)
from solver_lps.presentation.pages import estimation_2d_review_ui
from solver_lps.presentation.pages import estimation_2d_page
from solver_lps.presentation.pages.estimation_2d_review_ui import analytics_sample, compose_review_solution
from solver_lps.presentation.pages.estimation_2d_review_source import DistanceSource
from solver_lps.presentation.pages.review_clock import ReviewClock
from solver_lps.presentation.navigation import ANALYZE_GRAY_ZONE, REGENERATE_MERGED, RETURN_HOME


class FakePlaybackSource:
    def __init__(self, duration_s=10.0, frame_period_s=0.1):
        self.playback_duration_s = duration_s
        self.nominal_frame_period_s = frame_period_s
        self.samples = []

    def set_playback(self, position_s=None, paused=None):
        self.samples.append((position_s, paused))


class ReviewClockTests(unittest.TestCase):
    def test_all_sources_receive_the_same_position(self):
        now = [100.0]
        cv_source = FakePlaybackSource(duration_s=8.0)
        uwb_source = FakePlaybackSource(duration_s=12.0)
        clock = ReviewClock((cv_source, uwb_source), time_fn=lambda: now[0])

        now[0] = 102.5
        self.assertEqual(clock.sample(), 2.5)
        self.assertEqual(cv_source.samples[-1], (2.5, True))
        self.assertEqual(uwb_source.samples[-1], (2.5, True))
        self.assertEqual(clock.state["duration_s"], 8.0)

    def test_seek_and_frame_step_use_the_common_clock(self):
        source = FakePlaybackSource(frame_period_s=0.04)
        clock = ReviewClock((source,), time_fn=lambda: 50.0)

        clock.seek_absolute(3.0, paused=True)
        self.assertAlmostEqual(clock.seek_frames(5), 3.2)
        self.assertEqual(source.samples[-1], (3.2, True))

    def test_analytics_updates_once_per_source_frame_and_resets_on_rewind(self):
        state = {"ui": {}, "player_registry": {"profiles": {}}}
        self.assertEqual(analytics_sample(state, "cv", {"frame_index": 3, "position_s": 1.0}), (True, 1.0, 0.0))
        self.assertEqual(analytics_sample(state, "cv", {"frame_index": 3, "position_s": 1.0}), (False, 1.0, 0.0))
        changed, position_s, dt = analytics_sample(state, "cv", {"frame_index": 4, "position_s": 1.2})
        self.assertTrue(changed)
        self.assertEqual(1.2, position_s)
        self.assertAlmostEqual(0.2, dt)
        self.assertEqual(analytics_sample(state, "cv", {"frame_index": 1, "position_s": 0.2}), (True, 0.2, 0.0))

    def test_presentation_composes_cv_payload_outside_uwb_domain(self):
        solution = compose_review_solution({}, {"cv_positions": [{"player_id": "P1"}], "review_mode": "cv"})
        self.assertEqual(solution["review_mode"], "cv")
        self.assertEqual(solution["cv_positions"][0]["player_id"], "P1")


class DistanceSourceModeTests(unittest.TestCase):
    def test_uwb_review_switches_between_raw_anchors_and_merged_players(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "uwb_raw.csv"
            with raw_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("frame", "timestamp_s", "anchor_id", "distance_cm"))
                writer.writeheader()
                writer.writerow({"frame": 0, "timestamp_s": 0.0, "anchor_id": 1, "distance_cm": 500})
            merged_path = root / "positions_merged.csv"
            with merged_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("frame", "timestamp_s", "player_id", "x_cm", "y_cm", "valid"))
                writer.writeheader()
                writer.writerow({"frame": 0, "timestamp_s": 0.0, "player_id": "tag:uwb", "x_cm": 120, "y_cm": 340, "valid": 1})
            source = DistanceSource(
                review_data_mode="uwb",
                uwb_review_log_path=raw_path,
                uwb_merged_path=merged_path,
                sport="basket",
                asset_set="set1",
            )
            try:
                self.assertEqual("merged", source.uwb_view_mode)
                source.set_review_view_mode("raw")
                raw_packet = source.get_distances(None, {1: (0, 0)}, {})
                source.set_review_view_mode("merged")
                merged_packet = source.get_distances(None, {1: (0, 0)}, {})
            finally:
                source.close()

        self.assertEqual({1: 500.0}, raw_packet["raw"])
        self.assertFalse(raw_packet["uwb_merged_review"])
        self.assertEqual("raw", raw_packet["uwb_view_mode"])
        self.assertEqual("tag:uwb", merged_packet["cv_positions"][0]["player_id"])
        self.assertTrue(merged_packet["uwb_merged_review"])
        self.assertEqual("merged", merged_packet["uwb_view_mode"])

    def test_uwb_review_reads_merged_positions_through_players(self):
        with tempfile.TemporaryDirectory() as directory:
            merged_path = Path(directory) / "positions_merged.csv"
            with merged_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("frame", "timestamp_s", "player_id", "x_cm", "y_cm", "valid"))
                writer.writeheader()
                writer.writerow({"frame": 0, "timestamp_s": 0.0, "player_id": "tag:uwb", "x_cm": 120, "y_cm": 340, "valid": 1})
            source = DistanceSource(
                review_data_mode="uwb",
                uwb_merged_path=merged_path,
                sport="basket",
                asset_set="set1",
            )
            try:
                packet = source.get_distances(None, {}, {})
            finally:
                source.close()

        self.assertEqual("tag:uwb", packet["cv_positions"][0]["player_id"])
        self.assertAlmostEqual(50.16, packet["cv_positions"][0]["x"])
        self.assertAlmostEqual(70.875, packet["cv_positions"][0]["y"])
        self.assertEqual(120.0, packet["cv_positions"][0]["world_x_cm"])
        self.assertEqual(340.0, packet["cv_positions"][0]["world_y_cm"])
        self.assertIsNotNone(packet["uwb_playback"])

    def test_terrain_pixel_positions_use_double_mirror_without_axis_swap(self):
        source = DistanceSource.__new__(DistanceSource)
        source.assets = type("Assets", (), {"terrain_path": Path("terrain.png")})()
        source.ground_bounds = (0.0, 396.0, 0.0, 735.0)

        view = source.view_config

        self.assertEqual("flip_x_flip_y", view["coord_transform"])
        self.assertEqual(source.ground_bounds, view["bounds"])

    def test_raw_and_merged_resolve_to_distinct_csv_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "positions_raw.csv"
            merged_path = root / "positions_merged.csv"
            raw_path.touch()
            merged_path.touch()
            source = DistanceSource.__new__(DistanceSource)
            source.cv_raw_log_path = raw_path
            source.cv_merged_log_path = merged_path

            self.assertEqual(source._cv_log_path_for_mode("raw"), raw_path)
            self.assertEqual(source._cv_log_path_for_mode("merged"), merged_path)

    def test_raw_packet_exposes_unsmoothed_coordinates(self):
        player = {"x": 11.0, "y": 12.0, "raw_x": 101.0, "raw_y": 102.0}
        source = DistanceSource.__new__(DistanceSource)
        source.cv_view_mode = "raw"
        source.cv_review_source = Mock()
        source.cv_review_source.get_frame_packet.return_value = {
            "player_positions": [player],
            "playback": {"frame_index": 1},
        }
        source.review_clock = Mock(state={"position_s": 4.0, "duration_s": 9.0, "paused": True})

        packet = source._cv_packet()

        self.assertEqual(packet["player_positions"][0]["x"], 101.0)
        self.assertEqual(packet["player_positions"][0]["y"], 102.0)
        self.assertEqual(packet["playback"]["position_s"], 4.0)


class PageContractTests(unittest.TestCase):
    def test_home_reopens_menu_when_child_requests_return(self):
        from solver_lps.presentation.pages import home_page

        first_page = Mock()
        first_page.main.return_value = RETURN_HOME
        second_page = Mock()
        second_page.main.return_value = 0
        menu_selection = ("cv_tracking", "review", "cv", "basket", "set1", "2d")

        with patch.object(home_page, "_validate_session_selection"), patch.object(
            home_page, "choose_page_pygame", return_value=menu_selection
        ) as choose_page, patch.object(
            home_page.importlib, "import_module", side_effect=(first_page, second_page)
        ):
            result = home_page.main(
                ["--page", "udp_viewer", "--source", "realtime", "--sport", "basket", "--asset-set", "set1"]
            )

        self.assertEqual(0, result)
        choose_page.assert_called_once()
        first_page.main.assert_called_once()
        second_page.main.assert_called_once()

    @patch("solver_lps.presentation.pages.udp_viewer_page.realtime_main", return_value=0)
    def test_udp_realtime_routes_capture_to_selected_set(self, realtime_main):
        from solver_lps.presentation.pages import udp_viewer_page

        udp_viewer_page.main(["--source", "realtime", "--sport", "basket", "--asset-set", "set1"])

        argv = realtime_main.call_args.args[0]
        self.assertIn("--capture-output", argv)
        self.assertIn(str(udp_viewer_page.default_uwb_raw_path("basket", "set1")), argv)

    @patch.object(estimation_2d_page, "run_with_square_progress")
    @patch.object(estimation_2d_page, "generate_cv_positions")
    @patch.object(estimation_2d_page, "review_ui_main", side_effect=(REGENERATE_MERGED, 0))
    @patch.object(estimation_2d_page, "run_review")
    def test_uwb_solver_review_generates_merged_then_opens_terrain(
        self, run_review, review_ui_main, generate_cv_positions, run_with_square_progress
    ):
        merged_path = Path("positions_merged_three_d_to_2d.csv")
        run_review.return_value = merged_path
        run_with_square_progress.side_effect = lambda task, **_kwargs: task(Mock())

        result = estimation_2d_page.main(
            ["--source", "review", "--sport", "basket", "--asset-set", "set1", "--review-data", "uwb", "--solver-mode", "3d_to_2d"]
        )

        self.assertEqual(0, result)
        self.assertEqual("three_d_to_2d", run_review.call_args.kwargs["calculation_mode"])
        self.assertIn("progress_callback", run_review.call_args.kwargs)
        generate_cv_positions.assert_not_called()
        review_args = review_ui_main.call_args_list[-1].args[0]
        self.assertIn("--uwb-merged", review_args)
        self.assertIn(str(merged_path), review_args)

    @patch.object(estimation_2d_page, "run_with_square_progress")
    @patch.object(estimation_2d_page, "generate_cv_positions")
    @patch.object(estimation_2d_page, "review_ui_main", side_effect=(REGENERATE_MERGED, 0))
    def test_cv_solver_review_generates_cv_merged_before_opening(self, review_ui_main, generate_cv_positions, loading):
        loading.side_effect = lambda task, **_kwargs: task(Mock())

        result = estimation_2d_page.main(
            ["--source", "review", "--sport", "basket", "--asset-set", "set1", "--review-data", "cv"]
        )

        self.assertEqual(0, result)
        generate_cv_positions.assert_called_once()
        self.assertIn("progress_callback", generate_cv_positions.call_args.kwargs)
        self.assertEqual(2, review_ui_main.call_count)

    @patch.object(estimation_2d_page, "run_with_square_progress")
    @patch.object(estimation_2d_page, "generate_cv_positions")
    @patch.object(estimation_2d_page, "review_ui_main", return_value=0)
    def test_review_uses_existing_merged_without_regenerating(self, review_ui_main, generate_cv_positions, loading):
        result = estimation_2d_page.main(
            ["--source", "review", "--sport", "basket", "--asset-set", "set1", "--review-data", "cv"]
        )

        self.assertEqual(0, result)
        review_ui_main.assert_called_once()
        generate_cv_positions.assert_not_called()
        loading.assert_not_called()

    @patch.object(estimation_2d_page, "run_with_square_progress")
    @patch.object(estimation_2d_page, "_analyze_gray_zone")
    @patch.object(estimation_2d_page, "review_ui_main", side_effect=(ANALYZE_GRAY_ZONE, 0))
    def test_gray_zone_action_analyzes_sport_then_reopens_review(self, review_ui_main, analyze, loading):
        loading.side_effect = lambda task, **_kwargs: task(Mock())

        result = estimation_2d_page.main(
            ["--source", "review", "--sport", "basket", "--asset-set", "full", "--review-data", "cv"]
        )

        self.assertEqual(0, result)
        analyze.assert_called_once()
        self.assertEqual("basket", analyze.call_args.args[0])
        self.assertEqual(2, review_ui_main.call_count)

    def test_uwb_realtime_viewer_receives_only_network_arguments(self):
        args = estimation_2d_page.parse_args(["--source", "realtime", "--ip", "127.0.0.1", "--port", "5000"])

        self.assertEqual(["--ip", "127.0.0.1", "--port", "5000"], estimation_2d_page._build_realtime_args(args))

    def test_review_ui_uses_the_owned_uwb_calculation(self):
        self.assertIs(estimation_2d_review_ui.update_position_solution, uwb_update_position_solution)

    @patch("solver_lps.presentation.pages.home_page.SessionAssets")
    def test_home_creates_sets_through_session_assets(self, session_assets_type):
        assets = session_assets_type.return_value
        assets.sport_dir.is_dir.return_value = True
        assets.ensure_directories.return_value = Path("session")

        from solver_lps.presentation.pages.home_page import _create_set_dirs

        self.assertEqual(_create_set_dirs("basket", "set2"), Path("session"))
        session_assets_type.assert_called_once_with(sport="basket", asset_set="set2")
        assets.ensure_directories.assert_called_once_with()

    def test_home_lists_uwb_only_sets(self):
        from solver_lps.presentation.pages import home_page

        with tempfile.TemporaryDirectory() as directory, patch.object(
            home_page, "ASSETS_DIR", Path(directory)
        ):
            (Path(directory) / "basket" / "uwb_only" / "uwb" / "input").mkdir(parents=True)
            self.assertEqual(["uwb_only"], home_page._list_sets("basket"))


if __name__ == "__main__":
    unittest.main()
