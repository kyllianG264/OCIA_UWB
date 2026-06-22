import tempfile
import unittest
from pathlib import Path
from unittest import mock

from solver_lps.features.cv.review.domain import tracking_pipeline


def _write_temp_csv(rows):
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False, encoding="utf-8")
    try:
        writer = tracking_pipeline.csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        return handle.name
    finally:
        handle.close()


class TrackingPipelineTests(unittest.TestCase):
    def test_raw_id_alias_match_accepts_offset_pairs(self):
        self.assertTrue(tracking_pipeline.raw_id_alias_match("5", "35"))
        self.assertTrue(tracking_pipeline.raw_id_alias_match("35", "5"))
        self.assertFalse(tracking_pipeline.raw_id_alias_match("5", "36"))

    def test_load_calibration_geometry_handles_missing_file(self):
        geometry = tracking_pipeline.load_calibration_geometry("missing.json")
        self.assertEqual({"split_y": None, "bounds": None}, geometry)

    def test_default_paths_point_to_cv_review_feature(self):
        expected = str(Path("features") / "cv" / "review" / "data" / "cv_logs")
        self.assertIn(expected, tracking_pipeline.DEFAULT_INPUT)
        self.assertIn(expected, tracking_pipeline.DEFAULT_CALIBRATION)

    def test_run_tracking_keeps_same_track_across_split_occlusion(self):
        rows = [
            {"frame": "1", "timestamp_s": "0.00", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "520", "cam": "left", "on_terrain": "0"},
            {"frame": "2", "timestamp_s": "0.04", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "530", "cam": "left", "on_terrain": "0"},
            {"frame": "3", "timestamp_s": "0.08", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "540", "cam": "left", "on_terrain": "0"},
            {"frame": "4", "timestamp_s": "0.12", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "550", "cam": "left", "on_terrain": "0"},
            {"frame": "5", "timestamp_s": "0.16", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "560", "cam": "left", "on_terrain": "0"},
            {"frame": "6", "timestamp_s": "0.20", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "570", "cam": "left", "on_terrain": "0"},
            {"frame": "7", "timestamp_s": "0.24", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "580", "cam": "left", "on_terrain": "0"},
            {"frame": "8", "timestamp_s": "0.28", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "590", "cam": "left", "on_terrain": "0"},
            {"frame": "12", "timestamp_s": "0.44", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "600", "cam": "left", "on_terrain": "0"},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False, encoding="utf-8") as handle:
            csv_path = handle.name
            writer = tracking_pipeline.csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(lambda: Path(csv_path).unlink(missing_ok=True))

        args = tracking_pipeline.default_config(input_path=csv_path, min_hits=1, expected_players=10)
        geometry = {"split_y": 600.0, "bounds": (0.0, 2400.0, 0.0, 1800.0)}
        with mock.patch.object(tracking_pipeline, "load_calibration_geometry", return_value=geometry):
            tracked_rows, _stats = tracking_pipeline.run_tracking(args)

        self.assertEqual(8, min(int(row["frame"]) for row in tracked_rows))
        frame_eight_rows = [row for row in tracked_rows if int(row["frame"]) == 8]
        self.assertEqual(1, len(frame_eight_rows))
        self.assertEqual("T001", frame_eight_rows[0]["stable_id"])
        self.assertIn(frame_eight_rows[0]["status"], {"matched", "reattached"})

    def test_missing_frames_keep_confirmed_track_visible(self):
        rows = [
            {"frame": "1", "timestamp_s": "0.00", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "520", "cam": "left", "on_terrain": "0"},
            {"frame": "2", "timestamp_s": "0.04", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "530", "cam": "left", "on_terrain": "0"},
            {"frame": "3", "timestamp_s": "0.08", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "540", "cam": "left", "on_terrain": "0"},
            {"frame": "4", "timestamp_s": "0.12", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "550", "cam": "left", "on_terrain": "0"},
            {"frame": "5", "timestamp_s": "0.16", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "560", "cam": "left", "on_terrain": "0"},
            {"frame": "6", "timestamp_s": "0.20", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "570", "cam": "left", "on_terrain": "0"},
            {"frame": "7", "timestamp_s": "0.24", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "580", "cam": "left", "on_terrain": "0"},
            {"frame": "8", "timestamp_s": "0.28", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "590", "cam": "left", "on_terrain": "0"},
            {"frame": "12", "timestamp_s": "0.44", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "600", "cam": "left", "on_terrain": "0"},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False, encoding="utf-8") as handle:
            csv_path = handle.name
            writer = tracking_pipeline.csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(lambda: Path(csv_path).unlink(missing_ok=True))

        args = tracking_pipeline.default_config(input_path=csv_path, min_hits=1, expected_players=10)
        geometry = {"split_y": 600.0, "bounds": (0.0, 2400.0, 0.0, 1800.0)}
        with mock.patch.object(tracking_pipeline, "load_calibration_geometry", return_value=geometry):
            tracked_rows, _stats = tracking_pipeline.run_tracking(args)

        frame_ids = [int(row["frame"]) for row in tracked_rows]
        self.assertEqual([8, 9, 10, 11, 12], frame_ids)
        self.assertTrue(all(row["stable_id"] == "T001" for row in tracked_rows))

    def test_midcourt_spawn_is_rejected_when_isolated(self):
        rows = [
            {"frame": "1", "timestamp_s": "0.00", "timestamp_unix": "", "player_id": "99", "X": "-100", "Y": "-100", "cam": "left", "on_terrain": "0"},
            {"frame": "100", "timestamp_s": "10.00", "timestamp_unix": "", "player_id": "99", "X": "1200", "Y": "950", "cam": "left", "on_terrain": "1"},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False, encoding="utf-8") as handle:
            csv_path = handle.name
            writer = tracking_pipeline.csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(lambda: Path(csv_path).unlink(missing_ok=True))

        args = tracking_pipeline.default_config(input_path=csv_path, min_hits=1, expected_players=10)
        geometry = {"split_y": 600.0, "bounds": (0.0, 2400.0, 0.0, 1800.0)}
        with mock.patch.object(tracking_pipeline, "load_calibration_geometry", return_value=geometry):
            tracked_rows, _stats = tracking_pipeline.run_tracking(args)

        self.assertEqual([], tracked_rows)

    def test_on_terrain_spawn_without_outside_origin_is_blocked_after_bootstrap(self):
        rows = [
            {"frame": "11", "timestamp_s": "0.40", "timestamp_unix": "", "player_id": "2", "X": "140", "Y": "238", "cam": "gauche", "on_terrain": "1"},
            {"frame": "12", "timestamp_s": "0.44", "timestamp_unix": "", "player_id": "2", "X": "140", "Y": "237", "cam": "gauche", "on_terrain": "1"},
            {"frame": "13", "timestamp_s": "0.48", "timestamp_unix": "", "player_id": "2", "X": "141", "Y": "237", "cam": "gauche", "on_terrain": "1"},
        ]
        csv_path = _write_temp_csv(rows)
        self.addCleanup(lambda: Path(csv_path).unlink(missing_ok=True))

        args = tracking_pipeline.default_config(input_path=csv_path, min_hits=1, expected_players=10)
        geometry = {"split_y": 600.0, "bounds": (0.0, 2400.0, 0.0, 1800.0)}
        with mock.patch.object(tracking_pipeline, "load_calibration_geometry", return_value=geometry):
            tracked_rows, _stats = tracking_pipeline.run_tracking(args)

        self.assertEqual([], tracked_rows)

    def test_border_ghost_without_motion_stays_invisible(self):
        rows = [
            {"frame": str(frame), "timestamp_s": f"{(frame - 1) * 0.04:.2f}", "timestamp_unix": "", "player_id": "7", "X": "2", "Y": "22", "cam": "left", "on_terrain": "1"}
            for frame in range(11, 20)
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False, encoding="utf-8") as handle:
            csv_path = handle.name
            writer = tracking_pipeline.csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(lambda: Path(csv_path).unlink(missing_ok=True))

        args = tracking_pipeline.default_config(input_path=csv_path, min_hits=1, expected_players=10)
        geometry = {"split_y": 600.0, "bounds": (0.0, 2400.0, 0.0, 1800.0)}
        with mock.patch.object(tracking_pipeline, "load_calibration_geometry", return_value=geometry):
            tracked_rows, stats = tracking_pipeline.run_tracking(args)

        self.assertEqual([], tracked_rows)
        self.assertEqual(0, stats["candidate_created"])
        self.assertEqual(0, stats["candidate_promoted"])

    def test_on_terrain_player_can_bootstrap_and_continue_past_frame_ten(self):
        rows = [
            {
                "frame": str(frame),
                "timestamp_s": f"{(frame - 1) * 0.04:.2f}",
                "timestamp_unix": "",
                "player_id": "2",
                "X": str(140 + frame * 4),
                "Y": str(238 + frame * 8),
                "cam": "gauche",
                "on_terrain": "1",
            }
            for frame in range(1, 13)
        ]
        csv_path = _write_temp_csv(rows)
        self.addCleanup(lambda: Path(csv_path).unlink(missing_ok=True))

        args = tracking_pipeline.default_config(input_path=csv_path, min_hits=1, expected_players=10)
        geometry = {"split_y": 600.0, "bounds": (0.0, 2400.0, 0.0, 1800.0)}
        with mock.patch.object(tracking_pipeline, "load_calibration_geometry", return_value=geometry):
            tracked_rows, stats = tracking_pipeline.run_tracking(args)

        frame_ids = [int(row["frame"]) for row in tracked_rows]
        self.assertEqual([8, 9, 10, 11, 12], frame_ids)
        self.assertTrue(all(row["stable_id"] == "T001" for row in tracked_rows))
        self.assertEqual(1, stats["tracks_created"])
        self.assertEqual(1, stats["candidate_promoted"])

    def test_new_offterrain_player_after_bootstrap_can_still_confirm(self):
        rows = [
            {
                "frame": str(frame),
                "timestamp_s": f"{(frame - 11) * 0.04:.2f}",
                "timestamp_unix": "",
                "player_id": "9",
                "X": str(100 + frame * 5),
                "Y": str(520 + frame * 6),
                "cam": "right",
                "on_terrain": "0",
            }
            for frame in range(11, 20)
        ]
        csv_path = _write_temp_csv(rows)
        self.addCleanup(lambda: Path(csv_path).unlink(missing_ok=True))

        args = tracking_pipeline.default_config(input_path=csv_path, min_hits=1, expected_players=10)
        geometry = {"split_y": 600.0, "bounds": (0.0, 2400.0, 0.0, 1800.0)}
        with mock.patch.object(tracking_pipeline, "load_calibration_geometry", return_value=geometry):
            tracked_rows, stats = tracking_pipeline.run_tracking(args)

        frame_ids = [int(row["frame"]) for row in tracked_rows]
        self.assertEqual([18, 19], frame_ids)
        self.assertTrue(all(row["stable_id"] == "T001" for row in tracked_rows))
        self.assertEqual(1, stats["tracks_created"])
        self.assertGreaterEqual(stats["candidate_promoted"], 1)

    def test_midline_crossing_keeps_existing_track_ids_without_replacement(self):
        rows = []
        for frame in range(1, 13):
            rows.append(
                {
                    "frame": str(frame),
                    "timestamp_s": f"{(frame - 1) * 0.04:.2f}",
                    "timestamp_unix": "",
                    "player_id": "5",
                    "X": "200",
                    "Y": str(320 + frame * 28),
                    "cam": "left",
                    "on_terrain": "1",
                }
            )
            rows.append(
                {
                    "frame": str(frame),
                    "timestamp_s": f"{(frame - 1) * 0.04:.2f}",
                    "timestamp_unix": "",
                    "player_id": "7",
                    "X": "260",
                    "Y": str(700 - frame * 28),
                    "cam": "right",
                    "on_terrain": "1",
                }
            )

        csv_path = _write_temp_csv(rows)
        self.addCleanup(lambda: Path(csv_path).unlink(missing_ok=True))

        args = tracking_pipeline.default_config(input_path=csv_path, min_hits=1, expected_players=10)
        geometry = {"split_y": 600.0, "bounds": (0.0, 2400.0, 0.0, 1800.0)}
        with mock.patch.object(tracking_pipeline, "load_calibration_geometry", return_value=geometry):
            tracked_rows, stats = tracking_pipeline.run_tracking(args)

        window = [row for row in tracked_rows if 8 <= int(row["frame"]) <= 12]
        self.assertTrue(window)
        stable_ids = {row["stable_id"] for row in window}
        self.assertEqual(2, len(stable_ids))
        self.assertFalse({"T010", "T011"} & stable_ids)
        self.assertEqual(2, stats["tracks_created"])
        self.assertEqual(2, stats["candidate_promoted"])
        ids_by_raw = {}
        for row in window:
            ids_by_raw.setdefault(row["raw_player_id"], row["stable_id"])
            self.assertEqual(ids_by_raw[row["raw_player_id"]], row["stable_id"])

    def test_unconfirmed_track_blocks_same_spot_spawn(self):
        state, covariance = tracking_pipeline.create_filter_state(100.0, 100.0)
        track = tracking_pipeline.Track(
            track_id=1,
            state=state,
            covariance=covariance,
            last_frame=10,
            last_timestamp_s=0.4,
            hits=4,
            misses=0,
            age=4,
            confirmed=False,
            last_raw_player_id="7",
            last_cam="left",
            last_on_terrain=True,
            confidence=0.55,
            display_pos=(100.0, 100.0),
            spawn_category="border",
            spawn_position=(100.0, 100.0),
        )
        detection = tracking_pipeline.Detection(
            frame=11,
            timestamp_s=0.44,
            timestamp_unix="",
            raw_player_id="99",
            x=108.0,
            y=106.0,
            cam="left",
            on_terrain=True,
            spawn_allowed=True,
        )
        args = tracking_pipeline.default_config()
        args.court_bounds = (0.0, 2400.0, 0.0, 1800.0)
        args.split_y = 900.0

        self.assertFalse(
            tracking_pipeline.should_spawn_track(
                detection,
                {1: track},
                {},
                args,
                first_timestamp_s=0.0,
            )
        )

    def test_impossible_jump_is_rejected_by_coherence_gate(self):
        state, covariance = tracking_pipeline.create_filter_state(100.0, 100.0)
        track = tracking_pipeline.Track(
            track_id=1,
            state=state,
            covariance=covariance,
            last_frame=1,
            last_timestamp_s=0.0,
            confirmed=True,
            last_raw_player_id="7",
            last_cam="left",
            last_on_terrain=True,
            confidence=1.0,
            display_pos=(100.0, 100.0),
        )
        detection = tracking_pipeline.Detection(
            frame=2,
            timestamp_s=0.04,
            timestamp_unix="",
            raw_player_id="7",
            x=2500.0,
            y=1700.0,
            cam="left",
            on_terrain=True,
        )
        predicted_state, predicted_covariance = tracking_pipeline.kalman_predict(track, 1)
        args = tracking_pipeline.default_config()
        args.split_y = 900.0
        self.assertIsNone(tracking_pipeline.match_cost(track, detection, predicted_state, 1, args, reattach=False))

    def test_shadow_duplicate_same_frame_does_not_spawn_second_track(self):
        rows = [
            {"frame": "1", "timestamp_s": "0.00", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "520", "cam": "left", "on_terrain": "0"},
            {"frame": "2", "timestamp_s": "0.04", "timestamp_unix": "", "player_id": "7", "X": "104", "Y": "528", "cam": "left", "on_terrain": "0"},
            {"frame": "2", "timestamp_s": "0.04", "timestamp_unix": "", "player_id": "17", "X": "116", "Y": "536", "cam": "left", "on_terrain": "0"},
            {"frame": "3", "timestamp_s": "0.08", "timestamp_unix": "", "player_id": "7", "X": "108", "Y": "536", "cam": "left", "on_terrain": "0"},
            {"frame": "3", "timestamp_s": "0.08", "timestamp_unix": "", "player_id": "17", "X": "120", "Y": "544", "cam": "left", "on_terrain": "0"},
            {"frame": "4", "timestamp_s": "0.12", "timestamp_unix": "", "player_id": "7", "X": "112", "Y": "544", "cam": "left", "on_terrain": "0"},
            {"frame": "4", "timestamp_s": "0.12", "timestamp_unix": "", "player_id": "17", "X": "124", "Y": "552", "cam": "left", "on_terrain": "0"},
            {"frame": "5", "timestamp_s": "0.16", "timestamp_unix": "", "player_id": "7", "X": "116", "Y": "552", "cam": "left", "on_terrain": "0"},
            {"frame": "5", "timestamp_s": "0.16", "timestamp_unix": "", "player_id": "17", "X": "128", "Y": "560", "cam": "left", "on_terrain": "0"},
            {"frame": "6", "timestamp_s": "0.20", "timestamp_unix": "", "player_id": "7", "X": "120", "Y": "560", "cam": "left", "on_terrain": "0"},
            {"frame": "6", "timestamp_s": "0.20", "timestamp_unix": "", "player_id": "17", "X": "132", "Y": "568", "cam": "left", "on_terrain": "0"},
            {"frame": "7", "timestamp_s": "0.24", "timestamp_unix": "", "player_id": "7", "X": "124", "Y": "568", "cam": "left", "on_terrain": "0"},
            {"frame": "7", "timestamp_s": "0.24", "timestamp_unix": "", "player_id": "17", "X": "136", "Y": "576", "cam": "left", "on_terrain": "0"},
            {"frame": "8", "timestamp_s": "0.28", "timestamp_unix": "", "player_id": "7", "X": "128", "Y": "576", "cam": "left", "on_terrain": "0"},
            {"frame": "8", "timestamp_s": "0.28", "timestamp_unix": "", "player_id": "17", "X": "140", "Y": "584", "cam": "left", "on_terrain": "0"},
            {"frame": "9", "timestamp_s": "0.32", "timestamp_unix": "", "player_id": "7", "X": "132", "Y": "584", "cam": "left", "on_terrain": "0"},
            {"frame": "9", "timestamp_s": "0.32", "timestamp_unix": "", "player_id": "17", "X": "144", "Y": "592", "cam": "left", "on_terrain": "0"},
        ]
        csv_path = _write_temp_csv(rows)
        self.addCleanup(lambda: Path(csv_path).unlink(missing_ok=True))

        args = tracking_pipeline.default_config(input_path=csv_path, min_hits=1, expected_players=10)
        geometry = {"split_y": 600.0, "bounds": (0.0, 2400.0, 0.0, 1800.0)}
        with mock.patch.object(tracking_pipeline, "load_calibration_geometry", return_value=geometry):
            tracked_rows, stats = tracking_pipeline.run_tracking(args)

        stable_ids = {row["stable_id"] for row in tracked_rows}
        self.assertEqual({"T001"}, stable_ids)
        self.assertGreaterEqual(stats["detection_dedup_merged"], 1)
        self.assertGreaterEqual(stats["candidate_rejected_duplicate_track"], 1)

    def test_true_new_player_far_from_existing_tracks_can_confirm(self):
        rows = [
            {"frame": str(frame), "timestamp_s": f"{(frame - 1) * 0.04:.2f}", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": str(520 + frame * 8), "cam": "left", "on_terrain": "0"}
            for frame in range(1, 10)
        ]
        rows.extend(
            [
                {"frame": str(frame), "timestamp_s": f"{(frame - 1) * 0.04:.2f}", "timestamp_unix": "", "player_id": "9", "X": "2200", "Y": str(540 + frame * 8), "cam": "right", "on_terrain": "0"}
                for frame in range(1, 10)
            ]
        )
        csv_path = _write_temp_csv(rows)
        self.addCleanup(lambda: Path(csv_path).unlink(missing_ok=True))

        args = tracking_pipeline.default_config(input_path=csv_path, min_hits=1, expected_players=10)
        geometry = {"split_y": 600.0, "bounds": (0.0, 2400.0, 0.0, 1800.0)}
        with mock.patch.object(tracking_pipeline, "load_calibration_geometry", return_value=geometry):
            tracked_rows, _stats = tracking_pipeline.run_tracking(args)

        stable_ids = {row["stable_id"] for row in tracked_rows}
        self.assertIn("T001", stable_ids)
        self.assertIn("T002", stable_ids)

    def test_close_real_players_with_distinct_trajectories_remain_separate(self):
        rows = [
            {"frame": "1", "timestamp_s": "0.00", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "520", "cam": "left", "on_terrain": "0"},
            {"frame": "1", "timestamp_s": "0.00", "timestamp_unix": "", "player_id": "9", "X": "122", "Y": "526", "cam": "left", "on_terrain": "0"},
            {"frame": "2", "timestamp_s": "0.04", "timestamp_unix": "", "player_id": "7", "X": "108", "Y": "536", "cam": "left", "on_terrain": "0"},
            {"frame": "2", "timestamp_s": "0.04", "timestamp_unix": "", "player_id": "9", "X": "130", "Y": "546", "cam": "left", "on_terrain": "0"},
            {"frame": "3", "timestamp_s": "0.08", "timestamp_unix": "", "player_id": "7", "X": "116", "Y": "552", "cam": "left", "on_terrain": "0"},
            {"frame": "3", "timestamp_s": "0.08", "timestamp_unix": "", "player_id": "9", "X": "138", "Y": "566", "cam": "left", "on_terrain": "0"},
            {"frame": "4", "timestamp_s": "0.12", "timestamp_unix": "", "player_id": "7", "X": "124", "Y": "568", "cam": "left", "on_terrain": "0"},
            {"frame": "4", "timestamp_s": "0.12", "timestamp_unix": "", "player_id": "9", "X": "146", "Y": "586", "cam": "left", "on_terrain": "0"},
            {"frame": "5", "timestamp_s": "0.16", "timestamp_unix": "", "player_id": "7", "X": "132", "Y": "584", "cam": "left", "on_terrain": "0"},
            {"frame": "5", "timestamp_s": "0.16", "timestamp_unix": "", "player_id": "9", "X": "154", "Y": "606", "cam": "left", "on_terrain": "0"},
            {"frame": "6", "timestamp_s": "0.20", "timestamp_unix": "", "player_id": "7", "X": "140", "Y": "600", "cam": "left", "on_terrain": "0"},
            {"frame": "6", "timestamp_s": "0.20", "timestamp_unix": "", "player_id": "9", "X": "162", "Y": "626", "cam": "left", "on_terrain": "0"},
            {"frame": "7", "timestamp_s": "0.24", "timestamp_unix": "", "player_id": "7", "X": "148", "Y": "616", "cam": "left", "on_terrain": "0"},
            {"frame": "7", "timestamp_s": "0.24", "timestamp_unix": "", "player_id": "9", "X": "170", "Y": "646", "cam": "left", "on_terrain": "0"},
            {"frame": "8", "timestamp_s": "0.28", "timestamp_unix": "", "player_id": "7", "X": "156", "Y": "632", "cam": "left", "on_terrain": "0"},
            {"frame": "8", "timestamp_s": "0.28", "timestamp_unix": "", "player_id": "9", "X": "178", "Y": "666", "cam": "left", "on_terrain": "0"},
            {"frame": "9", "timestamp_s": "0.32", "timestamp_unix": "", "player_id": "7", "X": "164", "Y": "648", "cam": "left", "on_terrain": "0"},
            {"frame": "9", "timestamp_s": "0.32", "timestamp_unix": "", "player_id": "9", "X": "186", "Y": "686", "cam": "left", "on_terrain": "0"},
        ]
        csv_path = _write_temp_csv(rows)
        self.addCleanup(lambda: Path(csv_path).unlink(missing_ok=True))

        args = tracking_pipeline.default_config(input_path=csv_path, min_hits=1, expected_players=10)
        geometry = {"split_y": 600.0, "bounds": (0.0, 2400.0, 0.0, 1800.0)}
        with mock.patch.object(tracking_pipeline, "load_calibration_geometry", return_value=geometry):
            tracked_rows, _stats = tracking_pipeline.run_tracking(args)

        stable_ids = {row["stable_id"] for row in tracked_rows}
        self.assertEqual({"T001", "T002"}, stable_ids)


if __name__ == "__main__":
    unittest.main()
