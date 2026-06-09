import tempfile
import unittest
from pathlib import Path
from unittest import mock

from solver_lps.features.cv.review.domain import stable_id_tracker


class StableIdTrackerTests(unittest.TestCase):
    def test_raw_id_alias_match_accepts_offset_pairs(self):
        self.assertTrue(stable_id_tracker.raw_id_alias_match("5", "35"))
        self.assertTrue(stable_id_tracker.raw_id_alias_match("35", "5"))
        self.assertFalse(stable_id_tracker.raw_id_alias_match("5", "36"))

    def test_load_calibration_geometry_handles_missing_file(self):
        geometry = stable_id_tracker.load_calibration_geometry("missing.json")
        self.assertEqual({"split_y": None, "bounds": None}, geometry)

    def test_default_paths_point_to_cv_review_feature(self):
        expected = str(Path("features") / "cv" / "review" / "data" / "cv_logs")
        self.assertIn(expected, stable_id_tracker.DEFAULT_INPUT)
        self.assertIn(expected, stable_id_tracker.DEFAULT_CALIBRATION)

    def test_run_tracking_keeps_same_track_across_split_occlusion(self):
        rows = [
            {"frame": "1", "timestamp_s": "0.00", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "520", "cam": "left", "on_terrain": "1"},
            {"frame": "2", "timestamp_s": "0.04", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "560", "cam": "left", "on_terrain": "1"},
            {"frame": "3", "timestamp_s": "0.08", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "600", "cam": "left", "on_terrain": "1"},
            {"frame": "7", "timestamp_s": "0.24", "timestamp_unix": "", "player_id": "7", "X": "100", "Y": "760", "cam": "right", "on_terrain": "1"},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False, encoding="utf-8") as handle:
            csv_path = handle.name
            writer = stable_id_tracker.csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(lambda: Path(csv_path).unlink(missing_ok=True))

        args = stable_id_tracker.default_config(input_path=csv_path, min_hits=1, expected_players=10)
        geometry = {"split_y": 600.0, "bounds": (0.0, 200.0, 400.0, 800.0)}
        with mock.patch.object(stable_id_tracker, "load_calibration_geometry", return_value=geometry):
            tracked_rows, _stats = stable_id_tracker.run_tracking(args)

        frame_seven_rows = [row for row in tracked_rows if int(row["frame"]) == 7]
        self.assertEqual(1, len(frame_seven_rows))
        self.assertEqual("T001", frame_seven_rows[0]["stable_id"])
        self.assertIn(frame_seven_rows[0]["status"], {"matched", "reattached"})


if __name__ == "__main__":
    unittest.main()
