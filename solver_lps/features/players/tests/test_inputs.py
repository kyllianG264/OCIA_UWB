import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from solver_lps.features.players.data.merged_input import (
    TrackedPlayersCsvSource,
    load_merged_frames,
    load_uwb_merged_frames,
)


class PlayerInputTests(unittest.TestCase):
    def _write_csv(self, rows):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "positions_merged.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_normalizes_cv_uppercase_coordinates_and_stable_id(self):
        path = self._write_csv([{"frame": 4, "timestamp_s": 1.5, "stable_id": "T007", "X": 120, "Y": 340}])
        row = load_merged_frames(path, source="cv_merged")[0]
        self.assertEqual((row["player_id"], row["x_cm"], row["y_cm"]), ("T007", 120.0, 340.0))

    def test_normalizes_uwb_centimetre_coordinates_and_player_id(self):
        path = self._write_csv([{"frame_index": 2, "time_s": 0.25, "player_id": "tag:2", "x_cm": 10, "y_cm": 20, "z_cm": 30}])
        row = load_uwb_merged_frames(path)[0]
        self.assertEqual((row["frame"], row["player_id"], row["x_cm"], row["y_cm"], row["z_cm"]), (2, "tag:2", 10.0, 20.0, 30.0))

    def test_temporal_packet_is_positions_only(self):
        path = self._write_csv([
            {"frame": 10, "timestamp_s": 5.0, "stable_id": "P1", "X": 1, "Y": 2},
            {"frame": 11, "timestamp_s": 5.5, "stable_id": "P1", "X": 3, "Y": 4},
        ])
        source = TrackedPlayersCsvSource(path)
        packet = source.get_frame_packet(position_s=0.5)
        self.assertEqual(packet["frame"], 11)
        self.assertEqual((packet["player_positions"][0]["x_cm"], packet["player_positions"][0]["y_cm"]), (3.0, 4.0))
        self.assertNotIn("video_frame", packet)

    def test_invalid_uwb_solution_is_not_visible(self):
        path = self._write_csv([
            {"frame": 2, "timestamp_s": 0.25, "player_id": "tag:2", "x_cm": "", "y_cm": "", "valid": 0}
        ])

        packet = TrackedPlayersCsvSource(path).get_frame_packet(position_s=0.25)

        self.assertEqual([], packet["player_positions"])

    def test_large_merged_is_indexed_and_frames_are_loaded_on_demand(self):
        path = self._write_csv([
            {"frame": 10, "timestamp_s": 5.0, "stable_id": "P1", "X": 1, "Y": 2},
            {"frame": 10, "timestamp_s": 5.0, "stable_id": "P2", "X": 3, "Y": 4},
            {"frame": 11, "timestamp_s": 5.5, "stable_id": "P1", "X": 5, "Y": 6},
        ])
        with patch("solver_lps.features.players.data.merged_input.LARGE_MERGED_THRESHOLD_BYTES", 0):
            source = TrackedPlayersCsvSource(path)
        self.addCleanup(source.close)

        self.assertEqual(2, len(source.frames))
        self.assertEqual(["P1", "P2"], source.all_player_ids)
        packet = source.get_frame_packet(position_s=0.5)
        self.assertEqual(11, packet["frame"])
        self.assertEqual((5.0, 6.0), (packet["player_positions"][0]["x_cm"], packet["player_positions"][0]["y_cm"]))

    def test_merged_input_has_no_forbidden_feature_imports(self):
        module_path = Path(__file__).parents[1] / "data" / "merged_input.py"
        import_lines = [
            line.lower()
            for line in module_path.read_text(encoding="utf-8").splitlines()
            if line.startswith(("import ", "from "))
        ]
        imports = "\n".join(import_lines)
        for forbidden in ("features.cv", "features.uwb", "features.ground", "pygame", "cv2"):
            self.assertNotIn(forbidden, imports)


if __name__ == "__main__":
    unittest.main()
