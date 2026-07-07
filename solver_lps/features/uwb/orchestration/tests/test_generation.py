import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from solver_lps.features.uwb.orchestration import build_calculator, generate_uwb_positions
from solver_lps.features.players.application.player_timeline import build_player_timeline


class UwbGenerationTest(unittest.TestCase):
    def test_build_calculator_keeps_domain_symbol_contract(self):
        calculator = build_calculator("two_d")
        self.assertEqual(calculator.__name__, "update_position_solution")
        self.assertIn("calculus.domain.two_d", calculator.__module__)

    def _session(self, root):
        input_dir = root / "uwb" / "input"
        output_dir = root / "uwb" / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        raw_path = input_dir / "uwb_raw.csv"
        raw_path.write_text(
            "frame,timestamp_s,anchor_id,distance_cm\n"
            "0,0.0,1,7.0710678119\n"
            "0,0.0,2,7.0710678119\n"
            "0,0.0,3,7.0710678119\n"
            "0,0.0,4,7.0710678119\n",
            encoding="utf-8",
        )
        anchors_path = root / "anchors_layout.json"
        anchors_path.write_text(
            json.dumps(
                {
                    "layouts": {
                        "4": {
                            "anchors": {
                                "1": [0, 0],
                                "2": [10, 0],
                                "3": [0, 10],
                                "4": [10, 10],
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            sport="test",
            asset_set="set1",
            uwb_raw_path=raw_path,
            uwb_output_dir=output_dir,
            anchors_layout_path=anchors_path,
            uwb_positions_path=lambda mode: output_dir / (
                "positions_merged.csv" if mode == "two_d" else f"positions_merged_{mode}.csv"
            ),
        )

    def test_generate_two_d_returns_path_and_writes_merged_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = generate_uwb_positions(self._session(Path(directory)), "two_d")
            self.assertIsInstance(output_path, Path)
            self.assertEqual(output_path.name, "positions_merged.csv")
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["valid"], "True")
            self.assertEqual(row["player_id"], "tag:uwb")
            self.assertAlmostEqual(float(row["x_cm"]), 5.0, places=4)
            self.assertAlmostEqual(float(row["y_cm"]), 5.0, places=4)
            timeline = build_player_timeline({"uwb_merged_path": output_path}, ("uwb",))
            self.assertEqual("tag:uwb", timeline[0].player_id)
            self.assertAlmostEqual(5.0, timeline[0].x_cm, places=4)

    def test_compatibility_mode_does_not_claim_3d_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = generate_uwb_positions(
                self._session(Path(directory)),
                "three_d_to_2d",
            )
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "direct_2d_from_ranges_no_3d_projection")
            self.assertIn("direct_2d_compatibility", row["source"])

    def test_generate_three_d_uses_3d_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = generate_uwb_positions(self._session(Path(directory)), "three_d")
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertIn("z_cm", row)
            self.assertEqual(row["status"], "ok_coplanar_anchor_height_assumption")


if __name__ == "__main__":
    unittest.main()
