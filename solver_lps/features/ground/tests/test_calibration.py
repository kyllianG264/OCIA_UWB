import json
import tempfile
import unittest
from pathlib import Path

from solver_lps.features.ground.domain.calibration import (
    CalibrationWarning,
    load_calibration_geometry,
    load_terrain_calibration,
)


class CalibrationTests(unittest.TestCase):
    def _write_calibration(self, root, payload):
        path = Path(root) / "calibration.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_x_split_axis_does_not_fabricate_split_y(self):
        payload = {
            "split_axis": "x",
            "terrain_bounds": {
                "left": {"x_min": 0, "x_max": 50, "y_min": 0, "y_max": 100},
                "right": {"x_min": 50, "x_max": 100, "y_min": 0, "y_max": 100},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            calibration = load_terrain_calibration(self._write_calibration(temp_dir, payload))

        self.assertEqual("x", calibration["split_axis"])
        self.assertIsNone(calibration["split_y"])
        self.assertEqual((0.0, 100.0, 0.0, 100.0), calibration["bounds"])

    def test_y_split_axis_derives_split_from_halves(self):
        payload = {
            "split_axis": "y",
            "terrain_bounds": {
                "top": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 40},
                "bottom": {"x_min": 0, "x_max": 100, "y_min": 60, "y_max": 100},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            calibration = load_terrain_calibration(self._write_calibration(temp_dir, payload))

        self.assertEqual(50.0, calibration["split_y"])

    def test_missing_calibration_warns_and_preserves_empty_geometry(self):
        missing = Path("missing-ground-calibration.json")
        with self.assertWarnsRegex(CalibrationWarning, "does not exist"):
            geometry = load_calibration_geometry(missing)

        self.assertEqual(
            {"split_y": None, "bounds": None, "terrain_bounds": {}, "split_axis": None},
            geometry,
        )

    def test_invalid_explicit_split_warns_and_uses_bounds_fallback(self):
        payload = {
            "split_axis": "y",
            "split_y": "invalid",
            "terrain_bounds": {
                "top": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 40},
                "bottom": {"x_min": 0, "x_max": 100, "y_min": 60, "y_max": 100},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_calibration(temp_dir, payload)
            with self.assertWarnsRegex(CalibrationWarning, "Invalid split_y"):
                calibration = load_terrain_calibration(path)

        self.assertEqual(50.0, calibration["split_y"])

    def test_invalid_json_warns_and_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertWarnsRegex(CalibrationWarning, "Cannot read calibration"):
                calibration = load_terrain_calibration(path)

        self.assertIsNone(calibration)

    def test_relative_terrain_path_is_resolved_from_calibration(self):
        payload = {
            "terrain_image_path": "../terrain.png",
            "terrain_bounds": {
                "court": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 200},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output"
            output.mkdir()
            path = output / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            calibration = load_terrain_calibration(path)

        self.assertEqual(str(Path(temp_dir) / "terrain.png"), calibration["terrain_image_path"])


if __name__ == "__main__":
    unittest.main()
