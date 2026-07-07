import csv
import json
import tempfile
import unittest
from pathlib import Path

from solver_lps.features.cv.grayzone.application.gray_zone_analysis import build_gray_zone
from solver_lps.features.cv.grayzone.data.raw_input import read_raw_positions
from solver_lps.features.cv.grayzone.data.gray_zone_repository import (
    GrayZoneAssetRepository,
    default_gray_zone_repository,
    load_gray_zone_overlay,
)
from unittest.mock import patch


class GrayZoneAnalysisTests(unittest.TestCase):
    def _write_calibration(self, root):
        camera = {
            "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "frame_corners": [[0, 0], [100, 0], [100, 100], [0, 100]],
        }
        path = Path(root) / "calibration.json"
        path.write_text(
            json.dumps(
                {
                    "cam_gauche": camera,
                    "cam_droite": camera,
                    "terrain_bounds": {
                        "top": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 50},
                        "bottom": {"x_min": 0, "x_max": 100, "y_min": 50, "y_max": 100},
                    },
                    "split_y": 50,
                }
            ),
            encoding="utf-8",
        )
        return path

    def _write_csv(self, root, columns):
        path = Path(root) / "positions.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerow({column: 1 for column in columns})
        return path

    def test_builds_visibility_zone_from_raw_observations(self):
        with tempfile.TemporaryDirectory() as root:
            raw = self._write_csv(root, ["frame", "player_id", "X", "Y", "cam", "on_terrain"])
            zone = build_gray_zone(raw, self._write_calibration(root), grid_step=10)

        self.assertEqual("camera_visibility", zone.metadata["source"])
        self.assertEqual(1, zone.metadata["raw_observation_count"])
        self.assertEqual(2, zone.metadata["camera_count"])
        self.assertTrue(zone.is_in_gray_zone(50, 50))

    def test_rejects_post_processed_tracking_export(self):
        with tempfile.TemporaryDirectory() as root:
            tracked = self._write_csv(
                root,
                ["frame", "player_id", "X", "Y", "cam", "stable_id", "status"],
            )
            with self.assertRaisesRegex(ValueError, "tracking export"):
                read_raw_positions(tracked)

    def test_repository_paths_are_sport_level_not_set_level(self):
        repository = default_gray_zone_repository("basket")

        self.assertTrue(repository.raw_positions_path.endswith("basket\\ground\\gray_zone\\input\\positions_raw.csv"))
        self.assertTrue(repository.metadata_path.endswith("basket\\ground\\gray_zone\\analysis\\gray_zone_metadata.json"))
        self.assertNotIn("set1", repository.raw_positions_path)
        self.assertNotIn("full", repository.raw_positions_path)

    def test_loads_exported_polygons_as_overlay_cells(self):
        with tempfile.TemporaryDirectory() as root:
            metadata = Path(root) / "gray_zone_metadata.json"
            metadata.write_text(
                json.dumps({"polygons": [[[0, 10], [20, 10], [20, 30], [0, 30]]]}),
                encoding="utf-8",
            )
            repository = GrayZoneAssetRepository("", "", "", "", root)
            with patch(
                "solver_lps.features.cv.grayzone.data.gray_zone_repository.default_gray_zone_repository",
                return_value=repository,
            ):
                overlay = load_gray_zone_overlay("basket")

        self.assertEqual((10.0, 20.0), overlay["zone_cells"][0]["center"])
        self.assertEqual((20.0, 20.0), (overlay["zone_cells"][0]["width"], overlay["zone_cells"][0]["height"]))

    def test_raw_disappearance_adds_empirical_density_cell(self):
        with tempfile.TemporaryDirectory() as root:
            raw = Path(root) / "positions.csv"
            with raw.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["frame", "player_id", "X", "Y", "cam", "on_terrain"])
                writer.writeheader()
                for frame in range(5):
                    writer.writerow({"frame": frame, "player_id": 1, "X": 10, "Y": 10, "cam": "gauche", "on_terrain": 1})
                    writer.writerow({"frame": frame, "player_id": 2, "X": 70, "Y": 70, "cam": "gauche", "on_terrain": 1})
                writer.writerow({"frame": 5, "player_id": 2, "X": 71, "Y": 70, "cam": "gauche", "on_terrain": 1})
                writer.writerow({"frame": 6, "player_id": 2, "X": 72, "Y": 70, "cam": "gauche", "on_terrain": 1})
                writer.writerow({"frame": 7, "player_id": 2, "X": 73, "Y": 70, "cam": "gauche", "on_terrain": 1})
                writer.writerow({"frame": 7, "player_id": 9, "X": 12, "Y": 12, "cam": "droite", "on_terrain": 1})

            zone = build_gray_zone(raw, self._write_calibration(root), grid_step=10)

        self.assertEqual(14, zone.metadata["raw_observation_count"])
        self.assertEqual(1, zone.metadata["disappearance_event_count"])
        self.assertEqual(1, len(zone.handoff_points))
        self.assertEqual([10.0, 10.0], zone.handoff_points[0]["scene_point"])
        self.assertEqual(7, zone.handoff_points[0]["matched_frame"])
        self.assertEqual([10.0, 10.0], zone.handoff_ellipse["center"])

    def test_disappearance_keeps_frontier_point_when_trace_jitters_back(self):
        with tempfile.TemporaryDirectory() as root:
            raw = Path(root) / "positions.csv"
            with raw.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["frame", "player_id", "X", "Y", "cam", "on_terrain"])
                writer.writeheader()
                for frame, x_value in enumerate([10, 20, 30, 40, 35]):
                    writer.writerow(
                        {
                            "frame": frame,
                            "player_id": 1,
                            "X": x_value,
                            "Y": 50,
                            "cam": "gauche",
                            "on_terrain": 1,
                        }
                    )
                writer.writerow({"frame": 6, "player_id": 9, "X": 42, "Y": 50, "cam": "droite", "on_terrain": 1})

            zone = build_gray_zone(raw, self._write_calibration(root), grid_step=10)

        self.assertEqual(1, len(zone.handoff_points))
        self.assertEqual([40.0, 50.0], zone.handoff_points[0]["scene_point"])
        self.assertEqual([40.0, 50.0], zone.handoff_ellipse["center"])


if __name__ == "__main__":
    unittest.main()
