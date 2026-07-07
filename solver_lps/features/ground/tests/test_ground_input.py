import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from solver_lps.features.ground.data.ground_input import (
    ground_calibration_path,
    ground_terrain_path,
)


class GroundInputTests(unittest.TestCase):
    def test_calibration_prefers_existing_output_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            assets_dir = Path(temp_dir)
            legacy = assets_dir / "basket" / "ground" / "calibration.json"
            output = assets_dir / "basket" / "ground" / "output" / "calibration.json"
            legacy.parent.mkdir(parents=True)
            output.parent.mkdir(parents=True)
            legacy.touch()
            output.touch()

            with patch("solver_lps.session_assets.ASSETS_DIR", assets_dir):
                self.assertEqual(output, ground_calibration_path("basket"))

    def test_terrain_uses_existing_legacy_asset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            assets_dir = Path(temp_dir)
            terrain = assets_dir / "volley" / "ground" / "terrain.png"
            terrain.parent.mkdir(parents=True)
            terrain.touch()

            with patch("solver_lps.session_assets.ASSETS_DIR", assets_dir):
                self.assertEqual(terrain, ground_terrain_path("volley"))


if __name__ == "__main__":
    unittest.main()
