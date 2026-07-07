from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import solver_lps.session_assets as neutral_assets
from solver_lps.features.cv.review.data import session_assets as cv_assets
from solver_lps.features.cv.review import tracking_launcher


class CvSessionAssetsTest(unittest.TestCase):
    def test_public_helpers_follow_neutral_output_first_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            neutral_assets, "ASSETS_DIR", Path(temp_dir)
        ):
            ground = Path(temp_dir) / "basket" / "ground"
            (ground / "output").mkdir(parents=True)
            (ground / "input").mkdir(parents=True)
            calibration = ground / "output" / "calibration.json"
            terrain = ground / "terrain.png"
            calibration.touch()
            terrain.touch()

            self.assertEqual(cv_assets.ground_calibration_path("basket"), calibration)
            self.assertEqual(cv_assets.cv_review_calibration_path("basket"), calibration)
            self.assertEqual(cv_assets.ground_terrain_path("basket"), terrain)

    def test_legacy_helpers_keep_path_and_string_return_types(self):
        self.assertIsInstance(cv_assets.default_cv_positions_raw_path(), Path)
        self.assertIsInstance(cv_assets.default_cv_positions_merged_path(), Path)
        self.assertIsInstance(cv_assets.first_existing_path("missing.csv"), str)

    def test_models_and_uwb_ownership_are_canonical(self):
        assets = neutral_assets.SessionAssets()
        self.assertEqual(neutral_assets.ASSETS_DIR / "models" / "pose", assets.pose_models_dir)
        self.assertFalse(hasattr(cv_assets, "default_uwb_review_log_path"))

    def test_tracking_launcher_uses_session_contract(self):
        defaults = tracking_launcher.session_defaults("basket", "set1")
        assets = neutral_assets.SessionAssets("basket", "set1")
        self.assertEqual(assets.input_dir / "left_video.mp4", defaults["left_video"])
        self.assertEqual(assets.analysis_dir, defaults["analysis_root"])
        self.assertEqual(neutral_assets.SOLVER_ROOT.parent, tracking_launcher.REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
