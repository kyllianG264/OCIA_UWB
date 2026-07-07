from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from solver_lps.features.cv.review.generate_cv_positions import (
    generate_cv_positions,
    resolve_court_mode,
)


class GenerateCvPositionsTest(unittest.TestCase):
    def test_court_mode_is_explicit_or_derived_from_sport(self):
        self.assertEqual(resolve_court_mode({"sport": "basket"}), "split")
        self.assertEqual(resolve_court_mode({"sport": "volley"}), "full")
        self.assertEqual(resolve_court_mode({"sport": "basket", "court_mode": "full"}), "full")
        with self.assertRaises(ValueError):
            resolve_court_mode({"sport": "basket", "court_mode": "hybrid"})

    def test_split_generation_returns_merged_path_without_solved_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "analysis" / "positions_merged.csv"
            session = SimpleNamespace(
                sport="basket",
                asset_set="match1",
                court_mode="split",
                input_path=root / "output" / "positions_raw.csv",
                output_path=output_path,
                calibration_path=root / "ground" / "output" / "calibration.json",
            )

            with patch(
                "solver_lps.features.cv.review.analysis.split_court.domain.tracking_pipeline.run_tracking",
                return_value=([{"frame": 1}], {"tracks_created": 1}),
            ), patch(
                "solver_lps.features.cv.review.analysis.split_court.data.merged_output.write_output"
            ) as write_output:
                result = generate_cv_positions(session)

            self.assertEqual(result, output_path)
            write_output.assert_called_once_with([{"frame": 1}], str(output_path))

    def test_full_generation_uses_full_court_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "analysis" / "positions_merged.csv"
            session = {
                "sport": "volley",
                "asset_set": "set2",
                "input_path": root / "output" / "positions_raw.csv",
                "output_path": output_path,
                "calibration_path": root / "ground" / "output" / "calibration.json",
            }

            with patch(
                "solver_lps.features.cv.review.analysis.full_court.domain.tracking_pipeline.run_tracking",
                return_value=([{"frame": 2}], {"tracks_created": 1}),
            ), patch(
                "solver_lps.features.cv.review.analysis.full_court.data.track_exporter.write_output"
            ) as write_output:
                result = generate_cv_positions(session)

            self.assertEqual(result, output_path)
            write_output.assert_called_once_with([{"frame": 2}], str(output_path))

    def test_generation_passes_progress_callback_to_tracking_pipeline(self):
        callback = object()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = SimpleNamespace(
                sport="basket",
                court_mode="split",
                input_path=root / "positions_raw.csv",
                output_path=root / "positions_merged.csv",
                calibration_path=root / "calibration.json",
            )
            with patch(
                "solver_lps.features.cv.review.analysis.split_court.domain.tracking_pipeline.run_tracking",
                return_value=([], {}),
            ) as run_tracking, patch(
                "solver_lps.features.cv.review.analysis.split_court.data.merged_output.write_output"
            ):
                generate_cv_positions(session, progress_callback=callback)

        self.assertIs(callback, run_tracking.call_args.args[0].progress_callback)


if __name__ == "__main__":
    unittest.main()
