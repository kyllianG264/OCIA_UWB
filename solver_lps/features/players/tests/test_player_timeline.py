import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from solver_lps.features.players.application.player_timeline import build_player_timeline
from solver_lps.session_assets import SessionAssets


class PlayerTimelineTests(unittest.TestCase):
    def _write(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_uses_session_assets_and_sorts_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = SessionAssets("basket", "set_test")
            cv_path = root / "basket" / "set_test" / "analysis" / "positions_merged.csv"
            uwb_path = root / "basket" / "set_test" / "uwb" / "output" / "positions_merged.csv"
            self._write(cv_path, [
                {"frame": 2, "timestamp_s": 2, "stable_id": "P1", "X": 20, "Y": 20, "confidence": 0.5},
                {"frame": 1, "timestamp_s": 1, "stable_id": "P1", "X": 10, "Y": 10, "confidence": 0.5},
                {"frame": 1, "timestamp_s": 1, "stable_id": "P1", "X": 11, "Y": 11, "confidence": 0.9},
            ])
            self._write(uwb_path, [{"frame": 1, "timestamp_s": 1, "player_id": "tag:1", "x_cm": 30, "y_cm": 40}])
            with patch("solver_lps.features.players.application.player_timeline.SessionAssets.cv_positions_merged_path", new_callable=lambda: property(lambda _: cv_path)), patch("solver_lps.features.players.application.player_timeline.SessionAssets.uwb_positions_merged_path", new_callable=lambda: property(lambda _: uwb_path)):
                timeline = build_player_timeline(assets, ("uwb", "cv", "cv"))
        self.assertEqual([(item.timestamp_s, item.player_id) for item in timeline], [(1.0, "P1"), (1.0, "tag:1"), (2.0, "P1")])
        self.assertEqual(timeline[0].x_cm, 11.0)


if __name__ == "__main__":
    unittest.main()
