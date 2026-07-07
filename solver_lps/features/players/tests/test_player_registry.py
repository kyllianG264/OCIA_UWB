import unittest

from solver_lps.features.players.domain.player_registry import (
    create_player_registry,
    set_registry_bounds,
    update_registry_players_from_positions,
)


class PlayerRegistryTests(unittest.TestCase):
    def test_uwb_heatmap_uses_projected_position_but_keeps_metric_position(self):
        registry = create_player_registry(0.0, 0.0)
        set_registry_bounds(registry, (0.0, 396.0, 0.0, 735.0))

        profile = update_registry_players_from_positions(
            registry,
            [{
                "player_id": "tag:uwb",
                "x_cm": 1250.0,
                "y_cm": 900.0,
                "x": 198.0,
                "y": 367.5,
                "heatmap_x": 198.0,
                "heatmap_y": 367.5,
            }],
            t=1.0,
            dt=0.04,
        )[0]

        self.assertEqual((1250.0, 900.0), profile["last_pos"])
        self.assertEqual(1, sum(sum(row) for row in profile["heatmap"]))

    def test_accepts_canonical_coordinates_and_stable_id_alias(self):
        registry = create_player_registry(0.0, 0.0)
        updated = update_registry_players_from_positions(
            registry,
            [{"stable_id": "P9", "x_cm": 12.0, "y_cm": 34.0}],
            t=1.0,
            dt=0.04,
        )
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]["player_id"], "P9")
        self.assertEqual(updated[0]["last_pos"], (12.0, 34.0))


if __name__ == "__main__":
    unittest.main()
