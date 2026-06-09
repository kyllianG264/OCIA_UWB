import unittest

from solver_lps.features.players.domain.player_registry import (
    DEFAULT_PLAYER_ID,
    bind_selected_player_analytics,
    create_player_registry,
    select_player_profile,
    update_registry_player,
)


class PlayerRegistryTests(unittest.TestCase):
    def test_selecting_player_binds_selected_profile(self):
        state = {
            "player_registry": create_player_registry(1250.0, 900.0),
        }
        select_player_profile(state["player_registry"], "P001", show_card=True)
        analytics = bind_selected_player_analytics(state)
        self.assertEqual("P001", analytics["player_id"])
        self.assertTrue(analytics["card_visible"])

    def test_each_player_keeps_separate_heatmap(self):
        registry = create_player_registry(1250.0, 900.0)
        update_registry_player(
            registry,
            "P001",
            t=1.0,
            pos_xy=(100.0, 100.0),
            height_cm=None,
            jump_extra_cm=None,
            dt=0.1,
            name="P001",
            source_label="Tracking CV",
        )
        update_registry_player(
            registry,
            "P002",
            t=1.0,
            pos_xy=(200.0, 200.0),
            height_cm=None,
            jump_extra_cm=None,
            dt=0.1,
            name="P002",
            source_label="Tracking CV",
        )
        self.assertNotEqual(
            registry["profiles"]["P001"]["heatmap"],
            registry["profiles"]["P002"]["heatmap"],
        )
        self.assertEqual(DEFAULT_PLAYER_ID, registry["selected_player_id"])


if __name__ == "__main__":
    unittest.main()

