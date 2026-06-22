import unittest

from solver_lps.features.cv.review.domain.tracking_lifecycle import (
    BOOTSTRAP_IGNORE_SPAWN_GATE_FRAMES,
    SpawnGuard,
    TrackStage,
    should_promote_track,
    raw_id_alias_match,
    track_stage_from_hits,
)


class TrackingLifecycleTests(unittest.TestCase):
    def test_recent_raw_id_blocks_spawn(self):
        guard = SpawnGuard(raw_id_reuse_grace_frames=30)
        guard.observe("4", 100, on_terrain=False)

        detection = {"raw_player_id": "4", "timestamp_s": 11.0, "x": 57.0, "y": 553.0, "spawn_allowed": True}
        self.assertFalse(guard.should_allow_spawn(detection, 105, 0.0, (0.0, 400.0, 0.0, 800.0)))

    def test_alias_raw_id_blocks_spawn(self):
        guard = SpawnGuard(raw_id_reuse_grace_frames=30)
        guard.observe("4", 100, on_terrain=False)

        detection = {"raw_player_id": "34", "timestamp_s": 11.0, "x": 57.0, "y": 553.0, "spawn_allowed": True}
        self.assertFalse(guard.should_allow_spawn(detection, 105, 0.0, (0.0, 400.0, 0.0, 800.0)))

    def test_recent_duplicate_blocks_spawn(self):
        guard = SpawnGuard(raw_id_reuse_grace_frames=30)
        guard.mark_duplicate("17", 100)

        detection = {"raw_player_id": "17", "timestamp_s": 11.0, "x": 120.0, "y": 240.0, "spawn_allowed": True}
        self.assertFalse(guard.should_allow_spawn(detection, 105, 0.0, (0.0, 400.0, 0.0, 800.0)))

    def test_new_raw_id_can_spawn_near_border(self):
        guard = SpawnGuard(raw_id_reuse_grace_frames=30)
        detection = {"raw_player_id": "99", "timestamp_s": 1.0, "x": 8.0, "y": 12.0, "spawn_allowed": True, "on_terrain": False}
        self.assertTrue(guard.should_allow_spawn(detection, 3, 0.0, (0.0, 400.0, 0.0, 800.0)))

    def test_on_terrain_spawn_is_allowed_during_bootstrap_and_blocked_after(self):
        guard = SpawnGuard(raw_id_reuse_grace_frames=30)
        on_terrain_detection = {
            "raw_player_id": "7",
            "timestamp_s": 1.0,
            "x": 120.0,
            "y": 220.0,
            "spawn_allowed": True,
            "on_terrain": True,
        }
        self.assertTrue(
            guard.should_allow_spawn(
                on_terrain_detection,
                BOOTSTRAP_IGNORE_SPAWN_GATE_FRAMES,
                0.0,
                (0.0, 400.0, 0.0, 800.0),
            )
        )
        self.assertFalse(
            guard.should_allow_spawn(
                on_terrain_detection,
                BOOTSTRAP_IGNORE_SPAWN_GATE_FRAMES + 1,
                0.0,
                (0.0, 400.0, 0.0, 800.0),
            )
        )

        guard.observe("7", 2, on_terrain=False)
        self.assertFalse(
            guard.should_allow_spawn(
                on_terrain_detection,
                BOOTSTRAP_IGNORE_SPAWN_GATE_FRAMES + 1,
                0.0,
                (0.0, 400.0, 0.0, 800.0),
            )
        )

    def test_bootstrap_still_blocks_recent_duplicate_spawn(self):
        guard = SpawnGuard(raw_id_reuse_grace_frames=30)
        guard.mark_duplicate("17", 5)

        detection = {
            "raw_player_id": "17",
            "timestamp_s": 0.16,
            "x": 120.0,
            "y": 240.0,
            "spawn_allowed": True,
            "on_terrain": True,
        }

        self.assertFalse(
            guard.should_allow_spawn(
                detection,
                BOOTSTRAP_IGNORE_SPAWN_GATE_FRAMES,
                0.0,
                (0.0, 400.0, 0.0, 800.0),
            )
        )

    def test_track_stage_from_hits(self):
        self.assertEqual(TrackStage.CANDIDATE, track_stage_from_hits(1, False, 8))
        self.assertEqual(TrackStage.TENTATIVE, track_stage_from_hits(3, False, 8))
        self.assertEqual(TrackStage.CONFIRMED, track_stage_from_hits(8, False, 8))

    def test_raw_id_alias_match(self):
        self.assertTrue(raw_id_alias_match("4", "4"))
        self.assertTrue(raw_id_alias_match("4", "34"))
        self.assertTrue(raw_id_alias_match("34", "4"))
        self.assertFalse(raw_id_alias_match("4", "35"))

    def test_midfield_candidate_is_never_promoted(self):
        track = {
            "confirmed": False,
            "hits": 8,
            "age": 8,
            "misses": 0,
            "confidence": 0.9,
            "spawn_category": "midfield",
            "spawn_position": (200.0, 200.0),
            "display_pos": (260.0, 240.0),
        }

        allowed, reason = should_promote_track(
            track,
            current_confirmed_count=0,
            expected_players=10,
            min_confirm_hits=8,
        )

        self.assertFalse(allowed)
        self.assertEqual("midfield_spawn_blocked", reason)

    def test_border_candidate_needs_motion_before_promotion(self):
        track = {
            "confirmed": False,
            "hits": 8,
            "age": 8,
            "misses": 0,
            "confidence": 0.8,
            "spawn_category": "border",
            "spawn_position": (0.0, 0.0),
            "display_pos": (8.0, 10.0),
        }

        allowed, reason = should_promote_track(
            track,
            current_confirmed_count=0,
            expected_players=10,
            min_confirm_hits=8,
        )

        self.assertFalse(allowed)
        self.assertEqual("insufficient_entry_motion", reason)

        track["display_pos"] = (96.0, 24.0)
        allowed, reason = should_promote_track(
            track,
            current_confirmed_count=0,
            expected_players=10,
            min_confirm_hits=8,
        )

        self.assertTrue(allowed)
        self.assertEqual("confirmed", reason)

    def test_quota_full_requires_stronger_border_evidence(self):
        track = {
            "confirmed": False,
            "hits": 8,
            "age": 8,
            "misses": 0,
            "confidence": 0.35,
            "spawn_category": "border",
            "spawn_position": (0.0, 0.0),
            "display_pos": (120.0, 12.0),
        }

        allowed, reason = should_promote_track(
            track,
            current_confirmed_count=10,
            expected_players=10,
            min_confirm_hits=8,
        )

        self.assertFalse(allowed)
        self.assertEqual("quota_full_low_confidence", reason)


if __name__ == "__main__":
    unittest.main()
