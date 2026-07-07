from __future__ import annotations

from typing import Dict

from .tracking_lifecycle import TRACK_CONFIRM_MIN_HITS, should_promote_track


def filter_ghost_track(track, active_tracks, lost_tracks, args, stats: Dict[str, int]) -> None:
    from .tracking_pipeline import _confirmed_track_count

    if track.confirmed:
        return
    required_hits = max(int(getattr(args, "min_hits", 0) or 0), TRACK_CONFIRM_MIN_HITS)
    if track.hits < required_hits or track.age < required_hits:
        return

    confirmed_count = _confirmed_track_count(active_tracks, lost_tracks)
    expected_players = getattr(args, "expected_players", None)
    allowed, _reason = should_promote_track(
        track,
        current_confirmed_count=confirmed_count,
        expected_players=expected_players,
        min_confirm_hits=required_hits,
    )
    if allowed:
        track.confirmed = True
        stats["tracks_confirmed"] += 1
        stats["candidate_promoted"] += 1
        return

    stats["candidate_rejected"] += 1
