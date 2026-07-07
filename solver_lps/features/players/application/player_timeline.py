"""Build a sorted, deduplicated timeline of player positions."""

from __future__ import annotations

from pathlib import Path

from solver_lps.features.players.data.merged_input import load_merged_frames
from solver_lps.features.players.domain.player_frame import PlayerFrame
from solver_lps.session_assets import SessionAssets, session_assets


def _session_contract(session):
    if isinstance(session, SessionAssets):
        return session
    if isinstance(session, dict):
        return session_assets(session.get("sport", "basket"), session.get("asset_set", session.get("set", "set1")))
    return session_assets()


def _session_merged_path(session, source):
    if isinstance(session, dict):
        explicit = session.get(f"{source}_merged_path")
        if explicit:
            return Path(explicit)
    assets = _session_contract(session)
    if source == "cv":
        return assets.cv_positions_merged_path
    if source == "uwb":
        return assets.uwb_positions_merged_path
    return None


def build_player_timeline(session=None, sources=("cv", "uwb")):
    observations = {}
    for source in dict.fromkeys(str(item).strip().lower() for item in sources or []):
        path = _session_merged_path(session, source)
        if path is None or not path.exists():
            continue
        for row in load_merged_frames(path, source=f"{source}_merged"):
            frame = PlayerFrame.from_mapping(row)
            key = (frame.timestamp_s, frame.frame, frame.player_id, frame.source)
            previous = observations.get(key)
            if previous is None or frame.confidence >= previous.confidence:
                observations[key] = frame
    return sorted(
        observations.values(),
        key=lambda item: (item.timestamp_s, item.frame, item.player_id, item.source),
    )
