from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, Optional


RAW_ID_ALIAS_OFFSET = 30
RAW_ID_REUSE_GRACE_FRAMES = 30
SPAWN_WARMUP_SECONDS = 5.0
SPAWN_BORDER_MARGIN = 160.0
BOOTSTRAP_IGNORE_SPAWN_GATE_FRAMES = 10
TRACK_CONFIRM_MIN_HITS = 8
TRACK_CONFIRM_MIN_TRAVEL_WARMUP = 24.0
TRACK_CONFIRM_MIN_TRAVEL_BORDER = 48.0
TRACK_CONFIRM_MIN_TRAVEL_QUOTA = 72.0
TRACK_CONFIRM_MIN_CONFIDENCE = 0.32
TRACK_CONFIRM_MIN_CONFIDENCE_QUOTA = 0.42
TRACK_CONFIRM_MAX_MISSES = 1
TRACK_CONFIRM_MAX_SPEED_PX_PER_FRAME = 160.0


logger = logging.getLogger(__name__)


class TrackStage(str, Enum):
    CANDIDATE = "candidate"
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"


def _get_value(source: Any, key: str, default=None):
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def normalize_raw_player_id(raw_player_id) -> str:
    text = str(raw_player_id or "").strip()
    return text


def parse_int_raw_player_id(raw_player_id) -> Optional[int]:
    text = normalize_raw_player_id(raw_player_id)
    if not text.isdigit():
        return None
    return int(text)


def raw_id_alias_match(previous_raw_id, new_raw_id) -> bool:
    previous = parse_int_raw_player_id(previous_raw_id)
    new = parse_int_raw_player_id(new_raw_id)
    if previous is None or new is None:
        return False
    if previous == new:
        return True
    return abs(previous - new) == RAW_ID_ALIAS_OFFSET


def track_stage_from_hits(hits: int, confirmed: bool, min_confirm_hits: int) -> TrackStage:
    if confirmed or hits >= int(min_confirm_hits or 0):
        return TrackStage.CONFIRMED
    if int(hits or 0) <= 1:
        return TrackStage.CANDIDATE
    return TrackStage.TENTATIVE


def _track_position(track: Any) -> tuple[float, float]:
    display_pos = _get_value(track, "display_pos", None)
    if display_pos is not None:
        return float(display_pos[0]), float(display_pos[1])

    state = _get_value(track, "state", None)
    if state is not None:
        try:
            return float(state[0, 0]), float(state[1, 0])
        except Exception:
            pass

    if isinstance(track, dict):
        return float(track.get("x", 0.0)), float(track.get("y", 0.0))
    return 0.0, 0.0


def classify_spawn_category(detection, first_timestamp_s: float, bounds=None, *, warmup_seconds: float = SPAWN_WARMUP_SECONDS, border_margin: float = SPAWN_BORDER_MARGIN) -> str:
    elapsed_s = max(0.0, float(_get_value(detection, "timestamp_s", 0.0) or 0.0) - float(first_timestamp_s or 0.0))
    if elapsed_s <= float(warmup_seconds):
        return "warmup"
    if bounds is None:
        return "border"
    left, right, top, bottom = bounds
    x_value = float(_get_value(detection, "x", 0.0))
    y_value = float(_get_value(detection, "y", 0.0))
    if (
        abs(x_value - left) <= float(border_margin)
        or abs(x_value - right) <= float(border_margin)
        or abs(y_value - top) <= float(border_margin)
        or abs(y_value - bottom) <= float(border_margin)
    ):
        return "border"
    return "midfield"


def is_bootstrap_spawn_frame(frame: int, bootstrap_ignore_spawn_gate_frames: int = BOOTSTRAP_IGNORE_SPAWN_GATE_FRAMES) -> bool:
    current_frame = int(frame)
    return 1 <= current_frame <= int(bootstrap_ignore_spawn_gate_frames or 0)


def track_travel_distance(track: Any) -> float:
    spawn_position = _get_value(track, "spawn_position", None)
    if spawn_position is None:
        return 0.0
    current_x, current_y = _track_position(track)
    try:
        spawn_x, spawn_y = float(spawn_position[0]), float(spawn_position[1])
    except Exception:
        return 0.0
    return ((current_x - spawn_x) ** 2 + (current_y - spawn_y) ** 2) ** 0.5


def track_speed(track: Any) -> float:
    velocity = _get_value(track, "velocity", None)
    if velocity is not None:
        try:
            return (float(velocity[0]) ** 2 + float(velocity[1]) ** 2) ** 0.5
        except Exception:
            pass

    state = _get_value(track, "state", None)
    if state is not None:
        try:
            return (float(state[2, 0]) ** 2 + float(state[3, 0]) ** 2) ** 0.5
        except Exception:
            pass
    return 0.0


def should_promote_track(
    track: Any,
    *,
    current_confirmed_count: int,
    expected_players: Optional[int],
    min_confirm_hits: int,
) -> tuple[bool, str]:
    if bool(_get_value(track, "confirmed", False)):
        return True, "already_confirmed"

    hits = int(_get_value(track, "hits", 0) or 0)
    age = int(_get_value(track, "age", 0) or 0)
    misses = int(_get_value(track, "misses", 0) or 0)
    confidence = float(_get_value(track, "confidence", 0.0) or 0.0)
    spawn_category = str(_get_value(track, "spawn_category", "midfield") or "midfield")
    required_hits = max(int(min_confirm_hits or 0), TRACK_CONFIRM_MIN_HITS)

    if hits < required_hits or age < required_hits:
        return False, "insufficient_hits"
    if misses > TRACK_CONFIRM_MAX_MISSES:
        return False, "too_many_misses"
    if confidence < TRACK_CONFIRM_MIN_CONFIDENCE:
        return False, "low_confidence"
    if spawn_category == "midfield":
        return False, "midfield_spawn_blocked"

    travel_distance = track_travel_distance(track)
    if spawn_category == "warmup":
        min_travel = TRACK_CONFIRM_MIN_TRAVEL_WARMUP
    else:
        min_travel = TRACK_CONFIRM_MIN_TRAVEL_BORDER

    if expected_players is not None and expected_players > 0 and current_confirmed_count >= expected_players:
        min_travel = max(min_travel, TRACK_CONFIRM_MIN_TRAVEL_QUOTA)
        if confidence < TRACK_CONFIRM_MIN_CONFIDENCE_QUOTA:
            return False, "quota_full_low_confidence"

    if travel_distance < min_travel:
        return False, "insufficient_entry_motion"

    speed = track_speed(track)
    if speed > TRACK_CONFIRM_MAX_SPEED_PX_PER_FRAME:
        return False, "impossible_speed"

    return True, "confirmed"


@dataclass
class SpawnGuard:
    raw_id_reuse_grace_frames: int = RAW_ID_REUSE_GRACE_FRAMES
    alias_offset: int = RAW_ID_ALIAS_OFFSET
    last_seen_frame_by_raw_id: Dict[str, int] = field(default_factory=dict)
    last_duplicate_frame_by_raw_id: Dict[str, int] = field(default_factory=dict)
    last_offterrain_frame_by_raw_id: Dict[str, int] = field(default_factory=dict)

    def observe(self, raw_player_id, frame: int, on_terrain: bool = True):
        key = normalize_raw_player_id(raw_player_id)
        if not key:
            return
        current_frame = int(frame)
        self.last_seen_frame_by_raw_id[key] = current_frame
        if not on_terrain:
            self.last_offterrain_frame_by_raw_id[key] = current_frame

    def mark_duplicate(self, raw_player_id, frame: int):
        key = normalize_raw_player_id(raw_player_id)
        if not key:
            return
        current_frame = int(frame)
        self.last_duplicate_frame_by_raw_id[key] = current_frame
        parsed = parse_int_raw_player_id(key)
        if parsed is None:
            return
        self.last_duplicate_frame_by_raw_id[str(parsed + self.alias_offset)] = current_frame
        if parsed >= self.alias_offset:
            self.last_duplicate_frame_by_raw_id[str(parsed - self.alias_offset)] = current_frame

    def has_recent_offterrain_origin(self, raw_player_id, frame: int) -> bool:
        key = normalize_raw_player_id(raw_player_id)
        if not key:
            return False
        current_frame = int(frame)
        candidates = {key}
        parsed = parse_int_raw_player_id(key)
        if parsed is not None:
            candidates.add(str(parsed + self.alias_offset))
            if parsed >= self.alias_offset:
                candidates.add(str(parsed - self.alias_offset))
        for candidate in candidates:
            last_seen = self.last_offterrain_frame_by_raw_id.get(candidate)
            if last_seen is not None and current_frame - int(last_seen) <= self.raw_id_reuse_grace_frames:
                return True
        return False

    def is_recent(self, raw_player_id, frame: int) -> bool:
        key = normalize_raw_player_id(raw_player_id)
        if not key:
            return False
        current_frame = int(frame)
        candidates = {key}
        parsed = parse_int_raw_player_id(key)
        if parsed is not None:
            candidates.add(str(parsed + self.alias_offset))
            if parsed >= self.alias_offset:
                candidates.add(str(parsed - self.alias_offset))
        for candidate in candidates:
            last_seen = self.last_seen_frame_by_raw_id.get(candidate)
            if last_seen is not None and current_frame - int(last_seen) <= self.raw_id_reuse_grace_frames:
                return True
        return False

    def is_recent_duplicate(self, raw_player_id, frame: int) -> bool:
        key = normalize_raw_player_id(raw_player_id)
        if not key:
            return False
        current_frame = int(frame)
        candidates = {key}
        parsed = parse_int_raw_player_id(key)
        if parsed is not None:
            candidates.add(str(parsed + self.alias_offset))
            if parsed >= self.alias_offset:
                candidates.add(str(parsed - self.alias_offset))
        for candidate in candidates:
            last_seen = self.last_duplicate_frame_by_raw_id.get(candidate)
            if last_seen is not None and current_frame - int(last_seen) <= self.raw_id_reuse_grace_frames:
                return True
        return False

    def should_allow_spawn(
        self,
        detection,
        frame: int,
        first_timestamp_s: float,
        bounds=None,
        warmup_seconds: float = SPAWN_WARMUP_SECONDS,
        border_margin: float = SPAWN_BORDER_MARGIN,
        bootstrap_ignore_spawn_gate_frames: int = BOOTSTRAP_IGNORE_SPAWN_GATE_FRAMES,
    ) -> bool:
        if not _get_value(detection, "spawn_allowed", True):
            return False
        if self.is_recent_duplicate(_get_value(detection, "raw_player_id", ""), frame):
            return False
        if self.is_recent(_get_value(detection, "raw_player_id", ""), frame):
            return False
        if is_bootstrap_spawn_frame(frame, bootstrap_ignore_spawn_gate_frames):
            return True
        return not _get_value(detection, "on_terrain", True)
