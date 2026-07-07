"""Read merged CV/UWB player positions without presentation dependencies."""

from __future__ import annotations

import bisect
import csv
import io
import time
from collections import OrderedDict
from pathlib import Path


TRACKED_ALL_PLAYERS = "__all__"
LARGE_MERGED_THRESHOLD_BYTES = 10 * 1024 * 1024
LAZY_FRAME_CACHE_SIZE = 24


def _first_value(row, *names, default=None):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return default


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _to_bool(value, default=True):
    if isinstance(value, bool):
        return value
    if value is None or str(value).strip() == "":
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "none", "no", "off"}


def _normalize_player_id(row):
    value = _first_value(row, "stable_id", "player_id", "raw_player_id", "track_id", "id", default="player")
    return str(value).strip() or "player"


def normalize_merged_row(row, *, source="merged"):
    """Normalize known CV and UWB merged schemas to centimetre fields."""
    x_cm = _to_float(_first_value(row, "x_cm", "X", "x", "raw_x"))
    y_cm = _to_float(_first_value(row, "y_cm", "Y", "y", "raw_y"))
    z_raw = _first_value(row, "z_cm", "Z", "z", "raw_z")
    raw_player_id = _first_value(row, "raw_player_id")
    source_ids = [item.strip() for item in str(_first_value(row, "source_ids", default="")).split("|") if item.strip()]
    if raw_player_id is not None and str(raw_player_id) not in source_ids:
        source_ids.insert(0, str(raw_player_id))
    return {
        "frame": _to_int(_first_value(row, "frame", "frame_index")),
        "timestamp_s": _to_float(_first_value(row, "timestamp_s", "t", "time_s")),
        "player_id": _normalize_player_id(row),
        "raw_player_id": None if raw_player_id is None else str(raw_player_id),
        "source_player_ids": source_ids,
        "x_cm": x_cm,
        "y_cm": y_cm,
        "z_cm": None if z_raw is None else _to_float(z_raw),
        "x": x_cm,
        "y": y_cm,
        "raw_x": x_cm,
        "raw_y": y_cm,
        "vx": _to_float(_first_value(row, "vx", "vx_cm_s")),
        "vy": _to_float(_first_value(row, "vy", "vy_cm_s")),
        "source": source,
        "visible": _to_bool(_first_value(row, "visible", "on_terrain", "valid"), True),
        "confidence": _to_float(_first_value(row, "confidence"), 1.0),
        "track_age": _to_int(_first_value(row, "track_age")),
        "track_hits": _to_int(_first_value(row, "track_hits")),
        "track_misses": _to_int(_first_value(row, "track_misses")),
        "status": str(_first_value(row, "status", default="")),
    }


def load_merged_frames(csv_path, *, source="merged"):
    if not csv_path:
        return []
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [normalize_merged_row(row, source=source) for row in csv.DictReader(handle)]


def load_cv_merged_frames(csv_path):
    return load_merged_frames(csv_path, source="cv_merged")


def load_uwb_merged_frames(csv_path):
    return load_merged_frames(csv_path, source="uwb_merged")


def _ordered_unique(rows):
    ordered = sorted(rows, key=lambda row: (row["timestamp_s"], row["frame"], row["player_id"], row["source"]))
    unique = OrderedDict()
    for row in ordered:
        key = (row["timestamp_s"], row["frame"], row["player_id"], row["source"])
        previous = unique.get(key)
        if previous is None or row["confidence"] >= previous["confidence"]:
            unique[key] = row
    return list(unique.values())


def load_both_merged_frames(cv_csv_path, uwb_csv_path):
    return _ordered_unique(load_cv_merged_frames(cv_csv_path) + load_uwb_merged_frames(uwb_csv_path))


def load_players_merged_frames(mode, *, cv_csv_path=None, uwb_csv_path=None):
    normalized_mode = str(mode or "cv").strip().lower()
    if normalized_mode == "uwb":
        return load_uwb_merged_frames(uwb_csv_path)
    if normalized_mode == "both":
        return load_both_merged_frames(cv_csv_path, uwb_csv_path)
    return load_cv_merged_frames(cv_csv_path)


def _normalize_expected_player_count(count):
    value = _to_int(count, 0)
    return value if value > 0 else None


class _IndexedMergedFrames:
    """Compact frame index for long merged sessions, with rows loaded on demand."""

    def __init__(self, csv_path, *, source="players"):
        self.csv_path = Path(csv_path)
        self.source = source
        self.frame_numbers = []
        self.timestamps = []
        self.offsets = []
        self.end_offsets = []
        self.player_ids = []
        self._cache = OrderedDict()
        self._stream = None
        self._build_index()

    def _build_index(self):
        player_ids = OrderedDict()
        with self.csv_path.open("rb") as stream:
            header_line = stream.readline().decode("utf-8-sig").rstrip("\r\n")
            self.fieldnames = next(csv.reader([header_line]))
            frame_column = self.fieldnames.index("frame")
            timestamp_column = self.fieldnames.index("timestamp_s")
            player_column = next(
                (self.fieldnames.index(name) for name in ("stable_id", "player_id", "raw_player_id") if name in self.fieldnames),
                None,
            )
            max_column = max(frame_column, timestamp_column, player_column or 0)
            previous_frame = None
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                columns = line.split(b",", max_column + 1)
                try:
                    frame = int(float(columns[frame_column]))
                    timestamp_s = float(columns[timestamp_column] or 0.0)
                except (IndexError, ValueError):
                    continue
                if frame != previous_frame:
                    if self.offsets:
                        self.end_offsets.append(offset)
                    self.frame_numbers.append(frame)
                    self.timestamps.append(timestamp_s)
                    self.offsets.append(offset)
                    previous_frame = frame
                if player_column is not None and player_column < len(columns):
                    player_id = columns[player_column].decode("utf-8", errors="replace").strip().strip('"')
                    if player_id:
                        player_ids[player_id] = None
            if self.offsets:
                self.end_offsets.append(stream.tell())
        self.player_ids = list(player_ids)

    def __len__(self):
        return len(self.frame_numbers)

    def __bool__(self):
        return bool(self.frame_numbers)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        frame = self._read_frame(index)
        self._cache[index] = frame
        self._cache.move_to_end(index)
        while len(self._cache) > LAZY_FRAME_CACHE_SIZE:
            self._cache.popitem(last=False)
        return frame

    def _read_frame(self, index):
        if self._stream is None:
            self._stream = self.csv_path.open("rb")
        start = self.offsets[index]
        self._stream.seek(start)
        payload = self._stream.read(self.end_offsets[index] - start).decode("utf-8", errors="replace")
        rows = [
            normalize_merged_row(row, source=self.source)
            for row in csv.DictReader(io.StringIO(payload), fieldnames=self.fieldnames)
        ]
        players = {}
        for row in rows:
            previous = players.get(row["player_id"])
            if previous is None or row["confidence"] >= previous["confidence"]:
                players[row["player_id"]] = row
        return {
            "frame": self.frame_numbers[index],
            "timestamp_s": self.timestamps[index],
            "players": sorted(players.values(), key=lambda item: item["player_id"]),
        }

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None


class TrackedPlayersCsvSource:
    """Compatibility source exposing time-based, positions-only playback."""

    def __init__(
        self,
        csv_path,
        player_id=None,
        expected_player_count=None,
    ):
        self.csv_path = str(csv_path) if csv_path else ""
        self.player_id = str(player_id) if player_id else None
        self.expected_player_count = _normalize_expected_player_count(expected_player_count)
        path = Path(csv_path) if csv_path else None
        if path is not None and path.exists() and path.stat().st_size >= LARGE_MERGED_THRESHOLD_BYTES:
            self.frames = _IndexedMergedFrames(path, source="players")
        else:
            rows = load_merged_frames(csv_path, source="players")
            self.frames = self._group_frames(rows)
        self.normalized_timestamps = self._normalized_timestamps()
        self.nominal_frame_period_s = self._nominal_period()
        self.playback_duration_s = (
            self.normalized_timestamps[-1] + self.nominal_frame_period_s if self.normalized_timestamps else 0.0
        )
        self.frame_index = 0
        self.playback_position_s = 0.0
        self.playback_paused = False
        self.playback_started_at = time.monotonic()
        self.all_player_ids = self._collect_player_ids()

    @staticmethod
    def _group_frames(rows):
        grouped = {}
        for row in rows:
            key = row["frame"]
            frame = grouped.setdefault(key, {"frame": key, "timestamp_s": row["timestamp_s"], "players": {}})
            frame["timestamp_s"] = min(frame["timestamp_s"], row["timestamp_s"])
            previous = frame["players"].get(row["player_id"])
            if previous is None or row["confidence"] >= previous["confidence"]:
                frame["players"][row["player_id"]] = row
        result = []
        for frame in sorted(grouped.values(), key=lambda item: (item["timestamp_s"], item["frame"])):
            frame["players"] = sorted(frame["players"].values(), key=lambda item: item["player_id"])
            result.append(frame)
        return result

    def _normalized_timestamps(self):
        if not self.frames:
            return []
        timestamps = getattr(self.frames, "timestamps", None)
        if timestamps is None:
            timestamps = [frame["timestamp_s"] for frame in self.frames]
        first = timestamps[0]
        return [max(0.0, timestamp - first) for timestamp in timestamps]

    def _nominal_period(self):
        deltas = [
            current - previous
            for previous, current in zip(self.normalized_timestamps, self.normalized_timestamps[1:])
            if current > previous
        ]
        return min(deltas) if deltas else 1.0 / 25.0

    def _collect_player_ids(self):
        indexed_ids = getattr(self.frames, "player_ids", None)
        if indexed_ids is not None:
            return list(indexed_ids)
        return list(dict.fromkeys(player["player_id"] for frame in self.frames for player in frame["players"]))

    @property
    def status(self):
        if not self.frames:
            return "players merged: aucun frame"
        name = Path(self.csv_path).name
        return f"players merged: {name} | frame {self.frame_index + 1}/{len(self.frames)}"

    @property
    def playback_state(self):
        return {
            "paused": self.playback_paused,
            "position_s": self.playback_position_s,
            "duration_s": self.playback_duration_s,
            "frame_index": self.frame_index,
            "frame_count": len(self.frames),
        }

    def set_selected_player(self, player_id):
        self.player_id = TRACKED_ALL_PLAYERS if player_id in (None, "", TRACKED_ALL_PLAYERS) else str(player_id)

    def cycle_selected_player(self, step=1):
        selectable = [TRACKED_ALL_PLAYERS, *self.all_player_ids]
        if self.player_id not in selectable:
            self.player_id = selectable[0] if selectable else None
        elif selectable:
            self.player_id = selectable[(selectable.index(self.player_id) + int(step)) % len(selectable)]
        return self.player_id

    def set_expected_player_count(self, count):
        self.expected_player_count = _normalize_expected_player_count(count)
        return self.expected_player_count

    def set_playback(self, position_s=None, paused=None):
        if position_s is not None:
            self.playback_position_s = min(self.playback_duration_s, max(0.0, float(position_s)))
        if paused is not None:
            self.playback_paused = bool(paused)
        self.playback_started_at = time.monotonic() - self.playback_position_s

    def toggle_pause(self):
        if not self.playback_paused:
            self._update_playback_position()
        self.set_playback(paused=not self.playback_paused)
        return self.playback_paused

    def seek_relative(self, delta_s):
        return self.seek_absolute(self.playback_position_s + float(delta_s))

    def seek_absolute(self, position_s, paused=True):
        self.set_playback(position_s=position_s, paused=paused)
        return self.playback_position_s

    def seek_frames(self, delta_frames):
        if not self.frames:
            return self.playback_position_s
        target = max(0, min(len(self.frames) - 1, self.frame_index + int(delta_frames)))
        self.frame_index = target
        return self.seek_absolute(self.normalized_timestamps[target])

    def _update_playback_position(self):
        if not self.playback_paused:
            self.playback_position_s = min(self.playback_duration_s, time.monotonic() - self.playback_started_at)
            if self.playback_position_s >= self.playback_duration_s:
                self.playback_paused = True

    def _frame_index_at(self, position_s):
        if not self.frames:
            return 0
        index = bisect.bisect_right(self.normalized_timestamps, position_s) - 1
        return max(0, min(index, len(self.frames) - 1))

    def _selected_player(self, players):
        if self.player_id == TRACKED_ALL_PLAYERS:
            return None
        if self.player_id is None and players:
            self.player_id = players[0]["player_id"]
        for player in players:
            aliases = [player["player_id"], player.get("raw_player_id"), *player.get("source_player_ids", [])]
            if self.player_id in aliases:
                self.player_id = player["player_id"]
                return player
        return None

    def get_frame_packet(self, position_s=None):
        if position_s is None:
            self._update_playback_position()
            position_s = self.playback_position_s
        else:
            self.playback_position_s = min(self.playback_duration_s, max(0.0, float(position_s)))
        if not self.frames:
            return self._empty_packet()
        self.frame_index = self._frame_index_at(position_s)
        frame = self.frames[self.frame_index]
        visible = [player for player in frame["players"] if player["visible"]]
        selected = self._selected_player(visible)
        positions = [self._packet_position(player, selected) for player in visible]
        if self.expected_player_count is not None:
            positions = sorted(
                positions,
                key=lambda item: (not item["selected"], -item["confidence"], item["player_id"]),
            )[: self.expected_player_count]
        selected_visible = selected is not None and any(item["selected"] for item in positions)
        primary = next(((item["x_cm"], item["y_cm"]) for item in positions if item["selected"]), None)
        return {
            "raw": {},
            "valid": bool(positions) and (self.player_id == TRACKED_ALL_PLAYERS or selected_visible),
            "source": "players",
            "status": self.status,
            "frame": frame["frame"],
            "timestamp_s": frame["timestamp_s"],
            "player_positions": positions,
            "selected_player_id": None if self.player_id == TRACKED_ALL_PLAYERS else self.player_id,
            "selection_mode": "all" if self.player_id == TRACKED_ALL_PLAYERS else "single",
            "all_player_ids": list(self.all_player_ids),
            "selected_player_visible": selected_visible,
            "primary_position": primary,
            "playback": self.playback_state,
        }

    @staticmethod
    def _packet_position(player, selected):
        result = dict(player)
        result["selected"] = selected is not None and player["player_id"] == selected["player_id"]
        result["on_terrain"] = result["visible"]
        return result

    def _empty_packet(self):
        return {
            "raw": {},
            "valid": False,
            "source": "players",
            "status": self.status,
            "frame": 0,
            "timestamp_s": 0.0,
            "player_positions": [],
            "selected_player_id": None if self.player_id == TRACKED_ALL_PLAYERS else self.player_id,
            "selection_mode": "all" if self.player_id == TRACKED_ALL_PLAYERS else "single",
            "all_player_ids": [],
            "selected_player_visible": False,
            "primary_position": None,
            "playback": self.playback_state,
        }

    def close(self):
        close_frames = getattr(self.frames, "close", None)
        if close_frames is not None:
            close_frames()
