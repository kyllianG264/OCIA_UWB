"""Domain reader for normalized UWB UDP acquisition."""

import re
import time

from solver_lps.features.uwb.acquisition.data.udp_input import UdpInput


KEY_VALUE_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*[:=]\s*([^,;\s]+)", re.IGNORECASE)
A_INDEXED_DISTANCE_RE = re.compile(r"\bA\s*(\d+)\s*[:=]\s*([0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE)
D_INDEXED_DISTANCE_RE = re.compile(r"\bD\s*(\d+)\s*[:=]\s*([0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE)
ANCHOR_WITH_D1_RE = re.compile(r"\bA\s*(\d+)\s*(?:D1|DIST|DISTANCE)\s*[:=]\s*([0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE)
KEY_VALUE_DISTANCE_RE = re.compile(r"\b(?:dist|distance|range|r)\s*([1-9]\d*)\s*[:=]\s*([0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE)


def _to_float(value):
    return float(str(value).replace(",", "."))


def _to_float_or_none(value):
    text = str(value).strip()
    if not text:
        return None
    try:
        return _to_float(text)
    except ValueError:
        return None


def _to_int_or_none(value):
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _is_false_flag(value):
    return str(value).strip().lower() in {"0", "false", "off", "ko", "invalid", "none", "nan"}


def parse_message_fields(message):
    fields = {}
    for match in KEY_VALUE_RE.finditer(message):
        fields[match.group(1).strip().lower()] = match.group(2).strip()
    return fields


def parse_distance_message(message):
    distances = {}
    text = message.strip().upper()
    fields = parse_message_fields(message)

    anchor_id = _to_int_or_none(fields.get("anchor") or fields.get("id") or fields.get("a"))
    if anchor_id is not None and not _is_false_flag(fields.get("valid", "1")):
        for key in ("d1", "dist", "distance", "range", "r"):
            distance_cm = _to_float_or_none(fields.get(key))
            if distance_cm is not None:
                distances[anchor_id] = distance_cm
                break

    for pattern in (ANCHOR_WITH_D1_RE, A_INDEXED_DISTANCE_RE, D_INDEXED_DISTANCE_RE, KEY_VALUE_DISTANCE_RE):
        for match in pattern.finditer(text):
            distances[int(match.group(1))] = _to_float(match.group(2))

    return distances


class UdpReader:
    def __init__(self, udp_input, max_age_s=2.0, clock=None):
        self.udp_input = udp_input
        self.max_age_s = float(max_age_s)
        self.latest = {}
        self.last_poll_time = None
        self.packet_count = 0
        self.last_message = ""
        self._clock = clock or time.time

    def poll(self):
        events = []
        payloads = self.udp_input.poll_payloads()
        self.last_poll_time = payloads[-1]["received_at"] if payloads else self.last_poll_time
        for payload in payloads:
            message = payload["message"]
            addr = payload["addr"]
            received_at = payload["received_at"]
            parsed = parse_distance_message(message)
            for anchor_id, distance_cm in parsed.items():
                self.latest[int(anchor_id)] = {
                    "distance_cm": float(distance_cm),
                    "timestamp": received_at,
                    "addr": addr,
                    "message": message,
                }
            self.packet_count += 1
            self.last_message = message
            events.append((message, addr, parsed, received_at))
        return events

    def get_distances(self, active_ids):
        self.poll()
        now = self._clock()
        distances = {}
        for anchor_id in active_ids:
            record = self.latest.get(int(anchor_id))
            if record is None:
                continue
            if now - record["timestamp"] <= self.max_age_s:
                distances[int(anchor_id)] = record["distance_cm"]
            else:
                self.latest.pop(int(anchor_id), None)
        return distances

    def get_status_text(self, active_ids):
        active_ids = [int(anchor_id) for anchor_id in active_ids]
        available = sorted(aid for aid in active_ids if aid in self.get_distances(active_ids))
        if not available:
            return f"udp {self.udp_input.bind_ip}:{self.udp_input.port} - aucune distance recente"
        available_text = ", ".join(str(aid) for aid in available)
        return f"udp {self.udp_input.bind_ip}:{self.udp_input.port} - ancres recues: {available_text}"

    def close(self):
        self.udp_input.close()


class UdpDistanceReceiver:
    def __init__(self, bind_ip="0.0.0.0", port=4210, max_age_s=2.0):
        self._input = UdpInput(bind_ip=bind_ip, port=port)
        self._reader = UdpReader(self._input, max_age_s=max_age_s)

    @property
    def bind_ip(self):
        return self._input.bind_ip

    @property
    def port(self):
        return self._input.port

    @property
    def max_age_s(self):
        return self._reader.max_age_s

    @property
    def latest(self):
        return self._reader.latest

    @property
    def last_poll_time(self):
        return self._reader.last_poll_time

    @property
    def packet_count(self):
        return self._reader.packet_count

    @property
    def last_message(self):
        return self._reader.last_message

    def poll(self):
        return [(message, addr, parsed) for message, addr, parsed, _received_at in self._reader.poll()]

    def get_distances(self, active_ids):
        return self._reader.get_distances(active_ids)

    def get_status_text(self, active_ids):
        return self._reader.get_status_text(active_ids)

    def close(self):
        self._reader.close()
