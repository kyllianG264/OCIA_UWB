import re
import socket
import time


KEY_VALUE_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*[:=]\s*([^,;\s]+)", re.IGNORECASE)
A_INDEXED_DISTANCE_RE = re.compile(r"\bA\s*(\d+)\s*[:=]\s*([0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE)
D_INDEXED_DISTANCE_RE = re.compile(r"\bD\s*(\d+)\s*[:=]\s*([0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE)
ANCHOR_WITH_D1_RE = re.compile(
    r"\bA\s*(\d+)\s*(?:D1|DIST|DISTANCE)\s*[:=]\s*([0-9]+(?:[.,][0-9]+)?)",
    re.IGNORECASE,
)
KEY_VALUE_DISTANCE_RE = re.compile(
    r"\b(?:dist|distance|range|r)\s*([1-9]\d*)\s*[:=]\s*([0-9]+(?:[.,][0-9]+)?)",
    re.IGNORECASE,
)


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

    for pattern in (
        ANCHOR_WITH_D1_RE,
        A_INDEXED_DISTANCE_RE,
        D_INDEXED_DISTANCE_RE,
        KEY_VALUE_DISTANCE_RE,
    ):
        for match in pattern.finditer(text):
            anchor_id = int(match.group(1))
            distance_cm = _to_float(match.group(2))
            distances[anchor_id] = distance_cm

    return distances


class UdpDistanceReceiver:
    def __init__(self, bind_ip="0.0.0.0", port=4210, max_age_s=2.0):
        self.bind_ip = bind_ip
        self.port = int(port)
        self.max_age_s = float(max_age_s)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.bind_ip, self.port))
        self.socket.setblocking(False)
        self.latest = {}
        self.last_poll_time = None
        self.packet_count = 0
        self.last_message = ""

    def poll(self):
        events = []
        self.last_poll_time = time.time()
        while True:
            try:
                payload, addr = self.socket.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                break

            message = payload.decode("utf-8", errors="replace").strip()
            parsed = parse_distance_message(message)
            now = time.time()
            for anchor_id, distance_cm in parsed.items():
                self.latest[int(anchor_id)] = {
                    "distance_cm": float(distance_cm),
                    "timestamp": now,
                    "addr": addr,
                    "message": message,
                }
            self.packet_count += 1
            self.last_message = message
            events.append((message, addr, parsed))
        return events

    def get_distances(self, active_ids):
        self.poll()
        now = time.time()
        distances = {}
        for anchor_id in active_ids:
            record = self.latest.get(int(anchor_id))
            if record is None:
                continue
            if now - record["timestamp"] <= self.max_age_s:
                distances[int(anchor_id)] = record["distance_cm"]
        return distances

    def get_status_text(self, active_ids):
        active_ids = [int(anchor_id) for anchor_id in active_ids]
        available = sorted(aid for aid in active_ids if aid in self.get_distances(active_ids))
        if not available:
            return f"udp {self.bind_ip}:{self.port} - aucune distance recente"
        available_text = ", ".join(str(aid) for aid in available)
        return f"udp {self.bind_ip}:{self.port} - ancres recues: {available_text}"

    def close(self):
        try:
            self.socket.close()
        except OSError:
            pass
