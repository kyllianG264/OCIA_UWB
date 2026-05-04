import re
import socket
import time


A_INDEXED_DISTANCE_RE = re.compile(
    r"\bA\s*([1-9]\d*)\s*[:=]\s*(-?\d+(?:[\.,]\d+)?)",
    re.IGNORECASE,
)
D_INDEXED_DISTANCE_RE = re.compile(
    r"\bD\s*([1-9]\d*)\s*[:=]\s*(-?\d+(?:[\.,]\d+)?)",
    re.IGNORECASE,
)
ANCHOR_WITH_D1_RE = re.compile(
    r"\bA\s*[:=]\s*([1-9]\d*)\b.*?\bD1\s*[:=]\s*(-?\d+(?:[\.,]\d+)?)",
    re.IGNORECASE,
)
KEY_VALUE_DISTANCE_RE = re.compile(
    r"\banchor\s*[:=]\s*([1-9]\d*)\b.*?\bdistance(?:_cm)?\s*[:=]\s*(-?\d+(?:[\.,]\d+)?)",
    re.IGNORECASE,
)


def _to_float(value):
    return float(value.replace(",", "."))


def parse_distance_message(message):
    """Return {anchor_id: distance_cm} parsed from one UDP/serial line."""
    distances = {}
    text = message.strip()

    if not text or text.upper() == "NO_DATA":
        return distances

    anchor_with_d1_seen = False
    for match in ANCHOR_WITH_D1_RE.finditer(text):
        anchor_with_d1_seen = True
        distances[int(match.group(1))] = _to_float(match.group(2))

    for match in A_INDEXED_DISTANCE_RE.finditer(text):
        distances[int(match.group(1))] = _to_float(match.group(2))

    if not anchor_with_d1_seen:
        for match in D_INDEXED_DISTANCE_RE.finditer(text):
            distances[int(match.group(1))] = _to_float(match.group(2))

    for match in KEY_VALUE_DISTANCE_RE.finditer(text):
        distances[int(match.group(1))] = _to_float(match.group(2))

    return distances


class UdpDistanceReceiver:
    def __init__(self, bind_ip="0.0.0.0", port=4210, max_age_s=2.0):
        self.bind_ip = bind_ip
        self.port = int(port)
        self.max_age_s = float(max_age_s)
        self.last_values = {}
        self.last_sources = {}
        self.last_message = ""

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.bind_ip, self.port))
        self.sock.setblocking(False)

    def poll(self):
        received = []
        while True:
            try:
                data, addr = self.sock.recvfrom(2048)
            except BlockingIOError:
                break

            now = time.monotonic()
            message = data.decode("utf-8", errors="replace").strip()
            self.last_message = message

            parsed = parse_distance_message(message)
            for anchor_id, distance_cm in parsed.items():
                self.last_values[anchor_id] = (distance_cm, now)
                self.last_sources[anchor_id] = addr

            received.append((message, addr, parsed))

        return received

    def get_distances(self, active_ids=None):
        self.poll()
        now = time.monotonic()
        result = {}

        ids = active_ids if active_ids is not None else sorted(self.last_values.keys())
        for anchor_id in ids:
            entry = self.last_values.get(anchor_id)
            if entry is None:
                continue
            distance_cm, timestamp = entry
            if now - timestamp <= self.max_age_s:
                result[anchor_id] = distance_cm

        return result

    def get_status_text(self, active_ids=None):
        distances = self.get_distances(active_ids)
        if active_ids is None:
            expected = len(distances)
        else:
            expected = len(active_ids)
        return f"UDP {self.bind_ip}:{self.port} | {len(distances)}/{expected} distances"

    def close(self):
        self.sock.close()
