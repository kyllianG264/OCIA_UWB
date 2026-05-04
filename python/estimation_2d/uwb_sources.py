import math
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from uwb_udp import UdpDistanceReceiver


def distance_2d(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def noisy_distance(true_d, settings):
    d = true_d + random.gauss(0.0, settings["noise_std"])
    if random.random() < settings["spike_prob"]:
        d += random.uniform(-settings["spike_amplitude"], settings["spike_amplitude"])
    return max(1.0, d)


def get_simulated_distances(tag_real, anchors, settings):
    raw = {}
    for aid, anchor_pos in anchors.items():
        raw[aid] = noisy_distance(distance_2d(tag_real, anchor_pos), settings)
    return {
        "raw": raw,
        "valid": True,
        "source": "simulation",
        "status": "simulation",
    }


class DistanceSource:
    def __init__(self, mode="simulation", bind_ip="0.0.0.0", port=4210, max_age_s=2.0):
        self.mode = mode
        self.receiver = None
        if mode == "udp":
            self.receiver = UdpDistanceReceiver(bind_ip=bind_ip, port=port, max_age_s=max_age_s)

    @property
    def uses_simulated_tag(self):
        return self.mode != "udp"

    def get_distances(self, tag_real, anchors, settings):
        if self.mode != "udp":
            return get_simulated_distances(tag_real, anchors, settings)

        active_ids = sorted(anchors.keys())
        raw = self.receiver.get_distances(active_ids)
        return {
            "raw": raw,
            "valid": all(aid in raw for aid in active_ids),
            "source": "udp",
            "status": self.receiver.get_status_text(active_ids),
        }

    def close(self):
        if self.receiver is not None:
            self.receiver.close()
