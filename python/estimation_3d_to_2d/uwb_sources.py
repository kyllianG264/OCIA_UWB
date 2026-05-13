import math
import os
import random
import sys


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from uwb_udp import UdpDistanceReceiver


def distance_3d_xy_z(tag_xy, tag_z, anchor_xy, anchor_z):
    dx = anchor_xy[0] - tag_xy[0]
    dy = anchor_xy[1] - tag_xy[1]
    dz = anchor_z - tag_z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def noisy_distance(true_d, settings):
    d = true_d + random.gauss(0.0, settings["noise_std"])
    if random.random() < settings["spike_prob"]:
        d += random.uniform(-settings["spike_amplitude"], settings["spike_amplitude"])
    return max(1.0, d)


def project_3d_distance_to_2d(d3, dz_assumed):
    return math.sqrt(max(0.0, d3 * d3 - dz_assumed * dz_assumed))


def project_packet(raw_3d, settings):
    projected = {}
    dz_assumed = settings["anchor_height_cm"] - settings["tag_assumed_height_cm"]
    for aid, distance_cm in raw_3d.items():
        projected[aid] = project_3d_distance_to_2d(distance_cm, dz_assumed)
    return projected


def get_simulated_distances(tag_xy, tag_height, anchors, settings):
    raw_3d = {}
    for aid, anchor_xy in anchors.items():
        raw_3d[aid] = noisy_distance(distance_3d_xy_z(tag_xy, tag_height, anchor_xy, settings["anchor_height_cm"]), settings)
    return {"raw": project_packet(raw_3d, settings), "raw_3d": raw_3d, "valid": True, "source": "simulation", "status": "simulation"}


class DistanceSource:
    def __init__(self, mode="simulation", bind_ip="0.0.0.0", port=4210, max_age_s=2.0):
        self.mode = mode
        self.receiver = None
        if mode == "udp":
            self.receiver = UdpDistanceReceiver(bind_ip=bind_ip, port=port, max_age_s=max_age_s)

    @property
    def uses_simulated_tag(self):
        return self.mode != "udp"

    def get_distances(self, tag_xy, tag_height, anchors, settings):
        if self.mode != "udp":
            return get_simulated_distances(tag_xy, tag_height, anchors, settings)
        active_ids = sorted(anchors.keys())
        raw_3d = self.receiver.get_distances(active_ids)
        return {"raw": project_packet(raw_3d, settings), "raw_3d": raw_3d, "valid": all(aid in raw_3d for aid in active_ids), "source": "udp", "status": self.receiver.get_status_text(active_ids)}

    def close(self):
        if self.receiver is not None:
            self.receiver.close()
