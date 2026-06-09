import math

from solver_lps.features.uwb.realtime.data.udp_distance_receiver import UdpDistanceReceiver


def project_3d_distance_to_2d(d3, dz_assumed):
    return math.sqrt(max(0.0, d3 * d3 - dz_assumed * dz_assumed))


def project_packet(raw_3d, settings):
    projected = {}
    dz_assumed = settings["anchor_height_cm"] - settings["tag_assumed_height_cm"]
    for aid, distance_cm in raw_3d.items():
        projected[aid] = project_3d_distance_to_2d(distance_cm, dz_assumed)
    return projected


class DistanceSource:
    def __init__(self, bind_ip="0.0.0.0", port=4210, max_age_s=2.0, **_kwargs):
        self.receiver = UdpDistanceReceiver(bind_ip=bind_ip, port=port, max_age_s=max_age_s)

    def get_distances(self, _tag_xy, _tag_height, anchors, settings):
        active_ids = sorted(anchors.keys())
        raw_3d = self.receiver.get_distances(active_ids)
        return {
            "raw": project_packet(raw_3d, settings),
            "raw_3d": raw_3d,
            "valid": all(aid in raw_3d for aid in active_ids),
            "source": "realtime",
            "status": self.receiver.get_status_text(active_ids).replace("udp", "realtime", 1),
            "tag_real": None,
            "uwb_playback": None,
        }

    def close(self):
        self.receiver.close()
