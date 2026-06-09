from solver_lps.features.uwb.realtime.data.udp_distance_receiver import UdpDistanceReceiver


class DistanceSource:
    def __init__(self, bind_ip="0.0.0.0", port=4210, max_age_s=2.0, **_kwargs):
        self.receiver = UdpDistanceReceiver(bind_ip=bind_ip, port=port, max_age_s=max_age_s)

    def get_distances(self, anchors):
        active_ids = sorted(anchors.keys())
        raw = self.receiver.get_distances(active_ids)
        return {
            "raw": raw,
            "valid": all(aid in raw for aid in active_ids),
            "source": "realtime",
            "status": self.receiver.get_status_text(active_ids).replace("udp", "realtime", 1),
            "tag_real": None,
            "uwb_playback": None,
        }

    def close(self):
        self.receiver.close()
