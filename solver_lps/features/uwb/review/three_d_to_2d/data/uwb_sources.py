import math

from solver_lps.features.uwb.review.data.review_sources import (
    DEFAULT_UWB_TAG_REVIEW_PATH,
    TagReviewPlaybackSource,
)


def distance_3d_xy_z(tag_xy, tag_z, anchor_xy, anchor_z):
    dx = anchor_xy[0] - tag_xy[0]
    dy = anchor_xy[1] - tag_xy[1]
    dz = anchor_z - tag_z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def project_3d_distance_to_2d(d3, dz_assumed):
    return math.sqrt(max(0.0, d3 * d3 - dz_assumed * dz_assumed))


def project_packet(raw_3d, settings):
    projected = {}
    dz_assumed = settings["anchor_height_cm"] - settings["tag_assumed_height_cm"]
    for aid, distance_cm in raw_3d.items():
        projected[aid] = project_3d_distance_to_2d(distance_cm, dz_assumed)
    return projected


def build_review_3d_distances(tag_xy, tag_height, anchors, settings):
    raw_3d = {}
    for aid, anchor_xy in anchors.items():
        raw_3d[aid] = distance_3d_xy_z(tag_xy, tag_height, anchor_xy, settings["anchor_height_cm"])
    return raw_3d


class DistanceSource:
    def __init__(self, uwb_tag_review_path=DEFAULT_UWB_TAG_REVIEW_PATH, **_kwargs):
        self.review_source = TagReviewPlaybackSource(uwb_tag_review_path)

    def toggle_review_pause(self):
        return self.review_source.toggle_pause()

    def seek_review_relative(self, delta_s):
        return self.review_source.seek_relative(delta_s)

    def seek_review_frames(self, delta_frames):
        return self.review_source.seek_frames(delta_frames)

    def get_distances(self, _tag_xy, _tag_height, anchors, settings):
        review_tag_real = self.review_source.get_position()
        if review_tag_real is None:
            return {
                "raw": {},
                "raw_3d": {},
                "valid": False,
                "source": "review",
                "status": "review 3D->2D: position indisponible",
                "tag_real": None,
                "uwb_playback": self.review_source.playback_state,
            }
        review_xy = (review_tag_real[0], review_tag_real[1])
        review_height = review_tag_real[2]
        raw_3d = build_review_3d_distances(review_xy, review_height, anchors, settings)
        active_ids = sorted(anchors.keys())
        return {
            "raw": project_packet(raw_3d, settings),
            "raw_3d": raw_3d,
            "valid": all(aid in raw_3d for aid in active_ids),
            "source": "review",
            "status": f"review 3D->2D: {self.review_source.frame_index + 1}/{self.review_source.playback_state['frame_count']}",
            "tag_real": review_tag_real,
            "uwb_playback": self.review_source.playback_state,
        }

    def close(self):
        return None
