import math

from solver_lps.features.uwb.review.data.review_sources import (
    DEFAULT_UWB_TAG_REVIEW_PATH,
    TagReviewPlaybackSource,
)


def vec_sub(a, b):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def length(v):
    return math.sqrt(dot(v, v))


def distance_3d(a, b):
    return length(vec_sub(a, b))


def build_review_distances(tag_real, anchors):
    raw = {}
    for aid, anchor_pos in anchors.items():
        raw[aid] = distance_3d(tag_real, anchor_pos)
    return raw


class DistanceSource:
    def __init__(self, uwb_tag_review_path=DEFAULT_UWB_TAG_REVIEW_PATH, **_kwargs):
        self.review_source = TagReviewPlaybackSource(uwb_tag_review_path)

    def toggle_review_pause(self):
        return self.review_source.toggle_pause()

    def seek_review_relative(self, delta_s):
        return self.review_source.seek_relative(delta_s)

    def seek_review_frames(self, delta_frames):
        return self.review_source.seek_frames(delta_frames)

    def get_distances(self, anchors):
        tag_real = self.review_source.get_position()
        raw = build_review_distances(tag_real, anchors) if tag_real is not None else {}
        active_ids = sorted(anchors.keys())
        return {
            "raw": raw,
            "valid": all(aid in raw for aid in active_ids),
            "source": "review",
            "status": f"review 3D: {self.review_source.frame_index + 1}/{self.review_source.playback_state['frame_count']}",
            "tag_real": tag_real,
            "uwb_playback": self.review_source.playback_state,
        }

    def close(self):
        return None
