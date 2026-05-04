import math
import random


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def length(v):
    return math.sqrt(dot(v, v))


def distance_3d(a, b):
    return length(vec_sub(b, a))


def noisy_distance(true_d, settings):
    sigma = max(1e-6, true_d * settings["noise_ratio"])
    d = true_d + random.gauss(0, sigma)
    if random.random() < settings["spike_prob"]:
        d += random.uniform(-settings["spike_amplitude"], settings["spike_amplitude"])
    return max(1.0, d)


def get_simulated_distances(tag_real, anchors, settings):
    dist_raw = {}
    for aid, a_pos in anchors.items():
        d_true = distance_3d(tag_real, a_pos)
        dist_raw[aid] = noisy_distance(d_true, settings)
    return {
        "raw": dist_raw,
        "valid": True,
        "source": "simulation",
    }
