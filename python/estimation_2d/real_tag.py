import math

from scene import CENTER_X, CENTER_Y


def compute_real_tag_position(t):
    tag_x = CENTER_X + 420 * math.cos(t * 0.10)
    tag_y = CENTER_Y + 280 * math.sin(t * 0.08)
    return (tag_x, tag_y)
