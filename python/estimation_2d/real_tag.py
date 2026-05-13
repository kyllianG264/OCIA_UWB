from player_motion import compute_player_xy

from scene import CENTER_X, CENTER_Y


def compute_real_tag_position(t):
    return compute_player_xy(t, CENTER_X, CENTER_Y)
