from player_motion import compute_auto_jump_cm, compute_manual_jump_cm, compute_player_xy, trigger_manual_jump

from scene import CENTER_X, CENTER_Y, JUMP_EXTRA_HEIGHT_CM, TAG_BASE_Z_CM


def start_jump(state):
    trigger_manual_jump(state, JUMP_EXTRA_HEIGHT_CM, 0.62)


def update_jump(state, dt):
    return compute_manual_jump_cm(state, dt)


def compute_real_tag_position(t, state, dt):
    tag_x, tag_y = compute_player_xy(t, CENTER_X, CENTER_Y)
    jump_extra = compute_auto_jump_cm(t) + update_jump(state, dt)
    tag_z = TAG_BASE_Z_CM + jump_extra
    return (tag_x, tag_y, tag_z), jump_extra
