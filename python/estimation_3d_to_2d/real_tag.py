import math
import random

from scene import (
    AUTO_JUMP_PROB_PER_SEC,
    CENTER_X,
    CENTER_Y,
    JUMP_DURATION_MAX,
    JUMP_DURATION_MIN,
    JUMP_EXTRA_HEIGHT_CM,
)


def start_jump(state):
    state["jump_active"] = True
    state["jump_timer"] = 0.0
    state["jump_duration"] = random.uniform(JUMP_DURATION_MIN, JUMP_DURATION_MAX)
    state["jump_extra_height"] = JUMP_EXTRA_HEIGHT_CM * random.uniform(0.90, 1.10)


def update_jump(state, dt):
    if not state["jump_active"]:
        if random.random() < AUTO_JUMP_PROB_PER_SEC * dt:
            start_jump(state)
        return 0.0

    state["jump_timer"] += dt
    p = state["jump_timer"] / state["jump_duration"]
    if p >= 1.0:
        state["jump_active"] = False
        state["jump_timer"] = 0.0
        return 0.0

    return state["jump_extra_height"] * math.sin(math.pi * p)


def compute_real_tag_position(t, state, dt):
    tag_x = CENTER_X + 420 * math.cos(t * 0.10)
    tag_y = CENTER_Y + 280 * math.sin(t * 0.08)
    jump_extra = update_jump(state, dt)
    tag_height = state["settings"]["tag_base_height_cm"] + jump_extra
    return (tag_x, tag_y), tag_height, jump_extra
