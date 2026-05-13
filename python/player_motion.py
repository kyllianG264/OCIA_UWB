import math


def _clamp01(t):
    return max(0.0, min(1.0, t))


def _lerp(a, b, t):
    return a + (b - a) * t


def _smoothstep01(t):
    t = _clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def _ease_out_cubic(t):
    t = _clamp01(t)
    inv = 1.0 - t
    return 1.0 - inv * inv * inv


def _ease_in_out_sine(t):
    t = _clamp01(t)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _ease_sigmoid(t, slope=6.5):
    t = _clamp01(t)
    x = (t - 0.5) * slope
    y = 1.0 / (1.0 + math.exp(-x))
    y0 = 1.0 / (1.0 + math.exp(slope * 0.5))
    y1 = 1.0 / (1.0 + math.exp(-slope * 0.5))
    return (y - y0) / max(y1 - y0, 1e-6)


def _piecewise_progress(t, accel=0.22, decel=0.20):
    t = _clamp01(t)
    accel = max(0.05, min(0.45, accel))
    decel = max(0.05, min(0.45, decel))
    middle = max(0.05, 1.0 - accel - decel)
    v_peak = 2.0 / (accel + 2.0 * middle + decel)
    d_accel = 0.5 * v_peak * accel
    d_middle = v_peak * middle

    if t < accel:
        q = t / accel
        return d_accel * q * q
    if t < accel + middle:
        q = (t - accel) / middle
        return d_accel + d_middle * q
    q = (t - accel - middle) / decel
    return d_accel + d_middle + v_peak * decel * (q - 0.5 * q * q)


def _distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _unit_and_normal(a, b, fallback=(1.0, 0.0)):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        tx, ty = fallback
        tl = math.hypot(tx, ty)
        tx, ty = tx / max(tl, 1e-6), ty / max(tl, 1e-6)
        return (tx, ty), (-ty, tx)
    tx = dx / length
    ty = dy / length
    return (tx, ty), (-ty, tx)


def _segment_progress(kind, t):
    if kind == "hold":
        return _ease_in_out_sine(t)
    if kind == "burst":
        return _piecewise_progress(t, accel=0.18, decel=0.24)
    if kind == "sprint":
        return _piecewise_progress(t, accel=0.14, decel=0.16)
    if kind == "drive":
        return _piecewise_progress(t, accel=0.16, decel=0.26)
    if kind == "slide":
        return _ease_sigmoid(t, slope=7.5)
    if kind == "recovery":
        return _piecewise_progress(t, accel=0.26, decel=0.22)
    if kind == "curl":
        return _ease_in_out_sine(t)
    if kind == "stepback":
        return _piecewise_progress(t, accel=0.12, decel=0.34)
    if kind == "crash":
        return _piecewise_progress(t, accel=0.20, decel=0.20)
    if kind == "hesi":
        return 0.82 * _ease_out_cubic(t) + 0.18 * _smoothstep01(t)
    return _smoothstep01(t)


_SEGMENTS = (
    {
        "label": "receive outlet",
        "point": (-1110.0, 140.0),
        "duration": 1.15,
        "kind": "jog",
        "curve": 35.0,
        "sway": 10.0,
        "pace": 1.6,
    },
    {
        "label": "push in transition",
        "point": (-340.0, 120.0),
        "duration": 2.00,
        "kind": "sprint",
        "curve": -90.0,
        "sway": 15.0,
        "pace": 2.8,
    },
    {
        "label": "organize at top",
        "point": (120.0, -20.0),
        "duration": 1.35,
        "kind": "control",
        "curve": -130.0,
        "sway": 11.0,
        "pace": 1.8,
    },
    {
        "label": "hard cut to wing",
        "point": (560.0, -250.0),
        "duration": 0.92,
        "kind": "burst",
        "curve": -150.0,
        "sway": 16.0,
        "pace": 3.0,
    },
    {
        "label": "hesitation dribble",
        "point": (640.0, -180.0),
        "duration": 0.70,
        "kind": "hesi",
        "curve": 20.0,
        "sway": 18.0,
        "pace": 2.4,
    },
    {
        "label": "baseline drive",
        "point": (980.0, -470.0),
        "duration": 0.90,
        "kind": "drive",
        "curve": -110.0,
        "sway": 18.0,
        "pace": 3.1,
    },
    {
        "label": "attack rim",
        "point": (1030.0, -60.0),
        "duration": 0.80,
        "kind": "drive",
        "curve": 210.0,
        "sway": 14.0,
        "pace": 3.2,
    },
    {
        "label": "land and drift out",
        "point": (260.0, 210.0),
        "duration": 1.45,
        "kind": "recovery",
        "curve": 170.0,
        "sway": 12.0,
        "pace": 2.0,
    },
    {
        "label": "defensive slide",
        "point": (-470.0, 280.0),
        "duration": 1.35,
        "kind": "slide",
        "curve": 0.0,
        "sway": 14.0,
        "pace": 2.5,
    },
    {
        "label": "closeout corner",
        "point": (-940.0, 450.0),
        "duration": 0.96,
        "kind": "burst",
        "curve": 70.0,
        "sway": 15.0,
        "pace": 2.9,
    },
    {
        "label": "rebound crash",
        "point": (-140.0, 120.0),
        "duration": 1.02,
        "kind": "crash",
        "curve": 240.0,
        "sway": 13.0,
        "pace": 3.1,
    },
    {
        "label": "box out reset",
        "point": (-40.0, 80.0),
        "duration": 0.74,
        "kind": "hold",
        "curve": 10.0,
        "sway": 20.0,
        "pace": 1.7,
    },
    {
        "label": "outlet run right seam",
        "point": (780.0, 60.0),
        "duration": 1.42,
        "kind": "sprint",
        "curve": -120.0,
        "sway": 14.0,
        "pace": 2.9,
    },
    {
        "label": "curl to elbow",
        "point": (220.0, -240.0),
        "duration": 1.24,
        "kind": "curl",
        "curve": -220.0,
        "sway": 12.0,
        "pace": 2.2,
    },
    {
        "label": "step back slot",
        "point": (-280.0, -70.0),
        "duration": 0.98,
        "kind": "stepback",
        "curve": 120.0,
        "sway": 16.0,
        "pace": 2.6,
    },
    {
        "label": "relocate weak side",
        "point": (-760.0, -390.0),
        "duration": 1.36,
        "kind": "jog",
        "curve": -80.0,
        "sway": 11.0,
        "pace": 1.8,
    },
    {
        "label": "lift from corner",
        "point": (-300.0, -180.0),
        "duration": 1.06,
        "kind": "control",
        "curve": 130.0,
        "sway": 12.0,
        "pace": 2.0,
    },
    {
        "label": "middle attack pullup",
        "point": (340.0, 90.0),
        "duration": 1.05,
        "kind": "burst",
        "curve": 180.0,
        "sway": 15.0,
        "pace": 2.8,
    },
    {
        "label": "reset spacing",
        "point": (20.0, 170.0),
        "duration": 1.18,
        "kind": "recovery",
        "curve": -95.0,
        "sway": 10.0,
        "pace": 1.7,
    },
)

_START_POINT = (-1220.0, 160.0)
_CYCLE_DURATION = sum(segment["duration"] for segment in _SEGMENTS)


def _build_jump_windows():
    windows = []
    elapsed = 0.0
    for index, segment in enumerate(_SEGMENTS):
        start = elapsed
        if index == 6:
            windows.append((start + segment["duration"] * 0.30, 0.60, 74.0))
        elif index == 10:
            windows.append((start + segment["duration"] * 0.42, 0.66, 68.0))
        elif index == 17:
            windows.append((start + segment["duration"] * 0.54, 0.56, 56.0))
        elapsed += segment["duration"]
    return tuple(windows)


_JUMP_WINDOWS = _build_jump_windows()


def cycle_duration_seconds():
    return _CYCLE_DURATION


def _segment_state(local_t):
    elapsed = 0.0
    start = _START_POINT
    previous_dir = (1.0, 0.0)
    for segment in _SEGMENTS:
        duration = segment["duration"]
        end = segment["point"]
        if local_t <= elapsed + duration:
            return segment, start, end, local_t - elapsed, previous_dir
        direction, _ = _unit_and_normal(start, end, previous_dir)
        previous_dir = direction
        elapsed += duration
        start = end
    last = _SEGMENTS[-1]
    return last, _SEGMENTS[-2]["point"], last["point"], last["duration"], previous_dir


def _footwork_offset(segment_index, segment, local_t, progress, tangent, normal, path_length):
    stride_factor = min(1.0, path_length / 520.0)
    sway_amplitude = segment["sway"] * (0.35 + 0.65 * stride_factor)
    pace = segment["pace"]
    envelope = math.sin(math.pi * progress) if path_length > 5.0 else 0.55 + 0.45 * math.sin(math.pi * progress)
    phase = local_t * (2.2 + pace) + segment_index * 0.83

    lateral = math.sin(phase * 2.2) * sway_amplitude * envelope
    forward = math.cos(phase * 1.35) * sway_amplitude * 0.22 * envelope

    if segment["kind"] in {"slide", "hold", "hesi"}:
        lateral *= 1.28
        forward *= 0.55
    elif segment["kind"] in {"sprint", "burst", "drive", "crash"}:
        lateral *= 0.88
        forward *= 1.12

    return (
        tangent[0] * forward + normal[0] * lateral,
        tangent[1] * forward + normal[1] * lateral,
    )


def _interpolate_segment(segment_index, segment, start, end, local_t, fallback_dir):
    duration = segment["duration"]
    progress = _clamp01(local_t / max(duration, 1e-6))
    shaped = _segment_progress(segment["kind"], progress)

    base_x = _lerp(start[0], end[0], shaped)
    base_y = _lerp(start[1], end[1], shaped)
    tangent, normal = _unit_and_normal(start, end, fallback_dir)
    path_length = _distance(start, end)

    curve_amount = segment["curve"] * math.sin(math.pi * shaped)
    curve_x = normal[0] * curve_amount
    curve_y = normal[1] * curve_amount

    foot_x, foot_y = _footwork_offset(segment_index, segment, local_t, shaped, tangent, normal, path_length)
    return base_x + curve_x + foot_x, base_y + curve_y + foot_y


def compute_player_xy(t, center_x, center_y):
    local_t = t % _CYCLE_DURATION
    elapsed = 0.0
    start = _START_POINT
    previous_dir = (1.0, 0.0)

    for segment_index, segment in enumerate(_SEGMENTS):
        duration = segment["duration"]
        end = segment["point"]
        if local_t <= elapsed + duration:
            x, y = _interpolate_segment(segment_index, segment, start, end, local_t - elapsed, previous_dir)
            return center_x + x, center_y + y
        previous_dir, _ = _unit_and_normal(start, end, previous_dir)
        elapsed += duration
        start = end

    last = _SEGMENTS[-1]
    x, y = _interpolate_segment(len(_SEGMENTS) - 1, last, _SEGMENTS[-2]["point"], last["point"], last["duration"], previous_dir)
    return center_x + x, center_y + y


def _jump_profile(progress):
    progress = _clamp01(progress)
    if progress < 0.38:
        rise = progress / 0.38
        return rise * rise
    fall = (progress - 0.38) / 0.62
    return max(0.0, 1.0 - fall * fall)


def compute_auto_jump_cm(t):
    local_t = t % _CYCLE_DURATION
    jump_extra = 0.0
    for start_t, duration, amplitude in _JUMP_WINDOWS:
        if start_t <= local_t <= start_t + duration:
            p = (local_t - start_t) / max(duration, 1e-6)
            jump_extra = max(jump_extra, _jump_profile(p) * amplitude)
    return jump_extra


def trigger_manual_jump(state, jump_height_cm, jump_duration_s):
    state["jump"]["active"] = True
    state["jump"]["elapsed"] = 0.0
    state["jump"]["duration"] = jump_duration_s
    state["jump"]["height"] = jump_height_cm


def compute_manual_jump_cm(state, dt):
    jump = state["jump"]
    if not jump["active"]:
        return 0.0
    jump["elapsed"] += dt
    p = jump["elapsed"] / max(jump["duration"], 1e-6)
    if p >= 1.0:
        jump["active"] = False
        return 0.0
    return _jump_profile(p) * jump["height"]
