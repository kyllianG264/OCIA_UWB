import math


def clamp(v, a, b):
    return max(a, min(b, v))


def lerp(a, b, t):
    return a + (b - a) * t


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def length(v):
    return math.sqrt(dot(v, v))


def distance_3d(a, b):
    return length(vec_sub(b, a))


def distance_xy(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def smooth_value(old_val, new_val, alpha):
    if old_val is None:
        return new_val
    return alpha * new_val + (1 - alpha) * old_val


def smooth_point3(old_point, new_point, alpha):
    if new_point is None:
        return old_point
    if old_point is None:
        return new_point
    return (
        alpha * new_point[0] + (1 - alpha) * old_point[0],
        alpha * new_point[1] + (1 - alpha) * old_point[1],
        alpha * new_point[2] + (1 - alpha) * old_point[2],
    )


def error_to_precision_percent(err_cm, tolerance_cm):
    if err_cm is None:
        return None
    p = 100.0 * (1.0 - err_cm / tolerance_cm)
    return clamp(p, 0.0, 100.0)


def filter_delay_seconds(alpha, dt):
    if alpha >= 1.0:
        return 0.0
    if alpha <= 0.0:
        return 1e9
    return -dt / math.log(1.0 - alpha)


def get_delayed_real_position(history, target_time):
    if len(history) == 0:
        return None

    if target_time <= history[0][0]:
        return history[0][1]

    if target_time >= history[-1][0]:
        return history[-1][1]

    for i in range(len(history) - 1, 0, -1):
        t0, p0 = history[i - 1]
        t1, p1 = history[i]
        if t0 <= target_time <= t1:
            if abs(t1 - t0) < 1e-9:
                return p0
            u = (target_time - t0) / (t1 - t0)
            return (
                lerp(p0[0], p1[0], u),
                lerp(p0[1], p1[1], u),
                lerp(p0[2], p1[2], u),
            )
    return history[0][1]


def estimate_best_delay_seconds(history, tag_estimate, current_time, max_window=2.5):
    if len(history) == 0 or tag_estimate is None:
        return None

    best_time = None
    best_error = None
    for sample_time, sample_pos in reversed(history):
        delay = current_time - sample_time
        if delay < 0.0:
            continue
        if delay > max_window:
            break
        err = distance_3d(sample_pos, tag_estimate)
        if best_error is None or err < best_error:
            best_error = err
            best_time = sample_time

    if best_time is None:
        return None
    return current_time - best_time


def det3(m00, m01, m02, m10, m11, m12, m20, m21, m22):
    return (
        m00 * (m11 * m22 - m12 * m21)
        - m01 * (m10 * m22 - m12 * m20)
        + m02 * (m10 * m21 - m11 * m20)
    )


def solve_linear_3x3(M, V):
    m00, m01, m02 = M[0]
    m10, m11, m12 = M[1]
    m20, m21, m22 = M[2]
    v0, v1, v2 = V

    detM = det3(m00, m01, m02, m10, m11, m12, m20, m21, m22)
    if abs(detM) < 1e-9:
        return None

    dx = det3(v0, m01, m02, v1, m11, m12, v2, m21, m22) / detM
    dy = det3(m00, v0, m02, m10, v1, m12, m20, v2, m22) / detM
    dz = det3(m00, m01, v0, m10, m11, v1, m20, m21, v2) / detM
    return (dx, dy, dz)


def estimate_position_3d_least_squares(anchor_positions, distances):
    ids = sorted(anchor_positions.keys())
    if len(ids) < 4:
        return None

    ref_id = ids[0]
    x1, y1, z1 = anchor_positions[ref_id]
    r1 = distances[ref_id]

    rows = []
    vals = []

    for aid in ids[1:]:
        xi, yi, zi = anchor_positions[aid]
        ri = distances[aid]

        a = 2.0 * (xi - x1)
        b = 2.0 * (yi - y1)
        c = 2.0 * (zi - z1)
        d = (
            (r1 * r1 - ri * ri)
            - (x1 * x1 - xi * xi)
            - (y1 * y1 - yi * yi)
            - (z1 * z1 - zi * zi)
        )

        rows.append((a, b, c))
        vals.append(d)

    s_aa = sum(a * a for a, b, c in rows)
    s_ab = sum(a * b for a, b, c in rows)
    s_ac = sum(a * c for a, b, c in rows)
    s_bb = sum(b * b for a, b, c in rows)
    s_bc = sum(b * c for a, b, c in rows)
    s_cc = sum(c * c for a, b, c in rows)

    s_ad = sum(a * d for (a, b, c), d in zip(rows, vals))
    s_bd = sum(b * d for (a, b, c), d in zip(rows, vals))
    s_cd = sum(c * d for (a, b, c), d in zip(rows, vals))

    M = [
        [s_aa, s_ab, s_ac],
        [s_ab, s_bb, s_bc],
        [s_ac, s_bc, s_cc],
    ]
    V = [s_ad, s_bd, s_cd]

    return solve_linear_3x3(M, V)


def update_position_solution(anchors, distance_packet, state, dt, tag_real=None):
    settings = state["settings"]
    alpha_dist = settings["alpha_dist"]
    alpha_pos = settings["alpha_pos"]
    precision_tolerance_cm = settings["precision_tolerance_cm"]

    raw_distances = distance_packet["raw"]
    active_ids = sorted(anchors.keys())

    if set(state["dist_smooth"].keys()) != set(active_ids):
        state["dist_smooth"] = {aid: None for aid in active_ids}

    dist_smooth = {}
    for aid in active_ids:
        state["dist_smooth"][aid] = smooth_value(state["dist_smooth"][aid], raw_distances[aid], alpha_dist)
        dist_smooth[aid] = state["dist_smooth"][aid]

    tag_est_raw = estimate_position_3d_least_squares(anchors, raw_distances)
    tag_est_from_smooth = estimate_position_3d_least_squares(anchors, dist_smooth)
    state["tag_est_smooth"] = smooth_point3(state["tag_est_smooth"], tag_est_from_smooth, alpha_pos)
    tag_est_smooth = state["tag_est_smooth"]

    delay_dist = filter_delay_seconds(alpha_dist, dt)
    delay_pos = filter_delay_seconds(alpha_pos, dt)
    delay_total = delay_dist + delay_pos

    result = {
        "dist_raw": raw_distances,
        "dist_smooth": dist_smooth,
        "tag_est_raw": tag_est_raw,
        "tag_est_smooth": tag_est_smooth,
        "delay_dist": delay_dist,
        "delay_pos": delay_pos,
        "delay_total": delay_total,
        "avg_delay_est": None,
        "measured_delay": None,
        "avg_measured_delay": None,
        "max_measured_delay": None,
        "err3d": None,
        "errxy": None,
        "errz": None,
        "precision": None,
        "avg_err3d": None,
        "avg_errxy": None,
        "avg_errz": None,
        "avg_precision": None,
        "delayed_real": None,
        "traj_err_xy": None,
        "traj_err_3d": None,
        "traj_err_z": None,
        "traj_precision": None,
        "avg_traj_err_xy": None,
        "avg_traj_err_3d": None,
        "avg_traj_precision": None,
        "path_ratio": None,
    }

    if tag_est_smooth is None or tag_real is None:
        return result

    st = state["stats"]

    err3d = distance_3d(tag_real, tag_est_smooth)
    errxy = distance_xy(tag_real, tag_est_smooth)
    errz = abs(tag_real[2] - tag_est_smooth[2])
    precision = error_to_precision_percent(err3d, precision_tolerance_cm)

    st["err3d_sum"] += err3d
    st["errxy_sum"] += errxy
    st["errz_sum"] += errz
    st["prec_sum"] += precision
    st["count"] += 1
    st["err3d_max"] = max(st["err3d_max"], err3d)
    st["errz_max"] = max(st["errz_max"], errz)

    delayed_real = get_delayed_real_position(state["real_history"], state["t"] - delay_total)
    measured_delay = estimate_best_delay_seconds(state["real_history"], tag_est_smooth, state["t"])

    st["delay_est_sum"] += delay_total
    if measured_delay is not None:
        st["delay_measured_sum"] += measured_delay
        st["delay_measured_count"] += 1
        st["delay_measured_max"] = max(st["delay_measured_max"], measured_delay)
    st["delay_count"] += 1

    avg_err3d = st["err3d_sum"] / st["count"]
    avg_errxy = st["errxy_sum"] / st["count"]
    avg_errz = st["errz_sum"] / st["count"]
    avg_precision = st["prec_sum"] / st["count"]
    avg_delay_est = st["delay_est_sum"] / st["delay_count"]
    avg_measured_delay = None
    max_measured_delay = None
    if st["delay_measured_count"] > 0:
        avg_measured_delay = st["delay_measured_sum"] / st["delay_measured_count"]
        max_measured_delay = st["delay_measured_max"]

    traj_err_xy = None
    traj_err_3d = None
    traj_err_z = None
    traj_precision = None
    avg_traj_err_xy = None
    avg_traj_err_3d = None
    avg_traj_precision = None

    if delayed_real is not None:
        traj_err_xy = distance_xy(delayed_real, tag_est_smooth)
        traj_err_3d = distance_3d(delayed_real, tag_est_smooth)
        traj_err_z = abs(delayed_real[2] - tag_est_smooth[2])
        traj_precision = error_to_precision_percent(traj_err_xy, precision_tolerance_cm)

        st["traj_err_xy_sum"] += traj_err_xy
        st["traj_err_3d_sum"] += traj_err_3d
        st["traj_prec_sum"] += traj_precision
        st["traj_count"] += 1
        st["traj_err_xy_max"] = max(st["traj_err_xy_max"], traj_err_xy)
        st["traj_err_3d_max"] = max(st["traj_err_3d_max"], traj_err_3d)

        avg_traj_err_xy = st["traj_err_xy_sum"] / st["traj_count"]
        avg_traj_err_3d = st["traj_err_3d_sum"] / st["traj_count"]
        avg_traj_precision = st["traj_prec_sum"] / st["traj_count"]

    if st["last_real"] is not None:
        st["real_path_len"] += distance_3d(st["last_real"], tag_real)
    if st["last_est"] is not None:
        st["est_path_len"] += distance_3d(st["last_est"], tag_est_smooth)

    st["last_real"] = tag_real
    st["last_est"] = tag_est_smooth

    path_ratio = None
    if st["real_path_len"] > 1e-6:
        path_ratio = 100.0 * st["est_path_len"] / st["real_path_len"]

    result.update({
        "avg_delay_est": avg_delay_est,
        "measured_delay": measured_delay,
        "avg_measured_delay": avg_measured_delay,
        "max_measured_delay": max_measured_delay,
        "err3d": err3d,
        "errxy": errxy,
        "errz": errz,
        "precision": precision,
        "avg_err3d": avg_err3d,
        "avg_errxy": avg_errxy,
        "avg_errz": avg_errz,
        "avg_precision": avg_precision,
        "delayed_real": delayed_real,
        "traj_err_xy": traj_err_xy,
        "traj_err_3d": traj_err_3d,
        "traj_err_z": traj_err_z,
        "traj_precision": traj_precision,
        "avg_traj_err_xy": avg_traj_err_xy,
        "avg_traj_err_3d": avg_traj_err_3d,
        "avg_traj_precision": avg_traj_precision,
        "path_ratio": path_ratio,
    })
    return result
