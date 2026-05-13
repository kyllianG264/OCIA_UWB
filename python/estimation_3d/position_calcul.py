import math

from scene import CENTER_X, CENTER_Y, TAG_BASE_Z_CM, WORLD_MAX_X, WORLD_MAX_Y, WORLD_MIN_X, WORLD_MIN_Y


def clamp(v, a, b):
    return max(a, min(b, v))


def lerp(a, b, t):
    return a + (b - a) * t


def vec_sub(a, b):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def length(v):
    return math.sqrt(dot(v, v))


def distance_3d(a, b):
    return length(vec_sub(a, b))


def distance_xy(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def smooth_value(old_val, new_val, alpha):
    if old_val is None:
        return new_val
    return alpha * new_val + (1.0 - alpha) * old_val


def smooth_point3(old_point, new_point, alpha):
    if new_point is None:
        return old_point
    if old_point is None:
        return new_point
    return (
        alpha * new_point[0] + (1.0 - alpha) * old_point[0],
        alpha * new_point[1] + (1.0 - alpha) * old_point[1],
        alpha * new_point[2] + (1.0 - alpha) * old_point[2],
    )


def error_to_precision_percent(err_cm, tolerance_cm):
    return clamp(100.0 * (1.0 - err_cm / tolerance_cm), 0.0, 100.0)


def filter_delay_seconds(alpha, dt):
    if alpha <= 0.0 or dt <= 0.0:
        return 0.0
    return max(0.0, -dt / math.log(max(1e-6, 1.0 - alpha)))


def get_delayed_real_position(history, target_time):
    if not history:
        return None
    for i in range(len(history) - 1):
        t0, p0 = history[i]
        t1, p1 = history[i + 1]
        if t0 <= target_time <= t1 and t1 > t0:
            u = (target_time - t0) / (t1 - t0)
            return (lerp(p0[0], p1[0], u), lerp(p0[1], p1[1], u), lerp(p0[2], p1[2], u))
    return min(history, key=lambda item: abs(item[0] - target_time))[1]


def estimate_best_delay_seconds(history, tag_estimate, current_time, max_window=1.2):
    if len(history) < 2 or tag_estimate is None:
        return None, None
    best_time = None
    best_error = None
    for sample_time, sample_pos in reversed(history):
        delay = current_time - sample_time
        if delay < 0.0 or delay > max_window:
            continue
        err = distance_3d(sample_pos, tag_estimate)
        if best_error is None or err < best_error:
            best_time = sample_time
            best_error = err
    if best_time is None:
        return None, None
    return current_time - best_time, best_error


def det3(m00, m01, m02, m10, m11, m12, m20, m21, m22):
    return m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20)


def solve_linear_3x3(M, V):
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = M
    v0, v1, v2 = V
    detM = det3(m00, m01, m02, m10, m11, m12, m20, m21, m22)
    if abs(detM) < 1e-9:
        return None
    dx = det3(v0, m01, m02, v1, m11, m12, v2, m21, m22)
    dy = det3(m00, v0, m02, m10, v1, m12, m20, v2, m22)
    dz = det3(m00, m01, v0, m10, m11, v1, m20, m21, v2)
    return dx / detM, dy / detM, dz / detM


def _residual_and_jacobian(point, anchor_positions, distances):
    rows = []
    residuals = []
    px, py, pz = point
    for aid in sorted(anchor_positions.keys()):
        if aid not in distances:
            continue
        ax, ay, az = anchor_positions[aid]
        dx = px - ax
        dy = py - ay
        dz = pz - az
        predicted = math.sqrt(max(dx * dx + dy * dy + dz * dz, 1e-9))
        residuals.append(predicted - distances[aid])
        inv = 1.0 / max(predicted, 1e-6)
        rows.append((dx * inv, dy * inv, dz * inv))
    return rows, residuals


def estimate_position_3d_least_squares(anchor_positions, distances, initial_guess=None, iterations=18, damping=35.0):
    ids = sorted(anchor_positions.keys())
    if len(ids) < 4:
        return None
    for aid in ids:
        if aid not in distances:
            return None

    if initial_guess is None:
        point = [CENTER_X, CENTER_Y, TAG_BASE_Z_CM]
    else:
        point = [float(initial_guess[0]), float(initial_guess[1]), float(initial_guess[2])]

    best_point = tuple(point)
    best_error = None
    for _ in range(iterations):
        rows, residuals = _residual_and_jacobian(point, anchor_positions, distances)
        if len(rows) < 4:
            return None
        s_aa = sum(a * a for a, b, c in rows) + damping
        s_ab = sum(a * b for a, b, c in rows)
        s_ac = sum(a * c for a, b, c in rows)
        s_bb = sum(b * b for a, b, c in rows) + damping
        s_bc = sum(b * c for a, b, c in rows)
        s_cc = sum(c * c for a, b, c in rows) + damping
        s_ar = -sum(a * r for (a, b, c), r in zip(rows, residuals))
        s_br = -sum(b * r for (a, b, c), r in zip(rows, residuals))
        s_cr = -sum(c * r for (a, b, c), r in zip(rows, residuals))
        step = solve_linear_3x3(((s_aa, s_ab, s_ac), (s_ab, s_bb, s_bc), (s_ac, s_bc, s_cc)), (s_ar, s_br, s_cr))
        if step is None:
            break
        point[0] += step[0]
        point[1] += step[1]
        point[2] += step[2]
        point[0] = clamp(point[0], WORLD_MIN_X, WORLD_MAX_X)
        point[1] = clamp(point[1], WORLD_MIN_Y, WORLD_MAX_Y)
        point[2] = clamp(point[2], 0.0, 580.0)
        mean_abs_error = sum(abs(r) for r in residuals) / max(len(residuals), 1)
        if best_error is None or mean_abs_error < best_error:
            best_error = mean_abs_error
            best_point = tuple(point)
        if length(step) < 0.01:
            break
    return best_point


def update_position_solution(anchors, distance_packet, state, dt, tag_real):
    settings = state["settings"]
    raw_distances = dict(distance_packet.get("raw", {}))
    active_ids = sorted(anchors.keys())
    for aid in active_ids:
        if aid in raw_distances:
            state["dist_smooth"][aid] = smooth_value(state["dist_smooth"].get(aid), raw_distances[aid], settings["alpha_dist"])
    dist_smooth = {aid: value for aid, value in state["dist_smooth"].items() if value is not None}
    initial_guess = state.get("tag_est_smooth") or (CENTER_X, CENTER_Y, TAG_BASE_Z_CM)
    tag_est_raw = estimate_position_3d_least_squares(anchors, raw_distances, initial_guess=initial_guess) if len(active_ids) >= 4 else None
    tag_est_from_smooth = estimate_position_3d_least_squares(anchors, dist_smooth, initial_guess=initial_guess) if len(active_ids) >= 4 else None
    state["tag_est_smooth"] = smooth_point3(state["tag_est_smooth"], tag_est_from_smooth, settings["alpha_pos"])
    tag_est_smooth = state["tag_est_smooth"]
    delay_dist = filter_delay_seconds(settings["alpha_dist"], dt)
    delay_pos = filter_delay_seconds(settings["alpha_pos"], dt)
    delay_total = delay_dist + delay_pos
    state["delay_dist_s"] = delay_dist
    state["delay_pos_s"] = delay_pos
    delayed_real = get_delayed_real_position(state["real_history"], state["t"] - delay_total)
    result = {
        "dist_raw": raw_distances,
        "dist_smooth": dist_smooth,
        "tag_est_raw": tag_est_raw,
        "tag_est_smooth": tag_est_smooth,
        "delayed_real": delayed_real,
        "delay_dist_s": delay_dist,
        "delay_pos_s": delay_pos,
        "delay_total_s": delay_total,
        "valid": distance_packet.get("valid", False),
        "source": distance_packet.get("source", "simulation"),
        "status": distance_packet.get("status", "simulation"),
        "err3d": None,
        "errxy": None,
        "errz": None,
        "precision": None,
        "avg_err3d": None,
        "avg_errxy": None,
        "avg_errz": None,
        "avg_precision": None,
        "avg_delay_est": None,
        "avg_measured_delay": None,
        "max_measured_delay": None,
        "traj_err_xy": None,
        "traj_err_3d": None,
        "traj_err_z": None,
        "traj_precision": None,
        "avg_traj_err_xy": None,
        "avg_traj_err_3d": None,
        "avg_traj_precision": None,
        "path_ratio": None,
    }
    if tag_real is None or tag_est_raw is None or tag_est_smooth is None:
        return result
    err3d = distance_3d(tag_real, tag_est_smooth)
    errxy = distance_xy(tag_real, tag_est_smooth)
    errz = abs(tag_real[2] - tag_est_smooth[2])
    precision = error_to_precision_percent(err3d, settings["precision_tolerance_cm"])
    delayed_target = delayed_real or tag_real
    measured_delay, _ = estimate_best_delay_seconds(state["real_history"], tag_est_smooth, state["t"])
    traj_err_3d = distance_3d(delayed_target, tag_est_smooth)
    traj_err_xy = distance_xy(delayed_target, tag_est_smooth)
    traj_err_z = abs(delayed_target[2] - tag_est_smooth[2])
    traj_precision = error_to_precision_percent(traj_err_3d, settings["precision_tolerance_cm"])
    st = state["stats"]
    st["err3d_sum"] += err3d
    st["errxy_sum"] += errxy
    st["errz_sum"] += errz
    st["precision_sum"] += precision
    st["delay_est_sum"] += delay_total
    st["delay_measured_sum"] += measured_delay or 0.0
    st["traj_err3d_sum"] += traj_err_3d
    st["traj_errxy_sum"] += traj_err_xy
    st["traj_errz_sum"] += traj_err_z
    st["traj_precision_sum"] += traj_precision
    st["count"] += 1
    st["max_measured_delay"] = max(st["max_measured_delay"], measured_delay or 0.0)
    avg_err3d = st["err3d_sum"] / st["count"]
    avg_errxy = st["errxy_sum"] / st["count"]
    avg_errz = st["errz_sum"] / st["count"]
    avg_precision = st["precision_sum"] / st["count"]
    avg_delay_est = st["delay_est_sum"] / st["count"]
    avg_measured_delay = st["delay_measured_sum"] / st["count"]
    avg_traj_err_xy = st["traj_errxy_sum"] / st["count"]
    avg_traj_err_3d = st["traj_err3d_sum"] / st["count"]
    avg_traj_precision = st["traj_precision_sum"] / st["count"]
    path_ratio = None if avg_traj_err_3d == 0.0 else avg_err3d / avg_traj_err_3d
    result.update(
        err3d=err3d,
        errxy=errxy,
        errz=errz,
        precision=precision,
        avg_err3d=avg_err3d,
        avg_errxy=avg_errxy,
        avg_errz=avg_errz,
        avg_precision=avg_precision,
        avg_delay_est=avg_delay_est,
        avg_measured_delay=avg_measured_delay,
        max_measured_delay=st["max_measured_delay"],
        traj_err_xy=traj_err_xy,
        traj_err_3d=traj_err_3d,
        traj_err_z=traj_err_z,
        traj_precision=traj_precision,
        avg_traj_err_xy=avg_traj_err_xy,
        avg_traj_err_3d=avg_traj_err_3d,
        avg_traj_precision=avg_traj_precision,
        path_ratio=path_ratio,
    )
    return result
