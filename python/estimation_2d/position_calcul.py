import math

REFERENCE_DT = 1.0 / 60.0


def clamp(v, a, b):
    return max(a, min(b, v))


def distance_2d(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def smooth_value(old_val, new_val, alpha):
    if old_val is None:
        return new_val
    return alpha * new_val + (1.0 - alpha) * old_val


def smooth_point(old_point, new_point, alpha):
    if new_point is None:
        return old_point
    if old_point is None:
        return new_point
    return (
        alpha * new_point[0] + (1.0 - alpha) * old_point[0],
        alpha * new_point[1] + (1.0 - alpha) * old_point[1],
    )


def alpha_for_dt(alpha_reference, dt):
    alpha_reference = clamp(alpha_reference, 0.0, 0.999999)
    if dt <= 0.0:
        return alpha_reference
    return 1.0 - (1.0 - alpha_reference) ** (dt / REFERENCE_DT)


def error_to_precision_percent(err_cm, tolerance_cm):
    if err_cm is None:
        return None
    return clamp(100.0 * (1.0 - err_cm / tolerance_cm), 0.0, 100.0)


def circle_intersections(x0, y0, r0, x1, y1, r1):
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy)
    if d > r0 + r1 or d < abs(r0 - r1) or (d == 0 and r0 == r1):
        return []

    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h_sq = r0 * r0 - a * a
    h = math.sqrt(max(0.0, h_sq))

    xm = x0 + a * dx / d
    ym = y0 + a * dy / d
    rx = -dy * h / d
    ry = dx * h / d
    return [(xm + rx, ym + ry), (xm - rx, ym - ry)]


def estimate_position_least_squares(anchor_positions, distances):
    ids = sorted(anchor_positions.keys())
    if len(ids) < 3 or any(aid not in distances for aid in ids):
        return None

    ref_id = ids[0]
    x1, y1 = anchor_positions[ref_id]
    r1 = distances[ref_id]
    rows = []
    vals = []
    for aid in ids[1:]:
        xi, yi = anchor_positions[aid]
        ri = distances[aid]
        a = 2.0 * (xi - x1)
        b = 2.0 * (yi - y1)
        c = r1 * r1 - ri * ri - x1 * x1 + xi * xi - y1 * y1 + yi * yi
        rows.append((a, b))
        vals.append(c)

    s_aa = sum(a * a for a, b in rows)
    s_ab = sum(a * b for a, b in rows)
    s_bb = sum(b * b for a, b in rows)
    s_ac = sum(c * a for (a, b), c in zip(rows, vals))
    s_bc = sum(c * b for (a, b), c in zip(rows, vals))
    det = s_aa * s_bb - s_ab * s_ab
    if abs(det) < 1e-9:
        return None
    x = (s_ac * s_bb - s_bc * s_ab) / det
    y = (s_aa * s_bc - s_ab * s_ac) / det
    return x, y


def _base_result(raw_distances, dist_smooth, distance_packet):
    return {
        "dist_raw": dict(raw_distances),
        "dist_smooth": dict(dist_smooth),
        "raw_intersections": [],
        "smooth_intersections": [],
        "tag_est_raw": None,
        "tag_est_smooth": None,
        "valid": distance_packet.get("valid", False),
        "source": distance_packet.get("source", "simulation"),
        "status": distance_packet.get("status", "simulation"),
        "raw_err": None,
        "smooth_err": None,
        "raw_precision": None,
        "smooth_precision": None,
        "avg_raw": None,
        "avg_smooth": None,
        "avg_raw_precision": None,
        "avg_smooth_precision": None,
    }


def update_position_solution(anchors, distance_packet, state, dt, tag_real=None):
    settings = state["settings"]
    active_ids = sorted(anchors.keys())
    raw_distances = dict(distance_packet.get("raw", {}))
    alpha_dist = alpha_for_dt(settings["alpha_dist"], dt)
    alpha_pos = alpha_for_dt(settings["alpha_pos"], dt)

    for aid in active_ids:
        if aid in raw_distances:
            state["dist_smooth"][aid] = smooth_value(
                state["dist_smooth"].get(aid),
                raw_distances[aid],
                alpha_dist,
            )

    dist_smooth = {aid: value for aid, value in state["dist_smooth"].items() if value is not None}
    result = _base_result(raw_distances, dist_smooth, distance_packet)

    if len(active_ids) == 2 and all(aid in raw_distances for aid in active_ids):
        a1, a2 = active_ids
        result["raw_intersections"] = circle_intersections(
            anchors[a1][0],
            anchors[a1][1],
            raw_distances[a1],
            anchors[a2][0],
            anchors[a2][1],
            raw_distances[a2],
        )
    if len(active_ids) == 2 and all(aid in dist_smooth for aid in active_ids):
        a1, a2 = active_ids
        result["smooth_intersections"] = circle_intersections(
            anchors[a1][0],
            anchors[a1][1],
            dist_smooth[a1],
            anchors[a2][0],
            anchors[a2][1],
            dist_smooth[a2],
        )

    tag_est_raw = estimate_position_least_squares(anchors, raw_distances)
    tag_est_from_smooth = estimate_position_least_squares(anchors, dist_smooth)
    state["tag_est_smooth"] = smooth_point(state["tag_est_smooth"], tag_est_from_smooth, alpha_pos)
    tag_est_smooth = state["tag_est_smooth"]

    result["tag_est_raw"] = tag_est_raw
    result["tag_est_smooth"] = tag_est_smooth

    if tag_real is None or len(active_ids) < 3 or tag_est_raw is None or tag_est_smooth is None:
        return result

    raw_err = distance_2d(tag_real, tag_est_raw)
    smooth_err = distance_2d(tag_real, tag_est_smooth)
    raw_precision = error_to_precision_percent(raw_err, settings["precision_tolerance_cm"])
    smooth_precision = error_to_precision_percent(smooth_err, settings["precision_tolerance_cm"])

    st = state["stats"]
    st["raw_sum"] += raw_err
    st["smooth_sum"] += smooth_err
    st["raw_precision_sum"] += raw_precision
    st["smooth_precision_sum"] += smooth_precision
    st["count"] += 1
    st["raw_max"] = max(st["raw_max"], raw_err)
    st["smooth_max"] = max(st["smooth_max"], smooth_err)

    result.update(
        raw_err=raw_err,
        smooth_err=smooth_err,
        raw_precision=raw_precision,
        smooth_precision=smooth_precision,
        avg_raw=st["raw_sum"] / st["count"],
        avg_smooth=st["smooth_sum"] / st["count"],
        avg_raw_precision=st["raw_precision_sum"] / st["count"],
        avg_smooth_precision=st["smooth_precision_sum"] / st["count"],
    )
    return result
