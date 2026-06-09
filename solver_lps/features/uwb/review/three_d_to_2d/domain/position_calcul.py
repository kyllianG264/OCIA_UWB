from solver_lps.features.uwb.review.two_d.domain.position_calcul import (
    circle_intersections,
    clamp,
    distance_2d,
    error_to_precision_percent,
    estimate_position_least_squares,
    smooth_point,
    smooth_value,
    update_position_solution as update_position_solution_2d,
)


def update_position_solution(anchors, distance_packet, state, dt, tag_real):
    result = update_position_solution_2d(anchors, distance_packet, state, dt, tag_real)
    result["raw_3d"] = distance_packet.get("raw_3d", {})
    result["tag_real"] = tag_real
    return result
