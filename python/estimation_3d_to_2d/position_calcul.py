import importlib.util
import os


_CORE_PATH = os.path.join(os.path.dirname(__file__), "..", "estimation_2d", "position_calcul.py")
_SPEC = importlib.util.spec_from_file_location("position_calcul_2d_core", _CORE_PATH)
_CORE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CORE)

clamp = _CORE.clamp
distance_2d = _CORE.distance_2d
smooth_value = _CORE.smooth_value
smooth_point = _CORE.smooth_point
error_to_precision_percent = _CORE.error_to_precision_percent
circle_intersections = _CORE.circle_intersections
estimate_position_least_squares = _CORE.estimate_position_least_squares


def update_position_solution(anchors, distance_packet, state, dt, tag_real):
    result = _CORE.update_position_solution(anchors, distance_packet, state, dt, tag_real)
    result["raw_3d"] = distance_packet.get("raw_3d", {})
    result["tag_real"] = tag_real
    return result
