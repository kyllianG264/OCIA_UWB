from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenProjection:
    left: float
    top: float
    scale: float
    offset_x: float
    offset_y: float


def centered_screen_point(center_x, center_y, scale, width, height, x_cm, y_cm):
    sx = width // 2 + int((x_cm - center_x) * scale)
    sy = height // 2 + int((y_cm - center_y) * scale)
    return sx, sy


def fit_bounds_to_rect(bounds, target_rect):
    left, right, top, bottom = bounds
    width = max(float(right) - float(left), 1.0)
    height = max(float(bottom) - float(top), 1.0)
    scale = min(target_rect.width / width, target_rect.height / height)
    draw_w = width * scale
    draw_h = height * scale
    offset_x = target_rect.x + (target_rect.width - draw_w) / 2.0
    offset_y = target_rect.y + (target_rect.height - draw_h) / 2.0
    return ScreenProjection(
        left=float(left),
        top=float(top),
        scale=float(scale),
        offset_x=float(offset_x),
        offset_y=float(offset_y),
    )


def project_world_point(projection, x_cm, y_cm):
    sx = int(projection.offset_x + (float(x_cm) - projection.left) * projection.scale)
    sy = int(projection.offset_y + (float(y_cm) - projection.top) * projection.scale)
    return sx, sy
