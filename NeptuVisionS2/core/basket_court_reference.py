from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


COURT_LENGTH_CM = 2800.0
COURT_WIDTH_CM = 1500.0
COURT_HALF_LENGTH_CM = COURT_LENGTH_CM / 2.0
COURT_HALF_WIDTH_CM = COURT_WIDTH_CM / 2.0
MIDCOURT_X_CM = COURT_HALF_LENGTH_CM

COURT_BG = (16, 20, 24)
COURT_APRON = (64, 82, 68)
COURT_WOOD = (196, 140, 86)
COURT_WOOD_ALT = (206, 150, 96)
COURT_LINES = (248, 245, 236)


@dataclass(frozen=True)
class CourtReferenceLayout:
    image_width: int
    image_height: int
    court_left_px: int
    court_top_px: int
    court_right_px: int
    court_bottom_px: int
    scale_px_per_cm: float


def build_reference_layout(image_width: int = 1600, image_height: int = 900, padding_px: int = 70) -> CourtReferenceLayout:
    usable_width = max(1, image_width - (padding_px * 2))
    usable_height = max(1, image_height - (padding_px * 2))
    scale = min(usable_width / COURT_LENGTH_CM, usable_height / COURT_WIDTH_CM)
    draw_width = int(round(COURT_LENGTH_CM * scale))
    draw_height = int(round(COURT_WIDTH_CM * scale))
    left = (image_width - draw_width) // 2
    top = (image_height - draw_height) // 2
    return CourtReferenceLayout(
        image_width=image_width,
        image_height=image_height,
        court_left_px=left,
        court_top_px=top,
        court_right_px=left + draw_width,
        court_bottom_px=top + draw_height,
        scale_px_per_cm=scale,
    )


def court_cm_to_pixel(x_cm: float, y_cm: float, layout: CourtReferenceLayout) -> tuple[int, int]:
    px = int(round(layout.court_left_px + (float(x_cm) * layout.scale_px_per_cm)))
    py = int(round(layout.court_top_px + (float(y_cm) * layout.scale_px_per_cm)))
    return px, py


def pixel_to_court_cm(px: int, py: int, layout: CourtReferenceLayout) -> tuple[float, float]:
    x_cm = (float(px) - layout.court_left_px) / layout.scale_px_per_cm
    y_cm = (float(py) - layout.court_top_px) / layout.scale_px_per_cm
    x_cm = max(0.0, min(COURT_LENGTH_CM, x_cm))
    y_cm = max(0.0, min(COURT_WIDTH_CM, y_cm))
    return x_cm, y_cm


def render_reference_court(layout: CourtReferenceLayout) -> np.ndarray:
    image = np.full((layout.image_height, layout.image_width, 3), COURT_BG, dtype=np.uint8)
    cv2.rectangle(
        image,
        (max(0, layout.court_left_px - 36), max(0, layout.court_top_px - 36)),
        (min(layout.image_width - 1, layout.court_right_px + 36), min(layout.image_height - 1, layout.court_bottom_px + 36)),
        COURT_APRON,
        thickness=-1,
    )
    cv2.rectangle(
        image,
        (layout.court_left_px, layout.court_top_px),
        (layout.court_right_px, layout.court_bottom_px),
        COURT_WOOD,
        thickness=-1,
    )

    stripe_count = 12
    stripe_width_cm = COURT_LENGTH_CM / stripe_count
    for index in range(stripe_count):
        if index % 2 != 0:
            continue
        x0 = index * stripe_width_cm
        x1 = x0 + stripe_width_cm
        px0, py0 = court_cm_to_pixel(x0, 0.0, layout)
        px1, py1 = court_cm_to_pixel(x1, COURT_WIDTH_CM, layout)
        cv2.rectangle(image, (px0, py0), (px1, py1), COURT_WOOD_ALT, thickness=-1)

    left, top = court_cm_to_pixel(0.0, 0.0, layout)
    right, bottom = court_cm_to_pixel(COURT_LENGTH_CM, COURT_WIDTH_CM, layout)
    cv2.rectangle(image, (left, top), (right, bottom), COURT_LINES, thickness=3)

    mid_x, _ = court_cm_to_pixel(MIDCOURT_X_CM, 0.0, layout)
    cv2.line(image, (mid_x, top), (mid_x, bottom), COURT_LINES, thickness=3)

    circle_center = court_cm_to_pixel(COURT_HALF_LENGTH_CM, COURT_HALF_WIDTH_CM, layout)
    circle_radius = int(round(180.0 * layout.scale_px_per_cm))
    cv2.circle(image, circle_center, circle_radius, COURT_LINES, thickness=3)

    key_width_cm = 490.0
    key_length_cm = 580.0
    key_top_cm = (COURT_WIDTH_CM - key_width_cm) / 2.0
    key_bottom_cm = key_top_cm + key_width_cm
    left_key = np.array(
        [
            court_cm_to_pixel(0.0, key_top_cm, layout),
            court_cm_to_pixel(key_length_cm, key_top_cm, layout),
            court_cm_to_pixel(key_length_cm, key_bottom_cm, layout),
            court_cm_to_pixel(0.0, key_bottom_cm, layout),
        ],
        dtype=np.int32,
    )
    right_key = np.array(
        [
            court_cm_to_pixel(COURT_LENGTH_CM - key_length_cm, key_top_cm, layout),
            court_cm_to_pixel(COURT_LENGTH_CM, key_top_cm, layout),
            court_cm_to_pixel(COURT_LENGTH_CM, key_bottom_cm, layout),
            court_cm_to_pixel(COURT_LENGTH_CM - key_length_cm, key_bottom_cm, layout),
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [left_key], isClosed=True, color=COURT_LINES, thickness=3)
    cv2.polylines(image, [right_key], isClosed=True, color=COURT_LINES, thickness=3)

    return image
