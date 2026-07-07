"""
Logique pure d'overlay pour les rendus video de tracking.
"""

from __future__ import annotations

import cv2


def draw_tracking_overlay(frame_bgr, items, color):
    if not items:
        return frame_bgr
    annotated = frame_bgr.copy()
    for item in items:
        x1 = int(item["x1"])
        y1 = int(item["y1"])
        x2 = int(item["x2"])
        y2 = int(item["y2"])
        label = f"#{item['track_id']}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, ((x1 + x2) // 2, y2), 5, color, -1)
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - 26)),
            (x1 + 80, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 6, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (15, 20, 25),
            2,
            cv2.LINE_AA,
        )
    return annotated
