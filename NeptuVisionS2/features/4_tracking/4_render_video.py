"""
Genere une video composite avec les deux cameras et le terrain annote.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.paths import DEFAULT_TERRAIN_IMAGE
from core.utils import configure_utf8_stdout, hms_to_s, imread_unicode, require_file, sync_file_to_python_cv_logs

PREV_W = 640
PREV_H = 360
MAP_H = PREV_H * 2
OUT_W = PREV_W + MAP_H
OUT_H = MAP_H
CAMERA_COLORS = {
    "gauche": (50, 200, 50),
    "droite": (50, 100, 220),
}


def load_detections(path: str):
    detections = {}
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            detections[(int(row["frame"]), row["cam"].strip().lower(), str(row["detection_id"]))] = {
                "x1": int(float(row["x1"])),
                "y1": int(float(row["y1"])),
                "x2": int(float(row["x2"])),
                "y2": int(float(row["y2"])),
            }
    return detections


def load_assignments(path: str):
    by_frame = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            by_frame[int(row["frame"])].append(
                {
                    "camera": row["cam"].strip().lower(),
                    "track_id": str(row["track_id"]),
                    "detection_id": str(row["detection_id"]),
                    "x": int(float(row["X"])),
                    "y": int(float(row["Y"])),
                    "on_terrain": str(row.get("on_terrain", "1")).strip() not in ("0", "false", "False", ""),
                }
            )
    return by_frame


def draw_box(frame, box: dict, label: str, color, on_terrain: bool):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    suffix = "" if on_terrain else "*"
    full_label = f"{label}{suffix}"
    text_w = max(48, 12 * len(full_label))
    top = max(0, y1 - 28)
    cv2.rectangle(frame, (x1, top), (x1 + text_w, y1), color, -1)
    cv2.putText(frame, full_label, (x1 + 4, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_player(terrain, px: int, py: int, track_id: str, color, on_terrain: bool):
    ring_color = (255, 255, 255) if on_terrain else (180, 180, 180)
    fill_color = color if on_terrain else (90, 90, 90)
    cv2.circle(terrain, (px, py), 22, ring_color, -1)
    cv2.circle(terrain, (px, py), 19, fill_color, -1)
    size = cv2.getTextSize(track_id, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    cv2.putText(terrain, track_id, (px - size[0] // 2, py + size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def open_capture(path: str, start_s: float):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Flux inaccessible : {path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    return capture


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Rendu video composite du tracking")
    parser.add_argument("--video_gauche", required=True)
    parser.add_argument("--video_droite", required=True)
    parser.add_argument("--detections_csv", required=True)
    parser.add_argument("--assignments_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--terrain", default=str(DEFAULT_TERRAIN_IMAGE))
    parser.add_argument("--start", default="00:00:00")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    require_file(args.video_gauche, "video gauche")
    require_file(args.video_droite, "video droite")
    require_file(args.detections_csv, "detections CSV")
    require_file(args.assignments_csv, "assignments CSV")

    terrain_base = imread_unicode(args.terrain)
    if terrain_base is None:
        raise FileNotFoundError(f"Image terrain introuvable : {args.terrain}")

    detections = load_detections(args.detections_csv)
    assignments_by_frame = load_assignments(args.assignments_csv)
    start_s = hms_to_s(args.start)
    end_s = hms_to_s(args.end) if args.end else None

    capture_g = open_capture(args.video_gauche, start_s)
    capture_d = open_capture(args.video_droite, start_s)
    fps = capture_g.get(cv2.CAP_PROP_FPS) or 25.0

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))
    if not writer.isOpened():
        raise RuntimeError(f"Echec creation video : {args.output}")

    frame_idx = 0
    try:
        while True:
            ret_g, frame_g = capture_g.read()
            ret_d, frame_d = capture_d.read()
            if not ret_g or not ret_d:
                break

            timestamp_s = start_s + frame_idx / fps
            if end_s is not None and timestamp_s > end_s:
                break

            terrain = terrain_base.copy()
            for item in assignments_by_frame.get(frame_idx, []):
                camera = item["camera"]
                color = CAMERA_COLORS.get(camera, (200, 200, 200))
                box = detections.get((frame_idx, camera, item["detection_id"]))
                if box is not None:
                    frame = frame_g if camera == "gauche" else frame_d
                    draw_box(frame, box, item["track_id"], color, item["on_terrain"])
                draw_player(terrain, item["x"], item["y"], item["track_id"], color, item["on_terrain"])

            small_g = cv2.resize(frame_g, (PREV_W, PREV_H))
            small_d = cv2.resize(frame_d, (PREV_W, PREV_H))
            cameras = cv2.vconcat([small_g, small_d])
            terrain_resized = cv2.resize(terrain, (MAP_H, MAP_H))
            composite = cv2.hconcat([cameras, terrain_resized])
            cv2.putText(composite, f"Frame: {frame_idx}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2, cv2.LINE_AA)
            writer.write(composite)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx:6d} | t={timestamp_s:7.1f}s")
    finally:
        capture_g.release()
        capture_d.release()
        writer.release()

    print("=" * 60)
    print(f"Video composite exportee : {args.output}")
    mirrored_video_path = sync_file_to_python_cv_logs(args.output, "tracking_composite.mp4")
    if mirrored_video_path:
        print(f"Video composite copiee vers python/CV Logs : {mirrored_video_path}")


if __name__ == "__main__":
    main()
