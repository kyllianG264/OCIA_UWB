"""
Genere les videos undistorted par camera, avec option d'overlay tracking.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from solver_lps.features.cv.review.generation.raw_tracking.data.calibration_input import (
    load_calibration,
)
from solver_lps.features.cv.review.generation.raw_tracking.data.tracking_output import (
    configure_utf8_stdout,
    hms_to_s,
    require_file,
)
from solver_lps.features.cv.review.generation.raw_tracking.domain.render_overlay import (
    draw_tracking_overlay,
)
from solver_lps.features.cv.review.generation.raw_tracking.domain.video_remap import (
    remap_frame_with_distortion,
)
CAMERA_CONFIG = {
    "gauche": {
        "distortion_key": "distortion_g",
        "view_key": "undistort_view_g",
        "raw_output_name": "left_undistorted.mp4",
        "tracked_output_name": "left_undistorted_tracked.mp4",
        "color": (80, 220, 80),
    },
    "droite": {
        "distortion_key": "distortion_d",
        "view_key": "undistort_view_d",
        "raw_output_name": "right_undistorted.mp4",
        "tracked_output_name": "right_undistorted_tracked.mp4",
        "color": (60, 165, 255),
    },
}


def open_capture(path: str, start_s: float):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Flux inaccessible : {path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    return capture
def load_tracking_overlays(detections_csv: str, assignments_csv: str):
    require_file(detections_csv, "detections csv")
    require_file(assignments_csv, "tracking assignments csv")

    overlays = {}
    detections_index = {}

    with open(detections_csv, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_idx = int(row["frame"])
            camera = str(row["cam"]).strip().lower()
            detection_id = str(row["detection_id"]).strip()
            detections_index[(frame_idx, camera, detection_id)] = row

    with open(assignments_csv, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_idx = int(row["frame"])
            camera = str(row["cam"]).strip().lower()
            detection_id = str(row["detection_id"]).strip()
            detection = detections_index.get((frame_idx, camera, detection_id))
            if detection is None:
                continue
            overlays.setdefault((frame_idx, camera), []).append(
                {
                    "track_id": str(row["track_id"]).strip(),
                    "x1": int(float(detection["x1"])),
                    "y1": int(float(detection["y1"])),
                    "x2": int(float(detection["x2"])),
                    "y2": int(float(detection["y2"])),
                    "conf": float(detection.get("conf", 0.0) or 0.0),
                }
            )
    return overlays


def build_writer(path: str, fps: float, frame_bgr):
    return cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_bgr.shape[1], frame_bgr.shape[0]),
    )


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Export videos undistorted par camera")
    parser.add_argument("--video_gauche", required=True)
    parser.add_argument("--video_droite", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--start", default="00:00:00")
    parser.add_argument("--end", default=None)
    parser.add_argument("--detections_csv", default=None)
    parser.add_argument("--assignments_csv", default=None)
    args = parser.parse_args()

    require_file(args.video_gauche, "video gauche")
    require_file(args.video_droite, "video droite")
    calibration = load_calibration(args.calibration)

    overlays = None
    export_tracked = bool(args.detections_csv and args.assignments_csv)
    if export_tracked:
        overlays = load_tracking_overlays(args.detections_csv, args.assignments_csv)

    start_s = hms_to_s(args.start)
    end_s = hms_to_s(args.end) if args.end else None
    os.makedirs(args.output_dir, exist_ok=True)

    captures = {
        "gauche": open_capture(args.video_gauche, start_s),
        "droite": open_capture(args.video_droite, start_s),
    }
    fps = captures["gauche"].get(cv2.CAP_PROP_FPS) or 25.0

    raw_paths = {
        camera: os.path.join(args.output_dir, CAMERA_CONFIG[camera]["raw_output_name"])
        for camera in CAMERA_CONFIG
    }
    tracked_paths = {
        camera: os.path.join(args.output_dir, CAMERA_CONFIG[camera]["tracked_output_name"])
        for camera in CAMERA_CONFIG
    }
    raw_writers = {"gauche": None, "droite": None}
    tracked_writers = {"gauche": None, "droite": None}

    frame_idx = 0
    try:
        while True:
            frames = {}
            for camera, capture in captures.items():
                ok, frame = capture.read()
                if not ok:
                    frames = None
                    break
                frames[camera] = frame
            if frames is None:
                break

            timestamp_s = start_s + frame_idx / fps
            if end_s is not None and timestamp_s > end_s:
                break

            for camera, frame in frames.items():
                config = CAMERA_CONFIG[camera]
                undistorted = remap_frame_with_distortion(
                    frame,
                    calibration.get(config["distortion_key"]),
                    calibration.get(config["view_key"]),
                )

                if raw_writers[camera] is None:
                    raw_writers[camera] = build_writer(raw_paths[camera], fps, undistorted)
                raw_writers[camera].write(undistorted)

                if export_tracked:
                    if tracked_writers[camera] is None:
                        tracked_writers[camera] = build_writer(tracked_paths[camera], fps, undistorted)
                    annotated = draw_tracking_overlay(
                        undistorted,
                        overlays.get((frame_idx, camera), []),
                        config["color"],
                    )
                    tracked_writers[camera].write(annotated)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx:6d} | t={timestamp_s:7.1f}s")
    finally:
        for capture in captures.values():
            capture.release()
        for writer in raw_writers.values():
            if writer is not None:
                writer.release()
        for writer in tracked_writers.values():
            if writer is not None:
                writer.release()

    print("=" * 60)
    for camera, path in raw_paths.items():
        print(f"Video {camera} undistorted exportee : {path}")
    if export_tracked:
        for camera, path in tracked_paths.items():
            print(f"Video {camera} trackee exportee : {path}")


if __name__ == "__main__":
    main()
