from __future__ import annotations

import os

import cv2

from solver_lps.features.cv.review.generation.raw_tracking.data.tracking_output import infer_media_start_unix

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_source(source: str) -> bool:
    return os.path.splitext(source)[1].lower() in IMAGE_EXTENSIONS


def open_video_input(source: str, label: str) -> dict:
    if is_image_source(source):
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"Image {label} inaccessible : {source}")
        start_unix, start_source = infer_media_start_unix(source)
        return {
            "kind": "image",
            "path": source,
            "frame": frame,
            "consumed": False,
            "fps": 1.0,
            "frame_count": 1,
            "start_unix": start_unix,
            "start_unix_source": start_source,
        }

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Flux {label} inaccessible : {source}")
    start_unix, start_source = infer_media_start_unix(source)
    return {
        "kind": "video",
        "path": source,
        "capture": capture,
        "fps": capture.get(cv2.CAP_PROP_FPS) or 25.0,
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "start_unix": start_unix,
        "start_unix_source": start_source,
    }


def seek_video_input(source_state: dict, start_s: float) -> None:
    if source_state["kind"] == "video":
        source_state["capture"].set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)


def read_video_input_frame(source_state: dict):
    if source_state["kind"] == "image":
        if source_state["consumed"]:
            return False, None
        source_state["consumed"] = True
        return True, source_state["frame"].copy()
    return source_state["capture"].read()


def release_video_input(source_state: dict) -> None:
    if source_state["kind"] == "video":
        source_state["capture"].release()
