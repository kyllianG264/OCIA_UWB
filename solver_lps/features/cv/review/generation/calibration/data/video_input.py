from __future__ import annotations

import os

import cv2
import numpy as np


def open_video_capture(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video introuvable : {path}")
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video : {path}")
    return capture


def extract_frame(path: str, seconds: float) -> np.ndarray:
    capture = open_video_capture(path)
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Unable to read frame at {seconds:.1f}s from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
