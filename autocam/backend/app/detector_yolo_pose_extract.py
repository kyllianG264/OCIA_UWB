"""
detector_yolo_pose_extract.py
Pipeline 2-modeles compatible Windows :
  1) YOLO detection + tracking (personne dominante)
  2) YOLO11-pose sur le crop de cette personne

Objectif : separer explicitement le reperage de l'athlete et l'extraction biomecanique.
Retourne (n_frames, 33, 3) compatible feature_extractor.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from yolo_extract import _COCO_TO_MP33


def _resolve_device(device: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _resolve_model_path(model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)

    backend_base = Path(__file__).resolve().parent.parent
    candidate = backend_base / model_name
    if candidate.exists():
        return str(candidate)

    return model_name


# Cache de modèles : un chargement unique par (modèle, device), réutilisé pour
# tous les clips. Sans ça, chaque clip rechargeait 2 modèles YOLO → OOM/crash.
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _get_cached_model(model_name: str, device: str):
    from ultralytics import YOLO

    key = (model_name, device)
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model
    model = YOLO(_resolve_model_path(model_name))
    try:
        model.to(device)
    except Exception:
        pass
    _MODEL_CACHE[key] = model
    return model


def _extract_person_detections(result) -> list[dict]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(xyxy),), dtype=np.float32)
    cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros((len(xyxy),), dtype=np.float32)
    ids = boxes.id.cpu().numpy() if getattr(boxes, "id", None) is not None else None

    detections: list[dict] = []
    for i in range(len(xyxy)):
        if int(cls[i]) != 0:  # classe COCO person
            continue
        x1, y1, x2, y2 = xyxy[i].tolist()
        det = {
            "xyxy": (float(x1), float(y1), float(x2), float(y2)),
            "conf": float(conf[i]),
            "id": int(ids[i]) if ids is not None else None,
        }
        detections.append(det)
    return detections


def _choose_detection(
    detections: list[dict],
    dominant_track_id: int | None,
    prev_center: tuple[float, float] | None,
) -> dict | None:
    if not detections:
        return None

    if dominant_track_id is not None:
        same_track = [d for d in detections if d["id"] == dominant_track_id]
        if same_track:
            return max(same_track, key=lambda d: d["conf"])

    if prev_center is None:
        return max(
            detections,
            key=lambda d: (d["xyxy"][2] - d["xyxy"][0]) * (d["xyxy"][3] - d["xyxy"][1]),
        )

    px, py = prev_center

    def score(d: dict) -> float:
        x1, y1, x2, y2 = d["xyxy"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = max(1.0, (x2 - x1) * (y2 - y1))
        dist = np.hypot(cx - px, cy - py)
        return float(area - (dist * 25.0))

    return max(detections, key=score)


def _best_pose_person(kps_data: np.ndarray) -> np.ndarray | None:
    if kps_data.ndim != 3 or kps_data.shape[0] == 0:
        return None
    conf_mean = kps_data[:, :, 2].mean(axis=1)
    return kps_data[int(conf_mean.argmax())]


def _to_mp33_fullframe(person_kps_crop: np.ndarray, x1: int, y1: int, W: int, H: int) -> np.ndarray:
    frame33 = np.zeros((33, 3), dtype=np.float32)
    for coco_idx, mp_idx in _COCO_TO_MP33.items():
        px, py, conf = person_kps_crop[coco_idx]
        frame33[mp_idx] = [
            (float(px) + x1) / max(W, 1),
            (float(py) + y1) / max(H, 1),
            float(conf),
        ]
    return frame33


def extract_keypoints(
    video_path: str,
    max_frames: int = 300,
    detector_model_name: str = "yolo11n.pt",
    pose_model_name: str = "yolo11n-pose.pt",
    detector_conf_threshold: float = 0.35,
    pose_conf_threshold: float = 0.5,
    center_crop_ratio: float = 1.0,
    min_body_height_ratio: float = 0.0,
    min_hip_velocity: float = 0.015,
    device: str = "auto",
) -> np.ndarray | None:
    """Detecte/track l'athlete dominant puis extrait les keypoints pose sur son crop."""
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        raise ImportError("ultralytics n'est pas installe. Lancez : pip install ultralytics")

    resolved_device = _resolve_device(device)

    detector = _get_cached_model(detector_model_name, resolved_device)
    pose = _get_cached_model(pose_model_name, resolved_device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    center_crop_ratio = float(np.clip(center_crop_ratio, 0.1, 1.0))
    margin = (1.0 - center_crop_ratio) / 2.0
    crop_x0 = int(margin * W)
    crop_x1 = int((1.0 - margin) * W)

    dominant_track_id: int | None = None
    track_area_sum: dict[int, float] = {}
    track_area_count: dict[int, int] = {}

    prev_center: tuple[float, float] | None = None
    keypoints_list: list[np.ndarray] = []
    hip_y_history: list[float] = []

    init_frames = 40
    frame_idx = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_det = frame[:, crop_x0:crop_x1] if center_crop_ratio < 1.0 else frame

        det_result = None
        try:
            track_res = detector.track(
                frame_det,
                conf=float(detector_conf_threshold),
                classes=[0],
                tracker="bytetrack.yaml",
                # persist=False sur la 1ère frame réinitialise le tracker pour
                # cette nouvelle vidéo (le modèle est partagé entre clips via le
                # cache, il ne faut pas hériter du tracking du clip précédent).
                persist=frame_idx > 0,
                verbose=False,
                device=resolved_device,
            )
            if track_res:
                det_result = track_res[0]
        except Exception:
            det_result = None

        if det_result is None:
            det_res = detector(
                frame_det,
                conf=float(detector_conf_threshold),
                classes=[0],
                verbose=False,
                device=resolved_device,
            )
            det_result = det_res[0] if det_res else None

        detections = _extract_person_detections(det_result) if det_result is not None else []

        # Remap X depuis le crop central vers full-frame
        if center_crop_ratio < 1.0 and detections:
            remapped: list[dict] = []
            for d in detections:
                x1, y1, x2, y2 = d["xyxy"]
                d2 = dict(d)
                d2["xyxy"] = (x1 + crop_x0, y1, x2 + crop_x0, y2)
                remapped.append(d2)
            detections = remapped

        for d in detections:
            tid = d["id"]
            if tid is None:
                continue
            x1, y1, x2, y2 = d["xyxy"]
            area = max(1.0, (x2 - x1) * (y2 - y1))
            track_area_sum[tid] = track_area_sum.get(tid, 0.0) + area
            track_area_count[tid] = track_area_count.get(tid, 0) + 1

        if dominant_track_id is None and frame_idx >= init_frames and track_area_count:
            dominant_track_id = max(
                track_area_count.keys(),
                key=lambda tid: track_area_sum[tid] / max(1, track_area_count[tid]),
            )

        chosen = _choose_detection(detections, dominant_track_id, prev_center)
        if chosen is None:
            frame_idx += 1
            continue

        x1f, y1f, x2f, y2f = chosen["xyxy"]

        if dominant_track_id is None and chosen["id"] is not None and frame_idx >= 8:
            dominant_track_id = int(chosen["id"])

        cx = (x1f + x2f) / 2.0
        cy = (y1f + y2f) / 2.0
        prev_center = (cx, cy)

        pad_x = max(6, int((x2f - x1f) * 0.10))
        pad_y = max(6, int((y2f - y1f) * 0.10))

        x1 = max(0, int(x1f) - pad_x)
        y1 = max(0, int(y1f) - pad_y)
        x2 = min(W, int(x2f) + pad_x)
        y2 = min(H, int(y2f) + pad_y)

        if x2 <= x1 or y2 <= y1:
            frame_idx += 1
            continue

        if min_body_height_ratio > 0 and ((y2 - y1) / max(H, 1)) < float(min_body_height_ratio):
            frame_idx += 1
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            frame_idx += 1
            continue

        pose_res = pose(crop, conf=float(pose_conf_threshold), verbose=False, device=resolved_device)
        if not pose_res or pose_res[0].keypoints is None or pose_res[0].keypoints.data is None:
            frame_idx += 1
            continue

        kps_data = pose_res[0].keypoints.data.cpu().numpy()  # (N, 17, 3)
        best = _best_pose_person(kps_data)
        if best is None:
            frame_idx += 1
            continue

        frame33 = _to_mp33_fullframe(best, x1, y1, W, H)
        keypoints_list.append(frame33)

        if frame33[23, 2] > 0.1 or frame33[24, 2] > 0.1:
            hip_y_history.append(float((frame33[23, 1] + frame33[24, 1]) / 2.0))

        frame_idx += 1

    cap.release()

    if len(keypoints_list) < 5:
        return None

    keypoints_array = np.stack(keypoints_list, axis=0)

    if min_hip_velocity > 0 and len(hip_y_history) >= 3:
        diffs = [
            abs(hip_y_history[i + 1] - hip_y_history[i])
            for i in range(len(hip_y_history) - 1)
        ]
        avg_vel = float(sum(diffs) / max(len(diffs), 1))
        if avg_vel < float(min_hip_velocity):
            return None

    return keypoints_array
