"""
yolo_extract.py
Extrait les keypoints de pose YOLOv8-pose depuis un fichier vidéo.
Retourne le même format que mediapipe_extract (n_frames, 33, 3) pour
être compatible avec feature_extractor.py sans modification.

Avantages vs MediaPipe :
  - Détecte toutes les personnes dans le frame → on choisit la bonne
  - Meilleure robustesse aux occlusions partielles
  - Score de confiance par keypoint (pas seulement visibilité)
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import subprocess

# Mapping COCO 17 keypoints → indices MediaPipe 33
# Les indices MP non couverts par COCO restent à visibilité 0.
# COCO idx : [nose, l_eye, r_eye, l_ear, r_ear,
#              l_sho, r_sho, l_elb, r_elb, l_wri, r_wri,
#              l_hip, r_hip, l_kne, r_kne, l_ank, r_ank]
_COCO_TO_MP33: dict[int, int] = {
    0: 0,   # nose
    1: 2,   # left_eye
    2: 5,   # right_eye
    3: 7,   # left_ear
    4: 8,   # right_ear
    5: 11,  # left_shoulder
    6: 12,  # right_shoulder
    7: 13,  # left_elbow
    8: 14,  # right_elbow
    9: 15,  # left_wrist
    10: 16, # right_wrist
    11: 23, # left_hip
    12: 24, # right_hip
    13: 25, # left_knee
    14: 26, # right_knee
    15: 27, # left_ankle
    16: 28, # right_ankle
}

# Landmarks clés pour calculer la bounding box dominante
_BODY_COCO_IDX = [5, 6, 11, 12, 15, 16]  # épaules + hanches + chevilles

# Segments COCO pour visualisation du squelette
_COCO_SKELETON: list[tuple[int, int]] = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def _build_mp33_frame(coco_kps: np.ndarray) -> np.ndarray:
    """
    Convertit un array COCO (17, 3) en array MP33 (33, 3).
    coco_kps[:, 0] = x normalisé, [:, 1] = y normalisé, [:, 2] = confiance [0,1].
    Les positions MP non couvertes restent à (0, 0, 0).
    """
    mp33 = np.zeros((33, 3), dtype=np.float32)
    for coco_idx, mp_idx in _COCO_TO_MP33.items():
        mp33[mp_idx] = coco_kps[coco_idx]
    return mp33


def _bbox_area(coco_kps: np.ndarray, conf_threshold: float = 0.3) -> float:
    """Surface bbox à partir des keypoints COCO visibles."""
    visible = [i for i in _BODY_COCO_IDX if coco_kps[i, 2] > conf_threshold]
    if len(visible) < 3:
        return 0.0
    xs = coco_kps[visible, 0]
    ys = coco_kps[visible, 1]
    return float((xs.max() - xs.min()) * (ys.max() - ys.min()))


def _bbox_center_x(coco_kps: np.ndarray) -> float:
    """Centre horizontal COCO (moyenne épaules + hanches)."""
    return float(np.mean(coco_kps[[5, 6, 11, 12], 0]))


def _select_dominant_person(
    keypoints_list: list[np.ndarray],
    dominant_center_x: float,
    center_tolerance: float,
    dominant_bbox_area: float,
) -> np.ndarray | None:
    """
    Parmi plusieurs personnes détectées dans un frame, choisit celle
    qui est la plus cohérente avec la personne dominante identifiée.
    Critères (par ordre de priorité) :
      1. Centre horizontal proche de dominant_center_x (±center_tolerance)
      2. Plus grande bounding box parmi les candidats restants
    """
    if not keypoints_list:
        return None

    # Filtrer sur la cohérence de position si on a un dominant établi
    if dominant_bbox_area > 0.0:
        candidates = [
            kp for kp in keypoints_list
            if abs(_bbox_center_x(kp) - dominant_center_x) <= center_tolerance
        ]
        if not candidates:
            # Aucun candidat cohérent → prendre le plus grand
            candidates = keypoints_list
    else:
        candidates = keypoints_list

    # Parmi les candidats, prendre celui avec la plus grande bbox
    return max(candidates, key=_bbox_area)


def _resolve_device(device: str) -> str:
    """
    Résout le device d'inférence.
    "auto" → "cuda" si disponible, sinon "cpu".
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"

    if device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            return "cpu"

    return device


def _resolve_model_path(model_name: str) -> str:
    """Résout le chemin du modèle YOLO de façon robuste (backend/ ou cwd)."""
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)

    backend_base = Path(__file__).resolve().parent.parent
    candidate = backend_base / model_name
    if candidate.exists():
        return str(candidate)

    return model_name


# Cache de modèles : évite de recharger le réseau YOLO à chaque clip.
# Recharger un modèle par vidéo (144 fois) fait grimper la mémoire jusqu'à
# l'OOM qui tue le process. On garde donc une instance unique par
# (modèle, device) et on la réutilise pour tous les clips.
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _get_cached_model(model_name: str, device: str):
    """Retourne un modèle YOLO chargé une seule fois, placé sur le bon device."""
    from ultralytics import YOLO

    key = (model_name, device)
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model, device

    model = YOLO(_resolve_model_path(model_name))
    try:
        model.to(device)
    except Exception as e:
        print(f"⚠️ Impossible de charger le modèle sur '{device}': {e}. Utilisation du CPU.")
        device = "cpu"
        key = (model_name, device)
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached, device
        try:
            model.to(device)
        except Exception:
            pass

    _MODEL_CACHE[key] = model
    return model, device


def _draw_coco_pose(
    frame: np.ndarray,
    coco_kps_norm: np.ndarray,
    conf_threshold: float = 0.3,
    point_color: tuple[int, int, int] = (0, 255, 0),
    line_color: tuple[int, int, int] = (0, 200, 255),
) -> None:
    """Dessine les keypoints/segments COCO (coordonnées normalisées) sur l'image."""
    h, w = frame.shape[:2]

    # Segments
    for a, b in _COCO_SKELETON:
        if coco_kps_norm[a, 2] < conf_threshold or coco_kps_norm[b, 2] < conf_threshold:
            continue
        xa = int(np.clip(coco_kps_norm[a, 0] * w, 0, w - 1))
        ya = int(np.clip(coco_kps_norm[a, 1] * h, 0, h - 1))
        xb = int(np.clip(coco_kps_norm[b, 0] * w, 0, w - 1))
        yb = int(np.clip(coco_kps_norm[b, 1] * h, 0, h - 1))
        cv2.line(frame, (xa, ya), (xb, yb), line_color, 2, cv2.LINE_AA)

    # Points
    for i in range(coco_kps_norm.shape[0]):
        if coco_kps_norm[i, 2] < conf_threshold:
            continue
        x = int(np.clip(coco_kps_norm[i, 0] * w, 0, w - 1))
        y = int(np.clip(coco_kps_norm[i, 1] * h, 0, h - 1))
        cv2.circle(frame, (x, y), 4, point_color, -1, cv2.LINE_AA)


def _get_ffmpeg_executable() -> str | None:
    """Retourne le binaire ffmpeg (imageio-ffmpeg), ou None si indisponible."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except Exception:
        return None

    try:
        return get_ffmpeg_exe()
    except Exception:
        return None


def _transcode_h264(input_path: Path, output_path: Path) -> bool:
    ffmpeg_exe = _get_ffmpeg_executable()
    if ffmpeg_exe is None:
        return False

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return False

    return proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1024


def export_pose_preview(
    video_path: str,
    output_path: str,
    max_frames: int = 240,
    model_name: str = "yolov8n-pose.pt",
    conf_threshold: float = 0.5,
    center_crop_ratio: float = 1.0,
    device: str = "auto",
    draw_conf_threshold: float = 0.3,
    target_fps: float = 20.0,
    max_output_width: int = 960,
) -> str | None:
    """
    Génère une vidéo annotée compatible navigateur (H264) pour aperçu de pose.
    Retourne le chemin du fichier final ou None en cas d'échec.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics n'est pas installé. "
            "Lancez : pip install ultralytics"
        )

    resolved_device = _resolve_device(device)
    model = YOLO(_resolve_model_path(model_name))
    try:
        model.to(resolved_device)
    except Exception:
        resolved_device = "cpu"
        try:
            model.to(resolved_device)
        except Exception:
            pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    out_fps = max(5.0, min(float(fps), float(target_fps)))

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out_w, out_h = w, h
    if w > max_output_width:
        scale = float(max_output_width) / float(w)
        out_w = int(w * scale)
        out_h = int(h * scale)

    final_output = Path(output_path)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = final_output.with_name(f"{final_output.stem}.raw.mp4")

    writer = cv2.VideoWriter(
        str(temp_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (out_w, out_h),
    )

    if not writer.isOpened():
        cap.release()
        return None

    center_crop_ratio = float(np.clip(center_crop_ratio, 0.1, 1.0))
    margin = (1.0 - center_crop_ratio) / 2.0
    crop_x_start = margin
    crop_x_end = 1.0 - margin

    CENTER_TOLERANCE = 0.30
    INIT_FRAMES = 60
    dominant_bbox_area: float = 0.0
    dominant_center_x: float = 0.5

    # Scan initial
    scan_count = 0
    while scan_count < INIT_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        h_scan, fw_orig = frame.shape[:2]
        frame_inf = frame[:, int(crop_x_start * fw_orig):int(crop_x_end * fw_orig)] if center_crop_ratio < 1.0 else frame
        results = model(frame_inf, verbose=False, conf=conf_threshold, device=resolved_device)
        if results and results[0].keypoints is not None:
            kps_data = results[0].keypoints.data
            if len(kps_data) > 0:
                fh, fw2 = frame_inf.shape[:2]
                for person_kps in kps_data.cpu().numpy():
                    norm = person_kps.copy()
                    norm[:, 0] /= fw2
                    norm[:, 1] /= fh
                    area = _bbox_area(norm)
                    if area > dominant_bbox_area:
                        dominant_bbox_area = area
                        dominant_center_x = _bbox_center_x(norm)
        scan_count += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = 0
    written = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        h_frame, w_frame = frame.shape[:2]
        frame_inf = frame[:, int(crop_x_start * w_frame):int(crop_x_end * w_frame)] if center_crop_ratio < 1.0 else frame

        results = model(frame_inf, verbose=False, conf=conf_threshold, device=resolved_device)
        if results and results[0].keypoints is not None:
            kps_data = results[0].keypoints.data
            if len(kps_data) > 0:
                fh2, fw2 = frame_inf.shape[:2]
                all_persons_norm: list[np.ndarray] = []
                for person_kps in kps_data.cpu().numpy():
                    norm = person_kps.copy()
                    norm[:, 0] /= fw2
                    norm[:, 1] /= fh2
                    all_persons_norm.append(norm)

                best = _select_dominant_person(
                    all_persons_norm, dominant_center_x, CENTER_TOLERANCE, dominant_bbox_area
                )
                if best is not None:
                    to_draw = best.copy()
                    if center_crop_ratio < 1.0:
                        to_draw[:, 0] = crop_x_start + to_draw[:, 0] * (crop_x_end - crop_x_start)
                    _draw_coco_pose(frame, to_draw, conf_threshold=draw_conf_threshold)

        frame_to_write = frame
        if out_w != w_frame or out_h != h_frame:
            frame_to_write = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        writer.write(frame_to_write)
        written += 1
        frame_count += 1

    cap.release()
    writer.release()

    if written == 0 or not temp_output.exists() or temp_output.stat().st_size < 1024:
        temp_output.unlink(missing_ok=True)
        return None

    ok = _transcode_h264(temp_output, final_output)
    temp_output.unlink(missing_ok=True)
    if not ok:
        return None

    # Validation finale
    check_cap = cv2.VideoCapture(str(final_output))
    can_read, _ = check_cap.read()
    check_cap.release()
    if not can_read:
        return None

    return str(final_output)


def extract_keypoints(
    video_path: str,
    max_frames: int = 300,
    model_name: str = "yolov8n-pose.pt",
    conf_threshold: float = 0.5,
    center_crop_ratio: float = 1.0,
    min_body_height_ratio: float = 0.0,
    min_hip_velocity: float = 0.0,
    device: str = "auto",
) -> np.ndarray | None:
    """
    Extrait les keypoints YOLOv8-pose d'un clip vidéo.
    Sélectionne automatiquement la personne dominante (plus grande bbox)
    parmi toutes les personnes détectées dans chaque frame.

    Args:
        video_path: chemin vers le fichier vidéo.
        max_frames: nombre max de frames analysées.
        model_name: nom du modèle YOLO à utiliser.
            - "yolov8n-pose.pt" : nano, rapide, moins précis
            - "yolov8s-pose.pt" : small, bon équilibre
            - "yolov8m-pose.pt" : medium, meilleure précision
        conf_threshold: seuil de confiance minimum pour une personne.
        center_crop_ratio: idem mediapipe_extract.
        min_body_height_ratio: idem mediapipe_extract.
        min_hip_velocity: idem mediapipe_extract.

    Returns:
        np.ndarray de shape (n_frames, 33, 3) au format MediaPipe,
        ou None si aucune pose détectée.
    """
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        raise ImportError(
            "ultralytics n'est pas installé. "
            "Lancez : pip install ultralytics"
        )

    resolved_device = _resolve_device(device)
    model, resolved_device = _get_cached_model(model_name, resolved_device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    center_crop_ratio = float(np.clip(center_crop_ratio, 0.1, 1.0))
    margin = (1.0 - center_crop_ratio) / 2.0
    crop_x_start = margin
    crop_x_end = 1.0 - margin

    CENTER_TOLERANCE = 0.30
    INIT_FRAMES = 60

    # --- Phase 1 : scan initial pour trouver la personne dominante ---
    dominant_bbox_area: float = 0.0
    dominant_center_x: float = 0.5
    scan_count = 0

    while scan_count < INIT_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if center_crop_ratio < 1.0:
            frame = frame[:, int(crop_x_start * w):int(crop_x_end * w)]

        results = model(frame, verbose=False, conf=conf_threshold, device=resolved_device)
        if results and results[0].keypoints is not None:
            kps_data = results[0].keypoints.data  # (n_persons, 17, 3) [x, y, conf]
            if len(kps_data) > 0:
                # Normaliser les coordonnées en [0,1]
                fh, fw = frame.shape[:2]
                for person_kps in kps_data.cpu().numpy():
                    norm_kps = person_kps.copy()
                    norm_kps[:, 0] /= fw
                    norm_kps[:, 1] /= fh
                    area = _bbox_area(norm_kps)
                    if area > dominant_bbox_area:
                        dominant_bbox_area = area
                        dominant_center_x = _bbox_center_x(norm_kps)
        scan_count += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Phase 2 : extraction complète ---
    all_keypoints: list[np.ndarray] = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        orig_w = w
        if center_crop_ratio < 1.0:
            frame = frame[:, int(crop_x_start * w):int(crop_x_end * w)]

        results = model(frame, verbose=False, conf=conf_threshold, device=resolved_device)

        if results and results[0].keypoints is not None:
            kps_data = results[0].keypoints.data
            if len(kps_data) > 0:
                fh, fw = frame.shape[:2]
                # Normaliser toutes les personnes
                all_persons_norm: list[np.ndarray] = []
                for person_kps in kps_data.cpu().numpy():
                    norm_kps = person_kps.copy()
                    norm_kps[:, 0] /= fw
                    norm_kps[:, 1] /= fh
                    all_persons_norm.append(norm_kps)

                best = _select_dominant_person(
                    all_persons_norm,
                    dominant_center_x,
                    CENTER_TOLERANCE,
                    dominant_bbox_area,
                )

                if best is not None:
                    # Filtre taille
                    if min_body_height_ratio > 0.0:
                        shoulder_y = float(np.mean([best[5, 1], best[6, 1]]))
                        ankle_y = float(np.mean([best[15, 1], best[16, 1]]))
                        if abs(ankle_y - shoulder_y) < min_body_height_ratio:
                            frame_count += 1
                            continue

                    # Remap x si crop actif
                    if center_crop_ratio < 1.0:
                        best[:, 0] = crop_x_start + best[:, 0] * (crop_x_end - crop_x_start)

                    all_keypoints.append(_build_mp33_frame(best))

        frame_count += 1

    cap.release()

    if not all_keypoints:
        return None

    keypoints_array = np.stack(all_keypoints, axis=0)

    # Filtre vitesse : coupe le début statique
    if min_hip_velocity > 0.0 and len(keypoints_array) > 1:
        hip_centers = (keypoints_array[:, 23, :2] + keypoints_array[:, 24, :2]) / 2.0
        deltas = np.concatenate([[0.0], np.linalg.norm(np.diff(hip_centers, axis=0), axis=1)])
        window = min(15, max(3, len(deltas) // 8))
        rolling_avg = np.convolve(deltas, np.ones(window) / window, mode="same")
        active = np.where(rolling_avg >= min_hip_velocity)[0]
        if len(active) > 0:
            motion_start = max(0, int(active[0]) - window)
            keypoints_array = keypoints_array[motion_start:]

    return keypoints_array
