"""
Detection frame par frame sur deux cameras, sans tracking.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.paths import DEFAULT_POSE_MODEL
from core.performance import AGGRESSIVENESS_CHOICES, FPS_TARGET_CHOICES, resolve_performance_settings
from core.utils import configure_utf8_stdout, ensure_dir, hms_to_s, infer_media_start_unix, require_file, unix_to_iso8601

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_source(source: str) -> bool:
    return os.path.splitext(source)[1].lower() in IMAGE_EXTENSIONS


def open_source(source: str, label: str) -> dict:
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


def seek_source(source_state: dict, start_s: float) -> None:
    if source_state["kind"] == "video":
        source_state["capture"].set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)


def read_source_frame(source_state: dict):
    if source_state["kind"] == "image":
        if source_state["consumed"]:
            return False, None
        source_state["consumed"] = True
        return True, source_state["frame"].copy()
    return source_state["capture"].read()


def release_source(source_state: dict) -> None:
    if source_state["kind"] == "video":
        source_state["capture"].release()


def detect_frame(model: YOLO, frame, conf: float, imgsz: int, device: str | int):
    results = model.predict(
        frame,
        classes=[0],
        conf=conf,
        device=device,
        imgsz=imgsz,
        verbose=False,
    )
    return results[0]


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Detection joueurs sans tracking")
    parser.add_argument("--video_gauche", required=True, help="Video ou image gauche")
    parser.add_argument("--video_droite", required=True, help="Video ou image droite")
    parser.add_argument("--output", default="./resultats", help="Dossier de sortie CSV")
    parser.add_argument("--target_fps", type=int, choices=FPS_TARGET_CHOICES, default=15)
    parser.add_argument("--aggressiveness", choices=AGGRESSIVENESS_CHOICES, default="medium")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="0", help="0 = GPU 0, cpu = CPU")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf_gauche", type=float, default=None)
    parser.add_argument("--conf_droite", type=float, default=None)
    parser.add_argument("--imgsz_gauche", type=int, default=None)
    parser.add_argument("--imgsz_droite", type=int, default=None)
    parser.add_argument("--start", default="00:00:00")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    perf = resolve_performance_settings(args.target_fps, args.aggressiveness)
    args.model = args.model or perf.model or str(DEFAULT_POSE_MODEL)
    args.conf = args.conf if args.conf is not None else perf.conf
    args.imgsz = args.imgsz if args.imgsz is not None else perf.imgsz

    if not (0.0 < args.conf <= 1.0):
        raise SystemExit("--conf doit etre dans ]0, 1]")
    if args.imgsz <= 0:
        raise SystemExit("--imgsz doit etre > 0")

    ensure_dir(args.output)
    require_file(args.model, "modele")

    conf_gauche = args.conf_gauche if args.conf_gauche is not None else args.conf
    conf_droite = args.conf_droite if args.conf_droite is not None else args.conf
    imgsz_gauche = args.imgsz_gauche if args.imgsz_gauche is not None else args.imgsz
    imgsz_droite = args.imgsz_droite if args.imgsz_droite is not None else args.imgsz

    print(f"Chargement du modele : {os.path.basename(args.model)}")
    model_g = YOLO(args.model)
    model_d = YOLO(args.model)
    device = "cpu" if args.device.lower() == "cpu" else int(args.device)

    source_g = open_source(args.video_gauche, "gauche")
    source_d = open_source(args.video_droite, "droite")
    fps = source_g["fps"]
    total_g = source_g["frame_count"]
    start_s = hms_to_s(args.start)
    end_s = hms_to_s(args.end) if args.end else None
    total_frames = int((end_s - start_s) * fps) if end_s is not None else total_g

    seek_source(source_g, start_s)
    seek_source(source_d, start_s)

    csv_path = os.path.join(args.output, "detections.csv")
    meta_path = os.path.join(args.output, "run_metadata.json")
    csv_handle = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_handle)
    writer.writerow(
        [
            "frame",
            "timestamp_s",
            "timestamp_unix",
            "cam",
            "detection_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "foot_x",
            "foot_y",
            "conf",
        ]
    )

    print(f"Flux gauche : {args.video_gauche}")
    print(f"Flux droite : {args.video_droite}")
    print(f"Plage : {args.start} -> {args.end or 'fin'}")
    print(f"Profil perf : target_fps={args.target_fps} aggressiveness={args.aggressiveness}")

    frame_idx = 0
    start_wall_time = time.time()
    counts = {"gauche": 0, "droite": 0}

    try:
        while True:
            ret_g, frame_g = read_source_frame(source_g)
            ret_d, frame_d = read_source_frame(source_d)
            if not ret_g or not ret_d:
                break

            timestamp_s = start_s + frame_idx / fps
            if end_s is not None and timestamp_s > end_s:
                break

            timestamp_unix_g = source_g["start_unix"] + int(timestamp_s) if source_g["start_unix"] is not None else None
            timestamp_unix_d = source_d["start_unix"] + int(timestamp_s) if source_d["start_unix"] is not None else None

            for camera, frame, model, conf, imgsz, timestamp_unix in [
                ("gauche", frame_g, model_g, conf_gauche, imgsz_gauche, timestamp_unix_g),
                ("droite", frame_d, model_d, conf_droite, imgsz_droite, timestamp_unix_d),
            ]:
                result = detect_frame(model, frame, conf=conf, imgsz=imgsz, device=device)
                if result.boxes is None:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
                for detection_index, box in enumerate(boxes, start=1):
                    x1, y1, x2, y2 = map(int, box)
                    foot_x = int((x1 + x2) / 2)
                    foot_y = int(y2)
                    det_conf = float(confs[detection_index - 1]) if len(confs) >= detection_index else 0.0
                    writer.writerow(
                        [
                            frame_idx,
                            f"{timestamp_s:.3f}",
                            timestamp_unix or "",
                            camera,
                            detection_index,
                            x1,
                            y1,
                            x2,
                            y2,
                            foot_x,
                            foot_y,
                            f"{det_conf:.6f}",
                        ]
                    )
                    counts[camera] += 1

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_wall_time
                real_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                pct = frame_idx / total_frames * 100 if total_frames > 0 else 0.0
                print(f"  Frame {frame_idx:6d} | {timestamp_s:8.1f}s | {real_fps:5.1f} fps | {pct:5.1f}%")
    finally:
        release_source(source_g)
        release_source(source_d)
        csv_handle.close()

    elapsed_total = time.time() - start_wall_time
    summary = {
        "video_gauche": args.video_gauche,
        "video_droite": args.video_droite,
        "video_gauche_start_unix": source_g["start_unix"],
        "video_gauche_start_iso": unix_to_iso8601(source_g["start_unix"]),
        "video_gauche_start_source": source_g["start_unix_source"],
        "video_droite_start_unix": source_d["start_unix"],
        "video_droite_start_iso": unix_to_iso8601(source_d["start_unix"]),
        "video_droite_start_source": source_d["start_unix_source"],
        "model": args.model,
        "device": str(device),
        "start": args.start,
        "end": args.end,
        "target_fps": args.target_fps,
        "aggressiveness": args.aggressiveness,
        "frames_processed": frame_idx,
        "avg_fps": round(frame_idx / elapsed_total, 3) if elapsed_total > 0 else 0.0,
        "csv_path": csv_path,
        "conf_gauche": conf_gauche,
        "conf_droite": conf_droite,
        "imgsz_gauche": imgsz_gauche,
        "imgsz_droite": imgsz_droite,
        "detection_counts": counts,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 60)
    print(f"Detection terminee - {frame_idx} frames en {elapsed_total / 60:.1f} min ({summary['avg_fps']:.1f} fps moyen)")
    print(f"CSV exporte : {csv_path}")
    print(f"Metadonnees : {meta_path}")
    print(f"Detections gardees : gauche={counts['gauche']} droite={counts['droite']}")


if __name__ == "__main__":
    main()
