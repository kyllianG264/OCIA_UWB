"""
Detection frame par frame sur deux cameras, sans tracking.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from solver_lps.features.cv.review.generation.raw_tracking.data.tracking_output import (
    build_detection_output_paths,
    configure_utf8_stdout,
    DEFAULT_POSE_MODEL,
    ensure_dir,
    hms_to_s,
    open_detection_csv,
    require_file,
    write_detection_summary,
)
from solver_lps.features.cv.review.generation.raw_tracking.data.video_input import (
    open_video_input,
    read_video_input_frame,
    release_video_input,
    seek_video_input,
)


def _load_yolo_class():
    from ultralytics import YOLO

    return YOLO


def detect_frame(model, frame, conf: float, imgsz: int, device: str | int):
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
    parser.add_argument("--model", default=str(DEFAULT_POSE_MODEL))
    parser.add_argument("--device", default="0", help="0 = GPU 0, cpu = CPU")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=1408)
    parser.add_argument("--conf_gauche", type=float, default=None)
    parser.add_argument("--conf_droite", type=float, default=None)
    parser.add_argument("--imgsz_gauche", type=int, default=None)
    parser.add_argument("--imgsz_droite", type=int, default=None)
    parser.add_argument("--start", default="00:00:00")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

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
    YOLO = _load_yolo_class()
    model_g = YOLO(args.model)
    model_d = YOLO(args.model)
    device = "cpu" if args.device.lower() == "cpu" else int(args.device)

    source_g = open_video_input(args.video_gauche, "gauche")
    source_d = open_video_input(args.video_droite, "droite")
    fps = source_g["fps"]
    total_g = source_g["frame_count"]
    start_s = hms_to_s(args.start)
    end_s = hms_to_s(args.end) if args.end else None
    total_frames = int((end_s - start_s) * fps) if end_s is not None else total_g

    seek_video_input(source_g, start_s)
    seek_video_input(source_d, start_s)

    csv_path, meta_path = build_detection_output_paths(args.output)
    csv_handle, writer = open_detection_csv(csv_path)

    print(f"Flux gauche : {args.video_gauche}")
    print(f"Flux droite : {args.video_droite}")
    print(f"Plage : {args.start} -> {args.end or 'fin'}")
    print(f"Reglages detection : conf={args.conf} imgsz={args.imgsz} device={args.device}")

    frame_idx = 0
    start_wall_time = time.time()
    counts = {"gauche": 0, "droite": 0}

    try:
        while True:
            ret_g, frame_g = read_video_input_frame(source_g)
            ret_d, frame_d = read_video_input_frame(source_d)
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
        release_video_input(source_g)
        release_video_input(source_d)
        csv_handle.close()

    elapsed_total = time.time() - start_wall_time
    avg_fps = frame_idx / elapsed_total if elapsed_total > 0 else 0.0
    write_detection_summary(
        meta_path,
        video_gauche=args.video_gauche,
        video_droite=args.video_droite,
        source_g=source_g,
        source_d=source_d,
        model=args.model,
        device=str(device),
        start=args.start,
        end=args.end,
        frames_processed=frame_idx,
        avg_fps=avg_fps,
        csv_path=csv_path,
        conf_gauche=conf_gauche,
        conf_droite=conf_droite,
        imgsz_gauche=imgsz_gauche,
        imgsz_droite=imgsz_droite,
        detection_counts=counts,
    )

    print("=" * 60)
    print(f"Detection terminee - {frame_idx} frames en {elapsed_total / 60:.1f} min ({avg_fps:.1f} fps moyen)")
    print(f"CSV exporte : {csv_path}")
    print(f"Metadonnees : {meta_path}")
    print(f"Detections gardees : gauche={counts['gauche']} droite={counts['droite']}")


if __name__ == "__main__":
    main()
