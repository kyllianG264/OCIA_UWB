"""
Extraction de frames pour annotation ou constitution de dataset.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.utils import configure_utf8_stdout, ensure_dir, hms_to_s


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Extraction de frames pour annotation")
    parser.add_argument("--videos", nargs="+", required=True, help="Videos source")
    parser.add_argument("--output", default="./frames_annotation", help="Dossier de sortie")
    parser.add_argument("--interval", type=int, default=30, help="Extraire 1 frame toutes les N frames")
    parser.add_argument("--max_frames", type=int, default=300, help="Nombre max de frames par video")
    parser.add_argument("--start", default="00:00:00")
    parser.add_argument("--end", default=None)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    args = parser.parse_args()

    if args.interval <= 0:
        raise SystemExit("--interval doit etre > 0")
    if args.max_frames <= 0:
        raise SystemExit("--max_frames doit etre > 0")
    if not (1 <= args.jpeg_quality <= 100):
        raise SystemExit("--jpeg_quality doit etre compris entre 1 et 100")

    ensure_dir(args.output)
    start_s = hms_to_s(args.start)
    end_s = hms_to_s(args.end) if args.end else None

    total_extracted = 0
    processed_videos = 0

    for video_path in args.videos:
        if not os.path.isfile(video_path):
            print(f"ERREUR : video introuvable : {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERREUR : impossible d'ouvrir la video : {video_path}")
            continue

        processed_videos += 1
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps if fps > 0 else 0.0
        actual_end_s = end_s if end_s is not None else duration_s

        if actual_end_s < start_s:
            cap.release()
            print(f"ERREUR : plage invalide pour {video_path} (end < start)")
            continue

        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)

        frame_idx = int(start_s * fps)
        extracted = 0
        frames_seen = 0

        print(f"\n{'=' * 56}")
        print(f"Video : {video_path}")
        print(f"FPS : {fps:.2f} | Duree : {duration_s:.1f}s | Plage : {args.start} -> {args.end or 'fin'}")
        print(f"Intervalle : 1/{args.interval} | Max : {args.max_frames} | JPEG : {args.jpeg_quality}")

        while cap.isOpened() and extracted < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_s = frame_idx / fps if fps > 0 else 0.0
            if timestamp_s > actual_end_s:
                break

            if frames_seen % args.interval == 0:
                filename = f"{video_name}_f{frame_idx:06d}.jpg"
                output_path = os.path.join(args.output, filename)
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                extracted += 1

                if extracted == 1 or extracted % 20 == 0:
                    print(f"  {extracted} frames extraites... (t={timestamp_s:.1f}s)")

            frame_idx += 1
            frames_seen += 1

        cap.release()
        total_extracted += extracted
        print(f"  Termine : {extracted} frames -> {args.output}")

    print(f"\n{'=' * 56}")
    print(f"Videos traitees : {processed_videos}/{len(args.videos)}")
    print(f"Total extrait : {total_extracted} frames dans {args.output}")
    print("Prochaine etape : annoter ces images avec Roboflow ou LabelImg")


if __name__ == "__main__":
    main()
