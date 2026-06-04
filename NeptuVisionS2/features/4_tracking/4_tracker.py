"""
Tracking sur les positions projetees, avec export CSV/MQTT compatible solver.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.utils import configure_utf8_stdout, ensure_dir, require_file, sync_file_to_python_cv_logs

try:
    import paho.mqtt.client as mqtt

    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False


def connect_mqtt(host: str, port: int):
    if not HAS_MQTT:
        return None
    try:
        client = mqtt.Client()
        client.connect(host, port, 60)
        client.loop_start()
        print(f"MQTT connecte : {host}:{port}")
        return client
    except Exception as exc:
        print(f"MQTT non disponible ({exc}) - mode CSV uniquement")
        return None


def publish(client, topic: str, payload: dict) -> None:
    if client is not None:
        client.publish(topic, json.dumps(payload))


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Tracking sur positions projetees")
    parser.add_argument("--input_csv", required=True, help="CSV de positions projetees")
    parser.add_argument("--output", default="./resultats", help="Dossier de sortie CSV")
    parser.add_argument("--mqtt_host", default="localhost")
    parser.add_argument("--mqtt_port", type=int, default=1883)
    parser.add_argument("--max_track_distance", type=float, default=120.0, help="Distance max entre deux frames pour conserver un track")
    parser.add_argument("--max_idle_frames", type=int, default=15, help="Nombre max de frames sans match avant suppression du track")
    args = parser.parse_args()

    ensure_dir(args.output)
    require_file(args.input_csv, "positions projetees CSV")
    mqtt_client = connect_mqtt(args.mqtt_host, args.mqtt_port)

    grouped = defaultdict(lambda: {"gauche": [], "droite": [], "timestamp_s": 0.0, "timestamp_unix": None})
    with open(args.input_csv, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_idx = int(row["frame"])
            camera = row["cam"].strip().lower()
            if camera not in ("gauche", "droite"):
                continue
            grouped[frame_idx][camera].append(
                {
                    "x": int(float(row["X"])),
                    "y": int(float(row["Y"])),
                    "on_terrain": str(row.get("on_terrain", "1")).strip() not in ("0", "false", "False", ""),
                    "detection_id": row.get("detection_id", ""),
                    "source_foot_x": int(float(row.get("source_foot_x", 0) or 0)),
                    "source_foot_y": int(float(row.get("source_foot_y", 0) or 0)),
                }
            )
            grouped[frame_idx]["timestamp_s"] = float(row["timestamp_s"])
            raw_unix = str(row.get("timestamp_unix", "")).strip()
            if raw_unix:
                grouped[frame_idx]["timestamp_unix"] = int(float(raw_unix))

    csv_path = os.path.join(args.output, "positions_raw.csv")
    assignments_path = os.path.join(args.output, "tracking_assignments.csv")
    meta_path = os.path.join(args.output, "run_metadata.json")
    csv_handle = open(csv_path, "w", newline="", encoding="utf-8")
    assignments_handle = open(assignments_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_handle)
    assignments_writer = csv.writer(assignments_handle)
    writer.writerow(["frame", "timestamp_s", "timestamp_unix", "player_id", "X", "Y", "cam", "on_terrain"])
    assignments_writer.writerow(
        [
            "frame",
            "timestamp_s",
            "timestamp_unix",
            "cam",
            "track_id",
            "detection_id",
            "source_foot_x",
            "source_foot_y",
            "X",
            "Y",
            "on_terrain",
        ]
    )

    next_track_id = {"gauche": 1, "droite": 1}
    active_tracks = {"gauche": {}, "droite": {}}
    output_counts = {"gauche": 0, "droite": 0}

    try:
        for frame_idx in sorted(grouped):
            payload = grouped[frame_idx]
            timestamp_s = payload["timestamp_s"]
            timestamp_unix = payload["timestamp_unix"]

            for camera in ("gauche", "droite"):
                detections = payload[camera]
                tracks = active_tracks[camera]
                candidates = []

                for track_id, track in tracks.items():
                    frame_gap = frame_idx - track["last_frame"]
                    if frame_gap <= 0 or frame_gap > args.max_idle_frames:
                        continue
                    max_distance = args.max_track_distance * max(1, frame_gap)
                    for detection_index, detection in enumerate(detections):
                        dist = distance((track["x"], track["y"]), (detection["x"], detection["y"]))
                        if dist <= max_distance:
                            candidates.append((dist, track_id, detection_index))

                assignments = {}
                used_tracks = set()
                used_detections = set()
                for dist, track_id, detection_index in sorted(candidates):
                    if track_id in used_tracks or detection_index in used_detections:
                        continue
                    assignments[detection_index] = track_id
                    used_tracks.add(track_id)
                    used_detections.add(detection_index)

                seen_tracks = set()
                for detection_index, detection in enumerate(detections):
                    track_id = assignments.get(detection_index)
                    if track_id is None:
                        track_id = next_track_id[camera]
                        next_track_id[camera] += 1

                    tracks[track_id] = {
                        "x": detection["x"],
                        "y": detection["y"],
                        "last_frame": frame_idx,
                        "on_terrain": detection["on_terrain"],
                    }
                    seen_tracks.add(track_id)

                    publish(
                        mqtt_client,
                        "neptuvision/positions/raw",
                        {
                            "frame": frame_idx,
                            "t": round(timestamp_s, 3),
                            "timestamp_unix": timestamp_unix,
                            "cam": camera,
                            "id": int(track_id),
                            "x": detection["x"],
                            "y": detection["y"],
                            "on_terrain": detection["on_terrain"],
                        },
                    )
                    writer.writerow(
                        [
                            frame_idx,
                            f"{timestamp_s:.3f}",
                            timestamp_unix or "",
                            track_id,
                            detection["x"],
                            detection["y"],
                            camera,
                            1 if detection["on_terrain"] else 0,
                        ]
                    )
                    assignments_writer.writerow(
                        [
                            frame_idx,
                            f"{timestamp_s:.3f}",
                            timestamp_unix or "",
                            camera,
                            track_id,
                            detection["detection_id"],
                            detection["source_foot_x"],
                            detection["source_foot_y"],
                            detection["x"],
                            detection["y"],
                            1 if detection["on_terrain"] else 0,
                        ]
                    )
                    output_counts[camera] += 1

                for track_id in list(tracks.keys()):
                    if track_id in seen_tracks:
                        continue
                    if frame_idx - tracks[track_id]["last_frame"] > args.max_idle_frames:
                        del tracks[track_id]

            publish(
                mqtt_client,
                "neptuvision/frame_end",
                {
                    "frame": frame_idx,
                    "t": round(timestamp_s, 3),
                    "timestamp_unix_gauche": timestamp_unix,
                    "timestamp_unix_droite": timestamp_unix,
                },
            )

            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx:6d} | t={timestamp_s:7.1f}s")
    finally:
        csv_handle.close()
        assignments_handle.close()
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()

    summary = {
        "input_csv": args.input_csv,
        "csv_path": csv_path,
        "assignments_path": assignments_path,
        "mqtt_host": args.mqtt_host,
        "mqtt_port": args.mqtt_port,
        "max_track_distance": args.max_track_distance,
        "max_idle_frames": args.max_idle_frames,
        "output_counts": output_counts,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    mirrored_csv_path = sync_file_to_python_cv_logs(csv_path, "positions_raw.csv")
    mirrored_meta_path = sync_file_to_python_cv_logs(meta_path, "run_metadata.json")

    print("=" * 60)
    print("Tracking termine")
    print(f"CSV exporte : {csv_path}")
    print(f"Assignations exportees : {assignments_path}")
    print(f"Metadonnees : {meta_path}")
    if mirrored_csv_path:
        print(f"CSV copie vers python/CV Logs : {mirrored_csv_path}")
    if mirrored_meta_path:
        print(f"Metadonnees copiees vers python/CV Logs : {mirrored_meta_path}")
    print(f"Points suivis : gauche={output_counts['gauche']} droite={output_counts['droite']}")


if __name__ == "__main__":
    main()
