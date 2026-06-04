"""
Resolution des absences et doublons de position.

Deux modes sont supportes :
  - MQTT temps reel : consomme les topics publies par lps_publisher.py
  - hors ligne : lit positions_raw.csv et produit les sorties fusionnees
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import threading
from datetime import datetime
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.utils import configure_utf8_stdout, count_csv_rows, ensure_dir, sync_file_to_python_cv_logs

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


def open_output_csv(path: str, label: str) -> tuple[object, str, bool]:
    try:
        return open(path, "w", newline="", encoding="utf-8"), path, False
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = os.path.dirname(path)
        base, ext = os.path.splitext(os.path.basename(path))
        fallback_path = os.path.join(directory, f"{base}_{timestamp}{ext}")
        print(
            f"AVERTISSEMENT : {label} verrouille ou inaccessible ({path}). "
            f"Ecriture dans {fallback_path}"
        )
        return open(fallback_path, "w", newline="", encoding="utf-8"), fallback_path, True


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="IPS Solver - resolution absences + doublons")
    parser.add_argument("--mqtt_host", default="localhost")
    parser.add_argument("--mqtt_port", type=int, default=1883)
    parser.add_argument("--output", default="./resultats")
    parser.add_argument("--input_csv", default=None, help="Mode hors ligne : chemin vers positions_raw.csv")
    parser.add_argument("--max_absence", type=int, default=10, help="Nombre de frames sans detection avant de retirer un joueur")
    parser.add_argument("--merge_dist", type=int, default=80, help="Distance (px terrain) sous laquelle deux detections sont fusionnees")
    parser.add_argument(
        "--max_jump_dist",
        type=int,
        default=120,
        help="Distance max autorisee entre deux frames pour un meme joueur avant de considerer la detection comme un teleport",
    )
    args = parser.parse_args()

    ensure_dir(args.output)

    frame_buffer = defaultdict(lambda: {"gauche": [], "droite": []})
    lock = threading.Lock()
    last_positions: dict[str, dict] = {}

    csv_path = os.path.join(args.output, "positions_solved.csv")
    csv_compat_path = os.path.join(args.output, "positions_merged.csv")
    csv_handle, csv_path, _ = open_output_csv(csv_path, "positions_solved.csv")
    csv_compat_handle, csv_compat_path, _ = open_output_csv(csv_compat_path, "positions_merged.csv")
    writer = csv.writer(csv_handle)
    writer_compat = csv.writer(csv_compat_handle)
    writer.writerow(["frame", "timestamp_s", "timestamp_unix", "player_id", "X", "Y", "status", "on_terrain"])
    writer_compat.writerow(["frame", "timestamp_s", "timestamp_unix", "player_id", "X", "Y", "demi_terrain", "on_terrain"])

    def dist(p1: dict, p2: dict) -> float:
        return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)

    def stabilize_players(frame_idx: int, resolved: list[dict]) -> list[dict]:
        stabilized = []
        seen_ids = set()

        for player in resolved:
            player_id = player["player_id"]
            seen_ids.add(player_id)
            last_position = last_positions.get(player_id)

            if last_position is not None:
                frames_delta = max(1, frame_idx - last_position["frame"])
                jump_dist = math.sqrt((player["x"] - last_position["x"]) ** 2 + (player["y"] - last_position["y"]) ** 2)
                max_allowed = args.max_jump_dist * frames_delta
                if jump_dist > max_allowed:
                    stabilized.append(
                        {
                            "player_id": player_id,
                            "x": last_position["x"],
                            "y": last_position["y"],
                            "status": "held",
                            "on_terrain": last_position.get("on_terrain", True),
                        }
                    )
                    last_positions[player_id] = {
                        "x": last_position["x"],
                        "y": last_position["y"],
                        "frame": frame_idx,
                        "on_terrain": last_position.get("on_terrain", True),
                    }
                    continue

            stabilized.append(player)
            last_positions[player_id] = {
                "x": player["x"],
                "y": player["y"],
                "frame": frame_idx,
                "on_terrain": player.get("on_terrain", True),
            }

        for player_id, last_position in list(last_positions.items()):
            if player_id in seen_ids:
                continue
            frames_absent = frame_idx - last_position["frame"]
            if frames_absent <= args.max_absence:
                stabilized.append(
                    {
                        "player_id": player_id,
                        "x": last_position["x"],
                        "y": last_position["y"],
                        "status": "interpolated",
                        "on_terrain": last_position.get("on_terrain", True),
                    }
                )
            else:
                del last_positions[player_id]

        return stabilized

    def resolve_frame(frame_idx: int, dets_g: list, dets_d: list) -> list[dict]:
        resolved = []
        used_d = set()

        for det_g in dets_g:
            merged = False
            for index, det_d in enumerate(dets_d):
                if index in used_d:
                    continue
                if dist(det_g, det_d) >= args.merge_dist:
                    continue

                resolved.append(
                    {
                        "player_id": f"G{det_g['id']}",
                        "x": (det_g["x"] + det_d["x"]) // 2,
                        "y": (det_g["y"] + det_d["y"]) // 2,
                        "status": "merged",
                        "on_terrain": det_g.get("on_terrain", True) or det_d.get("on_terrain", True),
                    }
                )
                used_d.add(index)
                merged = True
                break

            if not merged:
                resolved.append(
                    {
                        "player_id": f"G{det_g['id']}",
                        "x": det_g["x"],
                        "y": det_g["y"],
                        "status": "ok",
                        "on_terrain": det_g.get("on_terrain", True),
                    }
                )

        for index, det_d in enumerate(dets_d):
            if index in used_d:
                continue
            resolved.append(
                {
                    "player_id": f"D{det_d['id']}",
                    "x": det_d["x"],
                    "y": det_d["y"],
                    "status": "ok",
                    "on_terrain": det_d.get("on_terrain", True),
                }
            )

        return stabilize_players(frame_idx, resolved)

    def persist_resolved(frame_idx: int, timestamp_s: float, timestamp_unix: int | None, resolved: list[dict], mqtt_client=None) -> None:
        for player in resolved:
            side = "gauche" if str(player["player_id"]).startswith("G") else "droite"
            writer.writerow(
                [
                    frame_idx,
                    f"{timestamp_s:.3f}",
                    timestamp_unix or "",
                    player["player_id"],
                    player["x"],
                    player["y"],
                    player["status"],
                    1 if player.get("on_terrain", True) else 0,
                ]
            )
            writer_compat.writerow(
                [
                    frame_idx,
                    f"{timestamp_s:.3f}",
                    timestamp_unix or "",
                    player["player_id"],
                    player["x"],
                    player["y"],
                    side,
                    1 if player.get("on_terrain", True) else 0,
                ]
            )
            if mqtt_client is not None:
                mqtt_client.publish(
                    "neptuvision/positions/solved",
                    json.dumps(
                        {
                            "frame": frame_idx,
                            "t": timestamp_s,
                            "timestamp_unix": timestamp_unix,
                            "player_id": player["player_id"],
                            "x": player["x"],
                            "y": player["y"],
                            "status": player["status"],
                            "on_terrain": player.get("on_terrain", True),
                        }
                    ),
                )

        if frame_idx % 100 == 0:
            n_ok = sum(1 for player in resolved if player["status"] == "ok")
            n_merged = sum(1 for player in resolved if player["status"] == "merged")
            n_interp = sum(1 for player in resolved if player["status"] == "interpolated")
            n_held = sum(1 for player in resolved if player["status"] == "held")
            print(
                f"  Frame {frame_idx:6d} | t={timestamp_s:7.1f}s | {len(resolved)} joueurs "
                f"(ok={n_ok} fusionnes={n_merged} interpoles={n_interp} retenus={n_held})"
            )

    def process_frame(
        frame_idx: int,
        timestamp_s: float,
        timestamp_unix: int | None,
        detections_g: list,
        detections_d: list,
        mqtt_client=None,
    ) -> None:
        resolved = resolve_frame(frame_idx, detections_g, detections_d)
        persist_resolved(frame_idx, timestamp_s, timestamp_unix, resolved, mqtt_client=mqtt_client)

    def on_position(client, userdata, msg):
        data = json.loads(msg.payload)
        frame_idx = data["frame"]
        cam = data["cam"]
        detection = {
            "id": data["id"],
            "x": data["x"],
            "y": data["y"],
            "on_terrain": bool(data.get("on_terrain", True)),
        }
        with lock:
            frame_buffer[frame_idx][cam].append(detection)

    def on_frame_end(client, userdata, msg):
        data = json.loads(msg.payload)
        frame_idx = data["frame"]
        timestamp_s = data["t"]
        timestamp_unix = data.get("timestamp_unix_gauche") or data.get("timestamp_unix_droite")
        with lock:
            buffered = frame_buffer.pop(frame_idx, {"gauche": [], "droite": []})
        process_frame(frame_idx, timestamp_s, timestamp_unix, buffered["gauche"], buffered["droite"], mqtt_client=client)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connecte au broker MQTT")
            client.subscribe("neptuvision/positions/raw")
            client.subscribe("neptuvision/frame_end")
        else:
            print(f"Echec connexion MQTT (code {rc})")

    def on_disconnect(client, userdata, rc):
        print("Deconnecte du broker MQTT")

    def run_offline(input_csv: str) -> None:
        if not os.path.isfile(input_csv):
            raise FileNotFoundError(f"CSV brut introuvable : {input_csv}")

        grouped: dict[int, dict] = defaultdict(lambda: {"gauche": [], "droite": [], "timestamp_s": 0.0, "timestamp_unix": None})
        with open(input_csv, encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                frame_idx = int(row["frame"])
                side = row.get("cam", "").strip().lower()
                if side not in ("gauche", "droite"):
                    continue
                grouped[frame_idx][side].append(
                    {
                        "id": int(row["player_id"]),
                        "x": int(row["X"]),
                        "y": int(row["Y"]),
                        "on_terrain": str(row.get("on_terrain", "1")).strip() not in ("0", "false", "False", ""),
                    }
                )
                grouped[frame_idx]["timestamp_s"] = float(row["timestamp_s"])
                raw_unix = str(row.get("timestamp_unix", "")).strip()
                if raw_unix:
                    grouped[frame_idx]["timestamp_unix"] = int(float(raw_unix))

        total_frames = len(grouped)
        total_rows = count_csv_rows(input_csv)
        print(f"Mode hors ligne - {input_csv}")
        print(f"Detections lues : {total_rows} | Frames : {total_frames}")

        for frame_idx in sorted(grouped):
            payload = grouped[frame_idx]
            process_frame(
                frame_idx,
                payload["timestamp_s"],
                payload["timestamp_unix"],
                payload["gauche"],
                payload["droite"],
                mqtt_client=None,
            )

    def run_mqtt() -> None:
        if mqtt is None:
            raise RuntimeError("paho-mqtt non installe. Utiliser --input_csv ou installer la dependance.")

        print(f"IPS Solver - connexion a {args.mqtt_host}:{args.mqtt_port}")
        print(
            f"Parametres : max_absence={args.max_absence} frames | "
            f"merge_dist={args.merge_dist} px | max_jump_dist={args.max_jump_dist} px/frame"
        )
        print("En attente de donnees... (Ctrl+C pour arreter)")
        print("=" * 60)

        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.message_callback_add("neptuvision/positions/raw", on_position)
        client.message_callback_add("neptuvision/frame_end", on_frame_end)
        client.connect(args.mqtt_host, args.mqtt_port, 60)
        client.loop_forever()

    print(f"Sortie CSV : {csv_path}")
    print(f"Alias compat : {csv_compat_path}")

    try:
        if args.input_csv:
            run_offline(args.input_csv)
        else:
            run_mqtt()
    except KeyboardInterrupt:
        print("\nArret demande.")
    finally:
        csv_handle.close()
        csv_compat_handle.close()
        mirrored_solved_path = sync_file_to_python_cv_logs(csv_path, "positions_solved.csv")
        mirrored_merged_path = sync_file_to_python_cv_logs(csv_compat_path, "positions_merged.csv")
        print(f"CSV sauvegarde : {csv_path}")
        print(f"CSV compatibilite : {csv_compat_path}")
        if mirrored_solved_path:
            print(f"CSV solved copie vers python/CV Logs : {mirrored_solved_path}")
        if mirrored_merged_path:
            print(f"CSV merged copie vers python/CV Logs : {mirrored_merged_path}")


if __name__ == "__main__":
    main()
