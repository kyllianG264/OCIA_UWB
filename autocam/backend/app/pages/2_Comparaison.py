"""
2_Comparaison.py
Compare YOLO11-pose seul vs pipeline 2-modeles (YOLO detecteur + YOLO11-pose).
Métriques : taux de détection, frames extraites, score silhouette du clustering.
"""

import re
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import streamlit as st


def _patch_torch_classes_path() -> None:
    """Évite un warning Streamlit/PyTorch lors de l'inspection des modules."""
    try:
        import torch
    except Exception:
        return

    classes = getattr(torch, "classes", None)
    if classes is None:
        return

    try:
        _ = classes.__path__
    except Exception:
        try:
            classes.__path__ = []
        except Exception:
            pass


_patch_torch_classes_path()

sys.path.insert(0, str(Path(__file__).parent.parent))

from classifier import ClusterManager
from config import load_config
from feature_extractor import extract_biomech_features, features_to_vector
from yolo_extract import extract_keypoints as yolo_extract
from detector_yolo_pose_extract import extract_keypoints as detector_pose_extract

SUPPORTED_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

st.set_page_config(
    page_title="Comparaison — AutoCam",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Comparaison YOLO11-pose vs YOLO détecteur + YOLO11-pose")
st.caption(
    "Lance les deux pipelines sur le même dossier de clips et compare "
    "le taux de détection, la vitesse et la qualité du clustering."
)

config = load_config()

# ── Paramètres ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paramètres de comparaison")

    folder_input = st.text_input(
        "📁 Dossier des clips",
        placeholder="C:/mes_videos/seance_01",
    )

    st.subheader("🟠 YOLO11-pose (seul)")
    yolo11_model = st.selectbox(
        "Modèle YOLO11",
        [
            "yolo11n-pose.pt",
            "yolo11s-pose.pt",
            "yolo11m-pose.pt",
            "yolo26n-pose.pt",
            "yolo26s-pose.pt",
            "yolo26m-pose.pt",
        ],
        index=0,
        help="n=nano (rapide), s=small (équilibré), m=medium (précis)",
    )
    yolo11_conf = st.slider("Confiance détection (YOLO11)", 0.1, 0.95,
                            float(config["yolo"].get("conf_threshold", 0.5)),
                            step=0.05, format="%.2f", key="yolo11_conf")

    st.subheader("🟢 YOLO détecteur + YOLO11-pose")
    detector_model_name = st.selectbox(
        "Modèle de repérage",
        [
            "yolo11n.pt",
            "yolo11s.pt",
            "yolo11m.pt",
            "yolo26n.pt",
            "yolo26s.pt",
            "yolo26m.pt",
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
        ],
        index=0,
        key="cmp_detector_model",
    )
    pose_model_name = st.selectbox(
        "Modèle YOLO11 pour les keypoints",
        [
            "yolo11n-pose.pt",
            "yolo11s-pose.pt",
            "yolo11m-pose.pt",
            "yolo26n-pose.pt",
            "yolo26s-pose.pt",
            "yolo26m-pose.pt",
        ],
        index=0,
        key="cmp_pose_model",
    )
    detector_conf = st.slider("Confiance détection (repérage)", 0.1, 0.95,
                              float(config.get("detector_pose", {}).get("detector_conf_threshold", 0.35)),
                              step=0.05, format="%.2f", key="cmp_detector_conf")
    pose_conf = st.slider("Confiance détection (pose)", 0.1, 0.95,
                          float(config.get("detector_pose", {}).get("pose_conf_threshold", 0.5)),
                          step=0.05, format="%.2f", key="cmp_pose_conf")

    st.subheader("Commun")
    max_frames = st.slider("Frames max par clip", 50, 600,
                           int(config["yolo"]["max_frames"]), step=50, key="cmp_max_frames")
    min_hip_vel = st.slider("Vitesse min hanches", 0.0, 0.1,
                            float(config["yolo"].get("min_hip_velocity", 0.015)),
                            step=0.005, format="%.3f", key="cmp_min_hip_vel")
    yolo_device = st.selectbox(
        "Device",
        ["auto", "cpu", "cuda"],
        index=["auto", "cpu", "cuda"].index(config["yolo"].get("device", "auto")),
        key="cmp_device",
    )
    vis_threshold = st.slider("Seuil visibilité features", 0.1, 0.9,
                               float(config["features"]["visibility_threshold"]),
                               step=0.05, format="%.2f", key="cmp_vis_threshold")
    min_valid = st.number_input("Frames valides minimum", 3, 50,
                                int(config["features"]["min_valid_frames"]))
    min_clusters = st.number_input("Min groupes (0=auto)", 0, 20,
                                   int(config["clustering"].get("min_clusters", 0)))

    run_btn = st.button("🚀 Lancer la comparaison", type="primary",
                        use_container_width=True)

# ── Corps ────────────────────────────────────────────────
if not folder_input:
    st.info("👈 Entrez le dossier de clips dans la barre latérale et cliquez sur **Lancer la comparaison**.")
    st.stop()

folder_path = Path(folder_input)
if not folder_path.exists() or not folder_path.is_dir():
    st.error(f"Dossier introuvable : `{folder_input}`")
    st.stop()

clips = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
if not clips:
    st.warning("Aucun clip vidéo trouvé.")
    st.stop()

st.info(f"**{len(clips)} clips** trouvés.")

ft_cfg = config["features"]
cl_cfg = config["clustering"]


def _athlete_name(filename: str) -> str:
    return re.split(r"[\(\.\s]", filename)[0].strip().lower()


def _worker_count(device: str, backend: str = "yolo") -> int:
    cpu_count = os.cpu_count() or 1
    if backend == "detector_yolo_pose":
        return 1 if str(device).lower() == "cuda" else max(1, min(2, cpu_count))
    return 1 if str(device).lower() == "cuda" else max(1, min(4, cpu_count))


def run_backend(
    name: str,
    extract_fn,
    extract_kwargs: dict,
    clips: list[Path],
    progress_col,
    backend_type: str = "yolo",
) -> dict:
    features_list = []
    valid_clips = []
    valid_athletes = []
    failed = []
    total_frames = 0
    t0 = time.perf_counter()

    bar = progress_col.progress(0, text=f"{name} — initialisation…")
    worker_count = _worker_count(extract_kwargs.get("device", "auto"), backend_type)
    clip_results: list[tuple[int, Path, dict | None, int]] = []

    def _process_clip(clip: Path) -> tuple[dict | None, int]:
        try:
            kp = extract_fn(str(clip), **extract_kwargs)
        except Exception:
            return None, 0
        if kp is None:
            return None, 0
        feats = extract_biomech_features(
            kp,
            visibility_threshold=vis_threshold,
            flight_ankle_y_threshold=ft_cfg["flight_ankle_y_threshold"],
            min_valid_frames=min_valid,
        )
        if feats is None or not any(v != 0.0 for v in feats.values()):
            return None, len(kp)
        return feats, len(kp)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_process_clip, clip): (i, clip)
            for i, clip in enumerate(clips)
        }
        done = 0
        for future in as_completed(future_map):
            i, clip = future_map[future]
            try:
                feats, frame_count = future.result()
            except Exception:
                feats, frame_count = None, 0
            clip_results.append((i, clip, feats, frame_count))
            done += 1
            bar.progress(done / len(clips), text=f"{name} — {done}/{len(clips)}")

    clip_results.sort(key=lambda item: item[0])

    for _, clip, feats, frame_count in clip_results:
        if feats is None:
            failed.append(clip.name)
            continue
        total_frames += frame_count
        features_list.append(feats)
        valid_clips.append(clip.name)
        valid_athletes.append(_athlete_name(clip.name))

    elapsed = time.perf_counter() - t0
    bar.empty()

    silhouette = None
    n_clusters_found = 0
    purity_avg = None
    recall_avg = None
    group_details = []
    athlete_details = []
    unassigned = []
    all_clips_counts = Counter(_athlete_name(c.name) for c in clips)

    if len(features_list) >= 2:
        X = np.stack([features_to_vector(f, config.get("feature_weights")) for f in features_list], axis=0)
        manager = ClusterManager(
            algorithm=cl_cfg.get("algorithm", "kmeans_auto"),
            n_init=int(cl_cfg["n_init"]),
            max_iter=int(cl_cfg["max_iter"]),
            random_state=int(cl_cfg["random_state"]),
            max_auto_clusters=int(cl_cfg["max_auto_clusters"]),
            min_silhouette_score=0.0,
            manual_distance_factor=float(cl_cfg["manual_distance_factor"]),
            min_clusters=int(min_clusters),
        )
        labels, _, manual_mask, meta = manager.fit_auto(X)
        silhouette = meta["global_silhouette"]
        n_clusters_found = meta["selected_clusters"]

        groups: dict[int, list[str]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            if not manual_mask[i]:
                groups[lbl].append(valid_athletes[i])

        purity_scores = []
        for g in sorted(groups.keys()):
            mc = Counter(groups[g])
            dominant, dom_count = mc.most_common(1)[0]
            p = dom_count / len(groups[g])
            purity_scores.append(p)
            group_details.append({"label": g, "n": len(groups[g]), "purity": p,
                                   "dominant": dominant, "counter": dict(mc)})
        purity_avg = float(np.mean(purity_scores)) if purity_scores else 0.0

        ath_groups: dict[str, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            if not manual_mask[i]:
                ath_groups[valid_athletes[i]].append(lbl)

        recalls = []
        for ath in sorted(all_clips_counts.keys()):
            total_real = all_clips_counts[ath]
            grps = Counter(ath_groups.get(ath, []))
            not_assigned = total_real - len(ath_groups.get(ath, []))
            if grps:
                dom_g, dom_count = grps.most_common(1)[0]
                recall = dom_count / total_real
            else:
                dom_g, dom_count, recall = -1, 0, 0.0
            recalls.append(recall)
            athlete_details.append({"name": ath, "total_real": total_real,
                                     "dominant_group": dom_g, "dominant_count": dom_count,
                                     "recall": recall, "n_groups": len(grps),
                                     "not_assigned": not_assigned})
        recall_avg = float(np.mean(recalls)) if recalls else 0.0
        unassigned = [valid_clips[i] for i in range(len(valid_clips)) if manual_mask[i]]

    return {
        "name": name, "detected": len(valid_clips), "failed": len(failed),
        "failed_names": failed,
        "detection_rate": len(valid_clips) / len(clips) * 100,
        "avg_frames": total_frames / max(len(valid_clips), 1),
        "elapsed_s": elapsed, "silhouette": silhouette, "n_clusters": n_clusters_found,
        "purity_avg": purity_avg, "recall_avg": recall_avg,
        "group_details": group_details, "athlete_details": athlete_details,
        "unassigned": unassigned,
    }


if run_btn:
    det_cfg = config.get("detector_pose", {})

    yolo11_kwargs = dict(
        max_frames=max_frames,
        model_name=yolo11_model,
        conf_threshold=float(yolo11_conf),
        center_crop_ratio=float(config["yolo"].get("center_crop_ratio", 1.0)),
        min_body_height_ratio=float(config["yolo"].get("min_body_height_ratio", 0.0)),
        min_hip_velocity=float(min_hip_vel),
        device=yolo_device,
    )
    two_stage_kwargs = dict(
        max_frames=max_frames,
        detector_model_name=detector_model_name,
        pose_model_name=pose_model_name,
        detector_conf_threshold=float(detector_conf),
        pose_conf_threshold=float(pose_conf),
        center_crop_ratio=float(det_cfg.get("center_crop_ratio", 1.0)),
        min_body_height_ratio=float(det_cfg.get("min_body_height_ratio", 0.0)),
        min_hip_velocity=float(min_hip_vel),
        device=yolo_device,
    )

    col_yolo, col_two_stage = st.columns(2)
    col_yolo.subheader("🟠 YOLO11-pose")
    col_two_stage.subheader("🟢 YOLO détecteur + YOLO11-pose")

    st.warning(
        "⏳ Le pipeline 2-modèles fait une étape de repérage puis une étape de pose. "
        "Comptez en général plus de temps que YOLO seul."
    )

    results_yolo = run_backend(
        f"YOLO11 ({yolo11_model})", yolo_extract, yolo11_kwargs, clips, col_yolo, "yolo"
    )
    results_two_stage = run_backend(
        f"Détecteur {detector_model_name} + {pose_model_name}",
        detector_pose_extract, two_stage_kwargs, clips, col_two_stage, "detector_yolo_pose"
    )

    st.divider()
    st.subheader("📊 Résultats comparatifs")

    def _delta(a, b, higher_better=True):
        if a is None or b is None:
            return None
        d = a - b
        return f"+{d:.2f}" if (d > 0) == higher_better else f"{d:.2f}"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Taux détection YOLO11", f"{results_yolo['detection_rate']:.0f}%")
    col1.metric("Taux détection 2-modèles", f"{results_two_stage['detection_rate']:.0f}%",
                _delta(results_two_stage['detection_rate'], results_yolo['detection_rate']))
    col2.metric("Clips détectés YOLO11", f"{results_yolo['detected']} / {len(clips)}")
    col2.metric("Clips détectés 2-modèles", f"{results_two_stage['detected']} / {len(clips)}")
    col3.metric("Frames moy./clip YOLO11", f"{results_yolo['avg_frames']:.0f}")
    col3.metric("Frames moy./clip 2-modèles", f"{results_two_stage['avg_frames']:.0f}")
    col4.metric("Temps total YOLO11", f"{results_yolo['elapsed_s']:.1f} s")
    col4.metric("Temps total 2-modèles", f"{results_two_stage['elapsed_s']:.1f} s",
                _delta(results_yolo['elapsed_s'], results_two_stage['elapsed_s'], higher_better=False))
    sil_yolo = f"{results_yolo['silhouette']:.3f}" if results_yolo['silhouette'] is not None else "N/A"
    sil_two_stage = f"{results_two_stage['silhouette']:.3f}" if results_two_stage['silhouette'] is not None else "N/A"
    col5.metric(f"Silhouette YOLO11 ({results_yolo['n_clusters']} gr.)", sil_yolo)
    col5.metric(f"Silhouette 2-modèles ({results_two_stage['n_clusters']} gr.)", sil_two_stage)

    pur_yolo = f"{results_yolo['purity_avg']:.1%}" if results_yolo['purity_avg'] is not None else "N/A"
    pur_two_stage = f"{results_two_stage['purity_avg']:.1%}" if results_two_stage['purity_avg'] is not None else "N/A"
    rec_yolo = f"{results_yolo['recall_avg']:.1%}" if results_yolo['recall_avg'] is not None else "N/A"
    rec_two_stage = f"{results_two_stage['recall_avg']:.1%}" if results_two_stage['recall_avg'] is not None else "N/A"
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Pureté moy. groupes",
                  f"YOLO11 : {pur_yolo}", help="% de clips du même athlète dans chaque groupe")
    col_p1.metric("Pureté moy. groupes (2-modèles)", f"2-modèles : {pur_two_stage}")
    col_p2.metric("Recall moy. par athlète",
                  f"YOLO11 : {rec_yolo}", help="% de clips d'un athlète dans son groupe dominant")
    col_p2.metric("Recall moy. athlète (2-modèles)", f"2-modèles : {rec_two_stage}")

    st.divider()

    # Verdict automatique
    yolo_score = 0
    two_stage_score = 0
    reasons = []

    if results_yolo["detection_rate"] > results_two_stage["detection_rate"] + 5:
        yolo_score += 2
        reasons.append(f"✅ YOLO11 détecte mieux ({results_yolo['detection_rate']:.0f}% vs {results_two_stage['detection_rate']:.0f}%)")
    elif results_two_stage["detection_rate"] > results_yolo["detection_rate"] + 5:
        two_stage_score += 2
        reasons.append(f"✅ Détecteur+Pose détecte mieux ({results_two_stage['detection_rate']:.0f}% vs {results_yolo['detection_rate']:.0f}%)")

    if results_yolo["purity_avg"] is not None and results_two_stage["purity_avg"] is not None:
        if results_yolo["purity_avg"] > results_two_stage["purity_avg"] + 0.05:
            yolo_score += 2
            reasons.append(f"✅ YOLO11 trie mieux par athlète (pureté {results_yolo['purity_avg']:.0%} vs {results_two_stage['purity_avg']:.0%})")
        elif results_two_stage["purity_avg"] > results_yolo["purity_avg"] + 0.05:
            two_stage_score += 2
            reasons.append(f"✅ Détecteur+Pose trie mieux par athlète (pureté {results_two_stage['purity_avg']:.0%} vs {results_yolo['purity_avg']:.0%})")

    if results_yolo["silhouette"] and results_two_stage["silhouette"]:
        if results_yolo["silhouette"] > results_two_stage["silhouette"] + 0.02:
            yolo_score += 1
            reasons.append(f"✅ YOLO11 clustering plus net ({results_yolo['silhouette']:.3f} vs {results_two_stage['silhouette']:.3f})")
        elif results_two_stage["silhouette"] > results_yolo["silhouette"] + 0.02:
            two_stage_score += 1
            reasons.append(f"✅ Détecteur+Pose clustering plus net ({results_two_stage['silhouette']:.3f} vs {results_yolo['silhouette']:.3f})")

    if results_yolo["elapsed_s"] < results_two_stage["elapsed_s"] * 0.6:
        yolo_score += 1
        reasons.append(f"✅ YOLO11 bien plus rapide ({results_yolo['elapsed_s']:.1f}s vs {results_two_stage['elapsed_s']:.1f}s)")
    elif results_two_stage["elapsed_s"] < results_yolo["elapsed_s"] * 0.8:
        two_stage_score += 1
        reasons.append(f"✅ Détecteur+Pose plus rapide ({results_two_stage['elapsed_s']:.1f}s vs {results_yolo['elapsed_s']:.1f}s)")

    total_score = yolo_score + two_stage_score
    if two_stage_score > yolo_score:
        st.success(f"### 🏆 YOLO détecteur + YOLO11-pose est le meilleur pipeline sur ce dossier (score {two_stage_score}/{total_score})")
    elif yolo_score > two_stage_score:
        st.info(f"### 🏆 YOLO11-pose seul est suffisant sur ce dossier (score {yolo_score}/{total_score}) — le pipeline 2-modèles n'apporte pas de gain mesurable")
    else:
        st.info("### 🤝 Les deux pipelines sont équivalents — préférez YOLO11 seul pour la rapidité.")

    for r in reasons:
        st.write(r)

    # Pureté par groupe
    st.subheader("🎯 Pureté des groupes")

    def _purity_table(result: dict):
        if not result["group_details"]:
            st.info("Pas assez de clips détectés pour le clustering.")
            return
        rows = []
        for g in result["group_details"]:
            membres = ", ".join(f"{k}×{v}" for k, v in sorted(g["counter"].items()))
            icon = "✅" if g["purity"] == 1.0 else ("⚠️" if g["purity"] >= 0.6 else "❌")
            rows.append({"Groupe": g["label"], "Clips": g["n"],
                         "Pureté": f"{g['purity']:.0%} {icon}",
                         "Dominant": g["dominant"], "Composition": membres})
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if result["unassigned"]:
            st.caption(f"Non assignés : {', '.join(result['unassigned'])}")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.caption(f"**{results_yolo['name']}**")
        _purity_table(results_yolo)
    with col_g2:
        st.caption(f"**{results_two_stage['name']}**")
        _purity_table(results_two_stage)

    # Recall par athlète
    if results_yolo["athlete_details"] or results_two_stage["athlete_details"]:
        st.subheader("🏃 Recall par athlète")

        def _recall_table(result: dict):
            if not result["athlete_details"]:
                st.info("Pas de données.")
                return
            rows = []
            for a in result["athlete_details"]:
                icon = "✅" if a["recall"] >= 0.8 else ("⚠️" if a["recall"] >= 0.5 else "❌")
                dom_group = int(a["dominant_group"])
                rows.append({"Athlète": a["name"], "Total clips": a["total_real"],
                             "Détectés": a["total_real"] - a["not_assigned"],
                             "Recall": f"{a['recall']:.0%} {icon}",
                             "Groupe dom.": str(dom_group) if dom_group >= 0 else "—",
                             "N groupes": int(a["n_groups"])})
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.caption(f"**{results_yolo['name']}**")
            _recall_table(results_yolo)
        with col_r2:
            st.caption(f"**{results_two_stage['name']}**")
            _recall_table(results_two_stage)

    # Clips échoués
    with st.expander("❌ Clips non détectés"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.caption(f"**{results_yolo['name']}**")
            for n in results_yolo["failed_names"]:
                st.write(f"- {n}")
            if not results_yolo["failed_names"]:
                st.write("Aucun ✅")
        with col_f2:
            st.caption(f"**{results_two_stage['name']}**")
            for n in results_two_stage["failed_names"]:
                st.write(f"- {n}")
            if not results_two_stage["failed_names"]:
                st.write("Aucun ✅")

