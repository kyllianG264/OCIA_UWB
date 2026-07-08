"""
streamlit_app.py
Interface Streamlit pour le tri automatique de clips de saut à la perche.

Lancement: streamlit run app/streamlit_app.py
"""

import json
import os
import re
import shutil
import sys
import hashlib
from datetime import datetime, timedelta
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

# Ajouter le dossier app/ au path pour les imports relatifs
sys.path.insert(0, str(Path(__file__).parent))

from classifier import ClusterManager
from config import load_config
from feature_extractor import extract_biomech_features, features_to_vector
from yolo_extract import extract_keypoints as _yolo_extract_keypoints, export_pose_preview as _yolo_export_pose_preview
from detector_yolo_pose_extract import extract_keypoints as _detector_pose_extract_keypoints

# --- Constantes ---
SUPPORTED_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "features_cache.json"
MODEL_FILE = DATA_DIR / "model.pkl"
LABELS_FILE = DATA_DIR / "athlete_labels.json"
SORTED_OUTPUT_DIRNAME = "tri_auto"
MANUAL_REVIEW_DIRNAME = "a trier a la main"
POSE_PREVIEW_DIR = DATA_DIR / "pose_previews"

PREVIEW_CACHE_MAX_FILES = 20
PREVIEW_CACHE_MAX_TOTAL_MB = 1500
PREVIEW_CACHE_MAX_AGE_DAYS = 7

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Charger la configuration (relit le JSON à chaque rerun Streamlit)
config = load_config()


# --- Cache features ---

def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def sanitize_folder_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {" ", "-", "_"} else "_" for char in name)
    cleaned = cleaned.strip().rstrip(".")
    return cleaned or "sans_nom"


def cluster_display_name(cluster_id: int, athlete_names: dict) -> str:
    custom_name = athlete_names.get(str(cluster_id), "").strip()
    return custom_name or f"Groupe {cluster_id + 1}"


def export_sorted_clips(
    valid_clips: list[Path],
    labels: list[int],
    manual_mask: list[bool],
    failed_clips: list[Path],
    athlete_names: dict,
    output_dir: Path,
) -> dict:
    warnings: list[str] = []
    skipped_files: list[str] = []
    effective_output_dir = output_dir

    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
        except (PermissionError, OSError) as exc:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            effective_output_dir = output_dir.parent / f"{output_dir.name}_{ts}"
            warnings.append(
                "Le dossier d'export principal est verrouillé par un autre processus "
                f"({exc}). Export redirigé vers : {effective_output_dir}"
            )

    effective_output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    for clip, label, is_manual in zip(valid_clips, labels, manual_mask):
        if is_manual:
            folder_name = MANUAL_REVIEW_DIRNAME
        else:
            folder_name = sanitize_folder_name(cluster_display_name(label, athlete_names))

        target_dir = effective_output_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(clip, target_dir / clip.name)
            counts[folder_name] = counts.get(folder_name, 0) + 1
        except (PermissionError, OSError) as exc:
            skipped_files.append(clip.name)
            warnings.append(f"Copie impossible pour {clip.name}: {exc}")

    if failed_clips:
        manual_dir = effective_output_dir / MANUAL_REVIEW_DIRNAME
        manual_dir.mkdir(parents=True, exist_ok=True)
        for clip in failed_clips:
            try:
                shutil.copy2(clip, manual_dir / clip.name)
                counts[MANUAL_REVIEW_DIRNAME] = counts.get(MANUAL_REVIEW_DIRNAME, 0) + 1
            except (PermissionError, OSError) as exc:
                skipped_files.append(clip.name)
                warnings.append(f"Copie impossible pour {clip.name}: {exc}")

    return {
        "export_dir": str(effective_output_dir),
        "requested_export_dir": str(output_dir),
        "counts": counts,
        "manual_count": counts.get(MANUAL_REVIEW_DIRNAME, 0),
        "warnings": warnings,
        "skipped_files": skipped_files,
    }


# --- Traitement d'un clip ---

def process_clip(video_path: Path, cache: dict, use_cache: bool) -> dict | None:
    """
    Extrait les features d'un clip. Utilise le cache si disponible.
    Retourne None si le clip ne peut pas être analysé.
    """
    # Clé de cache incluant le backend pour éviter les collisions
    backend = config.get("pose_backend", "detector_yolo_pose")
    key = f"{backend}|{video_path.name}"
    if use_cache and key in cache:
        return cache[key]

    if backend == "detector_yolo_pose":
        det_cfg = config["detector_pose"]
        keypoints = _detector_pose_extract_keypoints(
            str(video_path),
            max_frames=det_cfg.get("max_frames", 300),
            detector_model_name=det_cfg.get("detector_model_name", "yolo11n.pt"),
            pose_model_name=det_cfg.get("pose_model_name", "yolo11n-pose.pt"),
            detector_conf_threshold=det_cfg.get("detector_conf_threshold", 0.35),
            pose_conf_threshold=det_cfg.get("pose_conf_threshold", 0.5),
            center_crop_ratio=det_cfg.get("center_crop_ratio", 1.0),
            min_body_height_ratio=det_cfg.get("min_body_height_ratio", 0.0),
            min_hip_velocity=det_cfg.get("min_hip_velocity", 0.0),
            device=det_cfg.get("device", "auto"),
        )
    else:
        yolo_cfg = config["yolo"]
        keypoints = _yolo_extract_keypoints(
            str(video_path),
            max_frames=yolo_cfg["max_frames"],
            model_name=yolo_cfg.get("model_name", "yolov8n-pose.pt"),
            conf_threshold=yolo_cfg.get("conf_threshold", 0.5),
            center_crop_ratio=yolo_cfg.get("center_crop_ratio", 1.0),
            min_body_height_ratio=yolo_cfg.get("min_body_height_ratio", 0.0),
            min_hip_velocity=yolo_cfg.get("min_hip_velocity", 0.0),
            device=yolo_cfg.get("device", "auto"),
        )
    if keypoints is None:
        return None

    ft_cfg = config["features"]
    features = extract_biomech_features(
        keypoints,
        visibility_threshold=ft_cfg["visibility_threshold"],
        flight_ankle_y_threshold=ft_cfg["flight_ankle_y_threshold"],
        min_valid_frames=ft_cfg["min_valid_frames"],
    )
    # Stocker avec clé incluant le backend
    cache[key] = features
    return features


def _process_clip_without_cache(video_path: Path) -> dict | None:
    """Traite un clip sans écrire dans le cache; utile pour le parallélisme."""
    return process_clip(video_path, cache={}, use_cache=False)


def _clip_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    backend = config.get("pose_backend", "detector_yolo_pose")
    if backend == "detector_yolo_pose":
        device = config.get("detector_pose", {}).get("device", "auto")
        if str(device).lower() == "cuda":
            return 1
        return max(1, min(2, cpu_count))
    if config.get("yolo", {}).get("device", "auto") == "cuda":
        return max(1, min(2, cpu_count))
    return max(1, min(4, cpu_count))


def _build_preview_output_path(clip_path: Path, yolo_cfg: dict) -> Path:
    file_sig = f"{clip_path.resolve()}|{clip_path.stat().st_mtime_ns}|{clip_path.stat().st_size}"
    cfg_sig = (
        f"{yolo_cfg.get('model_name', '')}|{yolo_cfg.get('conf_threshold', 0.5)}|"
        f"{yolo_cfg.get('center_crop_ratio', 1.0)}|{yolo_cfg.get('device', 'auto')}|"
        "preview_h264_v1"
    )
    digest = hashlib.sha1(f"{file_sig}|{cfg_sig}".encode("utf-8")).hexdigest()[:10]
    safe_stem = sanitize_folder_name(clip_path.stem)
    return POSE_PREVIEW_DIR / f"{safe_stem}_{digest}.mp4"


def _is_valid_preview_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1024


def _preview_cache_stats(cache_dir: Path) -> tuple[int, int]:
    if not cache_dir.exists():
        return 0, 0
    files = [p for p in cache_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".jpg", ".jpeg"}]
    total_bytes = sum(p.stat().st_size for p in files)
    return len(files), total_bytes


def cleanup_pose_preview_cache(
    cache_dir: Path,
    max_files: int = PREVIEW_CACHE_MAX_FILES,
    max_total_mb: int = PREVIEW_CACHE_MAX_TOTAL_MB,
    max_age_days: int = PREVIEW_CACHE_MAX_AGE_DAYS,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    removed = 0
    removed_bytes = 0

    files = [p for p in cache_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".jpg", ".jpeg"}]
    if not files:
        return {"removed": 0, "removed_bytes": 0}

    now = datetime.now()
    expiry = now - timedelta(days=max_age_days)

    # 1) Supprimer les fichiers trop anciens
    for p in list(files):
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        if mtime < expiry:
            size = p.stat().st_size
            p.unlink(missing_ok=True)
            removed += 1
            removed_bytes += size

    files = [p for p in cache_dir.glob("*.mp4") if p.is_file()]
    max_total_bytes = int(max_total_mb * 1024 * 1024)

    # 2) Si dépassement, supprimer les plus anciens
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    total_bytes = sum(p.stat().st_size for p in files)

    while len(files) > max_files or total_bytes > max_total_bytes:
        victim = files.pop()  # plus ancien
        size = victim.stat().st_size
        victim.unlink(missing_ok=True)
        removed += 1
        removed_bytes += size
        total_bytes -= size

    return {"removed": removed, "removed_bytes": removed_bytes}


# --- UI ---

st.set_page_config(
    page_title="AutoCam — Tri Athlètes",
    page_icon="🏃",
    layout="wide",
)

st.title("🏃 AutoCam — Tri automatique de clips")
st.caption("Analyse les clips de saut à la perche et les groupe par athlète via signature biomécanique.")

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Paramètres")

    folder_input = st.text_input(
        "📁 Dossier des clips",
        placeholder="C:/mes_videos/seance_01",
        help="Chemin absolu vers le dossier contenant les clips vidéo.",
    )

    st.caption("Le nombre de groupes est détecté automatiquement à partir des similarités biomécaniques.")

    use_cache = st.checkbox(
        "⚡ Utiliser le cache features",
        value=True,
        help="Évite de re-traiter les clips déjà analysés. Désactiver si les clips ont changé.",
    )

    st.divider()

    analyze_btn = st.button("🔍 Analyser", type="primary", use_container_width=True)

    if MODEL_FILE.exists():
        st.success("Modèle sauvegardé disponible")
        if st.button("🗑️ Effacer le modèle", use_container_width=True):
            MODEL_FILE.unlink()
            if LABELS_FILE.exists():
                LABELS_FILE.unlink()
            st.rerun()

    st.divider()
    preview_count, preview_total_bytes = _preview_cache_stats(POSE_PREVIEW_DIR)
    st.caption(
        f"Cache aperçus: {preview_count} fichier(s), {preview_total_bytes / (1024 * 1024):.1f} MB"
    )
    if st.button("🧹 Nettoyer cache aperçus", use_container_width=True):
        cleanup = cleanup_pose_preview_cache(POSE_PREVIEW_DIR, max_files=0, max_total_mb=0, max_age_days=0)
        st.success(
            f"Cache nettoyé: {cleanup['removed']} fichier(s) supprimé(s), {cleanup['removed_bytes'] / (1024 * 1024):.1f} MB libérés."
        )
        st.rerun()


# ---- Corps principal ----

# Vérification dossier
if not folder_input:
    st.info("👈 Entrez le chemin du dossier de clips dans la barre latérale, puis cliquez sur **Analyser**.")
    st.stop()

folder_path = Path(folder_input)
if not folder_path.exists() or not folder_path.is_dir():
    st.error(f"Dossier introuvable : `{folder_input}`")
    st.stop()

clips = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
if not clips:
    st.warning(f"Aucun clip vidéo trouvé dans `{folder_input}` (formats supportés : mp4, mov, avi, mkv).")
    st.stop()

st.info(f"**{len(clips)} clips** trouvés dans `{folder_path.name}`")

# Nettoyage automatique de sécurité pour éviter l'accumulation infinie du cache previews.
cleanup_pose_preview_cache(POSE_PREVIEW_DIR)


st.subheader("🎯 Aperçu pose estimée")
if config.get("pose_backend", "mediapipe") != "yolo":
    st.info("L'aperçu annoté est disponible avec le backend YOLO. Active YOLO dans Paramètres.")
else:
    preview_candidates = clips[: min(len(clips), 12)]
    default_preview = [p.name for p in preview_candidates[: min(2, len(preview_candidates))]]

    selected_preview_names = st.multiselect(
        "Choisis 1 ou 2 clips à visualiser avec les points YOLO",
        options=[p.name for p in preview_candidates],
        default=default_preview,
        help="Le système génère une copie annotée (squelette + points) pour contrôle visuel.",
    )

    if "pose_preview_outputs" not in st.session_state:
        st.session_state["pose_preview_outputs"] = []

    if st.button("👁️ Générer l'aperçu pose", use_container_width=False):
        if not selected_preview_names:
            st.warning("Sélectionne au moins un clip.")
        else:
            selected_preview_names = selected_preview_names[:2]

            yolo_cfg = config.get("yolo", {})
            generated: list[dict[str, str]] = []
            failed: list[str] = []

            with st.spinner("Génération des vidéos annotées en cours..."):
                for clip in preview_candidates:
                    if clip.name not in selected_preview_names:
                        continue

                    output_path = _build_preview_output_path(clip, yolo_cfg)
                    preview_path: str | None = None

                    try:
                        if _is_valid_preview_file(output_path):
                            preview_path = str(output_path)
                        else:
                            preview_path = _yolo_export_pose_preview(
                                video_path=str(clip),
                                output_path=str(output_path),
                                max_frames=min(int(yolo_cfg.get("max_frames", 300)), 300),
                                model_name=yolo_cfg.get("model_name", "yolov8n-pose.pt"),
                                conf_threshold=float(yolo_cfg.get("conf_threshold", 0.5)),
                                center_crop_ratio=float(yolo_cfg.get("center_crop_ratio", 1.0)),
                                device=str(yolo_cfg.get("device", "auto")),
                                draw_conf_threshold=0.3,
                            )

                        # Fallback CPU si échec sur device configuré
                        if preview_path is None and str(yolo_cfg.get("device", "auto")).lower() != "cpu":
                            preview_path = _yolo_export_pose_preview(
                                video_path=str(clip),
                                output_path=str(output_path),
                                max_frames=min(int(yolo_cfg.get("max_frames", 300)), 300),
                                model_name=yolo_cfg.get("model_name", "yolov8n-pose.pt"),
                                conf_threshold=float(yolo_cfg.get("conf_threshold", 0.5)),
                                center_crop_ratio=float(yolo_cfg.get("center_crop_ratio", 1.0)),
                                device="cpu",
                                draw_conf_threshold=0.3,
                            )
                    except Exception as exc:
                        failed.append(f"{clip.name}: {exc}")

                    if preview_path is not None:
                        generated.append({"clip": str(clip), "preview": preview_path})
                    else:
                        failed.append(f"{clip.name}: aperçu non généré")

            st.session_state["pose_preview_outputs"] = generated
            st.session_state["pose_preview_failed"] = failed

    preview_outputs = st.session_state.get("pose_preview_outputs", [])
    preview_failed = st.session_state.get("pose_preview_failed", [])

    # Filtre les entrées dont le fichier a disparu
    preview_outputs = [p for p in preview_outputs if Path(p["preview"]).exists()]
    st.session_state["pose_preview_outputs"] = preview_outputs

    if preview_outputs:
        cols = st.columns(len(preview_outputs))
        for col, item in zip(cols, preview_outputs):
            with col:
                st.video(item["preview"])
                st.caption(f"Source : {Path(item['clip']).name}")
        st.caption(
            "Si les points sont mal placés, ajuste le modèle YOLO et le seuil de confiance dans Paramètres, puis régénère."
        )

    if preview_failed:
        with st.expander("⚠️ Détails des échecs d'aperçu"):
            for msg in preview_failed:
                st.write(f"- {msg}")

    if not preview_outputs and not preview_failed:
        st.caption("Sélectionne 1 ou 2 clips puis clique sur 'Générer l'aperçu pose'.")


# ---- Analyse ----
if analyze_btn:
    cache = load_cache()
    features_list = []
    valid_clips = []
    failed_clips = []

    progress_bar = st.progress(0, text="Initialisation...")
    status_text = st.empty()

    cached_results: list[tuple[int, Path, dict | None]] = []
    pending_jobs: list[tuple[int, Path]] = []

    for index, clip in enumerate(clips):
        key = clip.name
        if use_cache and key in cache:
            cached_results.append((index, clip, cache[key]))
        else:
            pending_jobs.append((index, clip))

    total_jobs = len(clips)

    if pending_jobs:
        worker_count = _clip_worker_count()
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_process_clip_without_cache, clip): (index, clip)
                for index, clip in pending_jobs
            }

            for future in as_completed(future_map):
                index, clip = future_map[future]
                try:
                    features = future.result()
                except Exception:
                    features = None
                cached_results.append((index, clip, features))
                if features is not None and use_cache:
                    cache[clip.name] = features

                done_jobs = len(cached_results)
                status_text.text(f"Analyse en parallèle : {done_jobs}/{total_jobs}")
                progress_bar.progress(done_jobs / total_jobs, text=f"{done_jobs}/{total_jobs}")
    else:
        progress_bar.progress(1.0, text=f"{total_jobs}/{total_jobs}")

    cached_results.sort(key=lambda item: item[0])

    for _, clip, features in cached_results:
        if features is not None and any(v != 0.0 for v in features.values()):
            features_list.append(features)
            valid_clips.append(clip)
        else:
            failed_clips.append(clip)

    save_cache(cache)
    progress_bar.empty()
    status_text.empty()

    if failed_clips:
        with st.expander(f"⚠️ {len(failed_clips)} clip(s) non analysés (pose non détectée)"):
            for c in failed_clips:
                st.write(f"- {c.name}")

    if not valid_clips:
        st.error("Aucun clip exploitable n'a été détecté. Vérifiez les vidéos ou les paramètres MediaPipe.")
        st.stop()

    existing_labels: dict = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            existing_labels = json.load(f)

    # Clustering
    X = np.stack([features_to_vector(f, config.get("feature_weights")) for f in features_list], axis=0)
    cl_cfg = config["clustering"]
    manager = ClusterManager(
        algorithm=cl_cfg.get("algorithm", "kmeans_auto"),
        n_init=cl_cfg["n_init"],
        max_iter=cl_cfg["max_iter"],
        random_state=cl_cfg["random_state"],
        max_auto_clusters=cl_cfg["max_auto_clusters"],
        min_silhouette_score=cl_cfg["min_silhouette_score"],
        manual_distance_factor=cl_cfg["manual_distance_factor"],
        min_clusters=cl_cfg.get("min_clusters", 0),
    )
    labels, distances, manual_mask, clustering_info = manager.fit_auto(X)

    # Sauvegarder le modèle
    manager.save(MODEL_FILE)

    export_summary = export_sorted_clips(
        valid_clips=valid_clips,
        labels=labels.tolist(),
        manual_mask=manual_mask.tolist(),
        failed_clips=failed_clips,
        athlete_names=existing_labels,
        output_dir=folder_path / SORTED_OUTPUT_DIRNAME,
    )

    # Stocker les résultats en session
    st.session_state["results"] = {
        "valid_clips": [str(c) for c in valid_clips],
        "failed_clips": [str(c) for c in failed_clips],
        "labels": labels.tolist(),
        "distances": distances.tolist(),
        "manual_mask": manual_mask.tolist(),
        "features": features_list,
        "n_clusters": int(manager.n_clusters),
        "athlete_names": existing_labels,
        "clustering_info": clustering_info,
        "export_dir": export_summary["export_dir"],
        "export_summary": export_summary,
    }
    st.rerun()


# ---- Résultats ----
if "results" not in st.session_state:
    st.stop()

res = st.session_state["results"]
valid_clips = [Path(p) for p in res["valid_clips"]]
failed_clips = [Path(p) for p in res.get("failed_clips", [])]
labels = res["labels"]
distances = res["distances"]
manual_mask = res.get("manual_mask", [False] * len(valid_clips))
n_clusters = res["n_clusters"]
athlete_names: dict = res.get("athlete_names", {})
clustering_info = res.get("clustering_info", {})
export_dir = Path(res.get("export_dir", folder_path / SORTED_OUTPUT_DIRNAME))
export_warnings = res.get("export_summary", {}).get("warnings", [])

cluster_ids = sorted({labels[i] for i in range(len(valid_clips)) if not manual_mask[i]})
manual_review_count = sum(manual_mask) + len(failed_clips)

st.divider()
st.subheader("📊 Résultats du clustering")

st.caption(
    f"{n_clusters} groupe(s) détecté(s) automatiquement. "
    f"Score global de séparation : {clustering_info.get('global_silhouette', 0.0):.2f}. "
    f"Dossiers exportés dans : {export_dir}"
)

if export_warnings:
    with st.expander("⚠️ Avertissements export"):
        for msg in export_warnings:
            st.write(f"- {msg}")

# ── Pureté (si les noms de fichiers contiennent le nom de l'athlète) ────────
def _athlete_name(filename: str) -> str:
    return re.split(r"[\(\.\s]", filename)[0].strip().lower()

clip_names = [c.name for c in valid_clips]
athlete_tags = [_athlete_name(n) for n in clip_names]
unique_athletes = sorted(set(athlete_tags))

# On calcule la pureté uniquement si les fichiers semblent nommés par athlète
# (au moins 2 athlètes distincts détectés dans les noms)
if len(unique_athletes) >= 2:
    groups_content: dict[int, list[str]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        if not manual_mask[i]:
            groups_content[lbl].append(athlete_tags[i])

    purity_scores = []
    for g in cluster_ids:
        mc = Counter(groups_content[g])
        if mc:
            dominant_count = mc.most_common(1)[0][1]
            purity_scores.append(dominant_count / len(groups_content[g]))

    if purity_scores:
        avg_purity = sum(purity_scores) / len(purity_scores)

        all_clip_counts = Counter(_athlete_name(c.name) for c in valid_clips + failed_clips)
        ath_groups: dict[str, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            if not manual_mask[i]:
                ath_groups[athlete_tags[i]].append(lbl)
        recalls = []
        for ath in unique_athletes:
            total = all_clip_counts[ath]
            grps = Counter(ath_groups.get(ath, []))
            dom = grps.most_common(1)[0][1] if grps else 0
            recalls.append(dom / total)
        avg_recall = sum(recalls) / len(recalls)

        pur_icon = "✅" if avg_purity >= 0.8 else ("⚠️" if avg_purity >= 0.5 else "❌")
        rec_icon = "✅" if avg_recall >= 0.8 else ("⚠️" if avg_recall >= 0.5 else "❌")

        col_pur1, col_pur2, col_pur3 = st.columns(3)
        col_pur1.metric(
            "Pureté moyenne des groupes",
            f"{avg_purity:.0%} {pur_icon}",
            help="% de clips du même athlète dans chaque groupe. 100% = groupes purs.",
        )
        col_pur2.metric(
            "Recall moyen par athlète",
            f"{avg_recall:.0%} {rec_icon}",
            help="% de clips d'un athlète regroupés dans son groupe dominant.",
        )
        col_pur3.metric(
            "Score silhouette",
            f"{clustering_info.get('global_silhouette', 0.0):.3f}",
        )

        with st.expander("🔍 Détail pureté par groupe"):
            import pandas as pd
            rows = []
            for g in cluster_ids:
                mc = Counter(groups_content[g])
                dominant, dom_count = mc.most_common(1)[0]
                p = dom_count / len(groups_content[g])
                icon = "✅" if p == 1.0 else ("⚠️" if p >= 0.6 else "❌")
                composition = ", ".join(f"{k}×{v}" for k, v in sorted(mc.items()))
                rows.append({
                    "Groupe": cluster_display_name(g, athlete_names),
                    "Clips": len(groups_content[g]),
                    "Pureté": f"{p:.0%} {icon}",
                    "Dominant": dominant,
                    "Composition": composition,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            rows_ath = []
            for ath in sorted(all_clip_counts.keys()):
                total = all_clip_counts[ath]
                grps = Counter(ath_groups.get(ath, []))
                if grps:
                    dom_g, dom_count = grps.most_common(1)[0]
                    recall = dom_count / total
                    split = f" ({len(grps)} groupes)" if len(grps) > 1 else ""
                    icon = "✅" if recall >= 0.8 else ("⚠️" if recall >= 0.5 else "❌")
                else:
                    dom_g, recall, split, icon = "—", 0.0, "", "❌"
                rows_ath.append({
                    "Athlète": ath,
                    "Clips réels": total,
                    "Groupe dominant": cluster_display_name(dom_g, athlete_names) if isinstance(dom_g, int) else dom_g,
                    "Recall": f"{recall:.0%} {icon}{split}",
                    "Non assignés": total - len(ath_groups.get(ath, [])),
                })
            st.caption("**Recall par athlète**")
            st.dataframe(pd.DataFrame(rows_ath), use_container_width=True, hide_index=True)


col_summary, col_spacer = st.columns([2, 3])
with col_summary:
    for cluster_id in cluster_ids:
        count = sum(1 for i in range(len(valid_clips)) if labels[i] == cluster_id and not manual_mask[i])
        st.metric(
            label=cluster_display_name(cluster_id, athlete_names),
            value=f"{count} clips",
        )
    if manual_review_count:
        st.metric(label=MANUAL_REVIEW_DIRNAME, value=f"{manual_review_count} clips")

st.divider()

# Tabs par cluster
tab_specs = [
    {"kind": "cluster", "cluster_id": cluster_id, "label": cluster_display_name(cluster_id, athlete_names)}
    for cluster_id in cluster_ids
]
if manual_review_count:
    tab_specs.append({"kind": "manual", "label": MANUAL_REVIEW_DIRNAME})

tabs = st.tabs([spec["label"] for spec in tab_specs])

for spec, tab in zip(tab_specs, tabs):
    with tab:
        if spec["kind"] == "cluster":
            cluster_id = spec["cluster_id"]
            cluster_clips = [
                (valid_clips[i], distances[i])
                for i in range(len(valid_clips))
                if labels[i] == cluster_id and not manual_mask[i]
            ]
            cluster_clips.sort(key=lambda x: x[1])

            current_name = athlete_names.get(str(cluster_id), "")
            new_name = st.text_input(
                "Nom de l'athlète pour ce groupe",
                value=current_name,
                key=f"name_{cluster_id}",
                placeholder="ex: Pierre, Athlète A...",
            )
            if new_name != current_name:
                athlete_names[str(cluster_id)] = new_name
                res["athlete_names"] = athlete_names

            st.caption(f"{len(cluster_clips)} clip(s) dans ce groupe")

            if not cluster_clips:
                st.write("Aucun clip dans ce groupe.")
                continue

            # Affichage texte uniquement — pas de player vidéo
            for clip_path, dist in cluster_clips:
                confidence_pct = max(0.0, 1.0 - dist / (dist + 1.0)) * 100
                st.markdown(f"📎 `{clip_path.name}` — Confiance : **{confidence_pct:.0f}%**")
        else:
            manual_clips = [
                (valid_clips[i], distances[i], "Clip ambigu par similarité")
                for i in range(len(valid_clips))
                if manual_mask[i]
            ]
            manual_clips.extend((clip, None, "Pose non détectée") for clip in failed_clips)

            st.caption(f"{len(manual_clips)} clip(s) à vérifier manuellement")

            if not manual_clips:
                st.write("Aucun clip à trier manuellement.")
                continue

            # Affichage texte uniquement — pas de player vidéo
            for clip_path, dist, reason in manual_clips:
                if dist is not None:
                    confidence_pct = max(0.0, 1.0 - dist / (dist + 1.0)) * 100
                    st.markdown(f"📎 `{clip_path.name}` — {reason} — Confiance : **{confidence_pct:.0f}%**")
                else:
                    st.markdown(f"📎 `{clip_path.name}` — {reason}")

st.divider()

# Bouton sauvegarder les labels
if st.button("💾 Sauvegarder les noms des athlètes", type="secondary"):
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(athlete_names, f, indent=2, ensure_ascii=False)
    export_summary = export_sorted_clips(
        valid_clips=valid_clips,
        labels=labels,
        manual_mask=manual_mask,
        failed_clips=failed_clips,
        athlete_names=athlete_names,
        output_dir=export_dir,
    )
    res["export_dir"] = export_summary["export_dir"]
    res["export_summary"] = export_summary
    st.success(f"Noms sauvegardés et dossiers ré-exportés dans {export_summary['export_dir']}")

    if export_summary.get("warnings"):
        for msg in export_summary["warnings"]:
            st.warning(msg)

# Tableau récapitulatif
with st.expander("🔬 Voir les features biomécaniques extraites"):
    import pandas as pd

    rows_data = []
    for i, (clip, feat) in enumerate(zip(valid_clips, res["features"])):
        if manual_mask[i]:
            cluster_name = MANUAL_REVIEW_DIRNAME
        else:
            cluster_name = cluster_display_name(labels[i], athlete_names)
        row = {"clip": clip.name, "cluster": cluster_name}
        row.update({k: round(v, 4) for k, v in feat.items()})
        rows_data.append(row)

    df = pd.DataFrame(rows_data)
    st.dataframe(df, use_container_width=True)
