"""
3_Entrainement.py
Page Streamlit pour entraîner le clustering sur une base maison.

Flux attendu :
- un dossier racine contenant les splits d'entraînement, test et vérification
- des vidéos dont le nom permet de vérifier la pureté des groupes
- un entraînement basé sur les features biomécaniques déjà utilisées par AutoCam
"""

import faulthandler
import json
import re
import sys
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Capture les crashs natifs (segfault torch/opencv) : sans ça le process meurt
# silencieusement. faulthandler imprime la pile C dans crash_log.txt + le terminal.
_CRASH_LOG = Path(__file__).parent.parent / "data" / "crash_log.txt"
try:
    _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
    faulthandler.enable(file=open(_CRASH_LOG, "w"), all_threads=True)
except Exception:
    faulthandler.enable(all_threads=True)

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
from yolo_extract import extract_keypoints as _yolo_extract_keypoints
from yolo_extract import _transcode_h264
from detector_yolo_pose_extract import extract_keypoints as _detector_pose_extract_keypoints


SUPPORTED_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
# Version du schéma de features : à incrémenter dès que extract_biomech_features
# change de sortie OU qu'un réglage d'extraction par défaut change (la clé de
# cache n'inclut pas les paramètres d'extraction), pour invalider les entrées
# obsolètes. v3 : min_body_height_ratio 0.1 -> 0.0 (récupère les clips de loin).
FEATURE_SCHEMA_VERSION = "v3-mbh0"
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "features_cache.json"
MODEL_FILE = DATA_DIR / "model.pkl"
REPORT_FILE = DATA_DIR / "training_report.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

config = load_config()

st.set_page_config(
    page_title="Entraînement — AutoCam",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Entraînement sur base maison")
st.caption(
    "Page dédiée au train/test/vérification d'une base vidéo personnelle. "
    "Le modèle entraîné est le clustering AutoCam basé sur les features biomécaniques."
)


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, ensure_ascii=False)


def _repair_mojibake(text: str) -> str:
    """Répare un nom mal décodé (UTF-8 relu en latin-1), ex. 'franÃ§ois' -> 'françois'.

    Sans ça, le même athlète est compté comme plusieurs identités distinctes
    (françois / franÃ§ois) et le recall s'effondre artificiellement.
    """
    if any(marker in text for marker in ("Ã", "Â", "â€")):
        try:
            return text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return text


def _infer_identity(filename: str) -> str:
    """Extrait une identité simple, normalisée (sans accents ni mojibake)."""
    stem = _repair_mojibake(Path(filename).stem)
    # Supprime les accents pour fusionner 'françois' et 'francois'.
    stem = unicodedata.normalize("NFKD", stem)
    stem = "".join(ch for ch in stem if not unicodedata.combining(ch))
    stem = stem.lower().strip()
    return re.split(r"[\(\)\.\s_\-]+", stem)[0].strip()


def _discover_split_dir(root: Path, split_name: str, aliases: list[str]) -> Path | None:
    candidates = [split_name, *aliases]
    for candidate in candidates:
        split_dir = root / candidate
        if split_dir.exists() and split_dir.is_dir():
            return split_dir
    return None


def _ensure_split_dir(root: Path, requested_name: str, fallback_name: str) -> Path:
    split_name = requested_name.strip() or fallback_name
    split_dir = root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir


def _collect_videos(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    videos = [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS]
    return sorted(videos)


def _process_mem_mb() -> float:
    """Mémoire (working set) du process en Mo, via ctypes Windows. 0.0 si indispo."""
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _clip_worker_count(device: str, backend: str) -> int:
    # PyTorch / ultralytics ne sont pas thread-safe : exécuter plusieurs
    # inférences YOLO en parallèle provoque des crashs natifs (segfault) et
    # une saturation mémoire qui tuent silencieusement le process sur Windows.
    # On force donc l'extraction séquentielle pour tous les backends torch.
    return 1


def _build_cache_key(video_path: Path, backend: str) -> str:
    return (
        f"{FEATURE_SCHEMA_VERSION}|{backend}|{video_path.resolve()}|"
        f"{video_path.stat().st_mtime_ns}|{video_path.stat().st_size}"
    )


def _extract_keypoints_for(path: Path, backend: str, extractor_options: dict):
    """Lance l'extraction de keypoints pour un chemin donné selon le backend."""
    if backend == "detector_yolo_pose":
        det_cfg = {**config["detector_pose"], **extractor_options}
        return _detector_pose_extract_keypoints(
            str(path),
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
    yolo_cfg = {**config["yolo"], **extractor_options}
    return _yolo_extract_keypoints(
        str(path),
        max_frames=yolo_cfg["max_frames"],
        model_name=yolo_cfg.get("model_name", "yolov8n-pose.pt"),
        conf_threshold=yolo_cfg.get("conf_threshold", 0.5),
        center_crop_ratio=yolo_cfg.get("center_crop_ratio", 1.0),
        min_body_height_ratio=yolo_cfg.get("min_body_height_ratio", 0.0),
        min_hip_velocity=yolo_cfg.get("min_hip_velocity", 0.0),
        device=yolo_cfg.get("device", "auto"),
    )


def _process_clip(
    video_path: Path,
    cache: dict,
    use_cache: bool,
    backend: str,
    ft_cfg: dict,
    extractor_options: dict,
    allow_transcode: bool = True,
) -> dict | None:
    key = _build_cache_key(video_path, backend)
    if use_cache and key in cache:
        return cache[key]

    keypoints = _extract_keypoints_for(video_path, backend, extractor_options)

    # Retry par transcodage H264 : les .MOV iPhone (HEVC) ne sont pas toujours
    # lisibles par OpenCV → cap.isOpened() échoue et l'extraction renvoie None.
    # On retente une fois sur une copie ré-encodée avant d'abandonner.
    if keypoints is None and allow_transcode:
        tmp_h264 = DATA_DIR / f"_transcode_{abs(hash(str(video_path)))}.mp4"
        try:
            if _transcode_h264(video_path, tmp_h264):
                print(f"[Entrainement] Transcodage H264 → retry extraction : {video_path.name}", file=sys.stderr, flush=True)
                keypoints = _extract_keypoints_for(tmp_h264, backend, extractor_options)
        finally:
            tmp_h264.unlink(missing_ok=True)

    if keypoints is None:
        return None

    features = extract_biomech_features(
        keypoints,
        visibility_threshold=ft_cfg["visibility_threshold"],
        flight_ankle_y_threshold=ft_cfg["flight_ankle_y_threshold"],
        min_valid_frames=ft_cfg["min_valid_frames"],
    )

    return features


def _extract_split_data(
    split_name: str,
    videos: list[Path],
    backend: str,
    ft_cfg: dict,
    use_cache: bool,
    progress_slot,
    cache: dict,
    extractor_options: dict,
) -> dict:
    if not videos:
        return {
            "split": split_name,
            "videos": [],
            "valid_videos": [],
            "failed_videos": [],
            "features": [],
            "identities": [],
            "elapsed_s": 0.0,
        }

    worker_count = _clip_worker_count(config.get("yolo", {}).get("device", "auto"), backend)
    results: list[tuple[int, Path, dict | None]] = []
    t0 = time.perf_counter()

    bar = progress_slot.progress(0.0, text=f"{split_name} — extraction en cours…")

    def _job(clip: Path) -> dict | None:
        try:
            return _process_clip(clip, cache, use_cache, backend, ft_cfg, extractor_options)
        except Exception:
            print(f"[Entrainement] Échec extraction sur {clip.name} :", file=sys.stderr)
            traceback.print_exc()
            return None

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(_job, clip): (index, clip) for index, clip in enumerate(videos)}
        done = 0
        for future in as_completed(future_map):
            index, clip = future_map[future]
            try:
                features = future.result()
            except Exception:
                print(f"[Entrainement] Erreur résultat extraction {clip.name} :", file=sys.stderr)
                traceback.print_exc()
                features = None
            if features is not None and use_cache:
                cache[_build_cache_key(clip, backend)] = features
            results.append((index, clip, features))
            done += 1
            mem_mb = _process_mem_mb()
            print(
                f"[Entrainement] {split_name} {done}/{len(videos)} "
                f"RAM={mem_mb:.0f} Mo — {clip.name}",
                file=sys.stderr,
                flush=True,
            )
            bar.progress(done / len(videos), text=f"{split_name} — {done}/{len(videos)} · RAM {mem_mb:.0f} Mo")

    results.sort(key=lambda item: item[0])

    valid_videos: list[Path] = []
    failed_videos: list[Path] = []
    failed_reasons: dict[str, str] = {}
    features_list: list[dict] = []
    identities: list[str] = []

    for _, clip, features in results:
        if features is None:
            failed_videos.append(clip)
            failed_reasons[clip.name] = "illisible / aucune pose détectée (même après transcodage)"
            continue
        if not any(value != 0.0 for value in features.values()):
            failed_videos.append(clip)
            failed_reasons[clip.name] = "features insuffisantes (trop peu de frames valides)"
            continue
        valid_videos.append(clip)
        features_list.append(features)
        identities.append(_infer_identity(clip.name))

    elapsed_s = time.perf_counter() - t0
    progress_slot.empty()

    return {
        "split": split_name,
        "videos": videos,
        "valid_videos": valid_videos,
        "failed_videos": failed_videos,
        "failed_reasons": failed_reasons,
        "features": features_list,
        "identities": identities,
        "elapsed_s": elapsed_s,
    }


def _compute_purity_metrics(labels: np.ndarray, identities: list[str], manual_mask: np.ndarray | None = None) -> dict:
    if manual_mask is None:
        manual_mask = np.zeros(len(labels), dtype=bool)

    valid_indices = [index for index in range(len(labels)) if not manual_mask[index]]
    if not valid_indices:
        return {
            "purety_avg": None,
            "recall_avg": None,
            "group_details": [],
            "athlete_details": [],
            "unique_identities": [],
        }

    valid_labels = [int(labels[index]) for index in valid_indices]
    valid_identities = [identities[index] for index in valid_indices]
    unique_identities = sorted(set(valid_identities))
    cluster_ids = sorted(set(label for label in valid_labels if label >= 0))

    if len(unique_identities) < 2:
        return {
            "purety_avg": None,
            "recall_avg": None,
            "group_details": [],
            "athlete_details": [],
            "unique_identities": unique_identities,
        }

    groups: dict[int, list[str]] = defaultdict(list)
    for label, identity in zip(valid_labels, valid_identities):
        if label >= 0:
            groups[label].append(identity)

    purity_scores: list[float] = []
    group_details: list[dict] = []
    for cluster_id in cluster_ids:
        members = groups.get(cluster_id, [])
        if not members:
            continue
        counts = Counter(members)
        dominant_name, dominant_count = counts.most_common(1)[0]
        purity = dominant_count / len(members)
        purity_scores.append(purity)
        group_details.append(
            {
                "cluster": cluster_id,
                "n": len(members),
                "dominant": dominant_name,
                "purity": purity,
                "composition": dict(sorted(counts.items())),
            }
        )

    avg_purity = float(np.mean(purity_scores)) if purity_scores else 0.0

    all_counts = Counter(valid_identities)
    athlete_groups: dict[str, list[int]] = defaultdict(list)
    for label, identity in zip(valid_labels, valid_identities):
        if label >= 0:
            athlete_groups[identity].append(label)

    athlete_details: list[dict] = []
    recalls: list[float] = []
    for athlete in unique_identities:
        total_real = all_counts[athlete]
        grouped = Counter(athlete_groups.get(athlete, []))
        dominant_group, dominant_count = grouped.most_common(1)[0] if grouped else (-1, 0)
        recall = dominant_count / total_real if total_real else 0.0
        recalls.append(recall)
        athlete_details.append(
            {
                "athlete": athlete,
                "total_real": total_real,
                "dominant_group": dominant_group,
                "recall": recall,
                "n_groups": len(grouped),
                "not_assigned": total_real - len(athlete_groups.get(athlete, [])),
            }
        )

    return {
        "purety_avg": avg_purity,
        "recall_avg": float(np.mean(recalls)) if recalls else 0.0,
        "group_details": group_details,
        "athlete_details": athlete_details,
        "unique_identities": unique_identities,
    }


def _render_split_summary(result: dict, title: str) -> None:
    valid_count = len(result["valid_videos"])
    failed_count = len(result["failed_videos"])
    total_count = len(result["videos"])

    cols = st.columns(4)
    cols[0].metric(title, f"{valid_count}/{total_count}")
    cols[1].metric("Échecs extraction", str(failed_count))
    cols[2].metric("Temps extraction", f"{result['elapsed_s']:.1f} s")
    cols[3].metric("Identités détectées", str(len(set(result["identities"]))))


def _render_result_block(result: dict, split_label: str) -> None:
    _render_split_summary(result, "Clips valides")

    if not result["valid_videos"]:
        st.warning("Aucune vidéo exploitable dans ce split.")
        return

    st.markdown("#### Vidéos non analysées")
    if result["failed_videos"]:
        reasons = result.get("failed_reasons", {})
        for clip in result["failed_videos"]:
            reason = reasons.get(clip.name)
            st.write(f"- {clip.name}" + (f" — _{reason}_" if reason else ""))
    else:
        st.caption("Aucun échec d'extraction sur ce split.")

    features_matrix = np.stack(
        [features_to_vector(feat, config.get("feature_weights")) for feat in result["features"]],
        axis=0,
    )

    manager = st.session_state["trained_model"]
    labels = manager.predict(features_matrix)
    distances = manager.cluster_distances(features_matrix)
    manual_mask = np.zeros(len(labels), dtype=bool)

    if split_label == "Entraînement":
        manual_mask = st.session_state["train_manual_mask"]

    metrics = _compute_purity_metrics(labels, result["identities"], manual_mask=manual_mask)

    cols = st.columns(4)
    cols[0].metric("Clusters trouvés", str(len(sorted(set(int(v) for v in labels if int(v) >= 0)))))
    cols[1].metric(
        "Pureté moyenne",
        f"{metrics['purety_avg']:.0%}" if metrics["purety_avg"] is not None else "N/A",
    )
    cols[2].metric(
        "Recall moyen",
        f"{metrics['recall_avg']:.0%}" if metrics["recall_avg"] is not None else "N/A",
    )
    cols[3].metric("Silhouette train", f"{st.session_state['train_meta'].get('global_silhouette', 0.0):.3f}")

    if metrics["group_details"]:
        import pandas as pd

        st.markdown("#### Groupes")
        rows = []
        for group in metrics["group_details"]:
            icon = "✅" if group["purity"] >= 0.8 else ("⚠️" if group["purity"] >= 0.6 else "❌")
            composition = ", ".join(f"{name}×{count}" for name, count in group["composition"].items())
            rows.append(
                {
                    "Groupe": group["cluster"],
                    "Clips": group["n"],
                    "Pureté": f"{group['purity']:.0%} {icon}",
                    "Dominant": group["dominant"],
                    "Composition": composition,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("#### Identités")
        rows = []
        for athlete in metrics["athlete_details"]:
            icon = "✅" if athlete["recall"] >= 0.8 else ("⚠️" if athlete["recall"] >= 0.6 else "❌")
            rows.append(
                {
                    "Identité": athlete["athlete"],
                    "Clips réels": athlete["total_real"],
                    "Groupe dominant": athlete["dominant_group"],
                    "Recall": f"{athlete['recall']:.0%} {icon}",
                    "Groupes distincts": athlete["n_groups"],
                    "Non assignés": athlete["not_assigned"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Vidéos et distances")
    import pandas as pd

    rows = []
    for clip, label, dist, identity in zip(result["valid_videos"], labels.tolist(), distances.tolist(), result["identities"]):
        confidence = max(0.0, 1.0 - dist / (dist + 1.0)) * 100
        rows.append(
            {
                "vidéo": clip.name,
                "identité": identity,
                "groupe": int(label),
                "distance": round(float(dist), 4),
                "confiance": f"{confidence:.0f}%",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


with st.sidebar:
    st.header("⚙️ Paramètres d'entraînement")

    dataset_root_input = st.text_input(
        "📁 Dossier racine de la base",
        placeholder="C:/chemin/vers/ma_base",
        help="Le dossier doit contenir les sous-dossiers d'entraînement, test et vérification.",
    )

    st.subheader("Découpage des splits")
    train_folder = st.text_input("Dossier entraînement", value="entrainement")
    test_folder = st.text_input("Dossier test", value="test")
    verification_folder = st.text_input("Dossier vérification", value="verification")

    st.subheader("Extraction des features")
    backend_options = {
        "🟢 Détecteur YOLO + YOLO11-pose": "detector_yolo_pose",
        "🟠 YOLO seul": "yolo",
    }
    backend_label = st.selectbox("Backend d'extraction", list(backend_options.keys()))
    backend = backend_options[backend_label]

    if backend == "detector_yolo_pose":
        detector_model_name = st.selectbox(
            "Modèle détecteur",
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
            key="train_detector_model",
        )
        pose_model_name = st.selectbox(
            "Modèle pose",
            [
                "yolo11n-pose.pt",
                "yolo11s-pose.pt",
                "yolo11m-pose.pt",
                "yolo26n-pose.pt",
                "yolo26s-pose.pt",
                "yolo26m-pose.pt",
            ],
            index=0,
            key="train_pose_model",
        )
    else:
        yolo_model_name = st.selectbox(
            "Modèle pose",
            [
                "yolov8n-pose.pt",
                "yolov8s-pose.pt",
                "yolov8m-pose.pt",
                "yolo11n-pose.pt",
                "yolo11s-pose.pt",
                "yolo11m-pose.pt",
                "yolo26n-pose.pt",
                "yolo26s-pose.pt",
                "yolo26m-pose.pt",
            ],
            index=0,
            key="train_yolo_model",
        )

    max_frames = st.slider(
        "Frames max par clip",
        min_value=50,
        max_value=600,
        value=int(config["yolo"]["max_frames"]),
        step=50,
    )
    vis_threshold = st.slider(
        "Seuil visibilité features",
        min_value=0.1,
        max_value=0.9,
        value=float(config["features"]["visibility_threshold"]),
        step=0.05,
        format="%.2f",
    )
    min_valid = st.number_input(
        "Frames valides minimum",
        min_value=3,
        max_value=50,
        value=int(config["features"]["min_valid_frames"]),
    )
    use_cache = st.checkbox("Utiliser le cache de features", value=True)

    st.subheader("Clustering")
    clustering_algorithm = st.selectbox(
        "Algorithme",
        ["kmeans_auto", "kmeans_fixed", "dbscan", "hierarchical", "gmm", "spectral"],
        index=["kmeans_auto", "kmeans_fixed", "dbscan", "hierarchical", "gmm", "spectral"].index(
            config["clustering"].get("algorithm", "kmeans_auto")
            if config["clustering"].get("algorithm", "kmeans_auto") in ["kmeans_auto", "kmeans_fixed", "dbscan", "hierarchical", "gmm", "spectral"]
            else "kmeans_auto"
        ),
    )
    n_clusters = st.number_input("Nombre de groupes", min_value=2, max_value=20, value=3)
    max_auto_clusters = st.slider(
        "Max groupes auto",
        min_value=2,
        max_value=20,
        value=int(config["clustering"].get("max_auto_clusters", 8)),
    )
    min_clusters = st.number_input(
        "Min groupes (0=auto)",
        min_value=0,
        max_value=20,
        value=int(config["clustering"].get("min_clusters", 0)),
    )
    min_silhouette = st.slider(
        "Seuil silhouette min",
        min_value=0.0,
        max_value=1.0,
        value=float(config["clustering"].get("min_silhouette_score", 0.0)),
        step=0.05,
    )

    run_training = st.button("🚀 Lancer l'entraînement", type="primary", use_container_width=True)


if run_training:
    if not dataset_root_input:
        st.error("Veuillez renseigner le dossier racine de la base.")
        st.stop()

    dataset_root = Path(dataset_root_input)
    if not dataset_root.exists() or not dataset_root.is_dir():
        st.error(f"Dossier introuvable : {dataset_root_input}")
        st.stop()

    train_split_dir = _ensure_split_dir(dataset_root, train_folder, "entrainement")
    test_split_dir = _ensure_split_dir(dataset_root, test_folder, "test")
    verification_split_dir = _ensure_split_dir(dataset_root, verification_folder, "verification")

    split_map = {
        "entrainement": _discover_split_dir(dataset_root, train_folder, ["train", "training", "entraînement"]) or train_split_dir,
        "test": _discover_split_dir(dataset_root, test_folder, ["eval", "evaluation"]) or test_split_dir,
        "verification": _discover_split_dir(dataset_root, verification_folder, ["validation", "val", "verify"]) or verification_split_dir,
    }

    fallback_videos = _collect_videos(dataset_root)
    if fallback_videos and not _collect_videos(train_split_dir):
        st.info(
            "Les dossiers de split ont été créés automatiquement. "
            "Déplacez ensuite vos vidéos dans le bon split avant l'entraînement si besoin."
        )

    split_videos = {name: _collect_videos(path) if path else [] for name, path in split_map.items()}
    total_videos = sum(len(items) for items in split_videos.values())
    if total_videos == 0:
        st.warning("Aucune vidéo trouvée dans la base fournie.")
        st.stop()

    st.session_state.pop("trained_model", None)
    st.session_state.pop("train_manual_mask", None)
    st.session_state.pop("train_meta", None)

    ft_cfg = {
        **config["features"],
        "visibility_threshold": float(vis_threshold),
        "min_valid_frames": int(min_valid),
    }
    extractor_options = {
        "max_frames": int(max_frames),
    }
    if backend == "detector_yolo_pose":
        extractor_options.update(
            {
                "detector_model_name": detector_model_name,
                "pose_model_name": pose_model_name,
            }
        )
    else:
        extractor_options.update({"model_name": yolo_model_name})

    cache = load_cache()
    cache_dirty = False
    extraction_placeholder = st.empty()
    progress_placeholder = st.empty()
    extraction_placeholder.info(f"Base détectée : {total_videos} vidéo(s). Démarrage de l'extraction…")

    split_results: dict[str, dict] = {}
    for split_name in ["entrainement", "test", "verification"]:
        videos = split_videos.get(split_name, [])
        if not videos:
            split_results[split_name] = {
                "split": split_name,
                "videos": [],
                "valid_videos": [],
                "failed_videos": [],
                "features": [],
                "identities": [],
                "elapsed_s": 0.0,
            }
            continue

        split_results[split_name] = _extract_split_data(
            split_name=split_name,
            videos=videos,
            backend=backend,
            ft_cfg=ft_cfg,
            use_cache=use_cache,
            progress_slot=progress_placeholder,
            cache=cache,
            extractor_options=extractor_options,
        )
        cache_dirty = True

    if cache_dirty:
        save_cache(cache)

    train_result = split_results["entrainement"]
    if not train_result["valid_videos"]:
        st.error("Aucune vidéo exploitable dans le split d'entraînement.")
        st.stop()

    train_matrix = np.stack(
        [features_to_vector(feat, config.get("feature_weights")) for feat in train_result["features"]],
        axis=0,
    )

    cl_cfg = config["clustering"]
    manager = ClusterManager(
        n_clusters=int(n_clusters),
        algorithm=clustering_algorithm,
        n_init=int(cl_cfg["n_init"]),
        max_iter=int(cl_cfg["max_iter"]),
        random_state=int(cl_cfg["random_state"]),
        max_auto_clusters=int(max_auto_clusters),
        min_silhouette_score=float(min_silhouette),
        manual_distance_factor=float(cl_cfg["manual_distance_factor"]),
        min_clusters=int(min_clusters),
    )
    train_labels, train_distances, train_manual_mask, train_meta = manager.fit_auto(train_matrix)
    manager.save(MODEL_FILE)

    st.session_state["trained_model"] = manager
    st.session_state["train_manual_mask"] = train_manual_mask
    st.session_state["train_meta"] = train_meta

    report = {
        "dataset_root": str(dataset_root),
        "backend": backend,
        "model_file": str(MODEL_FILE),
        "splits": {},
        "train_meta": train_meta,
    }

    for split_name, result in split_results.items():
        if result["features"]:
            split_matrix = np.stack(
                [features_to_vector(feat, config.get("feature_weights")) for feat in result["features"]],
                axis=0,
            )
            split_labels = manager.predict(split_matrix)
            split_distances = manager.cluster_distances(split_matrix)
        else:
            split_labels = np.array([], dtype=int)
            split_distances = np.array([], dtype=float)

        manual_mask = train_manual_mask if split_name == "entrainement" else np.zeros(len(split_labels), dtype=bool)
        metrics = _compute_purity_metrics(split_labels, result["identities"], manual_mask=manual_mask)

        report["splits"][split_name] = {
            "video_count": len(result["videos"]),
            "valid_count": len(result["valid_videos"]),
            "failed_count": len(result["failed_videos"]),
            "elapsed_s": result["elapsed_s"],
            "clusters": sorted(set(int(value) for value in split_labels if int(value) >= 0)),
            "purety_avg": metrics["purety_avg"],
            "recall_avg": metrics["recall_avg"],
            "labels": split_labels.tolist(),
            "distances": split_distances.tolist(),
            "identities": result["identities"],
            "failed_videos": [clip.name for clip in result["failed_videos"]],
            "failed_reasons": result.get("failed_reasons", {}),
            "group_details": metrics["group_details"],
            "athlete_details": metrics["athlete_details"],
            "manual_review_count": int(np.sum(manual_mask)),
        }

    with open(REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    st.session_state["training_report"] = report
    st.session_state["training_split_results"] = split_results
    st.success(f"Entraînement terminé. Modèle sauvegardé dans {MODEL_FILE}")


if "training_report" not in st.session_state:
    st.info("Lancez l'entraînement pour charger les splits, fitter le clustering et calculer la pureté par groupe.")
    st.stop()


report = st.session_state["training_report"]
split_results = st.session_state["training_split_results"]

st.divider()
st.subheader("📊 Résultats de l'entraînement")

top_cols = st.columns(4)
top_cols[0].metric("Backend", "2 modèles" if report["backend"] == "detector_yolo_pose" else "YOLO seul")
top_cols[1].metric("Modèle sauvegardé", MODEL_FILE.name)
top_cols[2].metric("Silhouette train", f"{report['train_meta'].get('global_silhouette', 0.0):.3f}")
top_cols[3].metric("Groupes", str(report["train_meta"].get("selected_clusters", 0)))

if REPORT_FILE.exists():
    st.download_button(
        "Télécharger le rapport JSON",
        data=json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name=REPORT_FILE.name,
        mime="application/json",
        use_container_width=True,
    )

for split_name, display_name in [
    ("entrainement", "Entraînement"),
    ("test", "Test"),
    ("verification", "Vérification"),
]:
    split_result = split_results.get(split_name, {})
    with st.expander(f"{display_name} — {len(split_result.get('videos', []))} vidéo(s)", expanded=(split_name == "entrainement")):
        if not split_result:
            st.write("Aucune donnée.")
            continue
        _render_result_block(split_result, display_name)
