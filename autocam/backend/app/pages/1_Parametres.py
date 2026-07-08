"""
1_Parametres.py
Page de réglages du modèle AutoCam — accessible via la sidebar Streamlit.
"""

import json
import sys
from pathlib import Path

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

# Ajouter app/ au path pour les imports relatifs
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULTS, DATA_DIR, load_config, reset_config, save_config

CACHE_FILE = DATA_DIR / "features_cache.json"
MODEL_FILE = DATA_DIR / "model.pkl"

st.set_page_config(
    page_title="Paramètres — AutoCam",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Paramètres du modèle")
st.caption("Les modifications sont appliquées à la prochaine analyse. Le cache de features doit être effacé si vous changez le modèle YOLO ou les features.")

config = load_config()


def _discover_backend_models() -> tuple[list[str], list[str]]:
    """Retourne (detector_models, pose_models) trouvés dans backend/*.pt."""
    backend_dir = Path(__file__).resolve().parent.parent.parent
    model_files = sorted(p.name for p in backend_dir.glob("*.pt"))

    detector_models = [m for m in model_files if m.endswith(".pt") and "-pose" not in m]
    pose_models = [m for m in model_files if m.endswith("-pose.pt")]
    return detector_models, pose_models

# ──────────────────────────────────────────────
# Formulaire principal
# ──────────────────────────────────────────────
with st.form("settings_form"):

    # ── Choix du backend ────────────────────────
    st.subheader("🧠 Backend d'analyse")
    BACKENDS = ["detector_yolo_pose", "yolo"]
    BACKEND_LABELS = [
        "🟢 YOLO détecteur + YOLO11-pose (2 modèles) — repérage puis extraction",
        "🟠 YOLO seul — plus rapide, pas de modèle distinct de repérage",
    ]
    BACKEND_LABEL_TO_KEY = dict(zip(BACKEND_LABELS, BACKENDS))
    current_backend = config.get("pose_backend", "detector_yolo_pose")
    if current_backend == "sam3_yolo":
        current_backend = "detector_yolo_pose"
    selected_backend_label = st.radio(
        "Backend actif",
        options=BACKEND_LABELS,
        index=BACKENDS.index(current_backend) if current_backend in BACKENDS else 0,
        help=(
            "**YOLO détecteur + YOLO11-pose** — un modèle détecte/suit la personne dominante, "
            "un second modèle extrait les keypoints sur le crop.\n"
            "**YOLO seul** — estimation de pose directe sur frame entière/crop central."
        ),
    )
    pose_backend = BACKEND_LABEL_TO_KEY[selected_backend_label]

    st.divider()

    # ── Détecteur YOLO + YOLO11-pose ──────────────────────────
    st.subheader("🟢 Détecteur YOLO + YOLO11-pose")
    det_cfg = config.get("detector_pose", DEFAULTS["detector_pose"])

    disk_detector_models, disk_pose_models = _discover_backend_models()

    DETECTOR_MODELS = list(dict.fromkeys([
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo26n.pt",
        "yolo26s.pt",
        "yolo26m.pt",
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        *disk_detector_models,
    ]))
    YOLO11_MODELS = list(dict.fromkeys([
        "yolo11n-pose.pt",
        "yolo11s-pose.pt",
        "yolo11m-pose.pt",
        "yolo26n-pose.pt",
        "yolo26s-pose.pt",
        "yolo26m-pose.pt",
        *disk_pose_models,
    ]))

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        detector_model_name = st.selectbox(
            "Modèle de repérage (détection/tracking)",
            DETECTOR_MODELS,
            index=DETECTOR_MODELS.index(det_cfg.get("detector_model_name", "yolo11n.pt"))
            if det_cfg.get("detector_model_name", "yolo11n.pt") in DETECTOR_MODELS else 0,
            help="Modèle dédié au repérage de la personne dominante (classe person).",
        )
        detector_conf = st.slider(
            "Confiance détection (repérage)",
            min_value=0.1, max_value=0.95,
            value=float(det_cfg.get("detector_conf_threshold", 0.35)),
            step=0.05, format="%.2f",
        )
        pose_model_name = st.selectbox(
            "Modèle YOLO-pose (extraction bioméca)",
            YOLO11_MODELS,
            index=YOLO11_MODELS.index(det_cfg.get("pose_model_name", "yolo11n-pose.pt"))
            if det_cfg.get("pose_model_name", "yolo11n-pose.pt") in YOLO11_MODELS else 0,
            help="Modèle dédié à l'extraction des keypoints sur le crop du sauteur.",
        )
        pose_conf = st.slider(
            "Confiance détection (pose)",
            min_value=0.1, max_value=0.95,
            value=float(det_cfg.get("pose_conf_threshold", 0.5)),
            step=0.05, format="%.2f",
        )
    with col_s2:
        det_max_frames = st.slider(
            "Frames max par clip",
            min_value=50, max_value=600,
            value=int(det_cfg.get("max_frames", 300)),
            step=50,
            key="det_max_frames",
        )
        det_min_hip = st.slider(
            "Vitesse min hanches",
            min_value=0.0, max_value=0.1,
            value=float(det_cfg.get("min_hip_velocity", 0.015)),
            step=0.005, format="%.3f",
        )
        det_min_body = st.slider(
            "Taille min. corps détecté",
            min_value=0.0, max_value=0.6,
            value=float(det_cfg.get("min_body_height_ratio", 0.0)),
            step=0.05, format="%.2f",
        )

    st.divider()

    # ── YOLOv8 / YOLO11 (fallback) ─────────────────
    st.subheader("🟠 Backend YOLO — estimation de pose (fallback)")
    YOLO_MODELS = list(dict.fromkeys([
        "yolov8n-pose.pt",
        "yolov8s-pose.pt",
        "yolov8m-pose.pt",
        "yolo11n-pose.pt",
        "yolo11s-pose.pt",
        "yolo11m-pose.pt",
        "yolo26n-pose.pt",
        "yolo26s-pose.pt",
        "yolo26m-pose.pt",
        *disk_pose_models,
    ]))
    col_y1, col_y2 = st.columns(2)
    with col_y1:
        yolo_model = st.selectbox(
            "Modèle",
            YOLO_MODELS,
            index=YOLO_MODELS.index(config["yolo"].get("model_name", "yolov8n-pose.pt"))
            if config["yolo"].get("model_name", "yolov8n-pose.pt") in YOLO_MODELS
            else 0,
            help="v8 : modèles éprouvés. v11 : nouvelle génération Ultralytics (plus précis à taille égale). n=nano, s=small, m=medium.",
        )
        yolo_conf = st.slider(
            "Confiance de détection",
            min_value=0.1, max_value=0.95,
            value=float(config["yolo"].get("conf_threshold", 0.5)),
            step=0.05, format="%.2f",
        )
    with col_y2:
        yolo_max_frames = st.slider(
            "Frames max par clip",
            min_value=50, max_value=600,
            value=int(config["yolo"].get("max_frames", 300)),
            step=50,
            key="yolo_max_frames",
        )

    # Détecter si CUDA est disponible pour informer l'utilisateur
    try:
        import torch
        _cuda_available = torch.cuda.is_available()
        _cuda_name = torch.cuda.get_device_name(0) if _cuda_available else None
    except ImportError:
        _cuda_available = False
        _cuda_name = None

    _device_options = ["auto", "cpu", "cuda"]
    _device_help = (
        "**auto** — utilise le GPU (CUDA) s'il est disponible, sinon CPU.  \n"
        "**cpu** — force le CPU (portable, pas de dépendance GPU).  \n"
        "**cuda** — force le GPU NVIDIA (CUDA). Nécessite PyTorch CUDA installé."
    )
    if _cuda_available:
        _device_help += f"  \n\n✅ GPU détecté : **{_cuda_name}**"
    else:
        _device_help += "  \n\n⚠️ Aucun GPU CUDA détecté — `auto` utilisera le CPU."

    yolo_device = st.selectbox(
        "🖥️ Device d'inférence",
        options=_device_options,
        index=_device_options.index(config["yolo"].get("device", "auto")),
        help=_device_help,
    )

    st.divider()

    # ── Filtres multi-personnes ─────────────────
    st.subheader("👥 Filtres — personnes parasites")
    st.caption("Élimine les frames où un entraîneur, du public ou un autre athlète est capturé à la place de l'athlète principal.")
    col1b, col2b = st.columns(2)
    with col1b:
        center_crop_ratio = st.slider(
            "Crop central (largeur conservée)",
            min_value=0.4, max_value=1.0,
            value=float(config["yolo"].get("center_crop_ratio", 1.0)),
            step=0.05,
            format="%.2f",
            help=(
                "Rogne les bords gauche/droit du frame avant analyse. "
                "0.75 = garder les 75% centraux → les personnes sur les côtés sont ignorées. "
                "1.0 = pas de crop."
            ),
        )
    with col2b:
        min_body_height_ratio = st.slider(
            "Taille minimale de la personne détectée",
            min_value=0.0, max_value=0.6,
            value=float(config["yolo"].get("min_body_height_ratio", 0.0)),
            step=0.05,
            format="%.2f",
            help=(
                "Rejette les frames où la personne détectée (épaule→cheville) occupe moins de N% de la hauteur de l'image. "
                "0.0 = désactivé."
            ),
        )

    min_hip_velocity = st.slider(
        "Vitesse minimale de déplacement (filtre personne statique)",
        min_value=0.0, max_value=0.1,
        value=float(config["yolo"].get("min_hip_velocity", 0.015)),
        step=0.005,
        format="%.3f",
        help=(
            "Rejette les frames où la personne détectée se déplace moins vite que ce seuil. "
            "0.0 = désactivé. Valeur recommandée : 0.015–0.03."
        ),
    )

    st.divider()

    # ── Features ───────────────────────────────
    st.subheader("📐 Features biomécaniques")
    col3, col4 = st.columns(2)
    with col3:
        visibility_threshold = st.slider(
            "Seuil de visibilité des landmarks",
            min_value=0.1, max_value=0.9,
            value=float(config["features"]["visibility_threshold"]),
            step=0.05,
            format="%.2f",
            help="Les frames où un landmark clé (épaules, hanches, chevilles) a une visibilité inférieure à ce seuil sont ignorées.",
        )
        min_valid_frames = st.number_input(
            "Frames valides minimum par clip",
            min_value=3, max_value=50,
            value=int(config["features"]["min_valid_frames"]),
            help="Un clip avec moins de N frames valides est rejeté (pose non détectée).",
        )
    with col4:
        flight_threshold = st.slider(
            "Seuil Y — phase aérienne",
            min_value=0.3, max_value=0.85,
            value=float(config["features"]["flight_ankle_y_threshold"]),
            step=0.05,
            format="%.2f",
            help="Coordonnée Y normalisée [0=haut image, 1=bas image]. Les frames où les chevilles sont au-dessus de ce seuil comptent comme 'phase aérienne'.",
        )
        st.markdown("")  # spacer
    st.info(
            "⚠️ Modifier le modèle ou les features **invalide le cache**. "
            "Effacez-le ci-dessous pour forcer le re-traitement."
        )

    st.divider()

    # ── Clustering automatique ─────────────────
    st.subheader("🤖 Clustering automatique")
    CLUSTERING_ALGOS = {
        "KMeans auto (recommandé)": "kmeans_auto",
        "KMeans (fixe)": "kmeans",
        "Spectral": "spectral",
        "GMM (Gaussian Mixture)": "gmm",
        "Agglomerative (hiérarchique)": "agglomerative",
        "DBSCAN": "dbscan",
    }
    current_algo = config["clustering"].get("algorithm", "kmeans_auto")
    if current_algo not in CLUSTERING_ALGOS.values():
        current_algo = "kmeans_auto"
    algo_labels = list(CLUSTERING_ALGOS.keys())
    selected_algo_label = st.selectbox(
        "Algorithme de clustering",
        options=algo_labels,
        index=[CLUSTERING_ALGOS[label] for label in algo_labels].index(current_algo),
        help="Choisis l'algorithme utilisé pour regrouper les clips par athlète.",
    )
    clustering_algorithm = CLUSTERING_ALGOS[selected_algo_label]

    col5, col6 = st.columns(2)
    with col5:
        n_init = st.slider(
            "Nombre d'initialisations (n_init)",
            min_value=5, max_value=50,
            value=int(config["clustering"]["n_init"]),
            help="Nombre de fois que K-Means est relancé avec une initialisation aléatoire différente. Le meilleur résultat est conservé. Plus élevé = plus stable mais plus lent.",
        )
        random_state = st.number_input(
            "Graine aléatoire (random_state)",
            min_value=0, max_value=9999,
            value=int(config["clustering"]["random_state"]),
            help="Valeur fixe pour des résultats reproductibles d'une session à l'autre.",
        )
        max_auto_clusters = st.slider(
            "Nombre maximum de groupes détectés",
            min_value=2, max_value=20,
            value=int(config["clustering"]["max_auto_clusters"]),
            help="Plafond utilisé quand l'application estime automatiquement le nombre d'athlètes présents.",
        )
    with col6:
        max_iter = st.slider(
            "Itérations max (max_iter)",
            min_value=100, max_value=1000,
            value=int(config["clustering"]["max_iter"]),
            step=50,
            help="Nombre maximal d'itérations par initialisation K-Means.",
        )
        min_silhouette_score = st.slider(
            "Seuil de similarité minimum avant tri manuel",
            min_value=-0.25, max_value=0.50,
            value=float(config["clustering"]["min_silhouette_score"]),
            step=0.05,
            format="%.2f",
            help="Si un clip est trop ambigu vis-à-vis de son groupe, il part dans le dossier 'a trier a la main'.",
        )
        manual_distance_factor = st.slider(
            "Tolérance d'éloignement dans un groupe",
            min_value=1.0, max_value=6.0,
            value=float(config["clustering"]["manual_distance_factor"]),
            step=0.5,
            format="%.1f",
            help="Plus la valeur est faible, plus l'application envoie facilement les clips atypiques dans 'a trier a la main'.",
        )

    st.divider()

    # ── Nombre minimum de groupes ─────────────────────────
    st.subheader("👥 Nombre d'athlètes attendus")
    min_clusters = st.number_input(
        "Nombre minimum de groupes (0 = automatique)",
        min_value=0, max_value=20,
        value=int(config["clustering"].get("min_clusters", 0)),
        help=(
            "Si vous connaissez le nombre d'athlètes présents dans le dossier, "
            "renseignez cette valeur pour forçer le modèle à créer au moins N groupes. "
            "0 = détection 100% automatique (heuristique sqrt(n/2))."
        ),
    )

    st.divider()

    # ── Poids des features ─────────────────────
    st.subheader("⚖️ Poids des features biomécaniques")
    st.caption(
        "Chaque poids détermine l'importance accordée à cette feature dans le clustering. "
        "**0.0** = feature ignorée. "
        "Les features marquées ❌ sont inutilisables en vue latérale (caméra de côté) : leur poids recommandé est 0."
    )

    fw = config.get("feature_weights", {})

    _FEATURE_META = [
        ("shoulder_width_ratio", "Largeur épaules / hauteur corps ❌ vue côté",    0.0, 4.0, "Invalide en vue latérale — L/R épaules superposées en 2D."),
        ("lateral_oscillation",  "Oscillation latérale des hanches",              0.0, 4.0, "Correspond à la profondeur en vue côté → bruité."),
        ("hip_height_max",       "Hauteur max des hanches",                        0.0, 4.0, "1 − min(hip_y) : plus grand = saut plus haut."),
        ("trunk_angle_mean",     "Angle tronc moyen (sagittal) ★",                0.0, 4.0, "Angle tronc / verticale dans le plan sagittal — très discriminant par athlète."),
        ("trunk_angle_std",      "Variabilité de l'angle tronc",                  0.0, 4.0, "Écart-type de l'angle tronc sur le clip."),
        ("flight_ratio",         "Ratio phase aérienne ★",                        0.0, 4.0, "Proportion de frames où les chevilles dépassent le seuil Y — distingue sauts vs approches."),
        ("stride_oscillation",   "Oscillation verticale (foulée) ★",              0.0, 4.0, "Rythme vertical des hanches : signature de la cadence de course."),
        ("hip_shoulder_ratio",   "Ratio hanches / épaules ❌ vue côté",           0.0, 4.0, "Invalide en vue latérale — projection 2D donne des valeurs aberrantes."),
        ("approach_speed_norm",  "Vitesse d'élan normalisée ★",                   0.0, 4.0, "Vitesse horizontale des hanches en phase d'approche — propre à chaque athlète."),
    ]

    col_w1, col_w2, col_w3 = st.columns(3)
    weight_cols = [col_w1, col_w2, col_w3]
    weight_values = {}
    for i, (key, label, wmin, wmax, help_txt) in enumerate(_FEATURE_META):
        default_val = float(fw.get(key, 0.0))
        with weight_cols[i % 3]:
            weight_values[key] = st.slider(
                label,
                min_value=wmin,
                max_value=wmax,
                value=default_val,
                step=0.1,
                format="%.1f",
                help=help_txt,
                key=f"fw_{key}",
            )

    st.divider()

    # ── Affichage ──────────────────────────────
    st.subheader("🖥️ Affichage")
    with st.container():
        cols_per_row = st.slider(
            "Colonnes par ligne (grille de clips)",
            min_value=1, max_value=5,
            value=int(config["display"]["cols_per_row"]),
        )

    st.divider()

    submitted = st.form_submit_button(
        "💾 Sauvegarder les paramètres",
        type="primary",
        use_container_width=True,
    )

# ──────────────────────────────────────────────
# Sauvegarde
# ──────────────────────────────────────────────
if submitted:
    new_config = {
        "pose_backend": pose_backend,
        "detector_pose": {
            "detector_model_name": detector_model_name,
            "pose_model_name": pose_model_name,
            "detector_conf_threshold": round(float(detector_conf), 2),
            "pose_conf_threshold": round(float(pose_conf), 2),
            "max_frames": int(det_max_frames),
            "center_crop_ratio": round(float(center_crop_ratio), 2),
            "min_body_height_ratio": round(float(det_min_body), 2),
            "min_hip_velocity": round(float(det_min_hip), 3),
            "device": yolo_device,
        },
        "yolo": {
            "model_name": yolo_model,
            "conf_threshold": round(float(yolo_conf), 2),
            "max_frames": int(yolo_max_frames),
            "center_crop_ratio": round(float(center_crop_ratio), 2),
            "min_body_height_ratio": round(float(min_body_height_ratio), 2),
            "min_hip_velocity": round(float(min_hip_velocity), 3),
            "device": yolo_device,
        },
        "features": {
            "visibility_threshold": round(float(visibility_threshold), 2),
            "flight_ankle_y_threshold": round(float(flight_threshold), 2),
            "min_valid_frames": int(min_valid_frames),
        },
        "clustering": {
            "algorithm": clustering_algorithm,
            "n_init": int(n_init),
            "max_iter": int(max_iter),
            "random_state": int(random_state),
            "max_auto_clusters": int(max_auto_clusters),
            "min_silhouette_score": round(float(min_silhouette_score), 2),
            "manual_distance_factor": round(float(manual_distance_factor), 2),
            "min_clusters": int(min_clusters),
        },
        "display": {
            "cols_per_row": int(cols_per_row),
        },
        "feature_weights": {k: round(float(v), 1) for k, v in weight_values.items()},
    }
    save_config(new_config)
    st.success("✅ Paramètres sauvegardés ! Retourne sur la page principale et relance une analyse.")

# ──────────────────────────────────────────────
# Actions de maintenance
# ──────────────────────────────────────────────
st.divider()
st.subheader("🛠️ Maintenance")

col_reset, col_cache, col_model = st.columns(3)

with col_reset:
    if st.button("🔄 Réinitialiser les valeurs par défaut", use_container_width=True):
        reset_config()
        st.success("Paramètres réinitialisés aux valeurs d'usine.")
        st.rerun()

with col_cache:
    cache_exists = CACHE_FILE.exists()
    if cache_exists:
        size_kb = CACHE_FILE.stat().st_size // 1024
        cache_label = f"🗑️ Effacer le cache features ({size_kb} KB)"
    else:
        cache_label = "🗑️ Effacer le cache features (vide)"
    if st.button(cache_label, disabled=not cache_exists, use_container_width=True):
        CACHE_FILE.unlink()
        st.success("Cache effacé. Les vidéos seront re-analysées au prochain lancement.")
        st.rerun()

with col_model:
    model_exists = MODEL_FILE.exists()
    model_label = "🗑️ Effacer le modèle sauvegardé" if model_exists else "🗑️ Effacer le modèle (aucun)"
    if st.button(model_label, disabled=not model_exists, use_container_width=True):
        MODEL_FILE.unlink()
        st.success("Modèle supprimé.")
        st.rerun()

# ──────────────────────────────────────────────
# Comparaison valeurs courantes vs défauts
# ──────────────────────────────────────────────
st.divider()
with st.expander("📋 Configuration actuelle vs valeurs par défaut"):
    current = load_config()
    col_cur, col_def = st.columns(2)
    with col_cur:
        st.caption("**Configuration actuelle**")
        st.code(json.dumps(current, indent=2), language="json")
    with col_def:
        st.caption("**Valeurs par défaut**")
        st.code(json.dumps(DEFAULTS, indent=2), language="json")
