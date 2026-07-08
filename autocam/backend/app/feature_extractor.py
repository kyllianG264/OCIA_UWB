"""
feature_extractor.py
Calcule les features biomécaniques à partir des keypoints MediaPipe.
Chaque feature est un scalaire représentant une caractéristique du saut ou de la morphologie.
"""

import numpy as np

# --- Indices landmarks (doivent rester cohérents avec mediapipe_extract.py) ---
_LS, _RS = 11, 12   # shoulders
_LE, _RE = 13, 14   # elbows
_LW, _RW = 15, 16   # wrists
_LH, _RH = 23, 24   # hips
_LK, _RK = 25, 26   # knees
_LA, _RA = 27, 28   # ankles


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """Distance euclidienne 2D entre deux points (x, y)."""
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:2] + b[:2]) / 2.0


def _robust_segment_length(
    keypoints: np.ndarray,
    pairs: list[tuple[int, int]],
    visibility_threshold: float,
    percentile: float = 90.0,
) -> float:
    """Estime la vraie longueur d'un segment osseux depuis une vidéo de profil.

    Un segment se raccourcit visuellement (foreshortening) quand il pointe
    vers/loin de la caméra. La longueur réelle est donc approchée par un haut
    percentile des distances observées (= segment vu perpendiculaire à l'axe
    caméra = extension maximale), et non par la moyenne (qui sous-estime).

    `pairs` regroupe les indices gauche ET droite du même os : on mutualise les
    observations des deux côtés et on ne garde que les frames où les deux
    extrémités sont suffisamment visibles.
    """
    samples: list[float] = []
    for idx_a, idx_b in pairs:
        va = keypoints[:, idx_a, 2]
        vb = keypoints[:, idx_b, 2]
        mask = (va > visibility_threshold) & (vb > visibility_threshold)
        if not np.any(mask):
            continue
        a = keypoints[mask, idx_a, :2]
        b = keypoints[mask, idx_b, :2]
        dists = np.sqrt(np.sum((a - b) ** 2, axis=1))
        samples.append(dists)
    if not samples:
        return 0.0
    pooled = np.concatenate(samples)
    return float(np.percentile(pooled, percentile))


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Ratio robuste : 0.0 si le dénominateur est négligeable (segment non mesuré)."""
    if denominator < 1e-5:
        return 0.0
    return float(numerator / denominator)


def extract_biomech_features(
    keypoints: np.ndarray,
    visibility_threshold: float = 0.4,
    flight_ankle_y_threshold: float = 0.6,
    min_valid_frames: int = 5,
) -> dict:
    """
    Calcule les features biomécaniques depuis les keypoints d'un clip.

    Args:
        keypoints: np.ndarray de shape (n_frames, 33, 3) — x, y, visibility en [0,1]
        visibility_threshold: seuil de visibilité des landmarks clés.
        flight_ankle_y_threshold: seuil Y pour détecter la phase aérienne.
        min_valid_frames: nombre minimal de frames valides pour accepter un clip.

    Returns:
        dict avec ~8 features scalaires.
        Retourne un dict de zéros si les données sont insuffisantes.
    """
    _ZERO = {
        "shoulder_width_ratio": 0.0,
        "lateral_oscillation": 0.0,
        "hip_height_max": 0.0,
        "trunk_angle_mean": 0.0,
        "trunk_angle_std": 0.0,
        "flight_ratio": 0.0,
        "stride_oscillation": 0.0,
        "hip_shoulder_ratio": 0.0,
        "approach_speed_norm": 0.0,
        # Ratios anthropométriques (signature personnelle, vue de profil)
        "crural_index": 0.0,
        "brachial_index": 0.0,
        "leg_torso_ratio": 0.0,
        "arm_torso_ratio": 0.0,
        "leg_arm_ratio": 0.0,
    }

    if keypoints is None or keypoints.shape[0] < min_valid_frames:
        return _ZERO

    n_frames = keypoints.shape[0]
    vis = keypoints[:, :, 2]  # (n_frames, 33)

    # Filtre: garder uniquement les frames où les landmarks clés sont visibles
    key_idxs = [_LS, _RS, _LH, _RH, _LA, _RA]
    visible_mask = np.all(vis[:, key_idxs] > visibility_threshold, axis=1)
    if visible_mask.sum() < min_valid_frames:
        return _ZERO

    kp = keypoints[visible_mask]  # (n_valid, 33, 3)

    # --- Extraction des séries temporelles ---
    ls = kp[:, _LS, :2]
    rs = kp[:, _RS, :2]
    lh = kp[:, _LH, :2]
    rh = kp[:, _RH, :2]
    lk = kp[:, _LK, :2]
    rk = kp[:, _RK, :2]
    la = kp[:, _LA, :2]
    ra = kp[:, _RA, :2]
    lw = kp[:, _LW, :2]
    rw = kp[:, _RW, :2]

    # Centre des hanches (trajectoire)
    hip_center = (lh + rh) / 2.0  # (n, 2)
    shoulder_center = (ls + rs) / 2.0

    # --- Feature 1: shoulder_width_ratio ---
    # Largeur épaules normalisée par la hauteur du corps (épaule → cheville)
    shoulder_width = np.mean([_dist(ls[i], rs[i]) for i in range(len(kp))])
    body_height = np.mean(
        [_dist((ls[i] + rs[i]) / 2, (la[i] + ra[i]) / 2) for i in range(len(kp))]
    )
    shoulder_width_ratio = shoulder_width / (body_height + 1e-6)

    # --- Feature 3: lateral_oscillation ---
    # Std du centre des hanches sur l'axe X, normalisée par la largeur des épaules
    # pour éliminer l'effet zoom/cadrage de la caméra
    lateral_oscillation = float(np.std(hip_center[:, 0]) / (shoulder_width + 1e-6))

    # --- Feature 4: hip_height_max ---
    # 1 - min(hip_center_y) → plus la valeur est grande, plus les hanches sont hautes
    # (y=0 = haut de l'image, y=1 = bas de l'image)
    hip_height_max = float(1.0 - np.min(hip_center[:, 1]))

    # --- Feature 5 & 6: trunk_angle ---
    # Angle du tronc par rapport à la verticale (en degrés)
    trunk_vectors = shoulder_center - hip_center  # (n, 2) — vecteur hanches→épaules
    # Angle avec la verticale [0, -1]
    trunk_angles = np.degrees(
        np.arctan2(np.abs(trunk_vectors[:, 0]), np.abs(trunk_vectors[:, 1]) + 1e-6)
    )
    trunk_angle_mean = float(np.mean(trunk_angles))
    trunk_angle_std = float(np.std(trunk_angles))

    # --- Feature 7: flight_ratio ---
    # Proportion de frames où les deux chevilles sont au-dessus du seuil Y
    # (y < seuil en coordonnées normalisées = haut de l'image)
    ankle_avg_y = (la[:, 1] + ra[:, 1]) / 2.0
    flight_ratio = float(np.mean(ankle_avg_y < flight_ankle_y_threshold))

    # --- Feature 8: stride_oscillation ---
    # Std du centre des hanches sur l'axe Y, normalisée par la hauteur du corps
    # pour éliminer l'effet zoom/cadrage de la caméra
    stride_oscillation = float(np.std(hip_center[:, 1]) / (body_height + 1e-6))

    # --- Feature 9: hip_shoulder_ratio (morphologie corporelle) ---
    # Largeur hanches / largeur épaules — proportion spécifique à chaque corps.
    # Invariant à la caméra car les deux mesures sont dans le même plan/frame.
    hip_width = np.mean([_dist(lh[i], rh[i]) for i in range(len(kp))])
    hip_shoulder_ratio = hip_width / (shoulder_width + 1e-6)

    # --- Feature 10: approach_speed_norm (vélocité d'approche) ---
    # Vitesse horizontale moyenne des hanches dans le premier tiers du clip.
    # Chaque athlète a une vitesse d'élan caractéristique.
    # Normalisée par la largeur d'épaules pour être invariante à l'échelle caméra.
    n_approach = max(2, len(kp) // 3)
    hip_x_approach = hip_center[:n_approach, 0]
    approach_speed = float(np.mean(np.abs(np.diff(hip_x_approach))))
    approach_speed_norm = approach_speed / (shoulder_width + 1e-6)

    # --- Features 11-15 : ratios anthropométriques (identité) ---
    # Longueurs de segments osseux estimées de façon robuste (haut percentile,
    # côtés gauche+droite mutualisés). Les ratios sont sans dimension donc
    # invariants au zoom/distance caméra, et bien visibles en vue de profil.
    thigh_len = _robust_segment_length(kp, [(_LH, _LK), (_RH, _RK)], visibility_threshold)
    shin_len = _robust_segment_length(kp, [(_LK, _LA), (_RK, _RA)], visibility_threshold)
    upperarm_len = _robust_segment_length(kp, [(_LS, _LE), (_RS, _RE)], visibility_threshold)
    forearm_len = _robust_segment_length(kp, [(_LE, _LW), (_RE, _RW)], visibility_threshold)
    torso_len = _robust_segment_length(kp, [(_LS, _LH), (_RS, _RH)], visibility_threshold)

    leg_len = thigh_len + shin_len
    arm_len = upperarm_len + forearm_len

    crural_index = _safe_ratio(shin_len, thigh_len)        # tibia / fémur
    brachial_index = _safe_ratio(forearm_len, upperarm_len)  # avant-bras / bras
    leg_torso_ratio = _safe_ratio(leg_len, torso_len)
    arm_torso_ratio = _safe_ratio(arm_len, torso_len)
    leg_arm_ratio = _safe_ratio(leg_len, arm_len)

    return {
        "shoulder_width_ratio": shoulder_width_ratio,
        "lateral_oscillation": lateral_oscillation,
        "hip_height_max": hip_height_max,
        "trunk_angle_mean": trunk_angle_mean,
        "trunk_angle_std": trunk_angle_std,
        "flight_ratio": flight_ratio,
        "stride_oscillation": stride_oscillation,
        "hip_shoulder_ratio": hip_shoulder_ratio,
        "approach_speed_norm": approach_speed_norm,
        "crural_index": crural_index,
        "brachial_index": brachial_index,
        "leg_torso_ratio": leg_torso_ratio,
        "arm_torso_ratio": arm_torso_ratio,
        "leg_arm_ratio": leg_arm_ratio,
    }


# Poids appliqués à chaque feature avant normalisation par StandardScaler.
# Un poids plus élevé force le clustering à accorder plus d'importance à cette feature.
#
# OBJECTIF : regrouper par IDENTITÉ (chaque athlète dans son groupe), caméra de
# PROFIL. On privilégie donc les ratios anthropométriques (proportions du corps,
# stables quel que soit le mouvement) et on neutralise les features de
# performance (hauteur du saut, phase aérienne) qui décrivent le geste et non la
# personne — deux athlètes faisant le même saut s'y confondent.
#
#   crural_index / leg_torso_ratio / leg_arm_ratio → 2.5 : proportions de jambe,
#       très personnelles et bien visibles de profil
#   brachial_index / arm_torso_ratio               → 2.0 : proportions de bras
#   approach_speed_norm                            → 1.0 : démarche (signal faible)
#   stride_oscillation                             → 0.8 : cadence de foulée
#   trunk_angle_mean / std                         → 0.5 / 0.3 : posture résiduelle
#   flight_ratio / hip_height_max                  → 0.0 : performance du saut = bruit
#   lateral_oscillation                            → 0.0 : profondeur, bruité de profil
#   shoulder_width_ratio / hip_shoulder_ratio      → 0.0 : largeurs L/R invalides de profil
# Poids UNIFORMES sur les features vivantes (mesuré : la séparabilité 1-NN est
# meilleure à poids égaux qu'en sur-pondérant l'anthropométrie — voir
# _diag_separability.py). Anthropométrie et bio du saut portent un signal
# d'identité comparable et complémentaire ; on les combine à parts égales.
# Seules les mesures invalides en vue de profil restent à 0.
_FEATURE_WEIGHTS = {
    "shoulder_width_ratio": 0.0,   # ❌ vue côté : L/R épaules superposées
    "hip_shoulder_ratio":   0.0,   # ❌ vue côté : ratio instable
    "lateral_oscillation":  0.0,   # ❌ profondeur, bruité de profil
    "hip_height_max":       1.0,   # détente / hauteur de saut
    "trunk_angle_mean":     1.0,
    "trunk_angle_std":      1.0,
    "flight_ratio":         1.0,   # durée de phase aérienne
    "stride_oscillation":   1.0,   # cadence de foulée
    "approach_speed_norm":  1.0,   # vitesse d'élan
    "crural_index":         1.0,   # tibia/fémur
    "brachial_index":       1.0,   # avant-bras/bras
    "leg_torso_ratio":      1.0,   # jambe/torse
    "arm_torso_ratio":      1.0,   # bras/torse
    "leg_arm_ratio":        1.0,   # plan corporel global
}

_FEATURE_KEYS = list(_FEATURE_WEIGHTS.keys())


def features_to_vector(features: dict, weights: dict | None = None) -> np.ndarray:
    """
    Convertit un dict de features en vecteur numpy pondéré (ordre fixe).

    Args:
        features: dict retourné par extract_biomech_features.
        weights: dict optionnel {feature_key: float} pour surcharger _FEATURE_WEIGHTS.
                 Les clés absentes sont prises dans _FEATURE_WEIGHTS.
    """
    w = _FEATURE_WEIGHTS if weights is None else {**_FEATURE_WEIGHTS, **weights}
    return np.array(
        [features.get(k, 0.0) * w.get(k, 0.0) for k in _FEATURE_KEYS],
        dtype=np.float32,
    )
