"""
features/1_calibration/1_CAL.py — Calibration manuelle du terrain handball (NeptuVision Tache 1)

Lancement : streamlit run features/1_calibration/1_CAL.py
Sortie    : calibration.json dans le dossier de sortie choisi

Flux par caméra :
  Étape A — Cliquer les 4 coins du demi-terrain sur le PLAN 2D
  Étape B — Cliquer les mêmes 4 coins dans la VIDEO

Ordre des 4 coins (même ordre sur plan ET vidéo) :
  1 → Haut-Gauche   2 → Haut-Droite
  4 → Bas-Gauche    3 → Bas-Droite

Dépendance : pip install streamlit-image-coordinates
"""

import streamlit as st
import cv2
import numpy as np
import json
import os
import sys
from PIL import Image, ImageDraw

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
if ROOT_DIR_PATH not in sys.path:
    sys.path.insert(0, ROOT_DIR_PATH)

from core.paths import DEFAULT_TERRAIN_IMAGE, ROOT_DIR, SAMPLE_VIDEOS_DIR
from core.utils import sync_file_to_python_cv_logs

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="NeptuVision — Calibration",
    page_icon="🎯",
    layout="wide"
)

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    _HAS_CLICK = True
except ImportError:
    _HAS_CLICK = False

# ─── Couleurs et labels ───────────────────────────────────────────────────────

COULEURS = [
    (34,  197,  94),   # 1 — vert
    (59,  130, 246),   # 2 — bleu
    (239,  68,  68),   # 3 — rouge
    (234, 179,   8),   # 4 — jaune
]
LABELS = ["1 Haut-Gauche", "2 Haut-Droite", "3 Bas-Droite", "4 Bas-Gauche"]

# ─── Chargement terrain ───────────────────────────────────────────────────────

terrain_path = str(DEFAULT_TERRAIN_IMAGE)
if not os.path.isfile(terrain_path):
    st.error(f"terrain-de-basket.jpg introuvable : {terrain_path}")
    st.stop()

terrain_pil  = Image.open(terrain_path).convert("RGB")
terrain_w, terrain_h = terrain_pil.size   # ex: 2000 × 2000

# ─── Utilitaires ─────────────────────────────────────────────────────────────

def extraire_frame(chemin: str, secondes: float) -> np.ndarray:
    cap = cv2.VideoCapture(chemin)
    cap.set(cv2.CAP_PROP_POS_MSEC, secondes * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Impossible de lire la frame à {secondes:.1f}s dans {chemin}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def annoter_image(img_rgb: np.ndarray, pts: list,
                  prochain_label: str | None = None,
                  source: str = "video") -> np.ndarray:
    """Dessine les points déjà cliqués et indique le prochain attendu."""
    out = img_rgb.copy()
    for i, (x, y) in enumerate(pts):
        c = COULEURS[i]
        cv2.circle(out, (x, y), 20, c, -1)
        cv2.circle(out, (x, y), 20, (255, 255, 255), 2)
        cv2.putText(out, str(i + 1), (x - 7, y + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if prochain_label:
        txt = f"Cliquez : {prochain_label}"
        cv2.putText(out, txt, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0),   4, cv2.LINE_AA)
        cv2.putText(out, txt, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 230, 0), 2, cv2.LINE_AA)
    elif len(pts) == 4:
        cv2.putText(out, "4 coins OK", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (50, 220, 50), 2, cv2.LINE_AA)
    return out


def dessiner_terrain(pts_terrain: list) -> np.ndarray:
    """Retourne le terrain-de-basket.jpg avec les points déjà cliqués."""
    img = terrain_pil.copy()
    draw = ImageDraw.Draw(img)
    for i, (x, y) in enumerate(pts_terrain):
        c = COULEURS[i]
        r = max(int(terrain_w / 50), 18)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(*c, 230),
                     outline=(255, 255, 255), width=3)
        draw.text((x - 9, y - 12), str(i + 1), fill=(255, 255, 255))
    return np.array(img)


def widget_clic(img_rgb: np.ndarray, display_w: int, key: str):
    """
    Affiche img_rgb redimensionnée et retourne (real_x, real_y) si clic détecté,
    sinon None.  Gère le fallback saisie manuelle si streamlit_image_coordinates absent.
    """
    h_orig, w_orig = img_rgb.shape[:2]
    scale    = display_w / w_orig
    disp_h   = int(h_orig * scale)
    resized  = cv2.resize(img_rgb, (display_w, disp_h))
    pil_disp = Image.fromarray(resized)

    if _HAS_CLICK:
        coord = streamlit_image_coordinates(pil_disp, key=key)
        if coord is not None:
            return int(coord["x"] / scale), int(coord["y"] / scale)
        return None
    else:
        st.image(pil_disp, use_container_width=True)
        c1, c2 = st.columns(2)
        rx = c1.number_input("X", 0, w_orig, w_orig // 2, key=f"{key}_x")
        ry = c2.number_input("Y", 0, h_orig, h_orig // 2, key=f"{key}_y")
        if st.button("Valider ce point", key=f"{key}_ok"):
            return int(rx), int(ry)
        return None


def calculer_H(src_pts: list, dst_pts: list) -> np.ndarray:
    src = np.array(src_pts, dtype=float)
    dst = np.array(dst_pts, dtype=float)
    H, _ = cv2.findHomography(src, dst)
    return H


def bounds_depuis_coins(terrain_pts: list) -> dict:
    """Calcule la bounding-box du demi-terrain depuis les 4 coins cliqués sur terrain."""
    xs = [p[0] for p in terrain_pts]
    ys = [p[1] for p in terrain_pts]
    return {"x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys)}


def sauvegarder(out_dir: str,
                v_g: list, t_g: list, H_g: np.ndarray,
                v_d: list, t_d: list, H_d: np.ndarray) -> str:
    os.makedirs(out_dir, exist_ok=True)
    data = {
        "cam_gauche": {
            "frame_corners":   v_g,
            "terrain_corners": t_g,
            "H": H_g.tolist()
        },
        "cam_droite": {
            "frame_corners":   v_d,
            "terrain_corners": t_d,
            "H": H_d.tolist()
        },
        "terrain_bounds": {
            "gauche": bounds_depuis_coins(t_g),
            "droite": bounds_depuis_coins(t_d)
        },
        "terrain_png_size": [terrain_w, terrain_h]
    }
    path = os.path.join(out_dir, "calibration.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    sync_file_to_python_cv_logs(path, "calibration.json")
    return path

# ─── Session state ────────────────────────────────────────────────────────────

def _init():
    for k, v in {
        "frame_g": None,   "frame_d": None,
        "vid_pts_g": [],   "vid_pts_d": [],
        "ter_pts_g": [],   "ter_pts_d": [],
        "prev_vid_g": None, "prev_vid_d": None,
        "prev_ter_g": None, "prev_ter_d": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ─── Interface ────────────────────────────────────────────────────────────────

st.title("Calibration manuelle du terrain")
st.markdown("""
Cliquez les **4 coins du demi-terrain** dans **le même ordre** sur le plan ET sur la vidéo :

| N° | Position |
|----|----------|
| **1** | Haut-Gauche |
| **2** | Haut-Droite |
| **3** | Bas-Droite  |
| **4** | Bas-Gauche  |
""")

if not _HAS_CLICK:
    st.warning("Installez `streamlit-image-coordinates` pour activer la sélection par clic. "
               "Mode saisie manuelle activé.")

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Paramètres")
    vid_g  = st.text_input("Vidéo gauche",  value=str(SAMPLE_VIDEOS_DIR / "gauche.mp4"),  key="sid_vg")
    vid_d  = st.text_input("Vidéo droite",  value=str(SAMPLE_VIDEOS_DIR / "droite.mp4"),  key="sid_vd")
    sec    = st.number_input("Horodatage extraction (s)", 0.0, value=5.0, step=1.0)
    out_dir = st.text_input("Dossier de sortie",
                             os.path.join(str(ROOT_DIR), "NeptuVisionResults", "2_Calibration"))

    if st.button("Extraire les frames", type="primary"):
        ok = True
        for label, path, key in [("gauche", vid_g, "frame_g"), ("droite", vid_d, "frame_d")]:
            if not path or not os.path.isfile(path):
                st.error(f"Vidéo {label} introuvable : {path}")
                ok = False
                continue
            try:
                st.session_state[key] = extraire_frame(path, sec)
                st.session_state[f"vid_pts_{'g' if label=='gauche' else 'd'}"] = []
                st.session_state[f"ter_pts_{'g' if label=='gauche' else 'd'}"] = []
                st.session_state[f"prev_vid_{'g' if label=='gauche' else 'd'}"] = None
                st.session_state[f"prev_ter_{'g' if label=='gauche' else 'd'}"] = None
            except Exception as e:
                st.error(str(e))
                ok = False
        if ok:
            st.success("Frames extraites ✓")


# ─── Section calibration d'une caméra ────────────────────────────────────────

def section_cam(cam: str):
    """cam = 'g' ou 'd'"""
    label     = "Gauche" if cam == "g" else "Droite"
    frame_key = f"frame_{cam}"
    vk        = f"vid_pts_{cam}"   # points vidéo
    tk        = f"ter_pts_{cam}"   # points terrain
    pvk       = f"prev_vid_{cam}"
    ptk       = f"prev_ter_{cam}"

    frame: np.ndarray | None = st.session_state[frame_key]
    vid_pts: list = st.session_state[vk]
    ter_pts: list = st.session_state[tk]

    st.subheader(f"Caméra {label}")

    if frame is None:
        st.info(f"Extrayez les frames via la barre latérale pour calibrer la caméra {label}.")
        return

    # — Étapes d'avancement —
    ter_ok  = len(ter_pts) == 4
    vid_ok  = len(vid_pts) == 4
    etape   = "terrain" if not ter_ok else ("video" if not vid_ok else "done")

    prog_col1, prog_col2 = st.columns(2)
    prog_col1.markdown(
        f"**Étape A — Plan 2D** : {'✅ 4/4' if ter_ok else f'{len(ter_pts)}/4 clics'}"
    )
    prog_col2.markdown(
        f"**Étape B — Vidéo** : {'✅ 4/4' if vid_ok else (f'{len(vid_pts)}/4 clics' if ter_ok else '⏳ après étape A')}"
    )

    col_ter, col_vid = st.columns([1, 2])

    # ── Colonne TERRAIN ──────────────────────────────────────────────────────
    with col_ter:
        prochain_ter = LABELS[len(ter_pts)] if not ter_ok else None
        ter_annote = dessiner_terrain(ter_pts)
        ter_annote_rgb = annoter_image(ter_annote, ter_pts, prochain_ter, "terrain")

        if etape == "terrain":
            st.markdown(f"**A. Cliquez sur le plan :**  `{prochain_ter}`")
            coord = widget_clic(ter_annote_rgb, 400, key=f"ter_{cam}")
            if coord is not None and coord != st.session_state[ptk]:
                st.session_state[ptk] = coord
                st.session_state[tk]  = ter_pts + [list(coord)]
                st.rerun()
        else:
            st.image(Image.fromarray(cv2.resize(ter_annote_rgb, (400, 400))),
                     use_container_width=True)

    # ── Colonne VIDEO ─────────────────────────────────────────────────────────
    with col_vid:
        prochain_vid = LABELS[len(vid_pts)] if (ter_ok and not vid_ok) else None
        vid_annote   = annoter_image(frame, vid_pts, prochain_vid, "video")

        if etape == "video":
            st.markdown(f"**B. Cliquez dans la vidéo :**  `{prochain_vid}`")
            coord = widget_clic(vid_annote, 760, key=f"vid_{cam}")
            if coord is not None and coord != st.session_state[pvk]:
                st.session_state[pvk] = coord
                st.session_state[vk]  = vid_pts + [list(coord)]
                st.rerun()
        else:
            h, w = frame.shape[:2]
            scale = 760 / w
            disp  = cv2.resize(vid_annote, (760, int(h * scale)))
            st.image(Image.fromarray(disp), use_container_width=True)

        if etape == "done":
            st.success(f"Caméra {label} calibrée ✓")

    # ── Contrôles ─────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    if c1.button("↩ Annuler dernier (plan)",  key=f"undo_t_{cam}", disabled=len(ter_pts)==0):
        st.session_state[tk] = ter_pts[:-1]; st.rerun()
    if c2.button("↩ Annuler dernier (vidéo)", key=f"undo_v_{cam}", disabled=len(vid_pts)==0):
        st.session_state[vk] = vid_pts[:-1]; st.rerun()
    if c3.button(f"Recommencer caméra {label}", key=f"reset_{cam}"):
        st.session_state[tk] = []; st.session_state[vk] = []
        st.session_state[ptk] = None; st.session_state[pvk] = None
        st.rerun()

    # ── Résumé coords ─────────────────────────────────────────────────────────
    if ter_pts or vid_pts:
        with st.expander("Coordonnées sélectionnées"):
            rows = []
            for i in range(4):
                tp = ter_pts[i] if i < len(ter_pts) else "—"
                vp = vid_pts[i] if i < len(vid_pts) else "—"
                rows.append(f"**{LABELS[i]}** : plan={tp}  vidéo={vp}")
            st.markdown("\n\n".join(rows))


# ─── Onglets ─────────────────────────────────────────────────────────────────

tab_g, tab_d, tab_prev = st.tabs(["Caméra Gauche", "Caméra Droite", "Apercu & Sauvegarde"])

with tab_g:
    section_cam("g")

with tab_d:
    section_cam("d")

with tab_prev:
    vg = st.session_state["vid_pts_g"]
    tg = st.session_state["ter_pts_g"]
    vd = st.session_state["vid_pts_d"]
    td = st.session_state["ter_pts_d"]

    ok_g = len(vg) == 4 and len(tg) == 4
    ok_d = len(vd) == 4 and len(td) == 4

    st.subheader("Aperçu de la projection")

    if not ok_g and not ok_d:
        st.info("Complétez les deux caméras (plan + vidéo) pour voir l'aperçu.")
    else:
        cols = st.columns(2)
        for i, (ok, vpts, tpts, fkey, label) in enumerate([
            (ok_g, vg, tg, "frame_g", "Gauche"),
            (ok_d, vd, td, "frame_d", "Droite"),
        ]):
            with cols[i]:
                if ok:
                    H = calculer_H(vpts, tpts)
                    frame = st.session_state[fkey]
                    h_f, w_f = frame.shape[:2]

                    # Projette une grille de test
                    terrain_test = np.array(terrain_pil)
                    draw = ImageDraw.Draw(Image.fromarray(terrain_test))
                    for r in range(6):
                        for c in range(9):
                            fx = int(w_f * c / 8)
                            fy = int(h_f * r / 5)
                            pt = np.array([[[fx, fy]]], dtype=float)
                            pr = cv2.perspectiveTransform(pt, H)
                            px, py = int(pr[0][0][0]), int(pr[0][0][1])
                            if 0 <= px < terrain_w and 0 <= py < terrain_h:
                                r2 = 8
                                draw.ellipse([px-r2, py-r2, px+r2, py+r2],
                                             fill=(100, 200, 255))
                    img_proj = np.array(draw._image)
                    img_small = cv2.resize(img_proj, (480, 480))
                    st.markdown(f"**Caméra {label}** — grille projetée sur terrain")
                    st.image(img_small, use_container_width=True)
                    st.caption("Points bleus = projection d'une grille régulière depuis la vidéo.")

                    # Affiche les bounds détectés
                    b = bounds_depuis_coins(tpts)
                    st.info(f"Zone terrain : X [{b['x_min']}–{b['x_max']}]  "
                            f"Y [{b['y_min']}–{b['y_max']}]")
                else:
                    st.warning(f"Caméra {label} incomplète.")

    st.divider()
    st.subheader("Sauvegarde")

    if ok_g and ok_d:
        H_g = calculer_H(vg, tg)
        H_d = calculer_H(vd, td)

        if st.button("Sauvegarder calibration.json", type="primary"):
            path = sauvegarder(out_dir, vg, tg, H_g, vd, td, H_d)
            st.success(f"Sauvegardé : `{path}`")

        export = {
            "cam_gauche": {"frame_corners": vg, "terrain_corners": tg, "H": calculer_H(vg, tg).tolist()},
            "cam_droite": {"frame_corners": vd, "terrain_corners": td, "H": calculer_H(vd, td).tolist()},
            "terrain_bounds": {
                "gauche": bounds_depuis_coins(tg),
                "droite": bounds_depuis_coins(td)
            },
            "terrain_png_size": [terrain_w, terrain_h]
        }
        st.download_button(
            "Telecharger calibration.json",
            data=json.dumps(export, indent=2),
            file_name="calibration.json",
            mime="application/json"
        )
        with st.expander("Aperçu JSON"):
            st.json({k: v for k, v in export.items() if k != "cam_gauche" or True})
    else:
        missing = []
        if not ok_g: missing.append(f"Gauche ({len(vg)}/4 vidéo, {len(tg)}/4 plan)")
        if not ok_d: missing.append(f"Droite ({len(vd)}/4 vidéo, {len(td)}/4 plan)")
        st.warning("Incomplet : " + " | ".join(missing))
