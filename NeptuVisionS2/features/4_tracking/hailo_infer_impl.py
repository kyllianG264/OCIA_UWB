"""
hailo_infer.py — Interface d'inférence Hailo pour NeptuVision
==============================================================
Remplace les appels YOLO CPU par la puce Hailo-8L (hailort API).

Architecture inspirée du module HailoDetector de référence (caméra fixe) :
- Même pattern de verrou threading (detection_lock)
- Même logique d'extraction bbox + coordonnées normalisées
- Adapté pour des frames individuelles issues de fichiers vidéo
  (vs. flux GStreamer temps réel du module de référence)

Prérequis Raspberry Pi :
    - SDK Hailo installé : pip install hailort
    - Fichier(s) .hef générés via export_to_hef.py + compilation Hailo DFC

Usage dans 2_DA.py / 4_H.py :
    from hailo_infer import HailoDetector, HailoPoseEstimator
"""

import numpy as np
import cv2
import threading
from typing import List, Optional, Tuple, Dict, Any

# Tentative d'import Hailo — si absent, fallback CPU automatique
try:
    from hailo_platform import (
        VDevice, HEF, ConfigureParams, HailoStreamInterface,
        InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams,
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("[hailo_infer] ⚠️ hailort non disponible — les classes Hailo ne pourront pas être instanciées.")


# =============================================================================
# CLASSE DE BASE — Gestion device + inférence brute
# =============================================================================

class _HailoInferenceBase:
    """
    Initialise le device Hailo-8L et expose _infer_raw().
    Inspiré de HailoDetector.__init__ / _run_detection_app du module de référence,
    sans la couche GStreamer (on traite des frames OpenCV directement).
    """

    def __init__(self, hef_path: str, input_size: Tuple[int, int] = (640, 640)):
        """
        hef_path   : chemin absolu vers le .hef compilé
        input_size : (largeur, hauteur) attendues par le modèle
        """
        if not HAILO_AVAILABLE:
            raise RuntimeError(
                "hailort n'est pas installé. "
                "Installez le SDK Hailo sur la Raspberry Pi avant d'utiliser cette classe."
            )

        self.hef_path   = hef_path
        self.input_size = input_size          # (W, H)

        # Verrou partagé — même rôle que detection_lock dans le module de référence
        self._lock = threading.Lock()

        # ── Initialisation device et chargement HEF ──────────────────────────
        self._target         = VDevice()
        self._hef            = HEF(hef_path)
        configure_params     = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        network_groups       = self._target.configure(self._hef, configure_params)
        self._ng             = network_groups[0]
        self._ng_params      = self._ng.create_params()

        # Infos sur les flux d'entrée/sortie
        self._in_info   = self._hef.get_input_vstream_infos()
        self._out_info  = self._hef.get_output_vstream_infos()
        self._in_name   = self._in_info[0].name

        print(f"[Hailo] ✅ HEF chargé     : {hef_path}")
        print(f"[Hailo]    Entrée         : {self._in_name}")
        print(f"[Hailo]    Sorties        : {[o.name for o in self._out_info]}")
        print(f"[Hailo]    Résolution HEF : {input_size[0]}x{input_size[1]}")

    # ── Prétraitement ─────────────────────────────────────────────────────────
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Redimensionne et prépare la frame BGR pour le HEF.
        Même logique que dans le module de référence (HEF_WIDTH / HEF_HEIGHT).
        Le modèle Hailo attend du UINT8 en format BHWC.
        """
        w, h = self.input_size
        resized = cv2.resize(frame, (w, h))           # (H, W, 3)  BGR uint8
        return resized[np.newaxis].astype(np.uint8)   # (1, H, W, 3)

    # ── Inférence brute ───────────────────────────────────────────────────────
    def _infer_raw(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Envoie une frame au Hailo et retourne {nom_sortie: tenseur}.
        Protégé par self._lock (cf. detection_lock du module de référence).
        """
        input_data    = self._preprocess(frame)
        input_params  = InputVStreamParams.make(self._ng, format_type=FormatType.UINT8)
        output_params = OutputVStreamParams.make(self._ng, format_type=FormatType.FLOAT32)

        with self._lock:
            with InferVStreams(self._ng, input_params, output_params) as pipeline:
                with self._ng.activate(self._ng_params):
                    return pipeline.infer({self._in_name: input_data})

    def close(self):
        """Libère le device Hailo proprement (équivalent de HailoDetector.stop())."""
        try:
            del self._target
            print("[Hailo] Device libéré.")
        except Exception:
            pass


# =============================================================================
# DÉTECTEUR — remplace YOLO dans 2_DA.py
# =============================================================================

class HailoDetector(_HailoInferenceBase):
    """
    Détection d'objets sur puce Hailo.
    Remplace `model = YOLO(path)` + `model(frame)` dans 2_DA.py.

    Format de sortie HEF attendu (NMS intégré dans le HEF) :
        Tenseur shape [1, max_detections, 6]
        Colonnes : [y_min, x_min, y_max, x_max, confidence, class_id]
        (coordonnées normalisées [0, 1])

    Ce format correspond aux HEF compilés via Hailo DFC avec NMS embedding.
    """

    def __init__(
        self,
        hef_path: str,
        class_names: Optional[Dict[int, str]] = None,
        input_size: Tuple[int, int] = (640, 640),
    ):
        """
        class_names : dict {class_id: "NomClasse"} — doit correspondre
                      aux classes du modèle d'origine (ex. {0: "ATTAQUE"})
        """
        super().__init__(hef_path, input_size)
        self.class_names  = class_names or {0: "ATTAQUE"}
        self._det_out     = self._out_info[0].name   # 1ère sortie = détections

    # ── API publique ──────────────────────────────────────────────────────────
    def predict(self, frame: np.ndarray, conf: float = 0.25) -> List[Dict]:
        """
        Détecte les objets dans une frame.
        Retourne une liste de dicts :
            [{'bbox': [x1,y1,x2,y2], 'conf': float, 'class_name': str, 'class_id': int}]

        Équivaut à : results = model(frame)
        """
        orig_h, orig_w = frame.shape[:2]
        raw            = self._infer_raw(frame)

        results = []
        detections_raw = raw[self._det_out][0]  # liste de classes
        for class_id, class_dets in enumerate(detections_raw):
            for det in class_dets:
                x1_n, y1_n, x2_n, y2_n, confidence = det
                if confidence < conf:
                    continue
                x1 = int(x1_n * orig_w)
                y1 = int(y1_n * orig_h)
                x2 = int(x2_n * orig_w)
                y2 = int(y2_n * orig_h)
                cls_name = self.class_names.get(int(class_id), f"class_{int(class_id)}")
                results.append({
                    "bbox":       [x1, y1, x2, y2],
                    "conf":       float(confidence),
                    "class_name": cls_name,
                    "class_id":   int(class_id),
                    })
        return results

    def __call__(self, frame: np.ndarray):
        """
        Émule l'API YOLO `model(frame)` pour que 2_DA.py n'ait besoin
        que d'un changement minimal.
        Retourne une liste [HailoDetectionResult].
        """
        return [_HailoDetectionResult(self.predict(frame), self.class_names)]


class _HailoDetectionResult:
    """
    Émule YOLO Results (results[0].boxes, results[0].names)
    pour une compatibilité maximale avec le code existant de 2_DA.py.
    """

    class _Boxes:
        def __init__(self, detections):
            self._detections = detections

        def __iter__(self):
            return iter(self._detections)

    class _Box:
        """Émule un objet YOLO box (box.cls, box.conf, box.xyxy)."""
        def __init__(self, det: Dict):
            x1, y1, x2, y2 = det["bbox"]
            self.xyxy = [[x1, y1, x2, y2]]
            self.cls  = [det["class_id"]]
            self.conf = [det["conf"]]

    def __init__(self, detections: List[Dict], class_names: Dict[int, str]):
        self.boxes = self._Boxes([self._Box(d) for d in detections])
        self.names = class_names


# =============================================================================
# ESTIMATEUR DE POSE — remplace YOLO Pose dans 4_H.py
# =============================================================================

class HailoPoseEstimator(_HailoInferenceBase):
    """
    Estimation de pose (squelette 17 keypoints COCO) sur puce Hailo.
    Remplace `model_pose = YOLO(path)` + `model_pose.predict(frame, ...)` dans 4_H.py.

    Format de sortie HEF attendu (2 sorties) :
        Sortie 0 — Détections : [1, max_det, 6]
            [y_min, x_min, y_max, x_max, confidence, class_id]
        Sortie 1 — Keypoints  : [1, max_det, 51]
            17 keypoints × 3 valeurs (x_norm, y_norm, visibilité)

    Keypoints COCO (indices) :
        0=nez, 1=oeil_g, 2=oeil_d, 3=oreille_g, 4=oreille_d,
        5=épaule_g, 6=épaule_d, 7=coude_g, 8=coude_d,
        9=poignet_g, 10=poignet_d, 11=hanche_g, 12=hanche_d,
        13=genou_g, 14=genou_d, 15=cheville_g, 16=cheville_d
    """

    N_KEYPOINTS = 17

    def __init__(self, hef_path: str, input_size: Tuple[int, int] = (640, 640)):
        super().__init__(hef_path, input_size)
        import onnxruntime as ort
        postprocess_path = hef_path.replace('.hef', '_postprocess.onnx')
        self._ort_session = ort.InferenceSession(postprocess_path)

        out_names      = [o.name for o in self._out_info]
        self._det_out  = out_names[0]
        self._kp_out   = out_names[1] if len(out_names) > 1 else None

        if self._kp_out is None:
            print("[HailoPoseEstimator] ⚠️  Aucune sortie keypoints dans le HEF."
                  " Seules les bboxes seront disponibles.")

    # ── API publique ──────────────────────────────────────────────────────────
    def predict(
        self,
        frame: np.ndarray,
        conf: float = 0.1,
        save: bool = False,
        verbose: bool = False,
    ) -> List[Any]:
        """
        API strictement compatible avec YOLO Pose :
            results = model_pose.predict(frame, conf=0.1, save=False, verbose=False)
            for r in results:
                for box, kp in zip(r.boxes.xyxy, r.keypoints.xy):
                    ...

        Retourne [HailoPoseResult] avec :
            .boxes.xyxy  → np.ndarray (N, 4)
            .keypoints.xy → liste de np.ndarray (N_kp, 2) par personne
        """
        orig_h, orig_w = frame.shape[:2]
        raw            = self._infer_raw(frame)
        
        hailo_output_names = [o.name for o in self._out_info]
        ort_input_names = [i.name for i in self._ort_session.get_inputs()]
        ort_input_shapes = {i.name: i.shape for i in self._ort_session.get_inputs()}
        ort_inputs = {}
        for hailo_name in hailo_output_names:
            tensor = np.transpose(raw[hailo_name], (0, 3, 1, 2))
            t_shape = tensor.shape
            for ort_name in ort_input_names:
                if ort_name in ort_inputs:
                    continue
                expected = ort_input_shapes[ort_name]
                if expected[1] == t_shape[1] and expected[2] == t_shape[2]:
                    ort_inputs[ort_name] = tensor
                    break
        ort_outputs = self._ort_session.run(None, ort_inputs)

        
        # Post-traitement via ONNX (décodage bboxes + keypoints)
        
        # Décodage du tenseur brut (1, 56, 8400)
        preds = ort_outputs[0][0].T  # (8400, 56)
        boxes, keypoints = [], []
        for pred in preds:
            confidence = pred[4]
            if confidence < conf:
                continue
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            x1 = (cx - w / 2) / 640 * orig_w
            y1 = (cy - h / 2) / 640 * orig_h
            x2 = (cx + w / 2) / 640 * orig_w
            y2 = (cy + h / 2) / 640 * orig_h
            boxes.append([x1, y1, x2, y2])
            kp_raw = pred[5:].reshape(self.N_KEYPOINTS, 3)
            kp_px = np.stack([kp_raw[:, 0] / 640 * orig_w, kp_raw[:, 1] / 640 * orig_h], axis=1)
            keypoints.append(kp_px)

        return [_HailoPoseResult(boxes, keypoints)]





class _HailoPoseResult:
    """
    Émule YOLO Pose Results pour compatibilité avec 4_H.py :
        results[0].boxes.xyxy   → coordonnées bboxes
        results[0].keypoints.xy → liste de keypoints par personne
    """

    class _Boxes:
        def __init__(self, xyxy: List):
            self.xyxy = (
                np.array(xyxy, dtype=np.float32) if xyxy
                else np.zeros((0, 4), dtype=np.float32)
            )

    class _Keypoints:
        def __init__(self, kps: List[np.ndarray]):
            self.xy = kps  # liste de (17, 2) — un tableau par personne détectée

    def __init__(self, boxes: List, keypoints: List[np.ndarray]):
        self.boxes     = self._Boxes(boxes)
        self.keypoints = self._Keypoints(keypoints)
