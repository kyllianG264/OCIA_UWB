# NeptuVision

NeptuVision est un projet Python de vision par ordinateur centre sur le pipeline suivant :

1. calibration manuelle de deux cameras,
2. detection des joueurs dans chaque camera,
3. projection des detections sur un plan 2D via homographie,
4. tracking des positions projetees,
5. fusion et resolution des positions.

## Architecture

```text
main.py                        point d'entree unique

app/
  pipeline.py                  orchestration CLI

assets/
  models/
    pose/                      poids et exports YOLO/Hailo
  terrain/                     image de reference 2D
  videos/
    samples/                   videos d'exemple

core/
  utils.py                     utilitaires partages
  paths.py                     chemins centralises

docs/
  README.md                    documentation
  requirements.txt             dependances Python

features/
  0_dataset/
    extract_frames.py          extraction de frames
  1_calibration/
    1_CAL.py                   point d'entree calibration
    streamlit_app.py           implementation Streamlit
  2_detection/
    2_detector.py              detection frame par frame sans tracking
  3_projection/
    3_projector.py             projection terrain 2D
  4_tracking/
    4_tracker.py               tracking des positions projetees
    4_render_video.py          rendu video composite
    publisher.py               wrapper de compatibilite
    hailo_infer_impl.py        implementation Hailo
  5_fusion/
    5_solver.py                fusion / resolution finale
```

## Pipeline actuel

```text
Videos brutes
    |
    +--> features/0_dataset/extract_frames.py      (optionnel)
    |
    +--> features/1_calibration/1_CAL.py           -> calibration.json
    |
    +--> features/2_detection/2_detector.py        -> detections.csv
    |
    +--> features/3_projection/3_projector.py      -> projected_positions.csv
    |
    +--> features/4_tracking/4_tracker.py          -> positions_raw.csv + tracking_assignments.csv + MQTT
    |
    +--> features/4_tracking/4_render_video.py     -> tracking_composite.mp4
    |
    `--> features/5_fusion/5_solver.py             -> positions_solved.csv + positions_merged.csv
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r docs/requirements.txt
```

## Lancement

Pipeline complet :

```powershell
python .\main.py --video_gauche .\assets\videos\samples\gauche.mp4 --video_droite .\assets\videos\samples\droite.mp4 --calibration .\NeptuVisionResults\2_Calibration\calibration.json
```

Cette commande genere aussi une video composite dans `NeptuVisionResults\5_Tracking\tracking_composite.mp4`.

Si tu veux sauter le rendu video :

```powershell
python .\main.py --video_gauche .\assets\videos\samples\gauche.mp4 --video_droite .\assets\videos\samples\droite.mp4 --calibration .\NeptuVisionResults\2_Calibration\calibration.json --skip_tracking_video
```

Calibration seule :

```powershell
python .\features\1_calibration\1_CAL.py
```

Si tu veux lancer Streamlit directement :

```powershell
streamlit run .\features\1_calibration\1_CAL.py
```
