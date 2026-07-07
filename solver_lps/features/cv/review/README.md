# review

Ce dossier contient la logique principale de review CV.

## Role

- charger une session CV
- relancer la generation des positions si besoin
- fournir les donnees necessaires a l'UI de review

## Sous-dossiers

- `analysis` : post-traitement et fusion des trajectoires
- `data` : acces aux assets de session
- `generation` : calibration et raw tracking
- `playback` : lecture video

## Fichiers importants

- `generate_cv_positions.py` : relance de generation CV
- `tracking_launcher.py` : point d'appui pour lancer le tracking
