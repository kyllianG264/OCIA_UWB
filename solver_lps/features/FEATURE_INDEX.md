# Solver LPS Feature Index

## Root Rule
- `presentation/pages` route vers une feature
- `features/...` contient la logique metier

## Features

### `features/uwb`
- `realtime/`
  - `data/`: UDP live
  - `two_d/`: solveur 2D live
  - `three_d/`: solveur 3D live
  - `three_d_to_2d/`: solveur 3D -> 2D live
- `review/`
  - `data/`: CSV review UWB
  - `two_d/`: solveur 2D review
  - `three_d/`: solveur 3D review
  - `three_d_to_2d/`: solveur 3D -> 2D review
- `tests/`: tests UWB partages

### `features/cv`
- `realtime/`
  - `data/`: sources live CV
  - `data/cv_realtime_source.py`: source vision live placeholder/interface
  - `domain/`: logique tracking live
  - `presentation/widgets/`: widgets CV live
- `review/`
  - `data/cv_logs/`: CSV, video, calibration review
  - `data/cv_log_source.py`: lecture review CV
  - `domain/`: tracking review
  - `presentation/cli/`: script tracking review
  - `tests/`: tests tracking review

### `features/players`
- `domain/`: registry, analytics, heatmaps, selection
- `presentation/widgets/`: UI joueur
- `tests/`: tests joueur

### `features/ground`
- `domain/`: geometrie, calibration, projection
- `presentation/widgets/`: rendu terrain
- `tests/`: tests terrain

### `features/udp_viewer`
- `realtime/`: viewer UDP live brut
- `review/`: viewer CSV review brut

## Pages
- `presentation/pages/estimation_2d_page.py`
  - route vers `uwb/realtime/two_d` ou `uwb/review/two_d`
- `presentation/pages/estimation_3d_page.py`
  - route vers `uwb/realtime/three_d` ou `uwb/review/three_d`
- `presentation/pages/estimation_3d_to_2d_page.py`
  - route vers `uwb/realtime/three_d_to_2d` ou `uwb/review/three_d_to_2d`
- `presentation/pages/udp_viewer_page.py`
  - route vers `udp_viewer/realtime` ou `udp_viewer/review`

## Quick Navigation
- UWB live 2D: `features/uwb/realtime/two_d`
- UWB review 2D: `features/uwb/review/two_d`
- CV live source: `features/cv/realtime/data/cv_realtime_source.py`
- CV review source: `features/cv/review/data/cv_log_source.py`
- CV logs: `features/cv/review/data/cv_logs`
- Player logic: `features/players/domain`
- Ground logic: `features/ground/domain`
