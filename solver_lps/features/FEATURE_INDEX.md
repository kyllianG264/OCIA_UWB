# Solver LPS Feature Index

## Root contracts

- `solver_lps/session_assets.py`: canonical sport/set paths shared without feature-to-feature imports.
- `presentation/pages`: routing, user commands and rendering only.
- `features/...`: data access, application use cases and domain rules.

## Features

### `features/cv`

- Owns CV videos, raw tracking, merged CV generation and review video playback.
- `review/generate_cv_positions.py`: `generate_cv_positions(session) -> Path`.
- `review/tracking_launcher.py`: validates inputs, synchronizes videos and launches CV tracking/calibration.
- `review/data/session_assets.py`: compatibility facade over the neutral session paths.
- `review/playback/video_playback.py`: the only review video playback implementation.
- `review/analysis/split_court` and `full_court`: explicit tracking strategies.
- `grayzone/data/raw_input.py`: validates raw CV observations and reads CV calibration.
- `grayzone/domain/gray_zone.py`: pure camera-visibility and blind-zone calculation.
- `grayzone/application/gray_zone_analysis.py`: `build_gray_zone(...)` use case.
- `grayzone/data/gray_zone_output.py`: grayzone metadata, mask and risk-map output.

### `features/uwb`

- Owns UDP acquisition, raw UWB files, review input, calculations and merged UWB output.
- `acquisition/data`: UDP input, raw UWB recording output, review input and UWB-owned path facade.
- `calculus/data/raw_input.py`: reads recorded raw UWB frames for calculations.
- `calculus`: 2D, 3D and compatibility 3D-to-2D calculations and merged writers.
- `orchestration/review_mode.py`: `generate_uwb_positions(session, mode) -> Path`.
- UWB has no dependency on CV.

### `features/players`

- Owns normalization of CV/UWB position rows and player-facing models.
- `data/merged_input.py`: positions-only merged reader and temporal packets.
- `application/player_timeline.py`: `build_player_timeline(session, sources) -> list[PlayerFrame]`.
- `domain`: `PlayerFrame`, registry and analytics.
- Players owns no video, calibration, terrain, projection or UI.

### `features/ground`

- Owns terrain assets, calibration loading, court geometry, projection and bounds.
- `data/ground_input.py`: ground paths through the neutral session contract.
- `domain/calibration.py`: calibrated geometry and split-axis interpretation.
- `domain/court_geometry.py` and `projection.py`: geometry and screen projection.

## Presentation

- `home_page.py`: routes CV tracking, UDP reader and the active 2D solver.
- `cv_tracking_page.py`: CV tracking UI and command assembly.
- `estimation_2d_page.py`: selects realtime or review presentation.
- `estimation_2d_review_source.py`: composes feature outputs without owning CSV rules.
- `review_clock.py`: one playback position shared by CV, UWB and video.
- `estimation_2d_review_ui.py`, `scene.py`, `display.py`: events, scene state and rendering.
- `udp_viewer_page.py`: routes acquisition viewers.

The former 3D pages are intentionally absent until a dedicated 3D presentation uses the real UWB orchestration.
