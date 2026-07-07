"""Application entry point for producing the canonical CV positions export."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from solver_lps.session_assets import DEFAULT_SET, DEFAULT_SPORT, SessionAssets


def _value(session, name, default=None):
    if isinstance(session, Mapping):
        return session.get(name, default)
    return getattr(session, name, default)


def _assets_for(session) -> SessionAssets:
    assets = _value(session, "assets")
    if isinstance(assets, SessionAssets):
        return assets
    if isinstance(session, SessionAssets):
        return session
    return SessionAssets(
        sport=_value(session, "sport", DEFAULT_SPORT),
        asset_set=_value(session, "asset_set", DEFAULT_SET),
    )


def resolve_court_mode(session) -> str:
    """Resolve an explicit mode, falling back to the sport's court topology."""
    mode = str(_value(session, "court_mode", "auto") or "auto").strip().lower().replace("_", "-")
    aliases = {"split": "split", "split-court": "split", "full": "full", "full-court": "full"}
    if mode != "auto":
        if mode not in aliases:
            raise ValueError(f"Unsupported CV court mode: {mode!r}")
        return aliases[mode]
    return "split" if _assets_for(session).sport.lower() == "basket" else "full"


def generate_cv_positions(session, progress_callback=None) -> Path:
    """Generate ``positions_merged.csv`` for a configured solver session."""
    assets = _assets_for(session)
    input_path = Path(_value(session, "input_path", assets.cv_positions_raw_path))
    output_path = Path(_value(session, "output_path", assets.cv_positions_merged_path))
    calibration_path = Path(_value(session, "calibration_path", assets.calibration_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = resolve_court_mode(session)
    expected_players = _value(session, "expected_players")
    if mode == "split":
        from solver_lps.features.cv.review.analysis.split_court.data.merged_output import write_output
        from solver_lps.features.cv.review.analysis.split_court.domain.tracking_pipeline import (
            _build_tracking_args,
            run_tracking,
        )

        overrides = {
            "calibration": str(calibration_path),
            "merge_distance": float(_value(session, "merge_distance", 40.0)),
        }
        if expected_players is not None:
            overrides["expected_players"] = int(expected_players)
        args = _build_tracking_args(str(input_path), str(output_path), **overrides)
    else:
        from solver_lps.features.cv.review.analysis.full_court.data.track_exporter import write_output
        from solver_lps.features.cv.review.analysis.full_court.domain.tracking_pipeline import (
            default_config,
            run_tracking,
        )

        overrides = {"calibration": str(calibration_path)}
        if expected_players is not None:
            overrides["expected_players"] = int(expected_players)
        args = default_config(str(input_path), str(output_path), **overrides)

    args.progress_callback = progress_callback
    rows, _stats = run_tracking(args)
    write_output(rows, str(output_path))
    return output_path
