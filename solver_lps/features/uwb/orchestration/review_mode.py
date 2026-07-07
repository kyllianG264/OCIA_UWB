"""Data-only generation of merged UWB position CSV files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from solver_lps.features.uwb.acquisition.data.session_assets import session_assets
from solver_lps.features.uwb.calculus.data.raw_input import load_raw_frames


CalculationMode = Literal["two_d", "three_d", "three_d_to_2d"]
CALCULATION_MODES = ("two_d", "three_d", "three_d_to_2d")


def normalize_calculation_mode(calculation_mode: str) -> str:
    mode = str(calculation_mode or "").strip().lower()
    if mode not in CALCULATION_MODES:
        raise ValueError(f"Unsupported UWB calculation mode: {calculation_mode!r}")
    return mode


def _coerce_session(session):
    if session is None:
        return session_assets()
    if hasattr(session, "uwb_raw_path") and hasattr(session, "uwb_output_dir"):
        return session
    if isinstance(session, dict):
        return session_assets(session.get("sport", "basket"), session.get("asset_set", "set1"))
    raise TypeError("session must follow solver_lps.session_assets.SessionAssets")


def _load_anchors(layout_path, active_ids, mode, settings):
    with Path(layout_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    layouts = payload.get("layouts", {})
    layout = layouts.get(str(len(active_ids)))
    if layout is None:
        raise ValueError(f"No anchor layout for {len(active_ids)} active anchors")
    raw_anchors = layout.get("anchors", {})
    anchors = {}
    anchor_height = float(settings.get("anchor_height_cm", 250.0))
    for anchor_id in active_ids:
        coordinates = raw_anchors.get(str(anchor_id))
        if coordinates is None:
            raise ValueError(f"Anchor {anchor_id} is missing from {layout_path}")
        if mode == "three_d":
            anchors[anchor_id] = tuple(float(value) for value in coordinates[:3])
            if len(anchors[anchor_id]) == 2:
                anchors[anchor_id] += (anchor_height,)
        else:
            anchors[anchor_id] = tuple(float(value) for value in coordinates[:2])
    return anchors


def _calculate_row(mode, frame, anchors, settings):
    raw = frame.get("distances", {})
    point = None
    source = f"uwb_{mode}"
    status = "insufficient_distances"

    if mode == "three_d":
        from solver_lps.features.uwb.calculus.domain.three_d.position_calcul import (
            estimate_position_3d_least_squares,
        )

        point = estimate_position_3d_least_squares(anchors, raw)
        status = "ok_coplanar_anchor_height_assumption" if point is not None else status
    else:
        from solver_lps.features.uwb.calculus.domain.two_d.position_calcul import (
            estimate_position_least_squares,
        )

        point = estimate_position_least_squares(anchors, raw)
        if point is not None:
            status = "ok"
        if mode == "three_d_to_2d":
            source = "uwb_three_d_to_2d_direct_2d_compatibility"
            status = (
                "direct_2d_from_ranges_no_3d_projection"
                if point is not None
                else "insufficient_distances_no_3d_projection"
            )

    row = {
        "frame": int(frame.get("frame", 0)),
        "timestamp_s": float(frame.get("timestamp_s", 0.0)),
        "player_id": str(frame.get("player_id") or settings.get("player_id") or "tag:uwb"),
        "x_cm": None if point is None else point[0],
        "y_cm": None if point is None else point[1],
        "valid": point is not None,
        "source": source,
        "status": status,
    }
    if mode == "three_d":
        row["z_cm"] = None if point is None else point[2]
    elif mode == "three_d_to_2d":
        row["projected_x_cm"] = None if point is None else point[0]
        row["projected_y_cm"] = None if point is None else point[1]
    return row


def build_calculator(calculation_mode, anchors=None, settings=None):
    mode = normalize_calculation_mode(calculation_mode)
    if anchors is None:
        if mode == "two_d":
            from solver_lps.features.uwb.calculus.domain.two_d.position_calcul import (
                update_position_solution,
            )
        elif mode == "three_d":
            from solver_lps.features.uwb.calculus.domain.three_d.position_calcul import (
                update_position_solution,
            )
        else:
            from solver_lps.features.uwb.calculus.domain.three_d_to_2d.position_calcul import (
                update_position_solution,
            )
        return update_position_solution
    settings = dict(settings or {})
    return lambda frame: _calculate_row(mode, frame, anchors, settings)


def _write_rows(mode, rows, output_path):
    if mode == "two_d":
        from solver_lps.features.uwb.calculus.data.merged_2d_output import write_2d_output

        write_2d_output(rows, output_path)
    elif mode == "three_d":
        from solver_lps.features.uwb.calculus.data.merged_3d_output import write_3d_output

        write_3d_output(rows, output_path)
    else:
        from solver_lps.features.uwb.calculus.data.merged_3d_to_2d_output import (
            write_3d_to_2d_output,
        )

        write_3d_to_2d_output(rows, output_path)


def run_review(
    *,
    calculation_mode: CalculationMode = "two_d",
    input_path=None,
    output_path=None,
    anchors=None,
    settings=None,
    session=None,
    progress_callback=None,
):
    """Calculate all raw frames, write the mode-specific merged CSV, and return its path."""
    mode = normalize_calculation_mode(calculation_mode)
    current_session = _coerce_session(session)
    input_path = Path(input_path or current_session.uwb_raw_path)
    output_path = Path(output_path or current_session.uwb_positions_path(mode))
    frames = load_raw_frames(input_path)
    settings = dict(settings or {})
    active_ids = sorted({anchor_id for frame in frames for anchor_id in frame["distances"]})
    if anchors is None and active_ids:
        anchors = _load_anchors(current_session.anchors_layout_path, active_ids, mode, settings)
    anchors = dict(anchors or {})
    calculator = build_calculator(mode, anchors, settings)
    total_frames = len(frames)
    if progress_callback is not None:
        progress_callback(0, total_frames)
    rows = []
    for index, frame in enumerate(frames, start=1):
        rows.append(calculator(frame))
        if progress_callback is not None:
            progress_callback(index, total_frames)
    _write_rows(mode, rows, output_path)
    return output_path


def generate_uwb_positions(session, mode) -> Path:
    """Stable session-level UWB generation contract."""
    return run_review(calculation_mode=mode, session=session)
