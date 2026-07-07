"""Data-only realtime UWB acquisition and position calculation."""

from __future__ import annotations

import time
from pathlib import Path

from solver_lps.features.uwb.acquisition.data.session_assets import session_assets
from solver_lps.features.uwb.acquisition.data.udp_input import UdpInput
from solver_lps.features.uwb.acquisition.domain.udp_reader import UdpReader
from solver_lps.features.uwb.orchestration.review_mode import (
    _load_anchors,
    _write_rows,
    build_calculator,
    normalize_calculation_mode,
)


def run_realtime(
    *,
    calculation_mode="two_d",
    udp_host="0.0.0.0",
    udp_port=4210,
    anchors=None,
    output_path=None,
    settings=None,
    stop_event=None,
    max_age_s=2.0,
):
    """Consume UDP until stopped, optionally write merged rows, and return rows or output path."""
    mode = normalize_calculation_mode(calculation_mode)
    settings = dict(settings or {})
    if anchors is None:
        current_session = session_assets()
        anchors = _load_anchors(current_session.anchors_layout_path, [1, 2, 3, 4], mode, settings)
    anchors = dict(anchors)
    calculator = build_calculator(mode, anchors, settings)
    reader = UdpReader(UdpInput(bind_ip=udp_host, port=udp_port), max_age_s=max_age_s)
    rows = []
    frame_index = 0
    started_at = time.time()
    try:
        while stop_event is None or not stop_event.is_set():
            distances = reader.get_distances(sorted(anchors))
            if distances:
                rows.append(
                    calculator(
                        {
                            "frame": frame_index,
                            "timestamp_s": time.time() - started_at,
                            "distances": distances,
                        }
                    )
                )
                frame_index += 1
            time.sleep(0.01)
    finally:
        reader.close()
    if output_path is not None:
        output_path = Path(output_path)
        _write_rows(mode, rows, output_path)
        return output_path
    return rows
