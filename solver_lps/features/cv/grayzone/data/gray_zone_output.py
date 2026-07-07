import json
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


def _save_map(path, values):
    if Image is None:
        return None
    maximum = float(np.max(values)) if values.size else 0.0
    normalized = values / maximum if maximum > 1e-9 else values
    image = Image.fromarray(np.uint8(np.clip(normalized, 0.0, 1.0) * 255), mode="L")
    image.save(path)
    return str(path)


def write_gray_zone(gray_zone, output_dir):
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)

    risk_path = _save_map(target / "gray_zone_risk.png", gray_zone.risk_map)
    mask_path = _save_map(target / "gray_zone_mask.png", gray_zone.gray_zone_mask.astype(float))
    metadata_path = target / "gray_zone_metadata.json"
    payload = {
        "bounds": list(gray_zone.bounds),
        "split_y": gray_zone.split_y,
        "grid_step": gray_zone.grid_step,
        "metadata": gray_zone.metadata,
        "polygons": gray_zone.gray_zone_polygons,
        "disappearance_cells": gray_zone.disappearance_cells,
        "disappearance_points": gray_zone.disappearance_points,
        "handoff_points": gray_zone.handoff_points,
        "handoff_rejected_points": gray_zone.handoff_rejected_points,
        "handoff_ellipse": gray_zone.handoff_ellipse,
        "files": {"risk": risk_path, "mask": mask_path},
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"metadata": metadata_path, "risk": risk_path, "mask": mask_path}
