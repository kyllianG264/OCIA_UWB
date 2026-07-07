from pathlib import Path

from solver_lps.features.cv.grayzone.data.gray_zone_output import write_gray_zone
from solver_lps.features.cv.grayzone.data.raw_input import iter_raw_positions, read_calibration
from solver_lps.features.cv.grayzone.domain.gray_zone import compute_gray_zone


def build_gray_zone(raw_csv_path, calibration_path, *, output_dir=None, grid_step=8.0):
    calibration, calibration_data = read_calibration(calibration_path)
    gray_zone = compute_gray_zone(
        calibration,
        calibration_data,
        raw_rows=iter_raw_positions(raw_csv_path),
        grid_step=grid_step,
    )
    gray_zone.metadata.update(
        {
            "raw_csv_path": str(Path(raw_csv_path).resolve()),
        }
    )
    if output_dir is None:
        return gray_zone
    return gray_zone, write_gray_zone(gray_zone, output_dir)
