import csv
import json
from pathlib import Path

from solver_lps.features.ground.domain.calibration import load_terrain_calibration


RAW_POSITION_COLUMNS = {"frame", "player_id", "X", "Y", "cam"}


def iter_raw_positions(path):
    source = Path(path).resolve()
    stream = source.open("r", encoding="utf-8-sig", newline="")
    try:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = RAW_POSITION_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{source} is not a CV raw positions file; missing columns: {', '.join(sorted(missing))}"
            )
        if "stable_id" in columns or "status" in columns:
            raise ValueError(f"{source} is a tracking export; grayzone accepts raw CV observations only")
        yield from reader
    finally:
        stream.close()


def read_raw_positions(path):
    return list(iter_raw_positions(path))


def read_calibration(path):
    source = Path(path).resolve()
    normalized = load_terrain_calibration(str(source))
    if normalized is None:
        return None, None
    with source.open("r", encoding="utf-8") as stream:
        return normalized, json.load(stream)
