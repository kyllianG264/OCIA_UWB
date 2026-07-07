"""2D merged output writers for UWB calculus."""

import csv
import os


FIELDNAMES = ("frame", "timestamp_s", "player_id", "x_cm", "y_cm", "valid", "source", "status")


def write_2d_output(rows, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
