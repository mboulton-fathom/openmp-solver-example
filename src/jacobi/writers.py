import csv
from pathlib import Path

import numpy as np


def write_csv(path: Path, grid: np.ndarray) -> None:
    """Write one CSV row per grid row, with full float precision."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in grid:
            writer.writerow(repr(float(v)) for v in row)
