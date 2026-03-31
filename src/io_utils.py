"""Input/output helpers for writing experiment metrics artifacts."""

from pathlib import Path
from typing import List

import numpy as np

from .config import METRICS_HISTORY_HEADER


History = List[List[float]]


class MetricsWriter:
    """Persist experiment metrics into reproducible CSV artifacts."""

    @staticmethod
    def save_history_csv(history: History, out_path: Path) -> None:
        """Save epoch history rows as a CSV with a fixed header schema."""
        np.savetxt(
            out_path,
            np.array(history),
            delimiter=",",
            header=METRICS_HISTORY_HEADER,
            comments="",
        )
