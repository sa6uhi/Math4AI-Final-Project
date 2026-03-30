from pathlib import Path
from typing import List

import numpy as np


History = List[List[float]]


class MetricsWriter:
    @staticmethod
    def save_history_csv(history: History, out_path: Path) -> None:
        np.savetxt(
            out_path,
            np.array(history),
            delimiter=",",
            header="epoch,train_loss,val_loss,val_accuracy",
            comments="",
        )
