from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

MODALITIES = ("image", "tabular", "timeseries")  # column order of every availability mask
TAB_FEATURES = [
    "age", "sex", "bmi", "sbp", "dbp", "heart_rate_adm",
    "creatinine", "lactate", "wbc", "hemoglobin", "sodium", "glucose",
]
TS_CHANNELS = ["heart_rate", "spo2"]

# Plausible physiological ranges used for validation (and clipping of synthetic data).
TAB_RANGES: Dict[str, Tuple[float, float]] = {
    "age": (18, 100), "sex": (0, 1), "bmi": (12, 60), "sbp": (50, 260), "dbp": (25, 150),
    "heart_rate_adm": (30, 220), "creatinine": (0.2, 15), "lactate": (0.2, 25),
    "wbc": (0.5, 60), "hemoglobin": (4, 20), "sodium": (110, 170), "glucose": (30, 800),
}
TS_RANGES: Dict[str, Tuple[float, float]] = {"heart_rate": (20, 250), "spo2": (50, 100)}


@dataclass
class Cohort:
    """One research cohort. Absent modalities are NaN (tabular/ts) or zeros (image) with a flag."""

    patient_id: np.ndarray  # (N,) str
    y: np.ndarray  # (N,) int {0,1}
    tab: pd.DataFrame  # (N, F), NaN = missing
    ts: np.ndarray  # (N, C, T) float32, NaN = missing
    img: np.ndarray  # (N, H, W) float32 in [0, 1]
    has_img: np.ndarray  # (N,) bool
    has_tab: np.ndarray
    has_ts: np.ndarray
    lesion_mask: "np.ndarray | None" = None  # (N, H, W) bool; ground truth, synthetic data only

    def __len__(self) -> int:
        return len(self.y)

    @property
    def avail(self) -> np.ndarray:
        """(N, 3) bool in MODALITIES order."""
        return np.stack([self.has_img, self.has_tab, self.has_ts], axis=1)

    def subset(self, idx: np.ndarray) -> "Cohort":
        return Cohort(
            patient_id=self.patient_id[idx], y=self.y[idx], tab=self.tab.iloc[idx].reset_index(drop=True),
            ts=self.ts[idx], img=self.img[idx], has_img=self.has_img[idx], has_tab=self.has_tab[idx],
            has_ts=self.has_ts[idx],
            lesion_mask=None if self.lesion_mask is None else self.lesion_mask[idx],
        )
