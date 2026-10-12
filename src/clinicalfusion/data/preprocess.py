"""Train-set-fitted normalisation and conversion of a Cohort into model-ready tensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .schema import TAB_FEATURES, Cohort


@dataclass
class Normalizers:
    tab_median: np.ndarray
    tab_mean: np.ndarray
    tab_std: np.ndarray
    ts_mean: np.ndarray
    ts_std: np.ndarray
    img_mean: float
    img_std: float

    @classmethod
    def fit(cls, c: Cohort) -> "Normalizers":
        """Fit on observed values of a (training) cohort only, to avoid leakage."""
        tab = c.tab.loc[c.has_tab, TAB_FEATURES].to_numpy(dtype=float)
        med = np.nanmedian(tab, axis=0)
        filled = np.where(np.isnan(tab), med, tab)
        ts = c.ts[c.has_ts]  # (n, C, T)
        ts_mean = np.nanmean(ts, axis=(0, 2))
        ts_std = np.nanstd(ts, axis=(0, 2)) + 1e-6
        imgs = c.img[c.has_img]
        return cls(
            tab_median=med, tab_mean=filled.mean(0), tab_std=filled.std(0) + 1e-6,
            ts_mean=ts_mean, ts_std=ts_std, img_mean=float(imgs.mean()), img_std=float(imgs.std() + 1e-6),
        )

    # -- encoders for the neural nets ---------------------------------------------------------
    def tab_nn(self, df: pd.DataFrame) -> np.ndarray:
        x = df[TAB_FEATURES].to_numpy(dtype=float)
        miss = np.isnan(x)
        x = np.where(miss, self.tab_median, x)
        x = (x - self.tab_mean) / self.tab_std
        return np.concatenate([x, miss.astype(float)], axis=1).astype(np.float32)  # (N, 2F)

    def ts_nn(self, ts: np.ndarray) -> np.ndarray:
        obs = ~np.isnan(ts)
        z = (ts - self.ts_mean[None, :, None]) / self.ts_std[None, :, None]
        z = np.where(obs, z, 0.0)
        return np.concatenate([z, obs.astype(float)], axis=1).astype(np.float32)  # (N, 2C, T)

    def img_nn(self, img: np.ndarray) -> np.ndarray:
        return ((img - self.img_mean) / self.img_std)[:, None].astype(np.float32)  # (N, 1, H, W)


@dataclass
class Tensors:
    img: torch.Tensor  # (N,1,H,W)
    tab: torch.Tensor  # (N,2F)
    ts: torch.Tensor  # (N,2C,T)
    avail: torch.Tensor  # (N,3) bool  [image, tabular, timeseries]
    y: torch.Tensor  # (N,) float

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: "Sequence[int] | np.ndarray | torch.Tensor") -> "Tensors":
        idx = torch.as_tensor(np.asarray(idx) if not isinstance(idx, torch.Tensor) else idx, dtype=torch.long)
        return Tensors(self.img[idx], self.tab[idx], self.ts[idx], self.avail[idx], self.y[idx])

    def with_avail(self, avail: torch.Tensor) -> "Tensors":
        return Tensors(self.img, self.tab, self.ts, avail, self.y)


def to_tensors(c: Cohort, norm: Normalizers, y: Optional[np.ndarray] = None) -> Tensors:
    return Tensors(
        img=torch.from_numpy(norm.img_nn(c.img)),
        tab=torch.from_numpy(norm.tab_nn(c.tab)),
        ts=torch.from_numpy(norm.ts_nn(c.ts)),
        avail=torch.from_numpy(c.avail.copy()),
        y=torch.from_numpy((c.y if y is None else y).astype(np.float32)),
    )


def stratified_split(y: np.ndarray, val_frac: float, test_frac: float, seed: int):
    """Patient-level stratified split (use group-aware splitting if one patient has many records)."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for cls in (0, 1):
        idx = rng.permutation(np.where(y == cls)[0])
        n_te, n_va = int(round(len(idx) * test_frac)), int(round(len(idx) * val_frac))
        te.append(idx[:n_te]); va.append(idx[n_te:n_te + n_va]); tr.append(idx[n_te + n_va:])
    cat = lambda parts: np.sort(np.concatenate(parts))  # noqa: E731
    return cat(tr), cat(va), cat(te)
