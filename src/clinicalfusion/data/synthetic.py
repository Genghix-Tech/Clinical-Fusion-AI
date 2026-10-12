"""Synthetic multimodal cohort.

Every modality is generated from its OWN independent latent factor, and the outcome depends on all
three. So each modality carries complementary information and fusion *should* help. Real data will
not behave this cleanly: this generator exists to exercise and test the pipeline, not to claim
clinical results. Results on it must never be reported as clinical findings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from ..config import Config
from .schema import TAB_FEATURES, TAB_RANGES, TS_CHANNELS, TS_RANGES, Cohort


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_images(rng: np.random.Generator, L0: np.ndarray, size: int):
    n = len(L0)
    ax = np.linspace(0, 1, size, dtype=np.float32)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    lungs = np.zeros((size, size), np.float32)
    for cx in (0.3, 0.7):  # two lung fields
        lungs = np.maximum(lungs, np.exp(-(((xx - cx) / 0.17) ** 4 + ((yy - 0.5) / 0.32) ** 4)))
    base = 0.55 - 0.3 * lungs  # body is bright, lung fields darker

    texture = gaussian_filter(rng.normal(size=(n, size, size)).astype(np.float32), sigma=(0, 3, 3))
    texture /= texture.std() + 1e-8
    img = base[None] * rng.uniform(0.9, 1.1, size=(n, 1, 1)).astype(np.float32)
    img = img + 0.04 * texture + 0.07 * rng.normal(size=(n, size, size)).astype(np.float32)

    amp = 0.4 * _sigmoid(2.0 * L0 - 0.5)  # opacity strength follows the image latent
    side = rng.integers(0, 2, size=n)
    cx = np.where(side == 0, 0.3, 0.7) + rng.uniform(-0.06, 0.06, size=n)
    cy = rng.uniform(0.45, 0.7, size=n)
    sig = rng.uniform(0.05, 0.09, size=n)
    g = np.exp(-(((xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2)
                 / (2 * sig[:, None, None] ** 2))).astype(np.float32)
    img = img + amp[:, None, None].astype(np.float32) * g
    lesion = (g > 0.5) & (amp[:, None, None] > 0.05)
    return np.clip(img, 0, 1).astype(np.float32), lesion


def _make_tabular(rng: np.random.Generator, L1: np.ndarray) -> pd.DataFrame:
    n = len(L1)
    n01 = lambda: rng.normal(size=n)  # noqa: E731
    d = {
        "age": 62 + 4 * L1 + 13 * n01(),
        "sex": (rng.random(n) < 0.5).astype(float),
        "bmi": 27 + 5 * n01(),
        "sbp": 122 - 7 * L1 + 16 * n01(),
        "dbp": 72 - 3 * L1 + 10 * n01(),
        "heart_rate_adm": 85 + 6 * L1 + 12 * n01(),
        "creatinine": np.exp(0.1 + 0.30 * L1 + 0.35 * n01()),
        "lactate": np.exp(0.3 + 0.35 * L1 + 0.30 * n01()),
        "wbc": 9 + 2.2 * L1 + 3 * n01(),
        "hemoglobin": 12.5 - 0.5 * L1 + 1.8 * n01(),
        "sodium": 139 - 0.8 * L1 + 3.5 * n01(),
        "glucose": 115 + 10 * L1 + 30 * np.abs(n01()),
    }
    df = pd.DataFrame(d)[TAB_FEATURES]
    for c, (lo, hi) in TAB_RANGES.items():
        df[c] = df[c].clip(lo, hi)
    # item-level missingness (labs are not always drawn)
    for c, p in {"lactate": 0.25, "creatinine": 0.05, "wbc": 0.05, "glucose": 0.08, "hemoglobin": 0.06}.items():
        df.loc[rng.random(n) < p, c] = np.nan
    return df


def _make_timeseries(rng: np.random.Generator, L2: np.ndarray, T: int) -> np.ndarray:
    n = len(L2)
    ramp = np.clip((np.arange(T) - (T - 16)) / 16.0, 0, 1)[None]  # deterioration only in the last 16h

    def ar1(phi: float, sd: float) -> np.ndarray:
        e = rng.normal(scale=sd, size=(n, T))
        out = np.zeros((n, T))
        for t in range(1, T):
            out[:, t] = phi * out[:, t - 1] + e[:, t]
        return out

    hr = (78 + 10 * rng.normal(size=(n, 1))) + 12 * L2[:, None] * ramp + ar1(0.7, 2.5)
    sp = (96.5 + 1.0 * rng.normal(size=(n, 1))) - 2.0 * L2[:, None] * ramp + ar1(0.6, 0.8)
    hr = np.clip(hr, *TS_RANGES["heart_rate"])
    sp = np.clip(sp, *TS_RANGES["spo2"])
    ts = np.stack([hr, sp], axis=1).astype(np.float32)
    ts[rng.random(ts.shape) < 0.08] = np.nan  # dropped measurements
    gap = rng.random(n) < 0.2  # sensor gaps
    start = rng.integers(0, T - 6, size=n)
    for i in np.where(gap)[0]:
        ts[i, :, start[i]:start[i] + 6] = np.nan
    return ts


def generate_cohort(cfg: Config) -> Cohort:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_patients
    L = rng.normal(size=(n, 3))
    logit = L @ np.array([1.7, 1.4, 1.1]) - 1.9  # ~28% prevalence
    y = (rng.random(n) < _sigmoid(logit)).astype(int)

    img, lesion = _make_images(rng, L[:, 0], cfg.img_size)
    tab = _make_tabular(rng, L[:, 1])
    ts = _make_timeseries(rng, L[:, 2], cfg.ts_len)

    has_img = rng.random(n) < cfg.p_img  # modality missingness is MCAR here (independent of outcome)
    has_tab = rng.random(n) < cfg.p_tab
    has_ts = rng.random(n) < cfg.p_ts
    img[~has_img] = 0.0
    lesion[~has_img] = False
    tab.loc[~has_tab, :] = np.nan
    ts[~has_ts] = np.nan

    return Cohort(
        patient_id=np.array([f"P{i:05d}" for i in range(n)]), y=y, tab=tab, ts=ts, img=img,
        has_img=has_img, has_tab=has_tab, has_ts=has_ts, lesion_mask=lesion,
    )
