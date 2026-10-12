"""File-based cohort I/O: the bridge for real data (e.g. MIMIC-IV + MIMIC-CXR after your own extraction).

Directory layout (see docs/REAL_DATA.md):

    cohort/
      labels.csv        patient_id,y
      tabular.csv       patient_id + the columns in TAB_FEATURES   (rows only for patients with tabular data)
      timeseries.npz    patient_id (M,), values (M, C, T) hourly grid, NaN = not measured
      images/<patient_id>.png   grayscale chest X-ray (any size; resized to cfg.img_size)

A patient missing from tabular.csv / timeseries.npz / images/ is treated as missing that modality.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from scipy.ndimage import zoom

from .schema import TAB_FEATURES, TS_CHANNELS, Cohort


def export_cohort(c: Cohort, out_dir: str) -> None:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": c.patient_id, "y": c.y}).to_csv(out / "labels.csv", index=False)
    tab = c.tab.loc[c.has_tab].copy()
    tab.insert(0, "patient_id", c.patient_id[c.has_tab])
    tab.to_csv(out / "tabular.csv", index=False)
    np.savez_compressed(out / "timeseries.npz", patient_id=c.patient_id[c.has_ts], values=c.ts[c.has_ts])
    for pid, im in zip(c.patient_id[c.has_img], c.img[c.has_img]):
        mpimg.imsave(out / "images" / f"{pid}.png", im, cmap="gray", vmin=0, vmax=1)


def load_cohort(data_dir: str, img_size: int = 64, ts_len: int = 48) -> Cohort:
    d = Path(data_dir)
    labels = pd.read_csv(d / "labels.csv", dtype={"patient_id": str})
    ids = labels["patient_id"].to_numpy()
    n = len(ids)
    pos = {p: i for i, p in enumerate(ids)}

    tab = pd.DataFrame(np.nan, index=range(n), columns=TAB_FEATURES)
    has_tab = np.zeros(n, bool)
    tdf = pd.read_csv(d / "tabular.csv", dtype={"patient_id": str})
    missing_cols = [c for c in TAB_FEATURES if c not in tdf.columns]
    if missing_cols:
        raise ValueError(f"tabular.csv is missing columns: {missing_cols}")
    for pid, row in zip(tdf["patient_id"], tdf[TAB_FEATURES].to_numpy(dtype=float)):
        if pid in pos:
            tab.iloc[pos[pid]] = row
            has_tab[pos[pid]] = True

    ts = np.full((n, len(TS_CHANNELS), ts_len), np.nan, np.float32)
    has_ts = np.zeros(n, bool)
    z = np.load(d / "timeseries.npz", allow_pickle=False)
    if z["values"].shape[1:] != (len(TS_CHANNELS), ts_len):
        raise ValueError(f"timeseries values must be (M,{len(TS_CHANNELS)},{ts_len}), got {z['values'].shape}")
    for pid, v in zip(z["patient_id"].astype(str), z["values"]):
        if pid in pos:
            ts[pos[pid]] = v
            has_ts[pos[pid]] = True

    img = np.zeros((n, img_size, img_size), np.float32)
    has_img = np.zeros(n, bool)
    for pid, i in pos.items():
        f = d / "images" / f"{pid}.png"
        if f.exists():
            a = mpimg.imread(f)
            if a.ndim == 3:
                a = a[..., :3].mean(-1)
            a = zoom(a.astype(np.float32), (img_size / a.shape[0], img_size / a.shape[1]), order=1)
            img[i] = np.clip(a, 0, 1)
            has_img[i] = True

    return Cohort(patient_id=ids, y=labels["y"].to_numpy(dtype=int), tab=tab, ts=ts, img=img,
                  has_img=has_img, has_tab=has_tab, has_ts=has_ts, lesion_mask=None)
