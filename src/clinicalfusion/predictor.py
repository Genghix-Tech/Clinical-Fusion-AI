"""Trained-system container: unimodal models + both fusion variants + calibrators.

Used by the experiment runner (batch evaluation) and by the API (single-patient inference with any
subset of modalities). Bundles are pickled: only load bundles you created yourself.
"""
from __future__ import annotations

import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from . import DISCLAIMER
from .config import Config
from .data.preprocess import Normalizers, Tensors
from .data.schema import TAB_FEATURES, TAB_RANGES, TS_CHANNELS, TS_RANGES
from .evaluation.metrics import PlattCalibrator, sigmoid
from .models.encoders import ImageEncoder, UnimodalNet, build_ts_encoder
from .models.fusion import AttentionFusion, LateFusionStacker
from .models.inference import attention_weights, mc_predict
from .models.tabular import TabularGBM
from .models.training import fwd_fusion, fwd_image, fwd_ts, predict_logits

EXPERIMENT_KEYS = ("A", "B", "C", "D1", "D2")


def build_nets(cfg: Config, tab_dim: int, ts_ch: int):
    d = cfg.embed_dim
    img_net = UnimodalNet(ImageEncoder(d), d)
    ts_net = UnimodalNet(build_ts_encoder(cfg.ts_arch, ts_ch, d), d)
    fusion = AttentionFusion(tab_dim, ts_ch, d, ts_arch=cfg.ts_arch, modality_dropout=cfg.modality_dropout)
    return img_net, ts_net, fusion


class Predictor:
    def __init__(self, cfg: Config, norm: Normalizers, img_net: UnimodalNet, ts_net: UnimodalNet, tab: TabularGBM,
                 stacker: LateFusionStacker, fusion: AttentionFusion,
                 calibrators: Optional[Dict[str, PlattCalibrator]] = None):
        self.cfg, self.norm = cfg, norm
        self.img_net, self.ts_net, self.tab, self.stacker, self.fusion = img_net, ts_net, tab, stacker, fusion
        self.calibrators = calibrators or {k: PlattCalibrator() for k in EXPERIMENT_KEYS}

    # -- batch inference ------------------------------------------------------------------------
    def unimodal_logits(self, tab_df: pd.DataFrame, T: Tensors) -> np.ndarray:
        """(N,3) logits for [image, tabular, timeseries]; NaN where the modality is absent."""
        L = np.stack([
            predict_logits(self.img_net, fwd_image, T), self.tab.logit(tab_df), predict_logits(self.ts_net, fwd_ts, T),
        ], axis=1).astype(np.float64)
        L[~T.avail.numpy()] = np.nan
        return L

    def logits(self, tab_df: pd.DataFrame, T: Tensors, avail: Optional[np.ndarray] = None,
               L: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Logits for experiments A, B, C, D1, D2. `avail` can only REMOVE modalities (what-if masking)."""
        nat = T.avail.numpy()
        av = nat if avail is None else (nat & avail)
        L = self.unimodal_logits(tab_df, T) if L is None else L
        Lm = np.where(av, L, np.nan)
        d2 = predict_logits(self.fusion, fwd_fusion, T.with_avail(torch.from_numpy(av)))
        return {"A": Lm[:, 0], "B": Lm[:, 1], "C": Lm[:, 2], "D1": self.stacker.decision(Lm, av), "D2": d2}, L

    def calibrated(self, logits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: self.calibrators[k].prob(v) for k, v in logits.items()}

    # -- single patient ---------------------------------------------------------------------------
    def predict_sample(self, tabular: Optional[Dict[str, Optional[float]]] = None,
                       timeseries: Optional[Dict[str, list]] = None, image: Optional[list] = None,
                       mc_samples: int = 30) -> Dict[str, Any]:
        cfg, warns = self.cfg, []
        tab_df = pd.DataFrame([{f: (tabular or {}).get(f) for f in TAB_FEATURES}]).astype(float)
        has_tab = bool(tab_df.notna().to_numpy().any())
        for f in TAB_FEATURES:
            v = tab_df[f].iloc[0]
            if not np.isnan(v) and not (TAB_RANGES[f][0] <= v <= TAB_RANGES[f][1]):
                warns.append(f"tabular '{f}'={v} is outside the plausible range {TAB_RANGES[f]}")

        ts = np.full((1, len(TS_CHANNELS), cfg.ts_len), np.nan, np.float32)
        if timeseries:
            for k, ch in enumerate(TS_CHANNELS):
                if timeseries.get(ch) is not None:
                    vals = timeseries[ch]
                    if len(vals) != cfg.ts_len:
                        raise ValueError(f"timeseries['{ch}'] needs {cfg.ts_len} hourly values, got {len(vals)}")
                    ts[0, k] = [np.nan if v is None else v for v in vals]
                    lo, hi = TS_RANGES[ch]
                    if np.nanmin(ts[0, k], initial=np.inf) < lo or np.nanmax(ts[0, k], initial=-np.inf) > hi:
                        warns.append(f"timeseries '{ch}' has values outside the plausible range {(lo, hi)}")
        has_ts = bool(np.isfinite(ts).any())

        img = np.zeros((1, cfg.img_size, cfg.img_size), np.float32)
        has_img = image is not None
        if has_img:
            a = np.asarray(image, dtype=np.float32)
            if a.shape != (cfg.img_size, cfg.img_size):
                raise ValueError(f"image must be {cfg.img_size}x{cfg.img_size} (rows of grey values in [0,1])")
            img[0] = np.clip(a, 0, 1)
        if not (has_img or has_tab or has_ts):
            raise ValueError("provide at least one modality (image, tabular or timeseries)")

        T = Tensors(
            img=torch.from_numpy(self.norm.img_nn(img)), tab=torch.from_numpy(self.norm.tab_nn(tab_df)),
            ts=torch.from_numpy(self.norm.ts_nn(ts)), avail=torch.tensor([[has_img, has_tab, has_ts]]),
            y=torch.zeros(1),
        )
        lg, _ = self.logits(tab_df, T)
        pc = self.calibrated(lg)
        mc_logit = mc_predict(self.fusion, T, mc_samples)[:, 0]
        mc = sigmoid(self.calibrators["D2"].a * mc_logit + self.calibrators["D2"].b)  # same calibration as the point estimate
        w = attention_weights(self.fusion, T)[0]
        uni = {k: float(pc[k][0]) for k in ("A", "B", "C") if np.isfinite(pc[k][0])}
        names = dict(A="image", B="tabular", C="timeseries")
        return {
            "modalities_used": [m for m, a in zip(("image", "tabular", "timeseries"), (has_img, has_tab, has_ts)) if a],
            "risk": {"fusion_attention": float(pc["D2"][0]), "fusion_late": float(pc["D1"][0])},
            "unimodal": {names[k]: v for k, v in uni.items()},
            "uncertainty": {
                "mc_mean": float(mc.mean()), "mc_std": float(mc.std()), "mc_logit_std": float(mc_logit.std()),
                "interval_95": [float(np.quantile(mc, 0.025)), float(np.quantile(mc, 0.975))],
            },
            "disagreement": float(max(uni.values()) - min(uni.values())) if len(uni) >= 2 else None,
            "attention": dict(zip(("image", "tabular", "timeseries", "none"), map(float, w))),
            "warnings": warns,
            "disclaimer": DISCLAIMER,
        }

    # -- persistence ----------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        state = {
            "cfg": self.cfg.to_dict(), "norm": self.norm, "img": self.img_net.state_dict(),
            "ts": self.ts_net.state_dict(), "fusion": self.fusion.state_dict(), "stacker": self.stacker,
            "tab_backend": self.tab.backend, "tab_raw": self.tab.to_bytes(),
            "tab_dim": len(TAB_FEATURES) * 2, "ts_ch": len(TS_CHANNELS) * 2,
            "cal": {k: (c.a, c.b) for k, c in self.calibrators.items()},
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "Predictor":
        with open(path, "rb") as f:
            s = pickle.load(f)  # noqa: S301 - trusted, self-produced bundle
        cfg = Config.from_dict(s["cfg"])
        img_net, ts_net, fusion = build_nets(cfg, s["tab_dim"], s["ts_ch"])
        img_net.load_state_dict(s["img"]); ts_net.load_state_dict(s["ts"]); fusion.load_state_dict(s["fusion"])
        cals = {}
        for k, (a, b) in s["cal"].items():
            c = PlattCalibrator(); c.a, c.b = a, b; cals[k] = c
        for m in (img_net, ts_net, fusion):
            m.eval()
        return cls(cfg, s["norm"], img_net, ts_net, TabularGBM.from_bytes(s["tab_backend"], s["tab_raw"]),
                   s["stacker"], fusion, cals)
