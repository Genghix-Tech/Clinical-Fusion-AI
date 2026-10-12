"""Multimodal fusion.

* ``AttentionFusion`` - feature-level fusion. Each modality is encoded to a token; a learned query
  attends over the *available* tokens (missing ones are masked out) plus an always-present learned
  "null" token, so any subset of modalities - including none - yields a valid prediction. Training
  applies modality dropout so the network learns to cope with absent inputs.
* ``LateFusionStacker`` - logistic regression over unimodal logits + availability flags.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from .encoders import ImageEncoder, TabEncoder, build_ts_encoder


class AttentionFusion(nn.Module):
    def __init__(self, tab_dim: int, ts_ch: int, d: int = 64, heads: int = 4, p: float = 0.2,
                 ts_arch: str = "tcn", modality_dropout: float = 0.25):
        super().__init__()
        self.img = ImageEncoder(d, p)
        self.tab = TabEncoder(tab_dim, d, p)
        self.ts = build_ts_encoder(ts_arch, ts_ch, d, p)
        self.mod_emb = nn.Parameter(torch.randn(4, d) * 0.02)  # image, tabular, timeseries, null
        self.null_token = nn.Parameter(torch.zeros(1, d))
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(d, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))
        self.modality_dropout = modality_dropout

    def forward(self, img: torch.Tensor, tab: torch.Tensor, ts: torch.Tensor, avail: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logit (B,), attention over [image, tabular, timeseries, null] (B,4))."""
        avail = avail.clone()
        if self.training and self.modality_dropout > 0:
            avail &= torch.rand(avail.shape, device=avail.device) >= self.modality_dropout
        b = img.shape[0]
        toks = torch.stack([self.img(img), self.tab(tab), self.ts(ts), self.null_token.expand(b, -1)], dim=1)
        toks = toks + self.mod_emb[None]
        keep = torch.cat([avail, torch.ones(b, 1, dtype=torch.bool, device=avail.device)], dim=1)
        out, w = self.attn(self.query.expand(b, -1, -1), toks, toks, key_padding_mask=~keep, need_weights=True)
        h = self.norm(out.squeeze(1))
        return self.head(h).squeeze(-1), w.squeeze(1)


def enable_mc_dropout(model: nn.Module) -> None:
    """Put the model in eval mode but keep Dropout active (BatchNorm stays in eval)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.MultiheadAttention)):
            m.train()
    if isinstance(model, AttentionFusion):
        model.modality_dropout = 0.0  # MC passes use the true availability only


class LateFusionStacker:
    """Logistic regression on [logit_m * avail_m ..., avail_m ...] (missing logit contributes 0)."""

    def __init__(self, C: float = 1.0):
        self.lr = LogisticRegression(C=C, max_iter=1000)

    @staticmethod
    def features(logits: np.ndarray, avail: np.ndarray) -> np.ndarray:
        logits = np.where(avail, logits, 0.0)
        return np.concatenate([logits, avail.astype(float)], axis=1)

    def fit(self, logits: np.ndarray, avail: np.ndarray, y: np.ndarray) -> "LateFusionStacker":
        self.lr.fit(self.features(logits, avail), y)
        return self

    def decision(self, logits: np.ndarray, avail: np.ndarray) -> np.ndarray:
        return self.lr.decision_function(self.features(logits, avail))
