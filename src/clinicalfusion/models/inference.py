"""Inference helpers shared by the experiment runner and the serving predictor."""
from __future__ import annotations

import numpy as np
import torch

from ..data.preprocess import Tensors
from .fusion import AttentionFusion, enable_mc_dropout


@torch.no_grad()
def mc_predict(model: AttentionFusion, data: Tensors, n_samples: int, seed: int = 0, batch_size: int = 256) -> np.ndarray:
    """Monte-Carlo dropout. Returns LOGITS (n_samples, N).

    Spread is measured in logit space on purpose: in probability space the spread shrinks towards 0 and 1
    simply because the sigmoid saturates, which would make confident predictions look "certain" regardless
    of how unstable the network is.
    """
    torch.manual_seed(seed)
    enable_mc_dropout(model)
    out = np.zeros((n_samples, len(data)), np.float32)
    for t in range(n_samples):
        for i in range(0, len(data), batch_size):
            b = data.subset(np.arange(i, min(i + batch_size, len(data))))
            out[t, i:i + batch_size] = model(b.img, b.tab, b.ts, b.avail)[0].numpy()
    model.eval()
    return out


@torch.no_grad()
def attention_weights(model: AttentionFusion, data: Tensors, batch_size: int = 256) -> np.ndarray:
    """(N,4) attention over [image, tabular, timeseries, null] with deterministic (eval) forward."""
    model.eval()
    out = []
    for i in range(0, len(data), batch_size):
        b = data.subset(np.arange(i, min(i + batch_size, len(data))))
        out.append(model(b.img, b.tab, b.ts, b.avail)[1].numpy())
    return np.concatenate(out)
