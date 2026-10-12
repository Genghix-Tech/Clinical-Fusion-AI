from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data.preprocess import Tensors

Forward = Callable[[nn.Module, Tensors], torch.Tensor]  # (model, batch) -> logits (B,)


def fwd_image(m: nn.Module, b: Tensors) -> torch.Tensor:
    return m(b.img)


def fwd_ts(m: nn.Module, b: Tensors) -> torch.Tensor:
    return m(b.ts)


def fwd_tab(m: nn.Module, b: Tensors) -> torch.Tensor:
    return m(b.tab)


def fwd_fusion(m: nn.Module, b: Tensors) -> torch.Tensor:
    return m(b.img, b.tab, b.ts, b.avail)[0]


@torch.no_grad()
def predict_logits(model: nn.Module, fwd: Forward, data: Tensors, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = [fwd(model, data.subset(np.arange(i, min(i + batch_size, len(data))))) for i in range(0, len(data), batch_size)]
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


def augment_images(b: Tensors) -> Tensors:
    """Random left-right flip (chest anatomy is roughly symmetric) and a small batch-wise shift."""
    img = b.img
    flip = (torch.rand(len(img)) < 0.5)[:, None, None, None]
    img = torch.where(flip, img.flip(-1), img)
    dy, dx = (int(v) for v in torch.randint(-3, 4, (2,)))
    return Tensors(torch.roll(img, shifts=(dy, dx), dims=(2, 3)), b.tab, b.ts, b.avail, b.y)


def fit(model: nn.Module, fwd: Forward, train: Tensors, val: Tensors, *, epochs: int, lr: float,
        batch_size: int, patience: int, seed: int = 0, augment: Optional[Callable[[Tensors], Tensors]] = None
        ) -> List[Dict[str, float]]:
    """AdamW + BCE, early stopping on validation loss; restores the best weights."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad, hist = float("inf"), copy.deepcopy(model.state_dict()), 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(train), generator=g).numpy()
        tl = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            b = train.subset(idx)
            if augment is not None:
                b = augment(b)
            loss = F.binary_cross_entropy_with_logits(fwd(model, b), b.y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item() * len(idx)
        vl = F.binary_cross_entropy_with_logits(torch.from_numpy(predict_logits(model, fwd, val)), val.y).item()
        hist.append({"epoch": ep + 1, "train_loss": tl / len(train), "val_loss": vl})
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return hist
