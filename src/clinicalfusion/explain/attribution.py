"""Model-behaviour explanations. These show what the models respond to - NOT medical causation."""
from __future__ import annotations

import base64
import io

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import image as mpimg

from ..models.encoders import UnimodalNet

matplotlib.use("Agg")


def gradcam(net: UnimodalNet, x: torch.Tensor) -> np.ndarray:
    """Grad-CAM of the image model's logit w.r.t. its last conv map. Returns (B,H,W) in [0,1]."""
    net.eval()
    fmap = net.encoder.features(x)
    fmap.retain_grad()
    net.head(net.encoder.embed(fmap)).squeeze(-1).sum().backward()
    w = fmap.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1).detach()
    flat = cam.flatten(1)
    lo, hi = flat.min(1).values[:, None, None], flat.max(1).values[:, None, None]
    net.zero_grad()
    return ((cam - lo) / (hi - lo + 1e-8)).numpy()


@torch.no_grad()
def ts_occlusion(net: UnimodalNet, ts: torch.Tensor, window: int = 6, stride: int = 3) -> np.ndarray:
    """Per-hour importance (N,T): average drop in logit when a window covering that hour is blanked."""
    net.eval()
    n, _, T = ts.shape
    base = net(ts)
    total, count = torch.zeros(n, T), torch.zeros(T)
    for s in range(0, T - window + 1, stride):
        occ = ts.clone()
        occ[:, :, s:s + window] = 0.0  # value=mean, observed-flag=0 -> "no measurement"
        total[:, s:s + window] += (base - net(occ))[:, None]
        count[s:s + window] += 1
    return (total / count.clamp(min=1)).numpy()


def _png_b64(rgb_or_gray: np.ndarray) -> str:
    buf = io.BytesIO()
    if rgb_or_gray.ndim == 2:
        mpimg.imsave(buf, rgb_or_gray, cmap="gray", vmin=0, vmax=1, format="png")
    else:
        mpimg.imsave(buf, np.clip(rgb_or_gray, 0, 1), format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def image_png(img: np.ndarray) -> str:
    return _png_b64(np.clip(img, 0, 1))


def overlay_png(img: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> str:
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    gray = np.repeat(np.clip(img, 0, 1)[..., None], 3, axis=-1)
    return _png_b64((1 - alpha * cam[..., None]) * gray + alpha * cam[..., None] * heat)


def cam_localisation(cam: np.ndarray, lesion: np.ndarray) -> dict:
    """Sanity check against the synthetic ground-truth lesion: pointing-game hit + CAM mass inside it."""
    hit = float(np.mean([lesion[i][np.unravel_index(cam[i].argmax(), cam[i].shape)] for i in range(len(cam))]))
    mass = float(np.mean([cam[i][lesion[i]].sum() / (cam[i].sum() + 1e-8) for i in range(len(cam))]))
    area = float(np.mean([lesion[i].mean() for i in range(len(cam))]))
    return {"pointing_game_hit_rate": hit, "cam_mass_in_lesion": mass, "lesion_area_fraction": area, "n": int(len(cam))}
