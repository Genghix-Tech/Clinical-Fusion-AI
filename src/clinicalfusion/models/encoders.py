"""Per-modality neural encoders. Each maps its input to a d-dim embedding."""
from __future__ import annotations

import torch
from torch import nn


def _conv_block(cin: int, cout: int, pool: bool) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ImageEncoder(nn.Module):
    """Small CNN. `features` returns the last conv map (used by Grad-CAM); `embed` pools it."""

    def __init__(self, d: int = 64, p: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 16, True), _conv_block(16, 32, True), _conv_block(32, 64, True), _conv_block(64, 64, False)
        )
        self.fc = nn.Sequential(nn.Dropout(p), nn.Linear(64, d), nn.ReLU(inplace=True))

    def embed(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.fc(fmap.mean(dim=(2, 3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.features(x))


class TCNEncoder(nn.Module):
    """Temporal CNN: residual dilated 1-D convolutions, then avg+max pooling over time."""

    def __init__(self, in_ch: int, d: int = 64, p: float = 0.2, width: int = 32):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, width, 1)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(width, width, 3, padding=dil, dilation=dil), nn.BatchNorm1d(width), nn.ReLU(inplace=True))
             for dil in (1, 2, 4, 8)]
        )
        self.fc = nn.Sequential(nn.Dropout(p), nn.Linear(2 * width, d), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.fc(torch.cat([h.mean(-1), h.amax(-1)], dim=1))


class LSTMEncoder(nn.Module):
    def __init__(self, in_ch: int, d: int = 64, p: float = 0.2, hidden: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(p), nn.Linear(2 * hidden, d), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, ch, T)
        out, _ = self.lstm(x.transpose(1, 2))
        return self.fc(out.mean(1))


class TabEncoder(nn.Module):
    def __init__(self, in_dim: int, d: int = 64, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96), nn.ReLU(inplace=True), nn.Dropout(p), nn.Linear(96, d), nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_ts_encoder(arch: str, in_ch: int, d: int, p: float = 0.2) -> nn.Module:
    if arch == "tcn":
        return TCNEncoder(in_ch, d, p)
    if arch == "lstm":
        return LSTMEncoder(in_ch, d, p)
    raise ValueError(f"unknown ts_arch {arch!r} (use 'tcn' or 'lstm')")


class UnimodalNet(nn.Module):
    """Encoder + linear head -> logit. Used for Experiments A (image) and C (time-series)."""

    def __init__(self, encoder: nn.Module, d: int = 64, p: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(d, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)
