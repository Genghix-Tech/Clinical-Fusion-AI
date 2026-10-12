from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class Config:
    # cohort
    n_patients: int = 4000
    seed: int = 42
    img_size: int = 64
    ts_len: int = 48
    p_img: float = 0.80  # probability a patient has an image
    p_tab: float = 0.97
    p_ts: float = 0.85
    data_dir: Optional[str] = None  # load a real cohort from disk instead of synthesising
    val_frac: float = 0.2
    test_frac: float = 0.2

    # models
    embed_dim: int = 64
    ts_arch: str = "tcn"  # tcn | lstm
    tab_backend: str = "xgboost"  # xgboost | catboost
    epochs_img: int = 20
    epochs_ts: int = 25
    epochs_fusion: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    patience: int = 8
    modality_dropout: float = 0.25
    pretrain_encoders: bool = True
    device: str = "cpu"

    # evaluation
    mc_samples: int = 30
    n_bootstrap: int = 500
    n_explain: int = 8

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    @classmethod
    def fast(cls, **overrides: Any) -> "Config":
        """Small, quick configuration for smoke tests and demos."""
        base = dict(
            n_patients=800, epochs_img=4, epochs_ts=6, epochs_fusion=6,
            mc_samples=8, n_bootstrap=100, n_explain=4, patience=3,
        )
        base.update(overrides)
        return cls.from_dict(base)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PRESETS = {
    "fast": Config.fast,
    "standard": Config,
}
