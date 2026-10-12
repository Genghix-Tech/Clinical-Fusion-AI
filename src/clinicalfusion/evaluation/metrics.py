"""Discrimination, calibration and uncertainty metrics, with bootstrap confidence intervals."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def clean(x: Any) -> Any:
    """Make nested results JSON-safe (numpy -> python, NaN -> None)."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, p)) if len(y) else float("nan")


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not len(y):
        return float("nan")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            err += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(err)


def calibration_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return [
        {"mean_pred": float(p[bins == b].mean()), "frac_pos": float(y[bins == b].mean()), "n": int((bins == b).sum())}
        for b in range(n_bins) if (bins == b).any()
    ]


def bootstrap_ci(fn, y: np.ndarray, *arrs: np.ndarray, n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], *[a[i] for a in arrs]))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]


def summarize(y: np.ndarray, p: np.ndarray, n_boot: int = 500, seed: int = 0) -> Dict[str, Any]:
    y, p = np.asarray(y), np.asarray(p, float)
    out: Dict[str, Any] = {"n": int(len(y)), "n_pos": int(y.sum())}
    if len(y) < 10 or len(np.unique(y)) < 2:
        return {**out, "auroc": None, "auroc_ci": None, "auprc": None, "brier": None, "ece": None}
    out.update(
        auroc=auroc(y, p), auroc_ci=bootstrap_ci(auroc, y, p, n_boot=n_boot, seed=seed),
        auprc=auprc(y, p), brier=brier(y, p), ece=ece(y, p),
    )
    return out


def paired_auroc_diff(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, n_boot: int = 500, seed: int = 0) -> Dict[str, Any]:
    """AUROC(a) - AUROC(b) on the same patients, with a paired-bootstrap 95% CI."""
    d = auroc(y, pa) - auroc(y, pb)
    ci = bootstrap_ci(lambda yy, a, b: auroc(yy, a) - auroc(yy, b), y, pa, pb, n_boot=n_boot, seed=seed)
    return {"diff": float(d), "ci": ci, "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0), "n": int(len(y))}


def selective_prediction(y: np.ndarray, p: np.ndarray, unc: np.ndarray,
                         coverages: Sequence[float] = (1.0, 0.8, 0.6, 0.4)) -> list:
    """AUROC / accuracy when keeping only the most certain `coverage` fraction of patients."""
    order = np.argsort(unc, kind="stable")
    rows = []
    for c in coverages:
        keep = order[: max(int(round(c * len(y))), 1)]
        yk, pk = y[keep], p[keep]
        rows.append({
            "coverage": float(c), "n": int(len(keep)),
            "auroc": auroc(yk, pk) if len(np.unique(yk)) == 2 else None,
            "accuracy": float(((pk >= 0.5) == yk).mean()),
        })
    return rows


class PlattCalibrator:
    """Logistic recalibration of a logit: p = sigmoid(a * logit + b). Fit on validation data only."""

    def __init__(self) -> None:
        self.a, self.b = 1.0, 0.0

    def fit(self, logit: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        if len(np.unique(y)) == 2 and len(y) >= 20:
            lr = LogisticRegression(C=1e4, max_iter=1000).fit(np.asarray(logit, float)[:, None], y)
            self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def prob(self, logit: np.ndarray) -> np.ndarray:
        return sigmoid(self.a * np.asarray(logit, float) + self.b)
