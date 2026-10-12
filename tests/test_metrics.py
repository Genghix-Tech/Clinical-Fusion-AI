import numpy as np
import pytest

from clinicalfusion.evaluation.metrics import (
    PlattCalibrator, auroc, brier, calibration_curve, clean, ece, paired_auroc_diff, selective_prediction, sigmoid, summarize,
)


def test_ece_zero_for_perfectly_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert ece(y, p) < 0.02
    assert ece(y, np.clip(p * 0.2 + 0.8, 0, 1)) > 0.3  # systematically over-predicting


def test_brier_known_value():
    assert brier(np.array([1, 0]), np.array([0.8, 0.2])) == pytest.approx(0.04)


def test_calibration_curve_bins_sum_to_n():
    rng = np.random.default_rng(1)
    p, y = rng.random(500), rng.integers(0, 2, 500)
    assert sum(b["n"] for b in calibration_curve(y, p)) == 500


def test_platt_fixes_overconfident_logits():
    rng = np.random.default_rng(2)
    z = rng.normal(size=4000)
    y = (rng.random(4000) < sigmoid(z)).astype(int)
    over = 3 * z  # 3x too confident
    cal = PlattCalibrator().fit(over, y)
    assert ece(y, cal.prob(over)) < ece(y, sigmoid(over))
    assert abs(cal.a - 1 / 3) < 0.08


def test_summarize_ci_contains_point_estimate_and_handles_one_class():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    s = summarize(y, sigmoid(y + rng.normal(scale=1.0, size=400)), n_boot=100)
    assert s["auroc_ci"][0] <= s["auroc"] <= s["auroc_ci"][1]
    assert summarize(np.zeros(50, int), rng.random(50))["auroc"] is None


def test_paired_diff_detects_better_model():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 1500)
    good, bad = y + rng.normal(scale=0.7, size=1500), y + rng.normal(scale=2.0, size=1500)
    d = paired_auroc_diff(y, good, bad, n_boot=200)
    assert d["diff"] > 0.1 and d["ci_excludes_zero"]


def test_selective_prediction_improves_when_uncertainty_is_informative():
    rng = np.random.default_rng(5)
    n = 4000
    y = rng.integers(0, 2, n)
    noise = rng.uniform(0.2, 3.0, n)  # per-patient noise level == uncertainty
    p = sigmoid(2 * (y - 0.5) + rng.normal(size=n) * noise)
    rows = selective_prediction(y, p, noise)
    assert rows[-1]["auroc"] > rows[0]["auroc"] + 0.05


def test_clean_makes_json_safe():
    import json

    out = clean({"a": np.float32("nan"), "b": np.array([1, 2]), "c": np.int64(3), "d": [np.inf]})
    json.dumps(out, allow_nan=False)
    assert out["a"] is None and out["c"] == 3
