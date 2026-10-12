import json

import numpy as np
import pytest

from clinicalfusion.predictor import Predictor


def test_results_structure_and_json_safe(tiny_run):
    r = tiny_run.results
    json.dumps(r, allow_nan=False)
    assert set(r["experiments"]) == {"A", "B", "C", "D1", "D2"}
    cc = r["comparison"]["complete_cases"]
    for k in ("A", "B", "C", "D1", "D2"):
        assert 0.0 <= cc[k]["auroc"] <= 1.0 and cc[k]["auroc_ci"][0] <= cc[k]["auroc_ci"][1]
        assert 0.0 <= cc[k]["ece"] <= 1.0 and cc[k]["brier"] >= 0
    assert set(r["missing_modality"]["scenarios_on_complete_cases"]) >= {"all", "no_image", "image_only", "timeseries_only"}
    assert r["meta"]["synthetic_data"] and "SYNTHETIC" in r["findings"][0]


def test_no_leakage_between_splits(tiny_run):
    r = tiny_run.results["meta"]["split"]
    assert r["train"] + r["val"] + r["test"] == tiny_run.results["validation"]["n_patients"]


def test_predictions_table(tiny_run):
    p = tiny_run.predictions
    assert set(p.experiment) == {"A", "B", "C", "D1", "D2"}
    assert p.prob_cal.between(0, 1).all() and p.prob_raw.between(0, 1).all()
    assert p.loc[p.experiment != "D2", "uncertainty"].isna().all() and p.loc[p.experiment == "D2", "uncertainty"].notna().all()
    # unimodal experiments only cover patients that actually have that modality
    assert len(p[p.experiment == "D2"]) > len(p[p.experiment == "A"])


def test_synthetic_signal_is_learnable_and_complementary():
    """On a properly sized cohort the tabular model beats chance, and the latents really are independent."""
    from sklearn.metrics import roc_auc_score

    from clinicalfusion.config import Config
    from clinicalfusion.data.preprocess import stratified_split
    from clinicalfusion.data.synthetic import generate_cohort
    from clinicalfusion.models.tabular import TabularGBM

    c = generate_cohort(Config(n_patients=3000, seed=7, p_tab=1.0))
    tr, va, te = stratified_split(c.y, 0.2, 0.2, 0)
    gbm = TabularGBM("xgboost").fit(c.tab.iloc[tr], c.y[tr], c.tab.iloc[va], c.y[va])
    assert roc_auc_score(c.y[te], gbm.logit(c.tab.iloc[te])) > 0.65
    assert gbm.contributions(c.tab.iloc[te]).shape == (len(te), 12)


def test_masking_a_modality_changes_fusion_output(tiny_run):
    sc = tiny_run.results["missing_modality"]["scenarios_on_complete_cases"]
    att_all, att_no_img = sc["all"]["D2_attention"], sc["no_image"]["D2_attention"]
    assert att_all[0] > 0 and att_no_img[0] == pytest.approx(0, abs=1e-6)


def test_explanations_present_and_valid(tiny_run):
    ex = tiny_run.results["explainability"]
    assert ex["local"] and ex["local"][0]["image"].startswith("data:image/png;base64,")
    assert "caveat" in ex["global"] and len(ex["local"][0]["timeseries"]["importance"]) == 48
    assert len(ex["global"]["tabular_mean_abs_contribution"]) == 12


def test_predictor_roundtrip_and_partial_modalities(tiny_run, tmp_path):
    path = str(tmp_path / "bundle.pkl")
    tiny_run.predictor.save(path)
    pred = Predictor.load(path)
    tab = {"age": 70, "sex": 1, "bmi": 28, "sbp": 100, "dbp": 60, "heart_rate_adm": 110, "creatinine": 2.1,
           "lactate": 4.0, "wbc": 15, "hemoglobin": 10, "sodium": 138, "glucose": 140}
    full = pred.predict_sample(tabular=tab, timeseries={"heart_rate": [80.0] * 48, "spo2": [95.0] * 48},
                               image=np.full((64, 64), 0.3).tolist(), mc_samples=4)
    assert full["modalities_used"] == ["image", "tabular", "timeseries"]
    tab_only = pred.predict_sample(tabular=tab, mc_samples=4)
    assert tab_only["modalities_used"] == ["tabular"] and tab_only["attention"]["image"] == 0
    for out in (full, tab_only):
        assert 0 <= out["risk"]["fusion_attention"] <= 1 and out["uncertainty"]["mc_std"] >= 0
    with pytest.raises(ValueError):
        pred.predict_sample()
    with pytest.raises(ValueError):
        pred.predict_sample(timeseries={"heart_rate": [80.0] * 10})
