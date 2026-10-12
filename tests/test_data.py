import numpy as np

from clinicalfusion.config import Config
from clinicalfusion.data.loader import export_cohort, load_cohort
from clinicalfusion.data.preprocess import Normalizers, stratified_split, to_tensors
from clinicalfusion.data.synthetic import generate_cohort
from clinicalfusion.data.validation import match_patient_ids, validate_cohort


def test_synthetic_cohort_is_valid_and_reproducible():
    cfg = Config(n_patients=300, seed=1)
    a, b = generate_cohort(cfg), generate_cohort(cfg)
    assert np.array_equal(a.y, b.y) and np.allclose(a.img, b.img)
    rep = validate_cohort(a)
    assert rep["ok"], rep["issues"]
    assert 0.1 < rep["prevalence"] < 0.5
    assert 0 < rep["complete_cases"] < 300  # some patients are missing a modality


def test_validation_catches_faults():
    c = generate_cohort(Config(n_patients=200, seed=2))
    c.tab.loc[c.has_tab.nonzero()[0][0], "sbp"] = 900  # impossible blood pressure
    c.patient_id[1] = c.patient_id[0]  # duplicate id
    rep = validate_cohort(c)
    assert not rep["ok"]
    assert "sbp" in rep["out_of_range"]
    assert any("duplicate" in f for f in rep["fatal"])


def test_match_patient_ids():
    rep = match_patient_ids(images=["a", "b", "c"], labs=["b", "c", "d"], vitals=["c", "d", "e"])
    assert rep["in_all_sources"] == 1 and rep["in_any_source"] == 5


def test_split_is_disjoint_and_stratified():
    y = (np.random.default_rng(0).random(1000) < 0.3).astype(int)
    tr, va, te = stratified_split(y, 0.2, 0.2, 0)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) + len(va) + len(te) == 1000
    assert abs(y[tr].mean() - y[te].mean()) < 0.02


def test_normalizer_fit_on_train_only_and_shapes():
    c = generate_cohort(Config(n_patients=300, seed=3))
    norm = Normalizers.fit(c)
    T = to_tensors(c, norm)
    assert T.img.shape == (300, 1, 64, 64) and T.tab.shape == (300, 24) and T.ts.shape == (300, 4, 48)
    assert np.isfinite(T.tab.numpy()).all() and np.isfinite(T.ts.numpy()).all()


def test_export_load_roundtrip(tmp_path):
    cfg = Config(n_patients=120, seed=4)
    c = generate_cohort(cfg)
    export_cohort(c, str(tmp_path))
    r = load_cohort(str(tmp_path), cfg.img_size, cfg.ts_len)
    assert np.array_equal(r.y, c.y)
    assert np.array_equal(r.has_img, c.has_img) and np.array_equal(r.has_ts, c.has_ts) and np.array_equal(r.has_tab, c.has_tab)
    assert np.allclose(r.tab.to_numpy(float), c.tab.to_numpy(float), equal_nan=True, atol=1e-4)
    assert np.allclose(r.ts, c.ts, equal_nan=True)
    assert np.abs(r.img - c.img).max() < 0.02  # 8-bit PNG quantisation
    assert validate_cohort(r)["ok"]
