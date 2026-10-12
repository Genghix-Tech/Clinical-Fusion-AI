"""Runs Experiments A-D end to end and assembles the research report.

A  image only          B  tabular only         C  time-series only
D1 late fusion (stacking)                      D2 attention fusion (feature level, missing-aware)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch

from .. import DISCLAIMER, __version__
from ..config import Config
from ..data.loader import load_cohort
from ..data.preprocess import Normalizers, Tensors, stratified_split, to_tensors
from ..data.schema import TAB_FEATURES, TS_CHANNELS, Cohort
from ..data.synthetic import generate_cohort
from ..data.validation import validate_cohort
from ..evaluation.metrics import (
    PlattCalibrator, auroc, brier, calibration_curve, clean, ece, paired_auroc_diff, selective_prediction,
    sigmoid, summarize,
)
from ..explain.attribution import cam_localisation, gradcam, image_png, overlay_png, ts_occlusion
from ..models.encoders import TabEncoder, UnimodalNet
from ..models.fusion import LateFusionStacker
from ..models.inference import attention_weights, mc_predict
from ..models.tabular import TabularGBM
from ..models.training import augment_images, fit, fwd_fusion, fwd_image, fwd_tab, fwd_ts
from ..predictor import EXPERIMENT_KEYS, Predictor, build_nets

Progress = Callable[[str, float], None]

EXPERIMENTS = {
    "A": ("Image only", "Chest X-ray CNN", ["image"]),
    "B": ("Tabular only", "Gradient-boosted trees on clinical values", ["tabular"]),
    "C": ("Time-series only", "Temporal model on hourly vitals", ["timeseries"]),
    "D1": ("Fusion - late", "Logistic stacker over A/B/C logits + availability flags", ["image", "tabular", "timeseries"]),
    "D2": ("Fusion - attention", "Feature-level attention over modality embeddings, missing-aware",
           ["image", "tabular", "timeseries"]),
}
SCENARIOS = {
    "all": (1, 1, 1), "no_image": (0, 1, 1), "no_tabular": (1, 0, 1), "no_timeseries": (1, 1, 0),
    "image_only": (1, 0, 0), "tabular_only": (0, 1, 0), "timeseries_only": (0, 0, 1),
}


@dataclass
class RunOutput:
    results: Dict[str, Any]
    predictions: pd.DataFrame
    predictor: Predictor


def _sub(T: Tensors, mask: np.ndarray) -> Tensors:
    return T.subset(np.where(mask)[0])


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def run_experiments(cfg: Config, progress: Optional[Progress] = None) -> RunOutput:
    P = progress or (lambda msg, frac: None)
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    # ---- data -------------------------------------------------------------------------------
    P("Loading cohort and validating data", 0.02)
    cohort: Cohort = load_cohort(cfg.data_dir, cfg.img_size, cfg.ts_len) if cfg.data_dir else generate_cohort(cfg)
    validation = validate_cohort(cohort)
    if validation["fatal"]:
        raise ValueError("data validation failed: " + "; ".join(validation["fatal"]))
    tr, va, te = stratified_split(cohort.y, cfg.val_frac, cfg.test_frac, cfg.seed)
    ctr, cva, cte = cohort.subset(tr), cohort.subset(va), cohort.subset(te)
    norm = Normalizers.fit(ctr)
    Ttr, Tva, Tte = (to_tensors(c, norm) for c in (ctr, cva, cte))
    atr, ava, ate = ctr.avail, cva.avail, cte.avail
    yva, yte = cva.y, cte.y
    d = cfg.embed_dim
    img_net, ts_net, fusion = build_nets(cfg, Ttr.tab.shape[1], Ttr.ts.shape[1])
    hist: Dict[str, Any] = {}

    # ---- train unimodal models (A, B, C) -----------------------------------------------------------
    kw = dict(lr=cfg.lr, batch_size=cfg.batch_size, patience=cfg.patience, seed=cfg.seed)
    P("Experiment A: training image model", 0.08)
    hist["A"] = fit(img_net, fwd_image, _sub(Ttr, atr[:, 0]), _sub(Tva, ava[:, 0]), epochs=cfg.epochs_img,
                    augment=augment_images, **kw)
    P("Experiment B: training tabular model", 0.30)
    tab = TabularGBM(cfg.tab_backend, cfg.seed).fit(ctr.tab[atr[:, 1]], ctr.y[atr[:, 1]], cva.tab[ava[:, 1]], cva.y[ava[:, 1]])
    P("Experiment C: training time-series model", 0.36)
    hist["C"] = fit(ts_net, fwd_ts, _sub(Ttr, atr[:, 2]), _sub(Tva, ava[:, 2]), epochs=cfg.epochs_ts, **kw)

    # ---- D1: late fusion (stacker fit on validation logits) --------------------------------------------
    P("Experiment D1: fitting late-fusion stacker", 0.52)
    pred = Predictor(cfg, norm, img_net, ts_net, tab, LateFusionStacker(), fusion)
    Lva = pred.unimodal_logits(cva.tab, Tva)
    pred.stacker.fit(Lva, ava, yva)

    # ---- D2: attention fusion, initialised from the unimodal encoders ---------------------------------------
    P("Experiment D2: training attention-fusion model", 0.56)
    if cfg.pretrain_encoders:
        tab_net = UnimodalNet(TabEncoder(Ttr.tab.shape[1], d), d)  # NN twin of B, only used to initialise the fusion encoder
        fit(tab_net, fwd_tab, _sub(Ttr, atr[:, 1]), _sub(Tva, ava[:, 1]), epochs=cfg.epochs_ts, **kw)
        fusion.img.load_state_dict(img_net.encoder.state_dict())
        fusion.ts.load_state_dict(ts_net.encoder.state_dict())
        fusion.tab.load_state_dict(tab_net.encoder.state_dict())
    kw["lr"] = cfg.lr * (0.5 if cfg.pretrain_encoders else 1.0)
    hist["D2"] = fit(fusion, fwd_fusion, Ttr, Tva, epochs=cfg.epochs_fusion, augment=augment_images, **kw)

    # ---- calibration (fit on validation only) -------------------------------------------------------------
    P("Calibrating and evaluating on held-out test data", 0.80)
    lv, _ = pred.logits(cva.tab, Tva, L=Lva)
    mask_for = lambda av: {"A": av[:, 0], "B": av[:, 1], "C": av[:, 2],  # noqa: E731
                           "D1": np.ones(len(av), bool), "D2": np.ones(len(av), bool)}
    mva = mask_for(ava)
    pred.calibrators = {k: PlattCalibrator().fit(lv[k][mva[k]], yva[mva[k]]) for k in EXPERIMENT_KEYS}
    lt, Lte = pred.logits(cte.tab, Tte)
    raw = {k: sigmoid(v) for k, v in lt.items()}
    cal = pred.calibrated(lt)
    mte = mask_for(ate)
    complete = ate.all(axis=1)

    def rec(k: str, m: np.ndarray) -> Dict[str, Any]:
        s = summarize(yte[m], cal[k][m], cfg.n_bootstrap, cfg.seed)
        s["raw_brier"] = brier(yte[m], raw[k][m]) if m.sum() else None
        s["raw_ece"] = ece(yte[m], raw[k][m]) if m.sum() else None
        return s

    comparison = {
        "complete_cases": {k: rec(k, complete) for k in EXPERIMENT_KEYS},
        "available_cases": {k: rec(k, mte[k]) for k in EXPERIMENT_KEYS},
        "incomplete_cases": {k: rec(k, ~complete) for k in ("D1", "D2")},
    }
    val_auc = {k: auroc(yva[mva[k]], lv[k][mva[k]]) for k in ("A", "B", "C")}
    best_uni = max(val_auc, key=lambda k: -1 if np.isnan(val_auc[k]) else val_auc[k])
    yc = yte[complete]
    comparison["best_unimodal_by_validation"] = best_uni
    comparison["paired_differences"] = {
        f"{a}_minus_{b}": paired_auroc_diff(yc, cal[a][complete], cal[b][complete], cfg.n_bootstrap, cfg.seed)
        for a, b in (("D2", best_uni), ("D1", best_uni), ("D2", "D1"))
    } if len(np.unique(yc)) == 2 else {}

    calibration = {
        "complete_cases": {k: {"raw": calibration_curve(yc, raw[k][complete]), "calibrated": calibration_curve(yc, cal[k][complete])}
                           for k in EXPERIMENT_KEYS}
    }

    # ---- missing-modality robustness --------------------------------------------------------------------------------
    P("Missing-modality analysis", 0.84)
    ci_ = np.where(complete)[0]
    Tc, tabc, Lc = Tte.subset(ci_), cte.tab.iloc[ci_], Lte[ci_]
    scen: Dict[str, Any] = {}
    for name, m in SCENARIOS.items():
        av = np.tile(np.array(m, bool), (len(ci_), 1))
        lg, _ = pred.logits(tabc, Tc, avail=av, L=Lc)
        pc = pred.calibrated(lg)
        scen[name] = {
            "modalities": [mod for mod, f in zip(("image", "tabular", "timeseries"), m) if f],
            "D1": summarize(yc, pc["D1"], cfg.n_bootstrap, cfg.seed), "D2": summarize(yc, pc["D2"], cfg.n_bootstrap, cfg.seed),
            "D2_attention": attention_weights(fusion, Tc.with_avail(torch.from_numpy(av))).mean(0).tolist(),
        }
    nmod = ate.sum(1)
    by_n = {}
    for k in ("D1", "D2"):
        for n in (1, 2, 3):
            m = nmod == n
            by_n.setdefault(str(n), {})[k] = summarize(yte[m], cal[k][m], cfg.n_bootstrap, cfg.seed) if m.sum() else None
    missing = {"scenarios_on_complete_cases": scen, "by_number_of_available_modalities": by_n}

    # ---- uncertainty & disagreement ---------------------------------------------------------------------------------------
    P("Uncertainty and disagreement", 0.88)
    mc_logits = mc_predict(fusion, Tte, cfg.mc_samples, cfg.seed)
    mc_mean = pred.calibrators["D2"].prob(mc_logits).mean(0)  # same recalibration as the point estimate
    mc_std = mc_logits.std(0)  # uncertainty = spread of the logit across dropout passes
    wrong = ((cal["D2"] >= 0.5) != (yte == 1)).astype(int)
    uncertainty = {
        "method": f"Monte-Carlo dropout, {cfg.mc_samples} passes; uncertainty = std of the logit (approximate, not a guaranteed interval)",
        "mc_mean_probability": summarize(yte, mc_mean, cfg.n_bootstrap, cfg.seed),
        "mean_std": float(mc_std.mean()),
        "mean_std_when_correct": float(mc_std[wrong == 0].mean()) if (wrong == 0).any() else None,
        "mean_std_when_wrong": float(mc_std[wrong == 1].mean()) if (wrong == 1).any() else None,
        "error_detection_auroc": auroc(wrong, mc_std) if 0 < wrong.sum() < len(wrong) else None,
        "selective_prediction": selective_prediction(yte, cal["D2"], mc_std),
        "mean_std_by_n_modalities": {str(n): (float(mc_std[nmod == n].mean()) if (nmod == n).any() else None) for n in (1, 2, 3)},
    }
    uni = np.stack([cal["A"], cal["B"], cal["C"]], axis=1)
    n_av = np.isfinite(uni).sum(1)
    spread = np.where(n_av >= 2, np.nanmax(np.where(np.isfinite(uni), uni, -np.inf), 1)
                      - np.nanmin(np.where(np.isfinite(uni), uni, np.inf), 1), np.nan)
    disagreement: Dict[str, Any] = {"definition": "max - min of calibrated unimodal probabilities among available modalities"}
    cs = spread[complete]
    if len(cs) > 20:
        hi = cs >= np.median(cs)
        for nm, m in (("low_disagreement_half", ~hi), ("high_disagreement_half", hi)):
            disagreement[nm] = summarize(yc[m], cal["D2"][complete][m], cfg.n_bootstrap, cfg.seed)
            disagreement[nm]["accuracy"] = float(((cal["D2"][complete][m] >= 0.5) == (yc[m] == 1)).mean())
        disagreement["median_spread"] = float(np.median(cs))

    # ---- explainability -------------------------------------------------------------------------------------------------------
    P("Generating explanations", 0.92)
    explain_g: Dict[str, Any] = {"caveat": "Attributions describe model behaviour, not causal medical relationships."}
    mt = ate[:, 1]
    if mt.any():
        contrib = tab.contributions(cte.tab[mt])
        mi = np.abs(contrib).mean(0)
        explain_g["tabular_mean_abs_contribution"] = dict(sorted(zip(TAB_FEATURES, mi.tolist()), key=lambda kv: -kv[1]))
    ms = ate[:, 2]
    if ms.any():
        imp = ts_occlusion(ts_net, Tte.ts[torch.from_numpy(ms)])
        ys = yte[ms]
        explain_g["timeseries_importance_by_hour"] = {
            "positive_class": imp[ys == 1].mean(0).tolist() if (ys == 1).any() else None,
            "negative_class": imp[ys == 0].mean(0).tolist() if (ys == 0).any() else None,
        }
    if cte.lesion_mask is not None:
        li = np.where(ate[:, 0] & cte.lesion_mask.reshape(len(cte), -1).any(1) & (yte == 1))[0]
        if len(li):
            cams = np.concatenate([gradcam(img_net, Tte.img[li[i:i + 64]]) for i in range(0, len(li), 64)])
            explain_g["image_localisation_vs_synthetic_lesion"] = cam_localisation(cams, cte.lesion_mask[li])
    explain_g["mean_attention_full_availability"] = scen["all"]["D2_attention"]
    explain_g["attention_order"] = ["image", "tabular", "timeseries", "none"]

    local = _local_explanations(cfg, cte, Tte, tab, img_net, ts_net, pred, cal, spread, mc_std, complete)
    explainability = {"global": explain_g, "local": local}

    # ---- predictions table -------------------------------------------------------------------------------------------------------
    rows = []
    for k in EXPERIMENT_KEYS:
        m = mte[k]
        rows.append(pd.DataFrame({
            "patient_id": cte.patient_id[m], "experiment": k, "y_true": yte[m], "prob_raw": raw[k][m],
            "prob_cal": cal[k][m], "n_modalities": nmod[m], "split": "test",
        }))
    predictions = pd.concat(rows, ignore_index=True)
    predictions["uncertainty"] = np.nan  # MC-dropout uncertainty exists for the attention-fusion model only
    predictions.loc[predictions.experiment == "D2", "uncertainty"] = mc_std

    results = {
        "meta": {
            "version": __version__, "created_at": datetime.now(timezone.utc).isoformat(), "runtime_seconds": round(time.time() - t0, 1),
            "synthetic_data": cfg.data_dir is None, "disclaimer": DISCLAIMER, "config": cfg.to_dict(),
            "split": {"train": len(tr), "val": len(va), "test": len(te), "test_complete_cases": int(complete.sum())},
        },
        "validation": validation,
        "experiments": {k: {"name": n, "description": desc, "modalities": mods} for k, (n, desc, mods) in EXPERIMENTS.items()},
        "training": hist,
        "calibrators": {k: {"a": c.a, "b": c.b} for k, c in pred.calibrators.items()},
        "comparison": comparison, "calibration": calibration, "missing_modality": missing,
        "uncertainty": uncertainty, "disagreement": disagreement, "explainability": explainability,
    }
    results["findings"] = _findings(results)
    P("Done", 1.0)
    return RunOutput(clean(results), predictions, pred)


def _findings(r: Dict[str, Any]) -> list:
    cc, pd_ = r["comparison"]["complete_cases"], r["comparison"].get("paired_differences", {})
    out = []
    synth = r["meta"]["synthetic_data"]
    if synth:
        out.append("These results come from SYNTHETIC data generated to exercise the pipeline; they are not clinical findings.")
    best = r["comparison"]["best_unimodal_by_validation"]
    if cc.get("D2", {}).get("auroc") is not None and cc.get(best, {}).get("auroc") is not None:
        out.append(
            f"On complete test cases (n={cc['D2']['n']}), attention fusion AUROC {_fmt(cc['D2']['auroc'])} vs best "
            f"unimodal ({EXPERIMENTS[best][0]}, chosen on validation) {_fmt(cc[best]['auroc'])}."
        )
        diff = pd_.get(f"D2_minus_{best}")
        if diff:
            verdict = "excludes" if diff["ci_excludes_zero"] else "includes"
            out.append(f"Paired bootstrap difference {diff['diff']:+.3f} (95% CI {diff['ci'][0]:+.3f} to {diff['ci'][1]:+.3f}); "
                       f"the interval {verdict} zero. A negative or null result would be equally reportable.")
    sc = r["missing_modality"]["scenarios_on_complete_cases"]
    if sc["all"]["D2"]["auroc"] is not None:
        worst = min((v["D2"]["auroc"], k) for k, v in sc.items() if v["D2"]["auroc"] is not None)
        out.append(f"Attention fusion AUROC falls from {_fmt(sc['all']['D2']['auroc'])} (all modalities) to {_fmt(worst[0])} in the worst "
                   f"masking scenario ('{worst[1]}').")
    u = r["uncertainty"]
    if u.get("error_detection_auroc") is not None:
        sp = u["selective_prediction"]
        out.append(f"MC-dropout uncertainty separates wrong from correct predictions with AUROC {_fmt(u['error_detection_auroc'])} "
                   f"(0.5 = uninformative); keeping the {sp[-1]['coverage']:.0%} most certain patients changes accuracy from "
                   f"{sp[0]['accuracy']:.1%} to {sp[-1]['accuracy']:.1%}.")
    return out


def _local_explanations(cfg, cte, Tte, tab, img_net, ts_net, pred, cal, spread, mc_std, complete) -> list:
    ci = np.where(complete)[0]
    if not len(ci):
        return []
    p, y = cal["D2"][ci], cte.y[ci]
    sp = np.nan_to_num(spread[ci], nan=-1)
    picks, used = [], set()

    def take(role: str, cand: Optional[int]) -> None:
        if cand is not None and int(ci[cand]) not in used and len(picks) < cfg.n_explain:
            used.add(int(ci[cand])); picks.append((role, int(ci[cand])))

    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    take("highest-risk true positive", pos[np.argmax(p[pos])] if len(pos) else None)
    take("lowest-risk true negative", neg[np.argmin(p[neg])] if len(neg) else None)
    take("false negative (positive, lowest risk)", pos[np.argmin(p[pos])] if len(pos) else None)
    take("false positive (negative, highest risk)", neg[np.argmax(p[neg])] if len(neg) else None)
    take("highest modality disagreement", int(np.argmax(sp)))
    rng = np.random.default_rng(cfg.seed)
    for j in rng.permutation(len(ci)):
        take("random complete case", int(j))

    out = []
    for role, i in picks:
        x = Tte.subset([i])
        cam = gradcam(img_net, x.img)[0]
        raw_img = cte.img[i]
        contrib = tab.contributions(cte.tab.iloc[[i]])[0]
        row = cte.tab.iloc[i]
        top = np.argsort(-np.abs(contrib))[:6]
        imp = ts_occlusion(ts_net, x.ts)[0]
        out.append({
            "patient_id": str(cte.patient_id[i]), "role": role, "y_true": int(cte.y[i]),
            "probability": {k: (None if not np.isfinite(cal[k][i]) else float(cal[k][i])) for k in EXPERIMENT_KEYS},
            "uncertainty_logit_std": float(mc_std[i]), "disagreement": None if spread[i] != spread[i] else float(spread[i]),
            "image": image_png(raw_img), "image_gradcam": overlay_png(raw_img, cam),
            "lesion_overlay_available": cte.lesion_mask is not None,
            "tabular_top_contributions": [
                {"feature": TAB_FEATURES[j], "value": None if np.isnan(row[TAB_FEATURES[j]]) else float(row[TAB_FEATURES[j]]),
                 "contribution_logodds": float(contrib[j])} for j in top
            ],
            "timeseries": {"channels": TS_CHANNELS, "values": cte.ts[i].tolist(), "importance": imp.tolist()},
        })
    return out
