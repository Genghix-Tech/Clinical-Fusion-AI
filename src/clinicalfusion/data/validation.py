"""Data validation: schema, ranges, duplicate/identifier matching, missingness and completeness."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable

import numpy as np

from .schema import TAB_FEATURES, TAB_RANGES, TS_CHANNELS, TS_RANGES, Cohort


def match_patient_ids(**id_sets: Iterable[str]) -> Dict[str, Any]:
    """Overlap report for identifiers coming from separate sources (e.g. images / labs / vitals).

    Combining datasets is only valid for patients present in all of them; report coverage first.
    """
    sets = {k: set(v) for k, v in id_sets.items()}
    inter = set.intersection(*sets.values()) if sets else set()
    union = set.union(*sets.values()) if sets else set()
    return {
        "sizes": {k: len(s) for k, s in sets.items()},
        "in_all_sources": len(inter),
        "in_any_source": len(union),
        "only_in": {k: len(s - set.union(*(o for kk, o in sets.items() if kk != k))) if len(sets) > 1 else len(s)
                    for k, s in sets.items()},
    }


def validate_cohort(c: Cohort) -> Dict[str, Any]:
    issues, fatal = [], []
    n = len(c)
    ids = np.asarray(c.patient_id)
    n_unique = len(set(ids.tolist()))
    if n_unique != n:
        fatal.append(f"{n - n_unique} duplicate patient ids (would leak across splits)")

    labels = set(np.unique(c.y).tolist())
    if not labels <= {0, 1}:
        fatal.append(f"labels must be binary, found {sorted(labels)}")
    prevalence = float(np.mean(c.y)) if n else float("nan")
    if n and (prevalence < 0.02 or prevalence > 0.98):
        issues.append(f"extreme class imbalance (prevalence={prevalence:.3f})")

    # shape consistency
    for name, arr in [("tab", c.tab), ("ts", c.ts), ("img", c.img), ("has_img", c.has_img)]:
        if len(arr) != n:
            fatal.append(f"{name} has {len(arr)} rows, expected {n}")

    # availability flags must agree with the data
    img_flag_bad = int(np.sum(c.has_img & (np.abs(c.img).reshape(n, -1).sum(1) == 0)))
    if img_flag_bad:
        issues.append(f"{img_flag_bad} patients flagged has_img but image is all zeros")
    ts_flag_bad = int(np.sum(c.has_ts & np.isnan(c.ts).reshape(n, -1).all(1)))
    if ts_flag_bad:
        issues.append(f"{ts_flag_bad} patients flagged has_ts but series is entirely missing")
    tab_flag_bad = int(np.sum(c.has_tab & c.tab.isna().all(axis=1).to_numpy()))
    if tab_flag_bad:
        issues.append(f"{tab_flag_bad} patients flagged has_tab but every value is missing")

    # ranges
    oor: Dict[str, int] = {}
    for f in TAB_FEATURES:
        lo, hi = TAB_RANGES[f]
        v = c.tab[f].to_numpy(dtype=float)
        cnt = int(np.sum((v < lo) | (v > hi)))  # NaN compares False
        if cnt:
            oor[f] = cnt
    for k, ch in enumerate(TS_CHANNELS):
        lo, hi = TS_RANGES[ch]
        v = c.ts[:, k, :]
        cnt = int(np.sum((v < lo) | (v > hi)))
        if cnt:
            oor[f"ts:{ch}"] = cnt
    if oor:
        issues.append(f"out-of-range values: {oor}")
    if np.isinf(c.ts).any():
        issues.append("infinite values in time-series")

    # missingness / completeness
    tab_missing = {f: round(float(c.tab.loc[c.has_tab, f].isna().mean()) * 100, 2) if c.has_tab.any() else 100.0
                   for f in TAB_FEATURES}
    ts_missing = {ch: round(float(np.isnan(c.ts[c.has_ts, k, :]).mean()) * 100, 2) if c.has_ts.any() else 100.0
                  for k, ch in enumerate(TS_CHANNELS)}
    availability = {
        m: {"n": int(a.sum()), "pct": round(float(a.mean()) * 100, 2) if n else 0.0}
        for m, a in zip(("image", "tabular", "timeseries"), (c.has_img, c.has_tab, c.has_ts))
    }
    short = {"image": "img", "tabular": "tab", "timeseries": "ts"}
    patterns = Counter(
        "+".join(short[m] for m, a in zip(short, row) if a) or "none" for row in c.avail
    )
    complete = int(np.sum(c.avail.all(axis=1)))

    return {
        "ok": not issues and not fatal,
        "fatal": fatal,
        "issues": issues,
        "n_patients": n,
        "n_unique_ids": n_unique,
        "prevalence": round(prevalence, 4),
        "n_positive": int(np.sum(c.y)),
        "modality_availability": availability,
        "completeness_patterns": dict(sorted(patterns.items(), key=lambda kv: -kv[1])),
        "complete_cases": complete,
        "complete_cases_pct": round(100.0 * complete / n, 2) if n else 0.0,
        "tabular_missing_pct": tab_missing,
        "timeseries_missing_pct": ts_missing,
        "out_of_range": oor,
    }
