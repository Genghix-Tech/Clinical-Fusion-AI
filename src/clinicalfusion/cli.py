from __future__ import annotations

import argparse
import json
import os
import sys

from .config import PRESETS, Config


def _cmd_run(a: argparse.Namespace) -> int:
    from .experiments.runner import run_experiments

    cfg = PRESETS[a.preset](seed=a.seed) if a.preset == "fast" else Config(seed=a.seed)
    if a.n_patients:
        cfg.n_patients = a.n_patients
    if a.data_dir:
        cfg.data_dir = a.data_dir
    if a.ts_arch:
        cfg.ts_arch = a.ts_arch
    if a.tab_backend:
        cfg.tab_backend = a.tab_backend
    os.makedirs(a.out, exist_ok=True)
    out = run_experiments(cfg, lambda m, f: print(f"[{f:4.0%}] {m}", flush=True))
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(out.results, f)
    out.predictions.to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    out.predictor.save(os.path.join(a.out, "bundle.pkl"))
    if a.mlflow:
        from .tracking import log_to_mlflow

        rid = log_to_mlflow(out.results, run_name=os.path.basename(os.path.abspath(a.out)))
        print("MLflow run:", rid or "mlflow not installed (pip install 'clinicalfusion[mlflow]')")
    print("\nFindings:")
    for line in out.results["findings"]:
        print(" -", line)
    print(f"\nWrote results.json, predictions.csv, bundle.pkl to {a.out}")
    return 0


def _cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("clinicalfusion.api.main:app", host=a.host, port=a.port)
    return 0


def _cmd_export(a: argparse.Namespace) -> int:
    from .data.loader import export_cohort
    from .data.synthetic import generate_cohort

    cfg = Config(n_patients=a.n_patients, seed=a.seed)
    export_cohort(generate_cohort(cfg), a.out)
    print(f"Exported synthetic cohort ({a.n_patients} patients) to {a.out} in the real-data layout.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="clinicalfusion", description="ClinicalFusion research platform (not a medical device)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run Experiments A-D and write a report")
    r.add_argument("--preset", choices=list(PRESETS), default="fast")
    r.add_argument("--out", default="runs/latest")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--n-patients", type=int)
    r.add_argument("--data-dir", help="load a real cohort (see docs/REAL_DATA.md) instead of synthetic data")
    r.add_argument("--ts-arch", choices=["tcn", "lstm"])
    r.add_argument("--tab-backend", choices=["xgboost", "catboost"])
    r.add_argument("--mlflow", action="store_true", help="log to MLflow (MLFLOW_TRACKING_URI, default ./mlruns)")
    r.set_defaults(fn=_cmd_run)

    s = sub.add_parser("serve", help="start the API + dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=_cmd_serve)

    e = sub.add_parser("export-synthetic", help="write a synthetic cohort in the on-disk real-data layout")
    e.add_argument("--out", required=True)
    e.add_argument("--n-patients", type=int, default=1000)
    e.add_argument("--seed", type=int, default=42)
    e.set_defaults(fn=_cmd_export)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
