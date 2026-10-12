"""Optional MLflow experiment tracking (pip install 'clinicalfusion[mlflow]')."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional


def _flatten(prefix: str, d: Any, out: Dict[str, float]) -> None:
    if isinstance(d, dict):
        for k, v in d.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix] = float(d)


def log_to_mlflow(results: Dict[str, Any], run_name: Optional[str] = None) -> Optional[str]:
    """Log config, headline metrics and the full report. Returns the MLflow run id, or None if unavailable."""
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "clinicalfusion"))
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in results["meta"]["config"].items()})
        metrics: Dict[str, float] = {}
        for exp, rec in results["comparison"]["complete_cases"].items():
            for m in ("auroc", "auprc", "brier", "ece", "raw_ece"):
                if rec.get(m) is not None:
                    metrics[f"complete.{exp}.{m}"] = rec[m]
        for scen, rec in results["missing_modality"]["scenarios_on_complete_cases"].items():
            if rec["D2"].get("auroc") is not None:
                metrics[f"missing.{scen}.D2.auroc"] = rec["D2"]["auroc"]
        mlflow.log_metrics({k.replace("/", "_"): v for k, v in metrics.items()})
        with tempfile.TemporaryDirectory() as d:
            slim = {k: v for k, v in results.items() if k != "explainability"}  # exclude embedded PNGs
            p = os.path.join(d, "report.json")
            with open(p, "w") as f:
                json.dump(slim, f)
            mlflow.log_artifact(p)
        return run.info.run_id
