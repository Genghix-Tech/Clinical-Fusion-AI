"""FastAPI service + static dashboard.

Security note: there is no user management. Set CF_API_KEY to require an `X-API-Key` header on every
/api call, keep the service on a private network, and do not load identifiable patient data unless
your governance and data-use agreements permit it.
"""
from __future__ import annotations

import os
import shutil
import threading
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import delete, select

from .. import DISCLAIMER, __version__
from ..config import PRESETS, Config
from .db import ARTIFACT_DIR, Prediction, Run, SessionLocal, init_db

STATIC_DIR = Path(__file__).parent / "static"
_predictors: Dict[str, Any] = {}
_run_lock = threading.Lock()


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.environ.get("CF_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "missing or invalid X-API-Key")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    with SessionLocal() as s:  # runs interrupted by a restart can never finish
        for r in s.scalars(select(Run).where(Run.status.in_(["queued", "running"]))):
            r.status, r.error = "failed", "server restarted while the run was in progress"
        s.commit()
    yield


app = FastAPI(title="ClinicalFusion", version=__version__, lifespan=lifespan,
              description=DISCLAIMER)


class RunRequest(BaseModel):
    preset: str = Field("fast", description="fast | standard")
    n_patients: Optional[int] = Field(None, ge=200, le=20000)
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)
    ts_arch: Optional[str] = Field(None, pattern="^(tcn|lstm)$")
    tab_backend: Optional[str] = Field(None, pattern="^(xgboost|catboost)$")
    epochs_img: Optional[int] = Field(None, ge=1, le=100)
    epochs_ts: Optional[int] = Field(None, ge=1, le=100)
    epochs_fusion: Optional[int] = Field(None, ge=1, le=100)
    mc_samples: Optional[int] = Field(None, ge=2, le=200)
    n_bootstrap: Optional[int] = Field(None, ge=20, le=5000)
    use_server_data_dir: bool = Field(False, description="load the cohort at the server's CF_DATA_DIR instead of synthetic data")


class PredictRequest(BaseModel):
    run_id: Optional[str] = None
    tabular: Optional[Dict[str, Optional[float]]] = None
    timeseries: Optional[Dict[str, List[Optional[float]]]] = None
    image: Optional[List[List[float]]] = None
    mc_samples: int = Field(30, ge=2, le=200)


def _execute(run_id: str) -> None:
    from ..experiments.runner import run_experiments

    with SessionLocal() as s:
        run = s.get(Run, run_id)
        run.status, run.message = "running", "starting"
        s.commit()
        cfg = Config.from_dict(run.config)

    last = [0.0]

    def progress(msg: str, frac: float) -> None:
        with SessionLocal() as s2:
            r = s2.get(Run, run_id)
            r.progress, r.message = float(frac), msg
            s2.commit()
        last[0] = frac

    try:
        out = run_experiments(cfg, progress)
        d = Path(ARTIFACT_DIR) / run_id
        d.mkdir(parents=True, exist_ok=True)
        out.predictor.save(str(d / "bundle.pkl"))
        out.predictions.to_csv(d / "predictions.csv", index=False)
        if os.environ.get("CF_MLFLOW") == "1":
            from ..tracking import log_to_mlflow
            log_to_mlflow(out.results, run_name=run_id)
        with SessionLocal() as s:
            preds = out.predictions
            s.bulk_insert_mappings(Prediction, [
                dict(run_id=run_id, patient_id=r.patient_id, experiment=r.experiment, split=r.split, y_true=int(r.y_true),
                     prob_raw=float(r.prob_raw), prob_cal=float(r.prob_cal),
                     uncertainty=None if r.uncertainty != r.uncertainty else float(r.uncertainty), n_modalities=int(r.n_modalities))
                for r in preds.itertuples()
            ])
            run = s.get(Run, run_id)
            run.status, run.progress, run.message = "done", 1.0, "done"
            run.results, run.bundle_path = out.results, str(d / "bundle.pkl")
            run.finished_at = datetime.now(timezone.utc)
            s.commit()
    except Exception as e:  # noqa: BLE001 - surfaced to the client
        traceback.print_exc()
        with SessionLocal() as s:
            run = s.get(Run, run_id)
            run.status, run.error, run.message = "failed", f"{type(e).__name__}: {e}", "failed"
            run.finished_at = datetime.now(timezone.utc)
            s.commit()
    finally:
        _run_lock.release()


def _summary(r: Run) -> Dict[str, Any]:
    head = None
    if r.results:
        cc = r.results["comparison"]["complete_cases"]
        head = {k: cc[k]["auroc"] for k in cc}
    return {"id": r.id, "status": r.status, "progress": r.progress, "message": r.message, "error": r.error,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            "config": r.config, "headline_auroc_complete_cases": head}


api = dict(dependencies=[Depends(require_key)])


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "version": __version__, "disclaimer": DISCLAIMER}


@app.get("/api/config/defaults", **api)
def defaults() -> Dict[str, Any]:
    return {k: v().to_dict() for k, v in PRESETS.items()}


@app.post("/api/runs", status_code=202, **api)
def create_run(req: RunRequest, background: BackgroundTasks) -> Dict[str, Any]:
    if req.preset not in PRESETS:
        raise HTTPException(422, f"preset must be one of {list(PRESETS)}")
    overrides = req.model_dump(exclude={"preset", "use_server_data_dir"}, exclude_none=True)
    cfg = PRESETS[req.preset](**overrides) if req.preset == "fast" else Config.from_dict(overrides)
    if req.use_server_data_dir:
        if not os.environ.get("CF_DATA_DIR"):
            raise HTTPException(400, "server has no CF_DATA_DIR configured")
        cfg.data_dir = os.environ["CF_DATA_DIR"]
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(409, "another run is in progress; wait for it to finish")
    try:
        with SessionLocal() as s:
            run = Run(config=cfg.to_dict())
            s.add(run)
            s.commit()
            rid = run.id
    except Exception:
        _run_lock.release()
        raise
    background.add_task(_execute, rid)
    return {"id": rid, "status": "queued"}


@app.get("/api/runs", **api)
def list_runs(limit: int = Query(50, ge=1, le=200)) -> List[Dict[str, Any]]:
    with SessionLocal() as s:
        return [_summary(r) for r in s.scalars(select(Run).order_by(Run.created_at.desc()).limit(limit))]


def _get_run(s, run_id: str) -> Run:
    run = s.get(Run, run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run


@app.get("/api/runs/{run_id}", **api)
def get_run(run_id: str) -> Dict[str, Any]:
    with SessionLocal() as s:
        return _summary(_get_run(s, run_id))


@app.get("/api/runs/{run_id}/results", **api)
def get_results(run_id: str) -> Dict[str, Any]:
    with SessionLocal() as s:
        run = _get_run(s, run_id)
        if run.status != "done":
            raise HTTPException(409, f"run is {run.status}")
        return run.results


@app.get("/api/runs/{run_id}/predictions", **api)
def get_predictions(run_id: str, experiment: Optional[str] = None, patient_id: Optional[str] = None,
                    limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)) -> Dict[str, Any]:
    with SessionLocal() as s:
        _get_run(s, run_id)
        q = select(Prediction).where(Prediction.run_id == run_id)
        if experiment:
            q = q.where(Prediction.experiment == experiment)
        if patient_id:
            q = q.where(Prediction.patient_id == patient_id)
        rows = s.scalars(q.order_by(Prediction.id).offset(offset).limit(limit)).all()
        return {"items": [{c: getattr(r, c) for c in ("patient_id", "experiment", "split", "y_true", "prob_raw", "prob_cal",
                                                     "uncertainty", "n_modalities")} for r in rows],
                "offset": offset, "limit": limit}


@app.delete("/api/runs/{run_id}", status_code=204, **api)
def delete_run(run_id: str) -> None:
    with SessionLocal() as s:
        run = _get_run(s, run_id)
        if run.status in ("queued", "running"):
            raise HTTPException(409, "cannot delete a run in progress")
        s.execute(delete(Prediction).where(Prediction.run_id == run_id))
        s.delete(run)
        s.commit()
    _predictors.pop(run_id, None)
    shutil.rmtree(Path(ARTIFACT_DIR) / run_id, ignore_errors=True)


@app.post("/api/predict", **api)
def predict(req: PredictRequest) -> Dict[str, Any]:
    """Single-patient inference with ANY subset of modalities (research demo, not for clinical use)."""
    from ..predictor import Predictor

    with SessionLocal() as s:
        if req.run_id:
            run = _get_run(s, req.run_id)
        else:
            run = s.scalars(select(Run).where(Run.status == "done").order_by(Run.created_at.desc()).limit(1)).first()
            if not run:
                raise HTTPException(409, "no finished run yet; start one with POST /api/runs")
        if run.status != "done" or not run.bundle_path:
            raise HTTPException(409, f"run is {run.status}")
        rid, bpath = run.id, run.bundle_path
    if rid not in _predictors:
        _predictors[rid] = Predictor.load(bpath)
    try:
        out = _predictors[rid].predict_sample(req.tabular, req.timeseries, req.image, req.mc_samples)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"run_id": rid, **out}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")
