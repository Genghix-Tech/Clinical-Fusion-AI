import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'cf.db'}")
    monkeypatch.setenv("CF_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    # db settings are read at import time: reload the modules that depend on them
    import importlib

    import clinicalfusion.api.db as db
    import clinicalfusion.api.main as main

    importlib.reload(db)
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def test_health_and_auth(client, monkeypatch):
    assert client.get("/api/health").json()["status"] == "ok"
    monkeypatch.setenv("CF_API_KEY", "secret")
    assert client.get("/api/runs").status_code == 401
    assert client.get("/api/runs", headers={"X-API-Key": "secret"}).status_code == 200


def test_validation_of_requests(client):
    assert client.post("/api/runs", json={"n_patients": 5}).status_code == 422
    assert client.post("/api/runs", json={"preset": "bogus"}).status_code == 422
    assert client.post("/api/runs", json={"use_server_data_dir": True}).status_code == 400
    assert client.get("/api/runs/nope").status_code == 404
    assert client.post("/api/predict", json={"tabular": {"age": 60}}).status_code == 409  # no finished run yet


def test_only_one_run_at_a_time(client):
    import clinicalfusion.api.main as main

    assert main._run_lock.acquire(blocking=False)  # simulate a run in progress
    try:
        assert client.post("/api/runs", json={}).status_code == 409
    finally:
        main._run_lock.release()


def test_full_run_lifecycle(client):
    # NB: TestClient executes background tasks before post() returns, so the run is finished here.
    r = client.post("/api/runs", json={"preset": "fast", "n_patients": 400, "epochs_img": 1, "epochs_ts": 2,
                                        "epochs_fusion": 2, "mc_samples": 3, "n_bootstrap": 20})
    assert r.status_code == 202, r.text
    rid = r.json()["id"]
    for _ in range(300):
        st = client.get(f"/api/runs/{rid}").json()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(1)
    assert st["status"] == "done", st.get("error")
    assert set(st["headline_auroc_complete_cases"]) == {"A", "B", "C", "D1", "D2"}

    res = client.get(f"/api/runs/{rid}/results").json()
    assert res["comparison"]["complete_cases"]["D2"]["n"] > 0
    preds = client.get(f"/api/runs/{rid}/predictions", params={"experiment": "D2", "limit": 5}).json()
    assert len(preds["items"]) == 5 and all(p["experiment"] == "D2" for p in preds["items"])

    p = client.post("/api/predict", json={"tabular": {"age": 70, "lactate": 4.2, "sbp": 95}, "mc_samples": 3})
    assert p.status_code == 200, p.text
    assert p.json()["modalities_used"] == ["tabular"] and 0 <= p.json()["risk"]["fusion_attention"] <= 1
    assert client.post("/api/predict", json={}).status_code == 422
    assert client.post("/api/predict", json={"timeseries": {"spo2": [90.0] * 5}}).status_code == 422

    assert client.delete(f"/api/runs/{rid}").status_code == 204
    assert client.get(f"/api/runs/{rid}").status_code == 404
