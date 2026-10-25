import numpy as np
import pandas as pd
from crm_score_ml import service
from crm_score_ml.features import FEATURES, raw_state, snapshots, to_frame
from crm_score_ml.generator import load
from fastapi.testclient import TestClient


def test_snapshots_never_see_the_outcome(dataset):
    data = load(dataset)
    snaps = snapshots(data)
    closed = data["leads"].set_index("lead_id")["closed"]
    snap_closed = snaps["lead_id"].map(closed)
    assert (snaps["at"][snap_closed.notna()] < snap_closed[snap_closed.notna()]).all()
    # No snapshot is taken in a won state: the open stages are 1-4 only.
    assert snaps["stage_seq"].between(1, 4).all()


def test_raw_state_counts_only_past_activities():
    lead = {
        "created": pd.Timestamp("2026-01-01"),
        "team": "Channel",
        "source": "event",
        "country": "JO",
        "industry": "retail",
        "company_size": "11-50",
        "email_state": "correct",
        "phone_state": "none",
        "lang": "en",
        "tags": "ERP",
        "expected_revenue": 1000.0,
    }
    stages = [(1, pd.Timestamp("2026-01-01")), (2, pd.Timestamp("2026-01-10"))]
    acts = [pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-12"), pd.Timestamp("2026-02-01")]
    s = raw_state(lead, stages, acts, pd.Timestamp("2026-01-15"))
    assert s["stage_seq"] == 2 and s["days_in_stage"] == 5 and s["activity_count"] == 2
    assert s["first_response_days"] == 2 and s["days_since_last_activity"] == 3
    assert list(to_frame([s]).columns) == FEATURES


def test_model_beats_stage_baseline_and_leak_is_detected(trained):
    _, meta = trained
    test = meta["metrics"]["test"]
    assert test["model"]["roc_auc"] > test["stage_win_rate"]["roc_auc"]
    assert test["model"]["roc_auc"] > 0.65
    assert meta["metrics"]["leakage_demo"]["roc_auc"] > 0.95  # the leaky setup looks (falsely) near perfect


def test_service_scores_with_reasons(trained, monkeypatch):
    path, _ = trained
    monkeypatch.setenv("MODEL_DIR", str(path))
    service.get_model.cache_clear()
    client = TestClient(service.app)
    lead = {
        "id": 7,
        "team": "Direct Sales",
        "source": "referral",
        "country": "SA",
        "industry": "manufacturing",
        "company_size": "51-200",
        "email_state": "correct",
        "phone_state": "correct",
        "lang": "ar",
        "tags": "ERP",
        "expected_revenue": 20000,
        "stage_seq": 4,
        "days_since_created": 30,
        "days_in_stage": 3,
        "activity_count": 6,
        "activities_last_14d": 3,
        "days_since_last_activity": 1,
        "first_response_days": 0,
    }
    cold = {
        **lead,
        "id": 8,
        "source": "cold_call",
        "stage_seq": 1,
        "activity_count": 0,
        "activities_last_14d": 0,
        "days_since_last_activity": None,
        "first_response_days": None,
        "email_state": "none",
        "company_size": "1-10",
        "country": "Mars",
    }  # unknown country is accepted
    res = client.post("/score", json={"leads": [lead, cold]})
    assert res.status_code == 200
    hot, cold_s = res.json()["scores"]
    assert 0 < cold_s["probability"] < hot["probability"] < 1
    assert hot["raises"] and isinstance(cold_s["lowers"], list)
    bad = {**lead, "stage_seq": 0}
    assert client.post("/score", json={"leads": [bad]}).status_code == 422


def test_odoo_baseline_probabilities_in_range(trained):
    _, meta = trained
    rows = meta["metrics"]["calibration"]["odoo_naive_bayes"]
    assert rows and all(0 <= r["predicted"] <= 1 for r in rows)
    assert np.isfinite(meta["metrics"]["test"]["odoo_naive_bayes"]["roc_auc"])
