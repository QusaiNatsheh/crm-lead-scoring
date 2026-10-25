"""Scoring service. Run: uvicorn crm_score_ml.service:app

Odoo sends the facts it knows about each open lead today; the service returns a calibrated win probability and
the main reasons up and down. Unknown category values (e.g. a source the model never saw) are allowed: the model
treats them as missing.
"""

import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from .features import CATEGORIES
from .model import Model

MAX_LEADS = 5000


class LeadIn(BaseModel):
    id: int
    team: str | None = Field(None, max_length=100)
    source: str | None = Field(None, max_length=100)
    country: str | None = Field(None, max_length=100)
    industry: str | None = Field(None, max_length=100)
    company_size: str | None = Field(None, max_length=20)
    email_state: str | None = Field(None, max_length=20)
    phone_state: str | None = Field(None, max_length=20)
    lang: str | None = Field(None, max_length=10)
    tags: str = Field("", max_length=1000)
    expected_revenue: float = Field(0.0, ge=0)
    stage_seq: int = Field(ge=1, le=10)
    days_since_created: int = Field(ge=0)
    days_in_stage: int = Field(ge=0)
    activity_count: int = Field(0, ge=0)
    activities_last_14d: int = Field(0, ge=0)
    days_since_last_activity: int | None = Field(None, ge=0)
    first_response_days: int | None = Field(None, ge=0)


class ScoreIn(BaseModel):
    leads: list[LeadIn] = Field(max_length=MAX_LEADS)


class ScoreOut(BaseModel):
    id: int
    probability: float
    raises: list[str]
    lowers: list[str]


class ScoreResponse(BaseModel):
    model_version: str
    scores: list[ScoreOut]


@lru_cache
def get_model() -> Model:
    return Model(Path(os.environ.get("MODEL_DIR", "models/latest")))


app = FastAPI(title="CRM lead scoring service", version="1.0.0")


@app.get("/health")
def health(model: Model = Depends(get_model)):
    return {"status": "ok", "model_version": model.version}


@app.post("/score", response_model=ScoreResponse)
def score(body: ScoreIn, model: Model = Depends(get_model)):
    states = [lead.model_dump() for lead in body.leads]
    for s in states:
        s["stage_seq"] = min(s["stage_seq"], 4)  # the model knows 4 open stages; later ones count as the last
        if s["country"] and s["country"] not in CATEGORIES["country"]:
            s["country"] = "Other"  # training labelled every country outside the list as "Other"
    results = model.score(states)
    return ScoreResponse(
        model_version=model.version, scores=[ScoreOut(id=s["id"], **r) for s, r in zip(states, results, strict=True)]
    )
