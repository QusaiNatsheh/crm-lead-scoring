"""Leakage-free features: a lead is described only by what was known on the scoring day.

`raw_state` builds the facts about one lead on one day from its dated history; Odoo sends the same facts for
open leads. `to_frame` turns facts into model features, identically in training and in the service.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from .generator import COUNTRIES, INDUSTRIES, SIZES, SOURCES, TAGS, TEAMS

CATEGORIES = {
    "team": TEAMS,
    "source": list(SOURCES),
    "country": list(COUNTRIES),
    "industry": list(INDUSTRIES),
    "company_size": list(SIZES),
    "email_state": ["correct", "incorrect", "none"],
    "phone_state": ["correct", "incorrect", "none"],
    "lang": ["ar", "en"],
}
NUMERIC = [
    "log_expected_revenue",
    "stage_seq",
    "days_since_created",
    "days_in_stage",
    "activity_count",
    "activities_last_14d",
    "days_since_last_activity",
    "first_response_days",
]
TAG_FEATURES = [f"tag_{t}" for t in TAGS]
FEATURES = list(CATEGORIES) + TAG_FEATURES + NUMERIC
SNAPSHOT_EVERY_DAYS = 14

# Facts that are only known once the deal is closed. Used ONLY to demonstrate leakage, never in the real model.
LEAKY = ["final_stage_reached", "total_activities", "days_to_close"]


def raw_state(
    lead: dict, stages: list[tuple[int, pd.Timestamp]], activities: list[pd.Timestamp], at: pd.Timestamp
) -> dict:
    """What a salesperson could see about the lead on day `at`."""
    current = [(s, d) for s, d in stages if d <= at]
    stage_seq, entered = current[-1]
    done = sorted(a for a in activities if a <= at)
    return {
        **{k: lead[k] for k in CATEGORIES},
        "tags": lead["tags"],
        "expected_revenue": lead["expected_revenue"],
        "stage_seq": stage_seq,
        "days_since_created": (at - lead["created"]).days,
        "days_in_stage": (at - entered).days,
        "activity_count": len(done),
        "activities_last_14d": sum(a > at - timedelta(days=14) for a in done),
        "days_since_last_activity": (at - done[-1]).days if done else None,
        "first_response_days": (done[0] - lead["created"]).days if done else None,
    }


def to_frame(states: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(states)
    out = pd.DataFrame(index=df.index)
    for col, cats in CATEGORIES.items():
        out[col] = pd.Categorical(df[col], categories=cats)
    tags = df["tags"].fillna("").astype(str)
    for t in TAGS:
        out[f"tag_{t}"] = tags.str.split(",").apply(lambda xs, t=t: float(t in xs))
    out["log_expected_revenue"] = np.log1p(df["expected_revenue"].astype(float))
    for col in NUMERIC[1:]:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return out[FEATURES]


def snapshots(data: dict) -> pd.DataFrame:
    """One row per (lead, scoring day): creation day, each stage entry, then every 14 days until the lead closes.

    Rows are only taken strictly before the close date, so the outcome is never visible in the features.
    """
    leads = data["leads"]
    stages = data["stage_changes"].sort_values(["lead_id", "date"]).groupby("lead_id")
    stage_map = {k: list(zip(g["stage_seq"], g["date"], strict=True)) for k, g in stages}
    acts = data["activities"].groupby("lead_id")["date"].apply(list).to_dict()
    end = data["stage_changes"]["date"].max()
    rows = []
    for lead in leads.to_dict("records"):
        lid = lead["lead_id"]
        st, ac = stage_map[lid], acts.get(lid, [])
        stop = lead["closed"] if pd.notna(lead["closed"]) else end
        days = set()
        for i, (_seq, entered) in enumerate(st):
            leave = st[i + 1][1] if i + 1 < len(st) else stop
            d = entered
            while d < leave:
                days.add(d)
                d += timedelta(days=SNAPSHOT_EVERY_DAYS)
        for d in sorted(x for x in days if x < stop):
            state = raw_state(lead, st, ac, d)
            final_stage = st[-1][0] + (1 if lead["outcome"] == "won" else 0)
            rows.append(
                {
                    "lead_id": lid,
                    "at": d,
                    "created": lead["created"],
                    "outcome": lead["outcome"],
                    "won": float(lead["outcome"] == "won") if lead["outcome"] else np.nan,
                    **state,
                    # leakage demonstration only
                    "final_stage_reached": final_stage,
                    "total_activities": len(ac),
                    "days_to_close": (lead["closed"] - lead["created"]).days if pd.notna(lead["closed"]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Odoo's native scoring (baseline)
ODOO_FIELDS = ["phone_state", "email_state", "country", "source", "lang"]


class OdooNaiveBayes:
    """Re-implementation of Odoo 20 predictive lead scoring (crm/models/crm_lead.py, _pls_*):

    - frequency tables of won/lost counts per field value and per team, from closed leads
    - stage: a won lead counts for every stage; a lost lead for its stage and all earlier stages
    - +0.1 on every count to avoid zero frequencies; tags with fewer than 50 closed leads are ignored
    - P = S_won / (S_won + S_lost), with S = P(team outcome) * product of P(value | outcome)
    """

    def fit(self, leads: pd.DataFrame, final_stage: pd.Series) -> "OdooNaiveBayes":
        self.freq = {}
        for team, g in leads.groupby("team"):
            table = {
                "won": float((g["outcome"] == "won").sum()),
                "lost": float((g["outcome"] == "lost").sum()),
                "fields": {},
            }
            for _, lead in g.iterrows():
                won = lead["outcome"] == "won"
                values = [(f, lead[f]) for f in ODOO_FIELDS] + [("tag", t) for t in str(lead["tags"]).split(",") if t]
                stages = range(1, 5) if won else range(1, int(final_stage[lead["lead_id"]]) + 1)
                values += [("stage", s) for s in stages]
                for key in values:
                    c = table["fields"].setdefault(key, [0.1, 0.1])
                    c[0 if won else 1] += 1
            for key in [k for k in table["fields"] if k[0] == "tag"]:
                if sum(table["fields"][key]) < 50:
                    del table["fields"][key]
            table["totals"] = {}
            for (field, _v), (w, lo) in table["fields"].items():
                t = table["totals"].setdefault(field, [0.0, 0.0])
                t[0] += w
                t[1] += lo
            self.freq[team] = table
        return self

    def predict(self, states: pd.DataFrame) -> np.ndarray:
        out = []
        for s in states.to_dict("records"):
            t = self.freq.get(s["team"])
            if t is None or not t["won"] or not t["lost"]:
                out.append(np.nan)
                continue
            total = t["won"] + t["lost"]
            sw, sl = t["won"] / total, t["lost"] / total
            values = [(f, s[f]) for f in ODOO_FIELDS] + [("tag", x) for x in str(s["tags"]).split(",") if x]
            values.append(("stage", s["stage_seq"]))
            for field, value in values:
                c = t["fields"].get((field, value))
                if c is None:
                    continue
                tw, tl = (t["won"], t["lost"]) if field == "stage" else t["totals"][field]
                sw *= c[0] / tw
                sl *= c[1] / tl
            out.append(min(max(sw / (sw + sl), 0.0001), 0.9999))
        return np.array(out)


def stage_prior(train: pd.DataFrame) -> dict:
    """Simplest baseline: historical win rate of snapshots at each stage."""
    return train.groupby("stage_seq")["won"].mean().to_dict()
