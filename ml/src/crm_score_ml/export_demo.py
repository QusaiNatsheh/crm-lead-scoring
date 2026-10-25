"""Export a demo for Odoo: leads closed before a cutoff day (to train Odoo's own scoring) and the leads open on
that day, with their history and their real outcome. Offline scores are included to check that Odoo computes
the same features. Usage: python -m crm_score_ml.export_demo --cutoff 2026-06-15 --out data/generated/demo.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .features import raw_state
from .generator import load
from .model import Model


def export(data_dir: Path, model_dir: Path, cutoff: pd.Timestamp) -> dict:
    data = load(data_dir)
    leads = data["leads"]
    stages = data["stage_changes"].sort_values(["lead_id", "date"]).groupby("lead_id")
    stage_map = {k: list(zip(g["stage_seq"], g["date"], strict=True)) for k, g in stages}
    acts = data["activities"].groupby("lead_id")["date"].apply(sorted).to_dict()
    cols = [
        "team",
        "source",
        "country",
        "industry",
        "company_size",
        "email_state",
        "phone_state",
        "lang",
        "tags",
        "expected_revenue",
    ]
    closed = leads[leads["closed"].notna() & (leads["closed"] <= cutoff)]
    open_ = leads[(leads["created"] <= cutoff) & ~(leads["closed"].notna() & (leads["closed"] <= cutoff))]
    model = Model(model_dir)
    open_rows, states = [], []
    for lead in open_.to_dict("records"):
        st = stage_map[lead["lead_id"]]
        state = raw_state(lead, st, acts.get(lead["lead_id"], []), cutoff)
        states.append(state)
        current = [d for s, d in st if d <= cutoff][-1]
        open_rows.append(
            {
                "ref": int(lead["lead_id"]),
                **{c: lead[c] for c in cols},
                "stage_seq": int(state["stage_seq"]),
                "created": lead["created"].isoformat(),
                "stage_entered": current.isoformat(),
                "activities": [a.isoformat() for a in acts.get(lead["lead_id"], []) if a <= cutoff],
                "outcome": lead["outcome"],
            }
        )
    for row, score in zip(open_rows, model.score(states), strict=True):
        row["offline_probability"] = score["probability"]
    final_stage = data["stage_changes"].groupby("lead_id")["stage_seq"].max()
    closed_rows = [
        {
            "ref": int(r["lead_id"]),
            **{c: r[c] for c in cols},
            "outcome": r["outcome"],
            "stage_seq": int(final_stage[r["lead_id"]]),
            "created": r["created"].isoformat(),
        }
        for r in closed.to_dict("records")
    ]
    return {"cutoff": cutoff.isoformat(), "closed": closed_rows, "open": open_rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/generated/v1"))
    parser.add_argument("--model", type=Path, default=Path("models/latest"))
    parser.add_argument("--cutoff", default="2026-06-15")
    parser.add_argument("--out", type=Path, default=Path("data/generated/demo.json"))
    args = parser.parse_args(argv)
    demo = export(args.data, args.model, pd.Timestamp(args.cutoff))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(demo, default=str))
    known = sum(r["outcome"] is not None for r in demo["open"])
    print(
        f"{len(demo['closed'])} closed leads, {len(demo['open'])} open on {args.cutoff} "
        f"({known} with a known outcome) -> {args.out}"
    )


if __name__ == "__main__":
    main()
