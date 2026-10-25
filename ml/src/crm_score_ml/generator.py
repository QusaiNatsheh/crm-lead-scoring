"""Synthetic CRM history: leads, dated stage changes, dated activities, final outcome.

Each lead has visible attributes (source, country, industry, size, contact quality, ...) and a hidden quality that
the model never sees. Leads move through New -> Qualified -> Proposition -> Negotiation -> Won, or get lost at any
stage. Every event is dated, so a lead can be "seen" as it looked on any day: this is what makes leakage-free
training snapshots possible.

Usage: python -m crm_score_ml.generator --out data/generated/v1 --seed 42
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ["New", "Qualified", "Proposition", "Negotiation"]  # sequence 1..4; Won = 5
SOURCES = {"website": 0.0, "referral": 0.9, "cold_call": -0.8, "email_campaign": -0.4, "event": 0.3, "partner": 0.6}
COUNTRIES = {"JO": 0.2, "SA": 0.3, "AE": 0.1, "EG": -0.3, "KW": 0.0, "QA": 0.1, "Other": -0.4}
INDUSTRIES = {
    "retail": 0.1,
    "manufacturing": 0.3,
    "services": 0.0,
    "healthcare": 0.2,
    "education": -0.3,
    "construction": -0.1,
    "government": -0.5,
}
SIZES = {"1-10": -0.5, "11-50": 0.0, "51-200": 0.3, "201-1000": 0.2, "1000+": -0.2}
SIZE_REVENUE = {"1-10": 2_000, "11-50": 6_000, "51-200": 15_000, "201-1000": 35_000, "1000+": 80_000}
TAGS = {"ERP": 0.2, "POS": 0.0, "eCommerce": -0.1, "Accounting": 0.1, "Manufacturing": 0.2}
TEAMS = ["Direct Sales", "Channel"]
SALESPEOPLE = {  # name: (team, diligence: how fast and how often they follow up)
    "Lina": ("Direct Sales", 1.3),
    "Omar": ("Direct Sales", 1.0),
    "Sara": ("Direct Sales", 0.7),
    "Khaled": ("Channel", 1.1),
    "Rami": ("Channel", 0.8),
    "Dana": ("Channel", 0.6),
}


@dataclass
class GeneratorConfig:
    seed: int = 42
    start: str = "2024-10-01"
    days: int = 730
    leads_per_day: float = 9.0
    base_win_logit: float = -0.4  # overall difficulty; tuned for a ~25-30% win rate among closed leads


class Generator:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.start = date.fromisoformat(cfg.start)
        self.end = self.start + timedelta(days=cfg.days)

    def _pick(self, options: dict | list, p=None):
        keys = list(options)
        return keys[int(self.rng.choice(len(keys), p=p))]

    def _lead(self, lead_id: int, created: date) -> tuple[dict, float]:
        rng = self.rng
        source = self._pick(SOURCES, p=[0.35, 0.12, 0.15, 0.2, 0.1, 0.08])
        country = self._pick(COUNTRIES, p=[0.3, 0.25, 0.12, 0.1, 0.08, 0.05, 0.1])
        industry = self._pick(INDUSTRIES)
        size = self._pick(SIZES, p=[0.3, 0.3, 0.2, 0.12, 0.08])
        email_state = self._pick(["correct", "incorrect", "none"], p=[0.75, 0.1, 0.15])
        phone_state = self._pick(["correct", "incorrect", "none"], p=[0.6, 0.1, 0.3])
        lang = "ar" if country in ("JO", "SA", "EG", "KW", "QA") and rng.random() < 0.6 else "en"
        tags = sorted(set(rng.choice(list(TAGS), size=int(rng.integers(0, 3)), replace=False).tolist()))
        salesperson = self._pick(SALESPEOPLE)
        team, diligence = SALESPEOPLE[salesperson]
        revenue = float(np.round(rng.lognormal(np.log(SIZE_REVENUE[size]), 0.6), -1))
        quality = (
            SOURCES[source]
            + COUNTRIES[country]
            + INDUSTRIES[industry]
            + SIZES[size]
            + sum(TAGS[t] for t in tags) * 0.5
            + (0.4 if email_state == "correct" else -0.3)
            + (0.3 if phone_state == "correct" else -0.1)
            + rng.normal(0, 0.9)
        )  # most of the hidden quality is not visible in the attributes
        lead = {
            "lead_id": lead_id,
            "created": created,
            "team": team,
            "salesperson": salesperson,
            "source": source,
            "country": country,
            "industry": industry,
            "company_size": size,
            "email_state": email_state,
            "phone_state": phone_state,
            "lang": lang,
            "tags": ",".join(tags),
            "expected_revenue": revenue,
        }
        return lead, quality, diligence

    def _simulate(self, lead: dict, quality: float, diligence: float):
        """Return (stage changes, activities, outcome, closed_date)."""
        rng = self.rng
        day = lead["created"]
        response_h = float(rng.lognormal(np.log(20 / diligence), 0.9))  # hours to first contact
        # Fast first contact helps: each doubling of response time costs a little.
        quality += -0.25 * np.log2(max(response_h, 1) / 12)
        stages = [(1, day)]
        activities = [day + timedelta(days=int(response_h // 24))]  # first contact
        seq = 1
        while True:
            dwell = max(1, int(rng.exponential([6, 10, 12, 9][seq - 1] / (0.8 + 0.2 * diligence))))
            # Engaged prospects get more follow-ups: activity rate depends on hidden quality and diligence.
            rate = 0.12 * diligence * np.exp(0.35 * quality)
            n_act = rng.poisson(rate * dwell)
            for _ in range(n_act):
                activities.append(day + timedelta(days=int(rng.integers(0, dwell + 1))))
            day = day + timedelta(days=dwell)
            if day > self.end:
                return stages, activities, None, None  # still open at the end of the data
            p_advance = 1 / (1 + np.exp(-(self.cfg.base_win_logit + quality + [1.3, 0.9, 0.6, 0.4][seq - 1])))
            if rng.random() < p_advance:
                seq += 1
                if seq == 5:
                    return stages, activities, "won", day
                stages.append((seq, day))
            else:
                return stages, activities, "lost", day

    def generate(self) -> dict[str, pd.DataFrame]:
        leads, stage_rows, activity_rows = [], [], []
        lead_id = 1
        for d in range(self.cfg.days):
            created = self.start + timedelta(days=d)
            season = 1 + 0.2 * np.sin(2 * np.pi * d / 365)
            for _ in range(
                self.rng.poisson(self.cfg.leads_per_day * season * (0.4 if created.weekday() in (4, 5) else 1.0))
            ):
                lead, quality, diligence = self._lead(lead_id, created)
                stages, activities, outcome, closed = self._simulate(lead, quality, diligence)
                lead.update({"outcome": outcome, "closed": closed, "_quality": quality})
                leads.append(lead)
                stage_rows += [{"lead_id": lead_id, "stage_seq": s, "date": dt} for s, dt in stages]
                activity_rows += [{"lead_id": lead_id, "date": a} for a in activities if a <= (closed or self.end)]
                lead_id += 1
        return {
            "leads": pd.DataFrame(leads),
            "stage_changes": pd.DataFrame(stage_rows),
            "activities": pd.DataFrame(activity_rows),
        }


def save(data: dict, out: Path, cfg: GeneratorConfig) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))


def load(path: Path) -> dict[str, pd.DataFrame]:
    data = {n: pd.read_parquet(Path(path) / f"{n}.parquet") for n in ("leads", "stage_changes", "activities")}
    for df, cols in (
        (data["leads"], ["created", "closed"]),
        (data["stage_changes"], ["date"]),
        (data["activities"], ["date"]),
    ):
        for c in cols:
            df[c] = pd.to_datetime(df[c])
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/generated/v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    cfg = GeneratorConfig(seed=args.seed)
    data = Generator(cfg).generate()
    save(data, args.out, cfg)
    leads = data["leads"]
    closed = leads[leads["outcome"].notna()]
    print(
        f"{len(leads)} leads, {len(closed)} closed, win rate {(closed['outcome'] == 'won').mean():.1%}, "
        f"{len(data['activities'])} activities"
    )


if __name__ == "__main__":
    main()
