"""Train the lead scoring model, compare it with Odoo's native scoring, and save artifacts and charts.

Split by lead creation date (never by row, so one lead's snapshots never sit on both sides):
  train: months 1-14, calibration: months 15-17, test: months 18-21.
Leads created in months 22-24 are left out of the evaluation: many are still open, and keeping only those that
closed quickly would bias the test toward fast losses.

Usage: python -m crm_score_ml.train --data data/generated/v1 --out models/latest
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .features import FEATURES, LEAKY, OdooNaiveBayes, snapshots, stage_prior, to_frame
from .generator import load

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 7,  # chosen on the calibration months (7/15/31)
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 7,
}


def split(snaps: pd.DataFrame) -> pd.DataFrame:
    start = snaps["created"].min()
    month = (snaps["created"].dt.year - start.year) * 12 + snaps["created"].dt.month - start.month
    snaps = snaps.assign(split=np.select([month < 14, month < 17, month < 21], ["train", "calib", "test"], "recent"))
    return snaps[snaps["won"].notna()]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    order = np.argsort(-p)
    top = order[: int(0.2 * len(p))]
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(
        abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()
    )
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_error": float(ece),
        "lift_top20": float(y[top].mean() / y.mean()),
        "win_rate_top20": float(y[top].mean()),
        "base_rate": float(y.mean()),
        "n": int(len(y)),
    }


def calibration_table(y: np.ndarray, p: np.ndarray) -> list[dict]:
    bins = np.clip((p * 10).astype(int), 0, 9)
    return [
        {
            "bin": b / 10,
            "predicted": float(p[bins == b].mean()),
            "actual": float(y[bins == b].mean()),
            "n": int((bins == b).sum()),
        }
        for b in range(10)
        if (bins == b).sum() >= 20
    ]


def train(data_dir: Path, out: Path) -> dict:
    cache = data_dir / "snapshots.parquet"
    snaps = pd.read_parquet(cache) if cache.exists() else snapshots(load(data_dir))
    snaps = split(snaps)
    parts = {k: snaps[snaps["split"] == k] for k in ("train", "calib", "test")}
    X = {k: to_frame(v.to_dict("records")) for k, v in parts.items()}
    y = {k: v["won"].to_numpy() for k, v in parts.items()}

    booster = lgb.train(
        PARAMS,
        lgb.Dataset(X["train"], y["train"]),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(X["calib"], y["calib"])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    raw = {k: booster.predict(X[k], num_iteration=booster.best_iteration) for k in ("calib", "test")}
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(raw["calib"], y["calib"])
    p_model = iso.predict(raw["test"])

    # Baselines, fitted on training leads only
    data = load(data_dir)
    train_leads = data["leads"][data["leads"]["lead_id"].isin(parts["train"]["lead_id"])]
    final_stage = data["stage_changes"].groupby("lead_id")["stage_seq"].max()
    odoo = OdooNaiveBayes().fit(train_leads, final_stage)
    p_odoo = odoo.predict(parts["test"])
    prior = stage_prior(parts["train"])
    p_stage = parts["test"]["stage_seq"].map(prior).to_numpy()

    results = {
        "test": {
            "model": metrics(y["test"], p_model),
            "odoo_naive_bayes": metrics(y["test"], p_odoo),
            "stage_win_rate": metrics(y["test"], p_stage),
        }
    }
    first = (parts["test"]["stage_seq"] == 1).to_numpy() & (parts["test"]["days_since_created"] == 0).to_numpy()
    results["test_new_leads_only"] = {
        "model": metrics(y["test"][first], p_model[first]),
        "odoo_naive_bayes": metrics(y["test"][first], p_odoo[first]),
    }
    results["bootstrap_auc_difference"] = bootstrap_difference(
        parts["test"]["lead_id"].to_numpy(), y["test"], p_model, p_odoo
    )
    results["calibration"] = {
        "model": calibration_table(y["test"], p_model),
        "odoo_naive_bayes": calibration_table(y["test"], p_odoo),
    }
    results["leakage_demo"] = leakage_demo(snaps)
    results["split_sizes"] = {k: {"snapshots": len(v), "leads": int(v["lead_id"].nunique())} for k, v in parts.items()}

    gain = booster.feature_importance("gain")
    importance = dict(sorted(zip(FEATURES, (gain / gain.sum()).round(4).tolist(), strict=True), key=lambda kv: -kv[1]))
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.txt"), num_iteration=booster.best_iteration)
    meta = {
        "version": datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
        "features": FEATURES,
        "params": PARAMS,
        "best_iteration": booster.best_iteration,
        "calibrator": {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()},
        "metrics": results,
        "feature_importance_gain": importance,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame({"y": y["test"], "model": p_model, "odoo": p_odoo}).to_parquet(out / "test_predictions.parquet")
    _log_mlflow(meta, out)
    return meta


def bootstrap_difference(groups: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 1000) -> dict:
    """ROC-AUC(a) - ROC-AUC(b) with a 95% interval, resampling whole leads (their snapshots are correlated)."""
    rng = np.random.default_rng(0)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    diffs = []
    for _ in range(n):
        rows = np.concatenate([index[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if y[rows].min() == y[rows].max():
            continue
        diffs.append(roc_auc_score(y[rows], a[rows]) - roc_auc_score(y[rows], b[rows]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean": float(np.mean(diffs)),
        "ci95": [float(lo), float(hi)],
        "share_positive": float(np.mean(np.array(diffs) > 0)),
    }


def leakage_demo(snaps: pd.DataFrame) -> dict:
    """The classic mistake: one row per closed lead with its final values, split at random.

    final_stage_reached, total_activities and days_to_close are only known after the deal closes.
    """
    rng = np.random.default_rng(0)
    closed = snaps.sort_values("at").groupby("lead_id").tail(1)
    X = to_frame(closed.to_dict("records")).assign(**{c: closed[c].to_numpy(float) for c in LEAKY})
    yv = closed["won"].to_numpy()
    mask = rng.random(len(closed)) < 0.7
    booster = lgb.train({**PARAMS, "seed": 1}, lgb.Dataset(X[mask], yv[mask]), num_boost_round=300)
    leaky_auc = roc_auc_score(yv[~mask], booster.predict(X[~mask]))
    return {"description": "final values of closed leads + random split", "roc_auc": float(leaky_auc)}


def _log_mlflow(meta: dict, out: Path) -> None:
    try:
        import mlflow
        import mlflow.lightgbm
    except ImportError:
        return
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("crm-lead-scoring")
    with mlflow.start_run(run_name=meta["version"]):
        mlflow.log_params({k: str(v) for k, v in meta["params"].items()})
        for who, m in meta["metrics"]["test"].items():
            for k, v in m.items():
                mlflow.log_metric(f"test_{who}_{k}", v)
        mlflow.log_metric("leaky_demo_roc_auc", meta["metrics"]["leakage_demo"]["roc_auc"])
        mlflow.log_artifact(str(out / "meta.json"))
        mlflow.lightgbm.log_model(
            lgb.Booster(model_file=str(out / "model.txt")), name="model", registered_model_name="crm-lead-scoring"
        )


def print_report(meta: dict) -> None:
    for title, block in (
        ("Test set (all snapshots)", meta["metrics"]["test"]),
        ("Test set, brand-new leads only (day 0)", meta["metrics"]["test_new_leads_only"]),
    ):
        n = next(iter(block.values()))["n"]
        print(f"\n{title}: {n} snapshots")
        print(f"{'':20}{'ROC-AUC':>9}{'PR-AUC':>9}{'Brier':>8}{'Cal.err':>9}{'Lift@20%':>10}")
        for who, m in block.items():
            print(
                f"{who:20}{m['roc_auc']:9.3f}{m['pr_auc']:9.3f}{m['brier']:8.3f}{m['calibration_error']:9.3f}"
                f"{m['lift_top20']:10.2f}"
            )
    bd = meta["metrics"]["bootstrap_auc_difference"]
    print(f"\nModel minus Odoo ROC-AUC: {bd['mean']:+.3f} (95% interval {bd['ci95'][0]:+.3f} to {bd['ci95'][1]:+.3f})")
    print(f"\nLeaky model (final values, random split): ROC-AUC {meta['metrics']['leakage_demo']['roc_auc']:.3f}")
    print("Top features:", list(meta["feature_importance_gain"].items())[:6])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/generated/v1"))
    parser.add_argument("--out", type=Path, default=Path("models/latest"))
    args = parser.parse_args(argv)
    print_report(train(args.data, args.out))


if __name__ == "__main__":
    main()
