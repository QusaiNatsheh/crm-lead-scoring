"""Load the trained model and score leads: calibrated win probability plus plain-language reasons."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .features import FEATURES, to_frame
from .generator import STAGES


def _describe(feature: str, state: dict, x: dict) -> str | None:
    v = x[feature]
    if feature == "stage_seq":
        return f"Stage: {STAGES[int(v) - 1]}" if 1 <= v <= 4 else None
    if feature in ("source", "country", "industry", "team", "lang"):
        return f"{feature.capitalize()}: {state[feature]}"
    if feature == "company_size":
        return f"Company size: {state['company_size']} employees"
    if feature in ("email_state", "phone_state"):
        kind = "Email" if feature == "email_state" else "Phone"
        return {"correct": f"{kind} is valid", "incorrect": f"{kind} looks invalid", "none": f"No {kind.lower()}"}.get(
            state[feature]
        )
    if feature.startswith("tag_"):
        return f"Tag {feature[4:]}" if v else None
    if feature == "log_expected_revenue":
        return f"Expected revenue {state['expected_revenue']:,.0f}"
    if np.isnan(v):
        return "No activity yet" if feature in ("days_since_last_activity", "first_response_days") else None
    labels = {
        "activity_count": f"{int(v)} activities so far",
        "activities_last_14d": f"{int(v)} activities in the last 14 days",
        "days_since_last_activity": f"Last activity {int(v)} days ago",
        "first_response_days": f"First contact after {int(v)} day(s)",
        "days_in_stage": f"{int(v)} days in the current stage",
        "days_since_created": f"Lead is {int(v)} days old",
    }
    return labels.get(feature)


class Model:
    def __init__(self, path: Path):
        path = Path(path)
        self.booster = lgb.Booster(model_file=str(path / "model.txt"))
        self.meta = json.loads((path / "meta.json").read_text())
        if self.meta["features"] != FEATURES:
            raise ValueError("Model features do not match this code version")
        cal = self.meta["calibrator"]
        self._cx, self._cy = np.array(cal["x"]), np.array(cal["y"])

    @property
    def version(self) -> str:
        return self.meta["version"]

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        return np.interp(raw, self._cx, self._cy)

    def score(self, states: list[dict], top: int = 3) -> list[dict]:
        if not states:
            return []
        X = to_frame(states)
        raw = self.booster.predict(X)
        prob = self.calibrate(raw)
        contrib = self.booster.predict(X, pred_contrib=True)[:, :-1]  # SHAP values per feature (log-odds)
        rows = X.to_dict("records")
        out = []
        for i, state in enumerate(states):
            order = np.argsort(-np.abs(contrib[i]))
            up, down = [], []
            for j in order:
                text = _describe(FEATURES[j], state, rows[i])
                if not text or abs(contrib[i, j]) < 0.05:
                    continue
                (up if contrib[i, j] > 0 else down).append(text)
            out.append({"probability": float(prob[i]), "raises": up[:top], "lowers": down[:top]})
        return out
