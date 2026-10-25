"""README charts: calibration (predicted vs actual win rate) and lift, model vs Odoo's native scoring."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#2b2b29", "#6b6b66", "#e6e5e0"
COLORS = {"model": "#2a78d6", "odoo": "#eb6834"}  # validated categorical slots 1 and 2
LABELS = {"model": "This model (LightGBM, calibrated)", "odoo": "Odoo native scoring (naive Bayes)"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def calibration(meta: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    for key, name in (("model", "model"), ("odoo", "odoo_naive_bayes")):
        rows = meta["metrics"]["calibration"][name]
        ax.plot(
            [r["predicted"] for r in rows],
            [r["actual"] for r in rows],
            marker="o",
            markersize=6,
            linewidth=2,
            color=COLORS[key],
            label=LABELS[key],
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted win probability", color=MUTED, fontsize=9)
    ax.set_ylabel("Actual win rate", color=MUTED, fontsize=9)
    ax.set_title("Calibration on the test months", color=TEXT, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def lift(preds: pd.DataFrame, path: Path) -> None:
    """Win rate by score decile (10 = highest scores)."""
    fig, ax = plt.subplots(figsize=(6.4, 4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)
    base = preds["y"].mean()
    width = 0.4
    for k, key in enumerate(("model", "odoo")):
        dec = pd.qcut(preds[key].rank(method="first"), 10, labels=False) + 1
        rates = preds.groupby(dec)["y"].mean()
        ax.bar(
            rates.index + (k - 0.5) * (width + 0.04),
            rates.values * 100,
            width=width,
            color=COLORS[key],
            label=LABELS[key],
        )
    ax.axhline(base * 100, color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.6, base * 100 + 1.5, f"average {base:.0%}", color=MUTED, fontsize=8)
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlabel("Score decile (10 = highest scored leads)", color=MUTED, fontsize=9)
    ax.set_ylabel("Actual win rate (%)", color=MUTED, fontsize=9)
    ax.set_title("Win rate by score decile", color=TEXT, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(model_dir: str = "models/latest", out: str = "../docs") -> None:
    model_dir, out = Path(model_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((model_dir / "meta.json").read_text())
    calibration(meta, out / "calibration.png")
    lift(pd.read_parquet(model_dir / "test_predictions.parquet"), out / "lift.png")
    print(f"charts -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
