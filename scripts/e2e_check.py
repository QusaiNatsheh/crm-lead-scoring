"""Score the demo's open leads through Odoo and the running service; compare with offline scores, with Odoo's
native probability, and with what actually happened. Run inside `odoo shell` after loading the demo."""

import json
import os

data = json.load(open(os.environ.get("DEMO_JSON", "/mnt/data/demo.json"), encoding="utf-8"))
by_ref = {r["ref"]: r for r in data["open"]}
Lead = env["crm.lead"]
Lead.action_ml_score_all()
env.cr.commit()
rows = []
for lead in Lead.search([("name", "=like", "% (#%)")]):
    r = by_ref[int(lead.name.rsplit("#", 1)[1].rstrip(")"))]
    rows.append(
        (r["offline_probability"], lead.ml_probability / 100, lead.automated_probability / 100, r["outcome"] == "won")
    )


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


diffs = [abs(o - m) for o, m, _, _ in rows]
labels = [y for *_, y in rows]
print(
    json.dumps(
        {
            "leads": len(rows),
            "won": sum(labels),
            "max_abs_diff_offline_vs_odoo": round(max(diffs), 4),
            "share_within_0.01": round(sum(d <= 0.01 for d in diffs) / len(diffs), 3),
            "roc_auc_ml": round(auc([m for _, m, _, _ in rows], labels), 3),
            "roc_auc_odoo_native": round(auc([n for _, _, n, _ in rows], labels), 3),
        }
    )
)
