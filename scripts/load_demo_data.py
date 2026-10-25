"""Load the exported CRM demo into an Odoo 20 database. Run inside `odoo shell`:

    DEMO_JSON=/path/demo.json odoo shell -d <db> < scripts/load_demo_data.py

All dates are shifted so that the export's cutoff day becomes today. Closed leads let Odoo's own predictive lead
scoring build its frequency tables; open leads keep their stage, stage date and completed activities.
"""

import json
import os
import time
from datetime import datetime, timedelta

data = json.load(open(os.environ.get("DEMO_JSON", "/mnt/data/demo.json"), encoding="utf-8"))
shift = datetime.now() - datetime.fromisoformat(data["cutoff"])
Lead = env["crm.lead"]


COMPANIES = [
    "Al Nour",
    "Al Hayat",
    "Al Khatib",
    "Al Masri",
    "Al Omari",
    "Al Zoubi",
    "Al Sharif",
    "Al Qahtani",
    "Al Otaibi",
    "Al Ghamdi",
    "Al Harbi",
    "Al Majali",
    "Al Fayez",
    "Al Safadi",
    "Al Rowad",
    "Al Waha",
]
ACTIVITY = {
    "retail": "Retail",
    "manufacturing": "Industries",
    "services": "Services",
    "healthcare": "Medical",
    "education": "Academy",
    "construction": "Contracting",
    "government": "Municipality",
}


def lead_name(row, prefix):
    """Readable demo names that keep the demo reference: 'Al Nour Industries · ERP (#4383)'."""
    company = f"{COMPANIES[row['ref'] % len(COMPANIES)]} {ACTIVITY[row['industry']]}"
    topic = (row["tags"] or "").split(",")[0] or "Odoo"
    return f"{company} · {topic} (#{row['ref']})" if prefix == "Demo" else f"{company} (closed #{row['ref']})"


def when(iso):
    return datetime.fromisoformat(iso) + shift


if Lead.search_count([("name", "=like", "% (#%)")], limit=1):
    print("Demo already loaded.")
else:
    stages = env["crm.stage"].search([("is_won", "=", False)], order="sequence, id")
    if len(stages) < 4:
        env["crm.stage"].create({"name": "Negotiation", "sequence": max(stages.mapped("sequence")) + 1})
        stages = env["crm.stage"].search([("is_won", "=", False)], order="sequence, id")
    won_stage = env["crm.stage"].search([("is_won", "=", True)], limit=1)
    teams = {}
    sources = {}
    tags = {}
    for lang in ("ar_001", "en_US"):
        env["res.lang"]._activate_lang(lang)
    Lang = env["res.lang"].with_context(active_test=False)
    langs = {
        "ar": Lang.search([("code", "=", "ar_001")], limit=1),
        "en": Lang.search([("code", "=", "en_US")], limit=1),
    }
    call = env.ref("mail.mail_activity_data_call")
    note = env.ref("mail.mt_note")

    def vals(row, prefix):
        team = teams.get(row["team"]) or teams.setdefault(
            row["team"],
            env["crm.team"].search([("name", "=", row["team"])], limit=1)
            or env["crm.team"].create({"name": row["team"]}),
        )
        src_name = row["source"].replace("_", " ").title()
        source = sources.get(src_name) or sources.setdefault(
            src_name,
            env["utm.source"].search([("name", "=", src_name)], limit=1)
            or env["utm.source"].create({"name": src_name}),
        )
        tag_ids = []
        for t in filter(None, (row["tags"] or "").split(",")):
            tag = tags.get(t) or tags.setdefault(
                t, env["crm.tag"].search([("name", "=", t)], limit=1) or env["crm.tag"].create({"name": t})
            )
            tag_ids.append(tag.id)
        code = "tr" if row["country"] == "Other" else row["country"].lower()
        email = {"correct": f"contact{row['ref']}@example.com", "incorrect": "not-an-email"}.get(row["email_state"])
        phone = {"correct": "+966 50 123 4567", "incorrect": "12"}.get(row["phone_state"])
        return {
            "name": lead_name(row, prefix),
            "partner_name": f"{COMPANIES[row['ref'] % len(COMPANIES)]} {ACTIVITY[row['industry']]}",
            "user_id": env.ref("base.user_admin").id,  # a real salesperson, so "My Opportunities" shows them
            "type": "opportunity",
            "team_id": team.id,
            "source_id": source.id,
            "country_id": env.ref(f"base.{code}").id,
            "lang_id": langs[row["lang"]].id,
            "tag_ids": [(6, 0, tag_ids)],
            "expected_revenue": row["expected_revenue"],
            "email_from": email,
            "phone": phone,
            "ml_industry": row["industry"],
            "ml_company_size": row["company_size"],
        }

    # Closed history first (for Odoo's native scoring)
    closed = data["closed"]
    for start in range(0, len(closed), 500):
        batch = closed[start : start + 500]
        recs = Lead.create(
            [
                {
                    **vals(r, "Closed"),
                    "stage_id": won_stage.id if r["outcome"] == "won" else stages[min(r["stage_seq"], 4) - 1].id,
                }
                for r in batch
            ]
        )
        lost = recs.filtered(lambda lead: not lead.stage_id.is_won)
        lost.action_set_lost()
        for rec, r in zip(recs, batch, strict=True):
            env.cr.execute("UPDATE crm_lead SET create_date = %s WHERE id = %s", (when(r["created"]), rec.id))
        env.cr.commit()
        print(f"closed {start + len(batch)}/{len(closed)}")
    # Open leads as they were on the cutoff day. Their dates are written last (see below), after every ORM write.
    date_updates = []
    for r in data["open"]:
        rec = Lead.create({**vals(r, "Demo"), "stage_id": stages[r["stage_seq"] - 1].id})
        date_updates.append((when(r["created"]), when(r["stage_entered"]), rec.id))
        for a in r["activities"]:
            env["mail.message"].create(
                {
                    "model": "crm.lead",
                    "res_id": rec.id,
                    "message_type": "notification",
                    "body": "Call done",
                    "mail_activity_type_id": call.id,
                    "subtype_id": note.id,
                    "date": when(a),
                }
            )
    # Rebuild Odoo's own predictive lead scoring from the closed history
    earliest = min(when(r["created"]) for r in closed)
    env["ir.config_parameter"].sudo().set_str("crm.pls_start_date", (earliest - timedelta(days=1)).date().isoformat())
    env.invalidate_all()
    env.cr.commit()
    # A running Odoo server (e.g. in Docker) may update the same leads from its scheduled jobs at the same time.
    # Odoo then logs "update transaction failed" and leaves the transaction aborted: roll back and try again.
    for attempt in range(5):
        Lead._cron_update_automated_probabilities()
        try:
            env.flush_all()
            env.cr.execute("SELECT 1")
            env.cr.commit()
            break
        except Exception as exc:  # psycopg2 errors: transaction aborted by a concurrent update
            env.cr.rollback()
            env.invalidate_all()
            print(f"scoring rebuild collided with the running server ({type(exc).__name__}); retry {attempt + 1}")
            time.sleep(5)
    # Raw SQL last: later ORM writes (e.g. the scoring rebuild above) would otherwise overwrite these dates.
    for created, entered, lead_id in date_updates:
        env.cr.execute(
            "UPDATE crm_lead SET create_date = %s, date_last_stage_update = %s WHERE id = %s",
            (created, entered, lead_id),
        )
    env.invalidate_all()
    env.cr.commit()
    print(f"Loaded {len(closed)} closed and {len(data['open'])} open leads")
