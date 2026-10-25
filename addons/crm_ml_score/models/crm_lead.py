import logging
import re
from datetime import timedelta

import requests

from odoo import api, fields, models
from odoo.exceptions import UserError

_logger = logging.getLogger(__name__)

SERVICE_TIMEOUT = 60
BATCH = 500
BANDS = [("hot", "Hot"), ("warm", "Warm"), ("cold", "Cold")]


def _key(name):
    """'Cold Call' -> 'cold_call'; 'Referral [2]' (Odoo's de-duplicated name) -> 'referral'."""
    name = re.sub(r"\s*\[\d+\]$", "", name or "").strip().lower()
    return re.sub(r"[\s-]+", "_", name) or None


class CrmLead(models.Model):
    _inherit = "crm.lead"

    # Lead facts the model uses that standard CRM does not record.
    ml_industry = fields.Selection(
        [
            ("retail", "Retail"),
            ("manufacturing", "Manufacturing"),
            ("services", "Services"),
            ("healthcare", "Healthcare"),
            ("education", "Education"),
            ("construction", "Construction"),
            ("government", "Government"),
        ],
        string="Industry",
    )
    ml_company_size = fields.Selection(
        [("1-10", "1-10"), ("11-50", "11-50"), ("51-200", "51-200"), ("201-1000", "201-1000"), ("1000+", "1000+")],
        string="Company Size",
    )
    ml_probability = fields.Float(
        string="ML Win Probability", digits=(5, 2), readonly=True, copy=False, aggregator="avg"
    )
    ml_band = fields.Selection(BANDS, string="ML Score", readonly=True, copy=False, index=True)
    ml_raises = fields.Text(string="Raises the Score", readonly=True, copy=False)
    ml_lowers = fields.Text(string="Lowers the Score", readonly=True, copy=False)
    ml_scored_on = fields.Datetime(string="Scored On", readonly=True, copy=False)
    ml_model_version = fields.Char(readonly=True, copy=False)

    # ------------------------------------------------------------------ facts sent to the model
    def _ml_stage_seq(self, stages):
        """1-based position of the lead's stage among the open (not won) stages, in sequence order."""
        ids = [s.id for s in stages]
        return ids.index(self.stage_id.id) + 1 if self.stage_id.id in ids else 1

    def _ml_activity_dates(self):
        """Dates of completed activities: Odoo posts a message with the activity type when one is marked done."""
        messages = (
            self.env["mail.message"]
            .sudo()
            .search_read(
                [("model", "=", "crm.lead"), ("res_id", "in", self.ids), ("mail_activity_type_id", "!=", False)],
                ["res_id", "date"],
                order="date",
            )
        )
        dates = {}
        for m in messages:
            dates.setdefault(m["res_id"], []).append(m["date"])
        return dates

    def _ml_payload(self):
        now = fields.Datetime.now()
        stages = self.env["crm.stage"].search([("is_won", "=", False)], order="sequence, id")
        activity_dates = self._ml_activity_dates()
        leads = []
        for lead in self:
            dates = activity_dates.get(lead.id, [])
            leads.append(
                {
                    "id": lead.id,
                    "team": lead.team_id.name or None,
                    "source": _key(lead.source_id.name),
                    "country": lead.country_id.code or None,
                    "industry": lead.ml_industry or None,
                    "company_size": lead.ml_company_size or None,
                    "email_state": lead.email_state or ("none" if not lead.email_from else None),
                    "phone_state": lead.phone_state or ("none" if not lead.phone else None),
                    "lang": (lead.lang_id.code or "")[:2] or None,
                    "tags": ",".join(lead.tag_ids.mapped("name")),
                    "expected_revenue": max(lead.expected_revenue, 0.0),
                    "stage_seq": lead._ml_stage_seq(stages),
                    "days_since_created": max((now - lead.create_date).days, 0),
                    "days_in_stage": max((now - (lead.date_last_stage_update or lead.create_date)).days, 0),
                    "activity_count": len(dates),
                    "activities_last_14d": sum(d > now - timedelta(days=14) for d in dates),
                    "days_since_last_activity": max((now - dates[-1]).days, 0) if dates else None,
                    "first_response_days": max((dates[0] - lead.create_date).days, 0) if dates else None,
                }
            )
        return leads

    # ------------------------------------------------------------------ scoring
    @api.model
    def _ml_service_url(self):
        url = self.env["ir.config_parameter"].sudo().get_str("crm_ml_score.service_url")
        return (url or "http://ml:8000").rstrip("/")

    def _ml_score(self):
        url = self._ml_service_url()
        scored = 0
        for start in range(0, len(self), BATCH):
            batch = self[start : start + BATCH]
            try:
                response = requests.post(f"{url}/score", json={"leads": batch._ml_payload()}, timeout=SERVICE_TIMEOUT)
                response.raise_for_status()
                body = response.json()
            except requests.RequestException as exc:
                _logger.warning("Lead scoring service error: %s", exc)
                raise UserError(self.env._("The lead scoring service at %(url)s is not available.", url=url)) from exc
            by_id = {s["id"]: s for s in body["scores"]}
            now = fields.Datetime.now()
            for lead in batch:
                s = by_id.get(lead.id)
                if not s:
                    continue
                p = s["probability"] * 100
                lead.write(
                    {
                        "ml_probability": p,
                        "ml_band": "hot" if p >= 60 else "warm" if p >= 30 else "cold",
                        "ml_raises": "\n".join(s["raises"]),
                        "ml_lowers": "\n".join(s["lowers"]),
                        "ml_scored_on": now,
                        "ml_model_version": body.get("model_version"),
                    }
                )
                scored += 1
        return scored

    @api.model
    def _ml_open_leads(self):
        return self.search([("type", "=", "opportunity"), ("active", "=", True), ("stage_id.is_won", "=", False)])

    def action_ml_score(self):
        self._ml_score()
        return True

    @api.model
    def action_ml_score_all(self):
        count = self._ml_open_leads()._ml_score()
        return {
            "type": "ir.actions.client",
            "tag": "display_notification",
            "params": {
                "title": self.env._("Lead scoring finished"),
                "message": self.env._("%(count)s open opportunities scored.", count=count),
                "type": "success",
                "next": {"type": "ir.actions.act_window_close"},
            },
        }

    @api.model
    def _cron_ml_score(self):
        try:
            self._ml_open_leads()._ml_score()
        except UserError as exc:
            _logger.warning("Scheduled lead scoring skipped: %s", exc)
