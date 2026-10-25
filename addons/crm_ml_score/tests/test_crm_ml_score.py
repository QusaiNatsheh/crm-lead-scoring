from datetime import timedelta
from unittest.mock import MagicMock, patch

import requests

from odoo import fields
from odoo.exceptions import UserError
from odoo.tests import TransactionCase, tagged

POST = "odoo.addons.crm_ml_score.models.crm_lead.requests.post"


def _response(payload):
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


@tagged("post_install", "-at_install")
class TestCrmMlScore(TransactionCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.stages = cls.env["crm.stage"].search([("is_won", "=", False)], order="sequence, id")
        cls.source = cls.env["utm.source"].create({"name": "Referral"})
        cls.lead = cls.env["crm.lead"].create(
            {
                "name": "Big deal",
                "type": "opportunity",
                "stage_id": cls.stages[1].id,
                "source_id": cls.source.id,
                "country_id": cls.env.ref("base.sa").id,
                "expected_revenue": 12000,
                "ml_industry": "manufacturing",
                "ml_company_size": "51-200",
                "email_from": "buyer@example.com",
            }
        )
        cls.won_stage = cls.env["crm.stage"].search([("is_won", "=", True)], limit=1)

    def test_payload(self):
        activity_type = self.env.ref("mail.mail_activity_data_call")
        self.lead.message_post(body="Called", mail_activity_type_id=activity_type.id)
        state = self.lead._ml_payload()[0]
        self.assertEqual(state["source"], "referral")
        self.assertEqual(state["country"], "SA")
        self.assertEqual(state["stage_seq"], 2)
        self.assertEqual(state["industry"], "manufacturing")
        self.assertEqual(state["activity_count"], 1)
        self.assertEqual(state["activities_last_14d"], 1)
        self.assertEqual(state["phone_state"], "none")
        self.assertEqual(state["days_since_created"], 0)

    def test_payload_ignores_messages_that_are_not_activities(self):
        self.lead.message_post(body="Just a note")
        state = self.lead._ml_payload()[0]
        self.assertEqual(state["activity_count"], 0)
        self.assertIsNone(state["days_since_last_activity"])

    def test_scores_written_with_band_and_reasons(self):
        payload = {
            "model_version": "test",
            "scores": [
                {"id": self.lead.id, "probability": 0.72, "raises": ["Source: referral"], "lowers": ["No phone"]}
            ],
        }
        with patch(POST, return_value=_response(payload)) as post:
            self.lead.action_ml_score()
        self.assertEqual(post.call_args.kwargs["timeout"], 60)
        self.assertAlmostEqual(self.lead.ml_probability, 72.0)
        self.assertEqual(self.lead.ml_band, "hot")
        self.assertEqual(self.lead.ml_raises, "Source: referral")
        self.assertTrue(self.lead.ml_scored_on)

    def test_bands(self):
        for p, band in ((0.65, "hot"), (0.45, "warm"), (0.05, "cold")):
            payload = {
                "model_version": "t",
                "scores": [{"id": self.lead.id, "probability": p, "raises": [], "lowers": []}],
            }
            with patch(POST, return_value=_response(payload)):
                self.lead._ml_score()
            self.assertEqual(self.lead.ml_band, band)

    def test_score_all_skips_won_leads(self):
        won = self.lead.copy({"name": "Done", "stage_id": self.won_stage.id})
        with patch(POST, return_value=_response({"model_version": "t", "scores": []})) as post:
            self.env["crm.lead"].action_ml_score_all()
        sent = {lead["id"] for lead in post.call_args.kwargs["json"]["leads"]}
        self.assertIn(self.lead.id, sent)
        self.assertNotIn(won.id, sent)

    def test_service_down(self):
        with patch(POST, side_effect=requests.ConnectionError("down")):
            with self.assertRaises(UserError):
                self.lead.action_ml_score()
            self.env["crm.lead"]._cron_ml_score()  # logs and continues

    def test_days_counted_from_dates(self):
        self.env.cr.execute(
            "UPDATE crm_lead SET create_date = %s, date_last_stage_update = %s WHERE id = %s",
            (fields.Datetime.now() - timedelta(days=20), fields.Datetime.now() - timedelta(days=5), self.lead.id),
        )
        self.lead.invalidate_recordset()
        state = self.lead._ml_payload()[0]
        self.assertEqual(state["days_since_created"], 20)
        self.assertEqual(state["days_in_stage"], 5)

    def test_salesperson_can_score_their_leads(self):
        """Runs as a real salesperson (not the superuser), so record rules apply."""
        from odoo.tests import new_test_user

        seller = new_test_user(self.env, login="seller", groups="sales_team.group_sale_salesman")
        mine = self.lead.copy({"name": "Mine", "user_id": seller.id})
        payload = {"model_version": "t", "scores": [{"id": mine.id, "probability": 0.4, "raises": [], "lowers": []}]}
        with patch(POST, return_value=_response(payload)):
            mine.with_user(seller).action_ml_score()
        self.assertEqual(mine.ml_band, "warm")
