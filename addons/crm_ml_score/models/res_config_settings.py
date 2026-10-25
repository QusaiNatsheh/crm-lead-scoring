from odoo import fields, models


class ResConfigSettings(models.TransientModel):
    _inherit = "res.config.settings"

    crm_ml_score_service_url = fields.Char(
        string="Lead Scoring Service URL", config_parameter="crm_ml_score.service_url", default="http://ml:8000"
    )

    def action_crm_ml_score_now(self):
        self.execute()
        return self.env["crm.lead"].action_ml_score_all()
