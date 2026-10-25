{
    "name": "CRM ML Lead Scoring",
    "summary": "Explainable win probability for CRM leads from a trained model, next to Odoo's native scoring",
    "version": "20.0.1.0.0",
    "category": "Sales/CRM",
    "author": "Qusai Natsheh",
    "website": "https://github.com/QusaiNatsheh/crm-lead-scoring",
    "license": "LGPL-3",
    "depends": ["crm"],
    "external_dependencies": {"python": ["requests"]},
    "data": [
        "data/ir_cron.xml",
        "views/crm_lead_views.xml",
        "views/res_config_settings_views.xml",
    ],
}
