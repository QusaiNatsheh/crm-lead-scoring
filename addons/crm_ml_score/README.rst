===================
CRM ML Lead Scoring
===================

Adds an explainable, calibrated win probability to CRM opportunities, computed by an external ML service, and
shows it next to Odoo's own predictive lead scoring.

Usage
=====

#. Set the service address in *CRM → Configuration → Settings → ML Lead Scoring*.
#. Optionally fill *Industry* and *Company Size* on opportunities (tab *ML Score*).
#. Scores refresh daily (scheduled action), or with *Score Now* / *Score All Open*.
#. Open *CRM → Reporting → Lead Scores* to compare the ML score and Odoo's probability, by band and stage.

What is sent
============

For each open opportunity: team, source, country, industry, company size, email/phone quality, language, tags,
expected revenue, stage position, days since creation and in stage, and the dates of completed activities
(messages carrying an activity type). Only facts known today are sent.

Credits
=======

Author: Qusai Natsheh. License: LGPL-3.
