# LinkedIn posts: CRM Lead Scoring (drafts)

Drafts for review before posting. Replace `<repo link>` with the GitHub URL once the repository is public.

---

## Post 1: the leakage mistake that makes lead scoring look perfect

**Attach:** an image of the "leakage trap" section of the README, or a simple two-row diagram (wrong vs right)

My lead-scoring model scored a perfect 1.000. That's how I knew it was wrong.

The most common way to build lead scoring:
1. Export your closed deals.
2. Train a model on what each deal looked like: its stage, number of activities, days until close.
3. Split the rows at random, test, celebrate.

The problem: those are the values **at the end**. A deal's final stage tells you whether it was won. Its total
number of activities includes all the follow-ups after the customer said yes. None of that exists while a lead is
still open, which is exactly when you need the score.

Done that way on my data, the model reached ROC-AUC 1.000. Useless.

What works instead:
→ **Snapshots:** train on each lead as it looked on the day it was scored (stage, activities and days so far),
  never after it closed.
→ **Split by date and by lead**, so the model is tested on later leads it has never seen.

The honest model scores 0.767. Lower, but real.

If a lead-scoring demo looks perfect, ask one question: "what did the model know, and when?"

<repo link>

#MachineLearning #DataScience #CRM #Sales #Odoo

---

## Post 2: I tried to beat Odoo's built-in lead scoring

**Attach:** `docs/calibration.png` and `docs/lift.png`

Odoo already scores leads. I tried to beat it with gradient boosting. Here's the honest result.

First, I re-implemented Odoo 20's native scoring (a naive Bayes model on fields like source, country, stage and
tags) and checked it against a real Odoo database. It matched exactly, on all 123 test leads.

Then I compared it with a LightGBM model on 2,554 scoring points from 853 test leads:

• Win rate of the current stage: ROC-AUC 0.684
• Odoo's native scoring: 0.754
• LightGBM: 0.767

That gap is within statistical noise: the 95% interval runs from −0.010 to +0.035. So, overall, not better.

Where the model did help:
• Brand-new leads (day 0): 0.729 vs 0.709. That's where prioritising matters most.
• Calibration: when it says 70%, about 70% win.
• Every score comes with reasons: "Source: referral", "16 activities so far", "No phone".

Lesson: Odoo's built-in scoring is a strong baseline. Measure against it before claiming anything. And note the
data was synthetic, built in a way that suits naive Bayes; real CRM data may differ.

<repo link>

#Odoo #MachineLearning #CRM #Sales #DataScience

---

## Post 3: what it looks like in Odoo CRM

**Attach:** `docs/screenshots/demo.gif` (or a short screen recording)

Lead scoring inside Odoo 20 CRM, next to Odoo's own number:

1. **Pipeline:** every opportunity shows a Hot / Warm / Cold badge and its win probability.
2. **Opportunity:** an "ML Score" tab with the probability next to Odoo's own, and what raises or lowers it.
   For example: "Stage: Negotiation", "16 activities so far", "Industry: government".
3. **Lead Scores:** one list sorted by ML score, with Odoo's percentage beside it, plus pipeline value by band.

Scores refresh every day, and only facts known today are sent to the model.

Built with Odoo 20 Community, FastAPI, LightGBM and MLflow. One `make demo` runs it.

<repo link>

#Odoo #CRM #Sales #Python
