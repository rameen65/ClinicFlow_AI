# ClinicFlow AI presentation speaker notes

## Slide 1 — ClinicFlow AI

We are presenting a local Python and Streamlit prototype for appointment-risk review. It uses a retained synthetic demo and a separately selected public Brazil dataset from 2016. It is not a clinical system and contains no current clinic patient records.

## Slide 2 — The problem

Missed appointments leave clinic capacity unused, and staff cannot manually check every booking. ClinicFlow AI organizes a list for optional follow-up. The score must never determine who receives care.

## Slide 3 — The workflow

The app validates one selected dataset, trains models on training records, then shows held-out predictions. Staff can explore a cutoff or a top-K queue. Nothing sends reminders or changes appointments.

## Slide 4 — The application

The seven sections are Executive Dashboard; Dataset and Data-Quality Intelligence; AI Model Laboratory; Individual Prediction Explanations; AI Copilot; Decision Simulator; and Responsible AI and Evaluation. After a model is trained in-session, users can upload a matching-schema CSV and score new rows locally. The copilot is deterministic and grounded in saved results because no LLM service credential is configured. No patient-level data leaves the app.

## Slide 5 — The datasets

The demo file has 1,200 generated synthetic appointments and is clearly labelled simulated. The public Medical Appointment No Shows dataset by Joni Hoppen on Kaggle has 110,527 original records and is licensed CC BY-NC-SA 4.0. One invalid row was excluded, leaving 110,526. We never concatenate these sources.

## Slide 6 — Evaluation design

For the public set, 88,420 rows train the models and 22,106 rows are held out; all appointments for a patient stay on one side. IDs and target labels are excluded from features, and preprocessing learns from training rows only. KNN and calibrated SVM use a stratified training sample capped at 25,000 on this dataset, then score the full holdout. The synthetic evaluation uses 960 training and 240 held-out records.

## Slide 7 — Held-out performance

All ten classifiers trained on each source with no fit failures. On the public holdout, HistGradientBoosting has the highest ROC-AUC at 0.740. At a 0.50 cutoff it catches only 0.8% of no-shows. Random Forest catches 83.7% at that same cutoff, but only 30.6% of its flags are actual no-shows. This class imbalance and cutoff trade-off is why accuracy or ROC-AUC alone is not an operational recommendation.

## Slide 8 — Capacity simulation

These are retrospective counts for the public holdout's 22,106 appointments, using HistGradientBoosting because it had the highest ROC-AUC. A top-5% queue flags 1,106 records, catches 502 no-shows (11.2% recall), and 45.4% of flags were no-shows. Top-10% flags 2,211, catches 940 (21.1% recall), and has 42.5% precision. Top-20% flags 4,422, catches 1,708 (38.3% recall), and has 38.6% precision. At a 0.50 threshold it flags 58, catches 36 (0.8% recall), and has 62.1% precision. The public holdout is not a clinic day's schedule, so these counts illustrate trade-offs only.

## Slide 9 — Responsible AI and limitations

This source is historical, geographically specific, and includes sensitive attributes. Results do not establish present-day clinic performance, calibration, fairness, or impact. Group comparisons are descriptive. One-feature perturbations are not causal effects or SHAP. There is no validated operational cutoff; any use would need authorized current data, independent prospective validation, privacy and governance review, and staff/patient involvement.

## Slide 10 — Safe next steps

Never deny, delay, or reduce access to care because of a risk score. Nothing sends messages or changes appointments. Before any pilot, clinics would need permissioned data, privacy controls, subgroup and calibration assessment, workflow review, and prospective validation with staff oversight.
