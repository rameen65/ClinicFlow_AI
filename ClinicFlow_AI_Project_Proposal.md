# ClinicFlow AI - Appointment Risk Review for Clinic Operations

**Project category:** AI in Healthcare - Clinic Operations Decision Support  
**Competition:** 4th International AI Championship  
**Team:** Rameen, Noor Amna, Abdullah Ali Khan, and Rana Ali Zain  
**Submission date:** 12 October 2026  
**Keywords:** appointment no-show prediction; explainable AI; clinic operations; capacity planning; responsible AI

## Project Overview & Problem Statement

Missed appointments leave clinic time unused and can make it harder for other patients to access available slots. Clinic staff have limited time to review every booking manually. ClinicFlow AI is a local prototype that organizes appointment data, estimates retrospective no-show risk, compares models on held-out records, and helps staff explore how a review queue changes with available capacity.

The project addresses a scheduling and follow-up planning problem. It does not diagnose patients, determine access to care, or claim that a risk score will prevent missed appointments.

## Proposed Solution

ClinicFlow AI is a Python and Streamlit application with seven sections: Executive Dashboard; Dataset and Data-Quality Intelligence; AI Model Laboratory; Individual Prediction Explanations; AI Copilot; Decision Simulator; and Responsible AI and Evaluation.

A user selects one dataset, imports or reviews appointment data, checks data quality, trains and compares classifiers, and explores individual risk explanations or threshold and top-K review queues. Processing is local. The deterministic copilot summarizes actual selected-dataset metrics because no LLM credentials are configured. The app does not transmit patient-level data, send reminders, or change appointments.

## Objectives & Expected Outcomes

- Provide a runnable local demonstration for appointment data review and validation.
- Train and evaluate ten classifiers using a held-out test set, with preprocessing learned on training data only.
- Prevent target and identifier leakage; keep each public-dataset patient on only one side of the train/test split.
- Show accuracy, precision, recall, F1, and ROC-AUC when calculable, plus local feature perturbation explanations.
- Let staff explore retrospective flag counts under configurable probability thresholds or a top-K capacity limit.
- Deliver source code, Windows launch instructions, tests, evaluation reports, proposal, presentation, speaker notes, and separately labelled datasets.

Expected outcomes are a functional competition prototype and transparent comparisons on the included data. No reduction in real clinic no-shows, improved patient outcomes, or real-world model performance is claimed.

## Target Users

Primary users are clinic operations staff, appointment coordinators, and front-desk teams who need to review a large schedule with limited follow-up capacity. Clinic managers and data analysts may use the dashboard to inspect data quality and compare model trade-offs. Judges and developers can use the documented methodology and tests to review the prototype. Any clinical or operational use would require local authorization and human oversight.

## Key Features & Deliverables

- CSV import, schema checks, missing-value and validity summaries, and source-specific data handling.
- Ten classifiers: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost, calibrated linear Support Vector Machine, K-Nearest Neighbors, and Gaussian Naive Bayes.
- Held-out evaluation and saved CSV/JSON audit results; failed models are reported rather than assigned invented scores.
- Individual risk scoring and local one-feature perturbation explanations. These explanations are sensitivity checks, not causal explanations or SHAP values.
- Threshold and staff-capacity top-K simulation using retrospective held-out predictions.
- Deterministic, metrics-grounded copilot; it is not presented as an LLM.
- Responsible-use guidance, Windows run/test scripts, automated tests, README, methodology, data-source notes, and a PowerPoint with speaker notes.

### Data and Verified Evaluation

The retained demo file contains 1,200 synthetic appointments and is labelled simulated. It is kept separate from the public dataset. The public **Medical Appointment No Shows** dataset by Joni Hoppen contains 110,527 Brazil appointment records from 2016; one invalid row was excluded, leaving 110,526 modelled rows. Its Kaggle page lists CC BY-NC-SA 4.0; attribution and reuse terms are documented in `DATA_SOURCES.md`.

All ten classifiers trained without failures on each included source. A seeded, stratified group holdout was used; the public patient groups do not overlap between train and test. Identifiers and the target are excluded from predictors, and preprocessing is fit on training records only. KNN and calibrated SVM use a stratified training sample capped at 25,000 rows on the larger source and are evaluated on the full holdout.

| Dataset / highest ROC-AUC model | Train / test | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Simulated / calibrated SVM | 960 / 240 | 0.642 | 0.631 | 0.398 | 0.488 | 0.699 |
| Brazil 2016 / HistGradientBoosting | 88,420 / 22,106 | 0.799 | 0.621 | 0.008 | 0.016 | 0.740 |

At the 0.50 threshold on the public holdout, Random Forest recall was 0.837 and precision was 0.306, while the highest-AUC HistGradientBoosting model recalled 0.008. This illustrates the effect of threshold choice; it does not establish a recommended cutoff. Full model metrics and split details are in `results/`.

## Implementation Plan & Timeline

### Competition prototype milestones - completed for the 12 October 2026 submission

1. Keep the provided simulated dataset labelled separately and document a public historical source.
2. Add data validation, leakage safeguards, shared preprocessing, and held-out evaluation.
3. Build the seven-section local Streamlit interface, model comparison, explanations, deterministic insights, and capacity simulator.
4. Run automated tests and model evaluations; prepare Windows instructions, methodology, proposal, and presentation package.

### Proposed future validation - not yet undertaken

- **Weeks 1-2 after clinic approval:** define intended workflow, obtain governance and privacy approval, and agree on authorized data access with staff and patient representatives.
- **Weeks 3-6:** conduct temporal and external validation on authorized, current clinic data; assess calibration, subgroup performance, data quality, and operational thresholds.
- **Weeks 7-12:** if results and approvals support it, run a staff-supervised shadow-mode evaluation without automated reminders or appointment changes; review workload and unintended effects before any further decision.

These are planning estimates, not completed work or a commitment to deploy.

## Resources or Requirements

- Windows computer with Python 3.13; the included `run_windows.bat` creates a local virtual environment and launches the app.
- Pinned Python packages: Streamlit 1.65.0, pandas 3.0.6, NumPy 2.5.4, and scikit-learn 1.9.1, listed in `requirements.txt`.
- The included synthetic and public CSV files, evaluation reports, and local disk space. The large public dataset is about 10.7 MB uncompressed and is bundled in the compressed project ZIP.
- No cloud service, LLM API key, or external data transfer is required for the prototype.
- Any future clinic evaluation requires authorized current data, privacy/security review, domain expertise, and clinic staff oversight.

## Potential Challenges or Risks

- **Limited generalizability:** the public records are historical and geographically specific; synthetic data is not evidence about real patients. Mitigation: keep results source-specific and require prospective, external validation.
- **Class imbalance and cutoff trade-offs:** a strong ROC-AUC does not guarantee useful recall at a chosen threshold. Mitigation: show multiple metrics and let staff inspect capacity scenarios without recommending a cutoff.
- **Privacy and sensitive features:** public data contains pseudonymous IDs and sensitive attributes. Mitigation: process locally, exclude identifiers from predictors, use patient IDs only for group splitting, and do not send records externally.
- **Timing and leakage:** reminder status may not be known at every review time. The public feature `SMS_received` is used only under the documented assumption that reminder status is available when risk is reviewed; its association is not causal.
- **Explanation limits:** one-feature perturbations do not establish causes or what will happen if staff change a field. Mitigation: label them as local sensitivity checks and preserve human review.
- **Operational harm:** a score could be misused to deny or delay care. Mitigation: the prototype cannot change bookings or send reminders and states that scores must never restrict care.

## Additional Details (Optional)

The copilot is deterministic and grounded in the selected dataset's actual saved evaluation results. There is no configured LLM service. The app does not automatically contact patients, alter appointments, or make staffing recommendations. Before any pilot, the clinic would need approved data access, privacy and governance controls, calibration and subgroup checks, prospective evaluation, and meaningful staff and patient involvement.
