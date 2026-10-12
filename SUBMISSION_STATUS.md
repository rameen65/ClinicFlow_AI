# ClinicFlow AI — verified Python submission status

Date: 12 October 2026

Team: Rameen, Noor Amna, Abdullah Ali Khan, and Rana Ali Zain

## What works

- Local Streamlit prototype with all seven required navigation sections.
- Separate simulated and public dataset selection; source audit and CSV quality review.
- Ten classifiers trained and scored on held-out records for each included dataset.
- Local scoring of new CSV rows after in-session training when the upload matches the selected dataset's predictor schema. IDs and any target column are ignored. Risk CSV output is only downloaded when the user requests it.
- Held-out example explanations using one-feature perturbations; threshold and top-K capacity simulation; deterministic metrics-grounded copilot.
- No LLM credential was configured. No patient data is sent to a service, no reminder is sent, and no appointment is changed.

## Data and verified evaluations

- **Simulated:** 1,200 synthetic rows, kept and labelled separately; 960 train / 240 holdout. All ten models trained without failures. Highest ROC-AUC was calibrated SVM at 0.699 (accuracy 0.642, precision 0.631, recall 0.398, F1 0.488 at threshold 0.50).
- **Public Brazil 2016:** Kaggle Medical Appointment No Shows by Joni Hoppen, 110,527 original rows. One invalid row was excluded (110,526 modelled). License shown by Kaggle: CC BY-NC-SA 4.0; noncommercial/share-alike terms and attribution are documented in `DATA_SOURCES.md`. The sources were never combined.
- Public split: 88,420 train / 22,106 test rows; 49,844 / 12,454 patient groups, with no group overlap. All ten models trained without failures. Highest ROC-AUC was HistGradientBoosting at 0.740 (accuracy 0.799, precision 0.621, recall 0.008, F1 0.016 at threshold 0.50). Random Forest recalled 0.837 but precision was 0.306 at the same cutoff. These differences show the operational threshold trade-off; no cutoff is recommended.
- KNN and calibrated SVM used stratified 25,000-row training subsets for the large public source and were evaluated on the entire holdout. Per-model counts and all other genuine scores are in the `results/` CSV and JSON reports.
- The best-AUC public model's holdout simulation at top-5%, top-10%, and top-20% capacities is recorded in the public evaluation JSON.

## Verification performed

- `python -m unittest discover -v`: **10 passed, 0 failed**. Includes a Streamlit render smoke test, source and leakage checks, all-ten-model sample training, new unlabeled CSV preparation and scoring, capacity selection, and explanations. See `results/TEST_RESULTS.md`.
- `python -m compileall -q .`: passed.
- Full-data CLI evaluation: 10/10 models trained on each dataset, zero estimator failures.
- Presentation package check: 10 slides and 10 embedded speaker-note pages; slide and notes XML parsed successfully and the PPTX archive passed integrity checking.

## Remaining limitations

This is a competition prototype. The public set is historical and geographically specific; neither source validates contemporary clinic performance. Results use one group-disjoint split, without a separate temporal or external validation cohort. The 0.50 threshold produces weak no-show recall for the highest-AUC public model; capacity simulations are retrospective, not staffing recommendations. Fairness summaries are descriptive, feature perturbations are not causal explanations, and probabilities have not been validated for clinical or operational calibration. Never deny, delay, or reduce care based on a score.

## Package paths

- App and source: `clinicflow_ai_python/`
- Presentation: `clinicflow_ai_python/ClinicFlow_AI_Presentation.pptx`
- Final ZIP: `outputs/ClinicFlow_AI_International_AI_Championship_Final.zip`
- The original Java submission ZIP is preserved byte-for-byte at `original/ClinicFlow_AI_Final_Submission.zip` inside the final bundle.
