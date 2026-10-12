# Model and evaluation methodology

## Prediction target and timing

The positive class is `no_show = 1`. The public source's `No-show=Yes` maps to 1. The model is a demonstration of retrospective appointment-risk ranking after reminder status is available. `SMS_received` is included in the public-data feature set only under that timing assumption. Its association is not a causal effect.

## Features and leakage safeguards

- `PatientId` is used only as a split group. `AppointmentID` is only an internal reference. Neither enters model inputs.
- `No-show`/`no_show` is the label and never a predictor.
- Raw `ScheduledDay` and `AppointmentDay` are dropped after deriving lead time, appointment weekday, and scheduled hour.
- Scikit-learn imputers, scaling and one-hot encoding are fitted within each model pipeline using training records only.
- For public records, a stratified group fold holds out entire patients, preventing the same patient from appearing in both train and test.
- The simulated sample has no patient identifier, so each synthetic appointment ID is a split group.
- A fixed 50% probability threshold is used for the leaderboard. The simulator lets staff explore other thresholds or a top-K review cap on the same held-out predictions.

## Classifiers

The laboratory trains Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost, a calibrated linear Support Vector Machine, K-Nearest Neighbors, and Gaussian Naive Bayes. Models share the same imputation/scaling/one-hot preprocessing structure where supported. K-Nearest Neighbors and calibrated SVM are each fit on a stratified sample of at most 25,000 training rows when the training set is larger; all models are evaluated on the complete holdout. The report records each model's actual training-row count.

## Metrics

For each successfully trained model the report calculates accuracy, precision, recall, F1, and ROC-AUC from held-out labels. Probability threshold metrics use 0.50 by default; ROC-AUC uses continuous predicted probabilities. Failed model fits are reported and excluded from the leaderboard rather than assigned invented scores.

## Explanations and decisions

Individual explanations perturb one input at a time to its training-set median or most frequent value and compare model scores. This is a local sensitivity check, not SHAP, a causal explanation, or a guarantee that changing the field changes attendance. The capacity simulator reports retrospective counts, precision and recall for threshold and top-K selection. CLI evaluation JSON includes observed top-5%, top-10%, and top-20% queues for the highest ROC-AUC model, plus 0.25/0.35/0.50 probability thresholds. These are holdout illustrations, not recommended daily staffing limits. It does not send messages, change appointments, or book additional slots.

## Responsible use

The public records are historical and contain pseudonymous IDs and sensitive fields. The interface suppresses source IDs, does not transmit records to a service, and labels the copilot deterministic because no LLM credentials are configured. Group comparisons are descriptive checks only. Neither dataset establishes current clinical performance, fairness, calibration, or operational impact. Never deny or delay care based on a score.
