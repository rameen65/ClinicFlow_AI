# Verified test results

Date: 12 October 2026

Command: `python -m unittest discover -v`

Result: **10 tests passed, 0 failed** (19.683 seconds in the final recorded run).

The suite checked that the Streamlit app renders with the seven required navigation sections, trains through the interface, and opens the prediction/scoring page; both source loaders validate their inputs and keep target/identifier fields out of model features; both source formats can be prepared for unlabeled scoring; a fitted model produces bounded risk probabilities for new CSV rows; all ten classifier families train on a sample with finite held-out metrics; train and test groups do not overlap; top-K simulation respects a chosen capacity; and local perturbation explanations return finite changes.

Separate full-data CLI evaluations also completed with **10/10 models trained and no model failures** on both datasets. Their actual metrics, row/group splits, data SHA-256 values, capacity scenarios, and audit counts are saved in `simulated_evaluation.json` and `public_brazil_2016_evaluation.json`. The app-render test may emit Streamlit's harmless bare-mode `ScriptRunContext` warning when run under `unittest`.
