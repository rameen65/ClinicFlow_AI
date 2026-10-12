# ClinicFlow AI — Python prototype

ClinicFlow AI is a local Streamlit application for reviewing appointment no-show risk, comparing classifiers, inspecting model behavior, and simulating staff review capacity. It is an operations prototype, not a clinical decision system.


## GitHub copy and public dataset

This repository copy intentionally omits the row-level public appointment CSV. It contains appointment-level health and demographic records, and the project instructions prohibit sending patient data to external services. The app runs with the included simulated sample. To reproduce public-data results locally, obtain the Kaggle dataset under its CC BY-NC-SA 4.0 terms and place it at `data/public/KaggleV2-May-2016.csv`. The CSV is not committed here.

## Run on Windows

1. Install Python 3.13 from [python.org](https://www.python.org/downloads/windows/). Keep the Python launcher (`py`) selected.
2. Extract this folder from the ZIP.
3. Double-click `run_windows.bat`. It creates a local virtual environment, installs the packages in `requirements.txt` once, and opens the app in your browser.
4. Select a dataset and choose **Train and evaluate 10 models** in the sidebar. Training the public dataset can take several minutes on a laptop.

For a manual setup, open Command Prompt in this folder and run:

```bat
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Run automated checks with `run_tests_windows.bat` after the first setup. From a terminal, the equivalent command is `.venv\Scripts\python.exe -m unittest discover -v`.

## Seven sections

1. **Executive Dashboard** — data summary and held-out classifier results.
2. **Dataset and Data-Quality Intelligence** — source details, exclusions, validation checks, and an in-memory CSV quality review.
3. **AI Model Laboratory** — performance comparison for up to ten trained classifiers.
4. **Individual Prediction Explanations** — local one-feature perturbation explanations for held-out examples, plus local scoring for new CSV rows with the selected model.
5. **AI Copilot** — deterministic findings calculated from application metrics. No LLM credentials are configured and no patient-level data is sent to an external service.
6. **Decision Simulator** — probability threshold or top-K review capacity, with retrospective precision/recall.
7. **Responsible AI and Evaluation** — group summaries, methodology, and limits.

## Data and evaluation

The simulated dataset and the public Brazil 2016 dataset are selected separately; the application never joins them. See [DATA_SOURCES.md](DATA_SOURCES.md) for attribution and reuse terms. Evaluation keeps all records for a patient on one side of a stratified group holdout; IDs and the target are not model features. Preprocessing is fit on training records only. K-Nearest Neighbors and calibrated SVM are trained on up to 25,000 stratified training rows for practical runtime and evaluated on the full held-out set; actual per-model training counts are reported.

The model laboratory reports accuracy, precision, recall, F1, ROC-AUC, and fit time for each model that successfully trains. Failed estimators are identified and receive no invented scores. The dashboard can read saved metrics in `results/`; use the sidebar button to train interactive models for explanations and simulation.

To score new appointments, first train models in the current session, then open **Individual Prediction Explanations** and upload a CSV matching the selected source. For the simulated source, use the eight predictor columns from `appointments_demo.csv`; for the public source, use the Kaggle predictor columns (target and ID columns are optional and ignored). Invalid rows are excluded and reported. The app does not save or send uploaded records; the score CSV is returned only when you select its download button.

## Safety and limitations

- The public dataset is historical and from Brazil in 2016; it does not establish present-day clinic performance.
- Risk scores must never be used to deny or delay care.
- Group comparisons are exploratory, not a fairness certification.
- Feature perturbations are not causal explanations or SHAP values.
- The simulator does not send reminders or alter appointments.
- CSV uploads are processed in memory. New appointment rows can be scored locally after training; uploaded records are not transmitted or saved by the application. A score CSV is created only if the user downloads it.
- The public source includes pseudonymous IDs and sensitive attributes. The app suppresses IDs and uses `PatientId` only as a split group.

## Files

- `app.py` — seven-section Streamlit user interface.
- `clinicflow/data.py` — separate source loading, feature engineering and quality checks.
- `clinicflow/modeling.py` — shared preprocessing, ten classifiers, group-aware evaluation, explanation and simulator functions.
- `train_models.py` — command-line evaluation and report generation.
- `tests/` — unittest checks.
- `results/` — actual model reports from this build.
- `ClinicFlow_AI_Project_Proposal.md`, `METHODOLOGY.md`, `DATA_SOURCES.md` — submission documentation.
- `ClinicFlow_AI_Presentation.pptx` and `Presentation_Speaker_Notes.md` — presentation and speaker notes.
- `results/TEST_RESULTS.md` — verified test command and result.
