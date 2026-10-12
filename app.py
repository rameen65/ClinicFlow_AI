"""ClinicFlow AI — seven-section Streamlit decision-support prototype."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from clinicflow.data import PROJECT_ROOT, load_dataset, prepare_prediction_features
from clinicflow.insights import copilot_insights, threshold_curve
from clinicflow.modeling import Evaluation, evaluate_all, local_feature_impacts, simulate_capacity

st.set_page_config(page_title="ClinicFlow AI", page_icon="🏥", layout="wide")

NAVIGATION = [
    "Executive Dashboard",
    "Dataset and Data-Quality Intelligence",
    "AI Model Laboratory",
    "Individual Prediction Explanations",
    "AI Copilot",
    "Decision Simulator",
    "Responsible AI and Evaluation",
]
DATASETS = {
    "Simulated clinic demo": "simulated",
    "Brazil public appointments (2016)": "public_brazil_2016",
}
REPORTS_DIR = PROJECT_ROOT / "results"

st.markdown(
    """
    <style>
      :root { --ink:#13283d; --muted:#52677c; --teal:#087e82; --pale:#eef6f5; }
      .stApp { background: #f7f9fb; color: var(--ink); }
      [data-testid="stSidebar"] { background: #edf3f6; }
      [data-testid="stMetric"] { background:#fff; border:1px solid #dce5eb;
        padding:14px 16px; border-radius:10px; }
      .cf-kicker { color:#087e82; font-size:.78rem; font-weight:700;
        letter-spacing:.08em; text-transform:uppercase; margin-bottom:.2rem; }
      .cf-note { color:#52677c; font-size:.9rem; }
      .cf-warning { border-left:4px solid #c17b22; background:#fff8ea;
        padding:.8rem 1rem; border-radius:4px; }
      h1, h2, h3 { color:#13283d; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _report_path(key: str) -> Path:
    return REPORTS_DIR / f"{key}_metrics.csv"


@st.cache_data(show_spinner=False)
def _load_source(key: str):
    return load_dataset(key)


def _saved_metrics(key: str) -> pd.DataFrame:
    path = _report_path(key)
    if path.is_file():
        try:
            return pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError):
            return pd.DataFrame()
    return pd.DataFrame()


def _evaluation_for(key: str) -> Evaluation | None:
    if st.session_state.get("active_eval_key") == key:
        return st.session_state.get("active_eval")
    return None


def _train(key: str) -> None:
    data = _load_source(key)
    progress = st.progress(0, text="Preparing shared preprocessing and held-out split…")
    status = st.empty()

    def update(step: int, total: int, model: str, seconds: float) -> None:
        progress.progress(step / total, text=f"Model {step}/{total}: {model}")
        status.caption(f"{model} fitted in {seconds:.1f}s")

    with st.spinner("Training and evaluating classifiers on the selected dataset…"):
        result = evaluate_all(data.X, data.y, data.groups, progress=update)
    st.session_state["active_eval"] = result
    st.session_state["active_eval_key"] = key
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(_report_path(key), index=False)
    progress.progress(1.0, text="Ten models trained and evaluated.")
    status.empty()
    st.success(
        f"Completed: {len(result.models)} classifiers trained successfully; "
        f"{result.metadata['train_rows']:,} training and {result.metadata['test_rows']:,} held-out rows."
    )
    if result.failures:
        st.warning("Some classifier fits failed and have no metric: " + "; ".join(result.failures))


def _display_metrics(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        st.info("Train the model set to show actual held-out metrics.")
        return
    view = metrics.copy()
    for column in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if column in view:
            view[column] = view[column].astype(float)
    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        column_config={
            "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
            "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
            "f1": st.column_config.NumberColumn("F1", format="%.3f"),
            "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
            "training_rows": st.column_config.NumberColumn("Train rows", format="%d"),
            "fit_seconds": st.column_config.NumberColumn("Fit seconds", format="%.1f"),
        },
    )


def _header(section: str, description: str) -> None:
    st.markdown('<div class="cf-kicker">ClinicFlow AI · decision support</div>', unsafe_allow_html=True)
    st.title(section)
    st.caption(description)


def _safe_source_label(data) -> None:
    st.markdown(f"**Selected source:** {data.label}")
    st.caption(data.source)


with st.sidebar:
    st.markdown("# ClinicFlow AI")
    st.caption("Appointment-risk review for clinic operations")
    dataset_label = st.selectbox("Dataset", list(DATASETS.keys()), index=0)
    dataset_key = DATASETS[dataset_label]
    section = st.radio("Workspace", NAVIGATION, label_visibility="collapsed")
    st.divider()
    st.caption("Local processing · no patient records are sent to an external service")
    if st.button("Train and evaluate 10 models", type="primary", width="stretch"):
        try:
            _train(dataset_key)
        except Exception as exc:  # surfaced with an actionable message
            st.error(f"Model run stopped: {exc}")
            st.exception(exc)

data = _load_source(dataset_key)
evaluation = _evaluation_for(dataset_key)
saved = _saved_metrics(dataset_key)
metrics = evaluation.metrics if evaluation else saved
default_model = (
    str(metrics.iloc[0]["model"]) if not metrics.empty and "model" in metrics else "Logistic Regression"
)


if section == "Executive Dashboard":
    _header(section, "A compact view of the selected dataset, model results and review workload.")
    _safe_source_label(data)
    total = int(data.audit["rows_modelled"])
    prevalence = float(data.audit["no_show_rate"])
    cols = st.columns(4)
    cols[0].metric("Appointments", f"{total:,}")
    cols[1].metric("Observed no-show rate", f"{prevalence:.1%}")
    cols[2].metric("Models evaluated", str(len(metrics)) if not metrics.empty else "Not run")
    if evaluation:
        best = evaluation.metrics.iloc[0]
        cols[3].metric("Best holdout ROC-AUC", f"{best['roc_auc']:.3f}", str(best["model"]))
    else:
        cols[3].metric("Quality issues removed", f"{data.audit['rows_removed']:,}")
    if evaluation is None and metrics.empty:
        st.info("Use **Train and evaluate 10 models** in the sidebar. The run is local and may take a few minutes.")
    elif not metrics.empty:
        st.subheader("Held-out model comparison")
        _display_metrics(metrics)
        st.caption(
            "These scores use labels from records held out from fitting. They describe this dataset and split only."
        )
        st.bar_chart(metrics.set_index("model")[["roc_auc", "f1"]])
    left, right = st.columns(2)
    with left:
        st.subheader("Outcome balance")
        st.bar_chart(pd.DataFrame({"Appointments": [total * (1 - prevalence), total * prevalence]}, index=["Attended", "No-show"]))
    with right:
        st.subheader("Data-quality snapshot")
        st.write(
            f"{data.audit['rows_removed']:,} rows were excluded for invalid outcomes, invalid dates/values, "
            f"or duplicate appointment IDs. {data.audit['missing_cells_raw']:,} missing cells were present in the raw file."
        )
    st.markdown(
        '<div class="cf-warning">This prototype supports staff review. A risk score must never be used to deny, '
        'delay, or reduce access to care.</div>',
        unsafe_allow_html=True,
    )


elif section == "Dataset and Data-Quality Intelligence":
    _header(section, "Inspect source, schema, exclusions and validation checks. Sources stay separate.")
    _safe_source_label(data)
    st.subheader("Quality indicators")
    audit = data.audit
    qcols = st.columns(6)
    qcols[0].metric("Rows received", f"{audit['rows_received']:,}")
    qcols[1].metric("Rows modelled", f"{audit['rows_modelled']:,}")
    qcols[2].metric("Rows excluded", f"{audit['rows_removed']:,}")
    qcols[3].metric("Duplicate IDs", f"{audit['duplicate_ids']:,}")
    qcols[4].metric("Missing raw cells", f"{audit['missing_cells_raw']:,}")
    qcols[5].metric("Missing target values", f"{audit['target_missing_raw']:,}")
    st.write(f"No-show prevalence after validation: **{audit['no_show_rate']:.1%}**")
    if "negative_lead_intervals" in audit:
        st.write(f"Negative scheduled-to-appointment intervals: **{audit['negative_lead_intervals']:,}** (clipped to zero and retained).")
    st.subheader("Model input fields")
    st.write(", ".join(data.X.columns))
    st.caption(
        "Appointment and patient IDs are excluded from predictors. The public dataset uses PatientId only to "
        "keep a patient's records together in the train/test split. Raw timestamps and outcome labels are not predictors."
    )
    st.subheader("Preview")
    preview = data.X.head(8).copy()
    preview.insert(0, "Local reference", data.references.head(8).to_numpy())
    st.dataframe(preview, hide_index=True, width="stretch")
    st.subheader("Import a CSV for data-quality review")
    upload = st.file_uploader("Choose a CSV file (kept in memory for this session)", type=["csv"])
    if upload is not None:
        try:
            user_frame = pd.read_csv(upload, low_memory=False)
            row_count, column_count = user_frame.shape
            blank_headers = int(sum(not str(c).strip() or str(c).startswith("Unnamed:") for c in user_frame.columns))
            missing_cells = int(user_frame.isna().sum().sum())
            duplicate_rows = int(user_frame.duplicated().sum())
            id_column = next((c for c in ["appointment_id", "AppointmentID"] if c in user_frame), None)
            duplicate_ids = int(user_frame[id_column].duplicated().sum()) if id_column else None
            target_column = next((c for c in ["no_show", "No-show"] if c in user_frame), None)
            st.write({
                "rows": row_count,
                "columns": column_count,
                "blank_headers": blank_headers,
                "missing_cells": missing_cells,
                "duplicate_rows": duplicate_rows,
                "duplicate_appointment_ids": duplicate_ids if id_column else "No recognized appointment ID column",
                "labelled_for_evaluation": bool(target_column),
            })
            if target_column is None:
                st.info("CSV checked in memory. Without a `no_show` label it cannot be used for model evaluation.")
            st.dataframe(user_frame.head(10), hide_index=True, width="stretch")
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            st.error(f"Could not validate this CSV: {exc}")


elif section == "AI Model Laboratory":
    _header(section, "Ten trained classifier families, one shared preprocessing pipeline and a group-disjoint holdout.")
    _safe_source_label(data)
    if evaluation is None and not metrics.empty:
        st.info("The saved score table is available. Train in this session to activate prediction explanations and simulation.")
    if metrics.empty:
        st.info("Start training from the sidebar to compare the ten classifiers on a held-out test split.")
    else:
        _display_metrics(metrics)
        st.bar_chart(metrics.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]])
        if evaluation:
            st.write(
                f"Split: {evaluation.metadata['train_rows']:,} train / {evaluation.metadata['test_rows']:,} test rows; "
                f"{evaluation.metadata['train_groups']:,} train / {evaluation.metadata['test_groups']:,} test groups; "
                f"patient/group overlap: {'none' if evaluation.metadata['groups_disjoint'] else 'detected'}."
            )
            st.caption(
                "The public-data split groups all appointments by PatientId. IDs are never predictors. "
                "K-Nearest Neighbors and calibrated SVM use up to 25,000 training rows for runtime; the full held-out test set is evaluated."
            )
            st.download_button(
                "Download this model report as CSV",
                evaluation.metrics.to_csv(index=False).encode("utf-8"),
                file_name=f"clinicflow_{dataset_key}_model_metrics.csv",
                mime="text/csv",
            )


elif section == "Individual Prediction Explanations":
    _header(section, "Inspect held-out predictions and see how one input change affects this model's score.")
    if evaluation is None:
        st.info("Train the model set in this session to inspect held-out examples.")
    else:
        selected_model = st.selectbox("Classifier", list(evaluation.models), index=list(evaluation.models).index(default_model) if default_model in evaluation.models else 0)
        probabilities = evaluation.probabilities[selected_model]
        order = np.argsort(probabilities)[::-1]
        shown = st.select_slider("Held-out appointment (ranked by predicted risk)", options=list(range(len(order))), value=min(0, len(order)-1), format_func=lambda x: f"Rank {x+1}")
        test_index = int(order[shown])
        risk = float(probabilities[test_index])
        actual = int(evaluation.y_test.iloc[test_index])
        original_index = int(evaluation.X_test.index[test_index])
        reference = data.references.iloc[original_index]
        c1, c2, c3 = st.columns(3)
        c1.metric("Local appointment reference", str(reference))
        c2.metric("Estimated no-show risk", f"{risk:.1%}")
        c3.metric("Observed holdout outcome", "No-show" if actual else "Attended")
        st.dataframe(evaluation.X_test.iloc[[test_index]], hide_index=True, width="stretch")
        impacts = local_feature_impacts(
            evaluation.models[selected_model],
            evaluation.X_test.iloc[test_index],
            evaluation.X_train,
        )
        st.subheader("One-feature-at-a-time score changes")
        st.dataframe(impacts, hide_index=True, width="stretch")
        st.caption(
            "Each row replaces one value with a training-set median or most common value and rescales the score. "
            "This is a local perturbation explanation, not a causal effect or SHAP value; correlated inputs can make it unstable."
        )

        st.divider()
        st.subheader("Score new appointments locally")
        st.caption(
            "Upload a CSV with the selected source's predictor columns. A fitted model scores valid rows in this session; "
            "IDs and any outcome label are ignored, and no record is sent outside this app."
        )
        future_upload = st.file_uploader(
            "Choose an appointment CSV to score", type=["csv"], key="future_score_upload"
        )
        if future_upload is not None:
            try:
                future_raw = pd.read_csv(future_upload, low_memory=False)
                future_X, future_refs, future_audit = prepare_prediction_features(
                    future_raw, dataset_key
                )
                future_model = evaluation.models[selected_model]
                future_all_probabilities = np.asarray(future_model.predict_proba(future_X))
                class_one = list(future_model.classes_).index(1)
                future_probabilities = future_all_probabilities[:, class_one]
                cutoff = st.slider(
                    "Show a review marker at estimated risk of at least", 5, 95, 50, 1,
                    key="future_score_cutoff",
                ) / 100
                future_result = pd.DataFrame(
                    {
                        "Local reference": future_refs,
                        "Estimated no-show risk": future_probabilities,
                        "At or above review cutoff": future_probabilities >= cutoff,
                    }
                ).sort_values("Estimated no-show risk", ascending=False)
                st.write(future_audit)
                st.dataframe(future_result, hide_index=True, width="stretch")
                st.download_button(
                    "Download local risk scores as CSV",
                    future_result.to_csv(index=False).encode("utf-8"),
                    file_name="clinicflow_local_risk_scores.csv",
                    mime="text/csv",
                )
                st.caption(
                    "These are model scores, not clinical judgments or validated future-clinic outcomes. "
                    "The app does not send reminders or change appointments."
                )
            except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
                st.error(f"Could not score this CSV: {exc}")


elif section == "AI Copilot":
    _header(section, "A local, deterministic assistant that summarizes this dataset's actual evaluation results.")
    if metrics.empty:
        st.info("Train classifiers first to generate findings from the application results.")
    else:
        for finding in copilot_insights(metrics, data.label):
            st.markdown(f"- {finding}")
        st.info(
            "Copilot mode: deterministic and data-grounded. No LLM credential is configured in this build; "
            "no patient-level data is sent to an external service."
        )
        if evaluation:
            best_name = str(evaluation.metrics.iloc[0]["model"])
            best_prob = evaluation.probabilities[best_name]
            review = simulate_capacity(
                evaluation.y_test, best_prob, mode="Top-K capacity", capacity=20
            )
            st.write(
                f"At a 20-booking review cap, the current holdout ranking would flag "
                f"{review['true_positives']} of {int(evaluation.y_test.sum())} held-out no-shows "
                f"({review['recall']:.1%} recall). This is retrospective, not a promise of future results."
            )


elif section == "Decision Simulator":
    _header(section, "Compare a probability cutoff with a daily top-K review limit using held-out labels.")
    if evaluation is None:
        st.info("Train classifiers in this session to activate the simulator.")
    else:
        model_name = st.selectbox("Classifier for simulation", list(evaluation.models), key="sim_model")
        mode = st.radio("Review policy", ["Risk threshold", "Top-K capacity"], horizontal=True)
        capacity = st.number_input("Staff review capacity (appointments)", min_value=1, max_value=len(evaluation.y_test), value=min(20, len(evaluation.y_test)), step=1)
        threshold = 0.5
        if mode == "Risk threshold":
            threshold_pct = st.slider("Flag when estimated risk is at least", 5, 95, 50, 1)
            threshold = threshold_pct / 100
        summary = simulate_capacity(
            evaluation.y_test,
            evaluation.probabilities[model_name],
            mode=mode,
            threshold=threshold,
            capacity=int(capacity),
        )
        c = st.columns(5)
        c[0].metric("Flagged", f"{summary['flagged']:,}")
        c[1].metric("Capacity used", f"{summary['capacity_utilization']:.1%}")
        c[2].metric("Precision", f"{summary['precision']:.1%}")
        c[3].metric("Recall", f"{summary['recall']:.1%}")
        c[4].metric("Missed no-shows", f"{summary['missed_no_shows']:,}")
        curve = threshold_curve(evaluation.y_test, evaluation.probabilities[model_name])
        st.subheader("Threshold trade-off on the same held-out appointments")
        st.line_chart(curve.set_index("threshold")[["precision", "recall"]])
        st.bar_chart(curve.set_index("threshold")[["flagged"]])
        st.caption(
            "Capacity and threshold runs are retrospective simulations. They do not message patients, change bookings, "
            "or reserve extra appointments. Imported scores also require local validation before operational use."
        )


elif section == "Responsible AI and Evaluation":
    _header(section, "What the evaluation supports, what it does not, and how the data should be handled.")
    _safe_source_label(data)
    st.markdown(
        "- This is an operations prototype, not a clinical system. A score must not deny or delay care.\n"
        "- Metrics describe this historical or simulated dataset and its group-disjoint holdout only.\n"
        "- `PatientId` and `AppointmentID` are excluded from predictors; patient IDs are used only to keep a patient's rows together across the split.\n"
        "- `No-show` is the label, never an input. Raw appointment timestamps are converted to lead time, weekday and booking hour, then removed.\n"
        "- `SMS_received` is recorded after booking; the model's prediction point assumes this status is known. It does not establish that SMS caused attendance.\n"
        "- The public data are from Brazil in 2016. Results may not transfer to a different clinic, country, population or workflow.\n"
        "- Group rates are exploratory comparisons, not a fairness certification. Small groups and historical patterns can hide harms.\n"
        "- No real clinic data is transmitted. The copilot uses fixed, deterministic calculations; no LLM is configured.\n"
        "- Reminder text is not sent. The app does not edit or cancel appointments."
    )
    if evaluation is not None:
        selected_model = st.selectbox("Evaluation model", list(evaluation.models), key="fairness_model")
        flagged = evaluation.probabilities[selected_model] >= 0.5
        sensitive = data.sensitive.iloc[evaluation.X_test.index].reset_index(drop=True)
        y_test = evaluation.y_test.reset_index(drop=True)
        group_col = st.selectbox("Exploratory group", list(sensitive.columns))
        rows = []
        for value, indexes in sensitive.groupby(group_col, dropna=False).groups.items():
            indexes = list(indexes)
            truth = y_test.iloc[indexes].to_numpy(dtype=int) == 1
            select = flagged[indexes]
            positive_count = int(truth.sum())
            rows.append({
                "group": str(value),
                "n": len(indexes),
                "observed_no_show_rate": float(truth.mean()) if len(indexes) else 0.0,
                "flag_rate": float(select.mean()) if len(indexes) else 0.0,
                "recall": float(np.sum(select & truth) / positive_count) if positive_count else np.nan,
            })
        group_table = pd.DataFrame(rows).sort_values("n", ascending=False)
        st.dataframe(group_table, hide_index=True, width="stretch")
        st.caption("Groups below n=30 are unstable. Differences warrant review; they do not establish discrimination or fairness.")
        st.subheader("Evaluation protocol")
        st.write(json.dumps(evaluation.metadata, indent=2))
        st.caption("The held-out records have labels used only for retrospective evaluation and simulation.")
    elif not metrics.empty:
        st.subheader("Saved held-out report")
        _display_metrics(metrics)
        st.info("For subgroup checks, explanations, and capacity simulation, train the classifiers in this session.")
