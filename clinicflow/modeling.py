"""Shared preprocessing, group-safe evaluation, and capacity simulation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
KNN_TRAIN_LIMIT = 25_000
SVM_TRAIN_LIMIT = 25_000


@dataclass
class Evaluation:
    metrics: pd.DataFrame
    models: dict[str, Pipeline]
    probabilities: dict[str, np.ndarray]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_groups: pd.Series
    metadata: dict[str, Any]
    failures: dict[str, str]


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])]
    categorical = [column for column in X.columns if column not in numeric]
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _model_specs() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=500, class_weight="balanced", solver="lbfgs", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=30, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=60, max_depth=16, min_samples_leaf=20, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=2, random_state=RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=80, max_depth=16, min_samples_leaf=20, max_features="sqrt",
            class_weight="balanced", n_jobs=2, random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=60, learning_rate=0.06, max_depth=2, subsample=0.85,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=70, learning_rate=0.08, max_leaf_nodes=19, l2_regularization=0.5,
            early_stopping=True, random_state=RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
            n_estimators=50, learning_rate=0.5, random_state=RANDOM_STATE,
        ),
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=Pipeline(
                [
                    ("preprocessor", _preprocessor_placeholder()),
                    ("classifier", LinearSVC(
                        class_weight="balanced", dual="auto", max_iter=2500,
                        random_state=RANDOM_STATE,
                    )),
                ]
            ),
            method="sigmoid",
            cv=3,
            n_jobs=1,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=15, weights="distance", n_jobs=2, leaf_size=40
        ),
        "Gaussian Naive Bayes": GaussianNB(var_smoothing=1e-8),
    }


def _preprocessor_placeholder() -> ColumnTransformer:
    # CalibratedClassifierCV clones this estimator on internal folds. It is replaced with the
    # dataset-specific shared transformer in _pipeline_for_model before fitting.
    return _preprocessor(pd.DataFrame({"placeholder": pd.Series(dtype=float)}))


def model_names() -> list[str]:
    return list(_model_specs())


def _pipeline_for_model(name: str, estimator: Any, prep: ColumnTransformer) -> Any:
    if name == "Support Vector Machine":
        estimator = clone(estimator)
        base = estimator.estimator
        base.set_params(preprocessor=clone(prep))
        estimator.set_params(estimator=base)
        return estimator
    return Pipeline([("preprocessor", clone(prep)), ("classifier", clone(estimator))])


def _positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    p = np.asarray(model.predict_proba(X))
    classes = list(model.classes_)
    positive_column = classes.index(1)
    return p[:, positive_column].astype(float)


def _metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predicted = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def evaluate_all(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    progress: Any = None,
) -> Evaluation:
    """Train ten classifiers on a patient/group-disjoint 80/20 holdout."""
    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("Features, labels and split groups must have the same length.")
    if len(X) < 40 or y.nunique() != 2:
        raise ValueError("At least 40 rows and both target classes are required.")
    if X.columns.intersection(["no_show", "No-show", "PatientId", "AppointmentID", "appointment_id"]).size:
        raise ValueError("Target and identifier fields must not be model features.")

    fold_count = min(5, int(groups.nunique()))
    if fold_count < 2:
        raise ValueError("At least two distinct split groups are required.")
    splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
    if set(groups_train.astype(str)).intersection(set(groups_test.astype(str))):
        raise RuntimeError("Train/test split contains overlapping groups.")

    prep = _preprocessor(X_train)
    specs = _model_specs()
    fitted: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    for number, (name, estimator) in enumerate(specs.items(), start=1):
        model_X = X_train
        model_y = y_train
        train_rows = len(X_train)
        limit = KNN_TRAIN_LIMIT if name == "K-Nearest Neighbors" else (
            SVM_TRAIN_LIMIT if name == "Support Vector Machine" else None
        )
        if limit is not None and len(X_train) > limit:
            chosen, _ = train_test_split(
                np.arange(len(X_train)), train_size=limit,
                random_state=RANDOM_STATE, stratify=y_train,
            )
            model_X = X_train.iloc[chosen]
            model_y = y_train.iloc[chosen]
            train_rows = len(model_X)
        model = _pipeline_for_model(name, estimator, prep)
        start = perf_counter()
        try:
            model.fit(model_X, model_y)
            elapsed = perf_counter() - start
            probability = _positive_probability(model, X_test)
            fitted[name] = model
            predictions[name] = probability
            row = {"model": name, **_metrics(y_test, probability)}
            row["training_rows"] = int(train_rows)
            row["fit_seconds"] = round(elapsed, 2)
            records.append(row)
        except Exception as exc:  # one optional estimator must not discard successful runs
            elapsed = perf_counter() - start
            failures[name] = f"{type(exc).__name__}: {exc}"
        if progress:
            progress(number, len(specs), name, elapsed)

    if not records:
        raise RuntimeError(f"All classifier runs failed: {failures}")
    metrics = pd.DataFrame(records).sort_values(
        ["roc_auc", "f1"], ascending=False, kind="stable"
    ).reset_index(drop=True)
    return Evaluation(
        metrics=metrics,
        models=fitted,
        probabilities=predictions,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_groups=groups_test,
        metadata={
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_groups": int(groups_train.nunique()),
            "test_groups": int(groups_test.nunique()),
            "groups_disjoint": True,
            "split": "StratifiedGroupKFold first fold; 80/20 group-disjoint holdout",
            "random_state": RANDOM_STATE,
            "knn_training_cap": KNN_TRAIN_LIMIT,
            "svm_training_cap": SVM_TRAIN_LIMIT,
        },
        failures=failures,
    )


def simulate_capacity(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    mode: str,
    threshold: float = 0.5,
    capacity: int = 20,
) -> dict[str, float | int]:
    """Retrospective holdout simulation; no appointments are changed or contacted."""
    probabilities = np.asarray(probabilities, dtype=float)
    if len(probabilities) != len(y_true):
        raise ValueError("Predictions and labels must have equal length.")
    if mode == "Top-K capacity":
        count = max(0, min(int(capacity), len(probabilities)))
        selected = np.zeros(len(probabilities), dtype=bool)
        if count:
            selected[np.argsort(probabilities)[-count:]] = True
    elif mode == "Risk threshold":
        selected = probabilities >= float(threshold)
        count = int(selected.sum())
    else:
        raise ValueError(f"Unknown simulation mode: {mode}")

    truth = y_true.to_numpy(dtype=int) == 1
    tp = int(np.sum(selected & truth))
    fp = int(np.sum(selected & ~truth))
    fn = int(np.sum(~selected & truth))
    return {
        "flagged": count,
        "capacity": int(capacity),
        "capacity_utilization": float(count / capacity) if capacity else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "missed_no_shows": fn,
        "observed_no_show_rate_flagged": float(tp / count) if count else 0.0,
        "median_risk_flagged": float(np.median(probabilities[selected])) if count else 0.0,
    }


def local_feature_impacts(
    model: Any,
    row: pd.Series,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """One-at-a-time model-agnostic feature perturbation; descriptive, not causal/SHAP."""
    current = row.to_frame().T
    base = float(_positive_probability(model, current)[0])
    replacement: dict[str, Any] = {}
    for column in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            replacement[column] = float(reference[column].median())
        else:
            modes = reference[column].mode(dropna=True)
            replacement[column] = modes.iloc[0] if not modes.empty else "Unknown"
    impacts: list[dict[str, Any]] = []
    for column in reference.columns:
        changed = current.copy()
        changed.loc[:, column] = replacement[column]
        counterfactual = float(_positive_probability(model, changed)[0])
        impacts.append(
            {
                "feature": column,
                "current_value": str(row[column]),
                "reference_value": str(replacement[column]),
                "risk_change_pp": round((base - counterfactual) * 100, 2),
                "reference_risk": round(counterfactual * 100, 2),
            }
        )
    return pd.DataFrame(impacts).assign(abs_effect=lambda df: df["risk_change_pp"].abs()).sort_values(
        "abs_effect", ascending=False
    ).drop(columns="abs_effect").reset_index(drop=True)

