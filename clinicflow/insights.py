"""Deterministic, dataset-grounded language for the optional AI copilot page."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .modeling import simulate_capacity


def copilot_insights(metrics: pd.DataFrame, dataset_label: str) -> list[str]:
    """Return plain-language findings calculated from the supplied evaluation table."""
    if metrics.empty:
        return ["Train and evaluate the classifiers to create data-backed findings."]
    ranked = metrics.sort_values(["roc_auc", "f1"], ascending=False)
    best = ranked.iloc[0]
    logistic = metrics.loc[metrics["model"] == "Logistic Regression"]
    findings = [
        f"On the held-out {dataset_label} records, {best['model']} ranked first by ROC-AUC "
        f"({best['roc_auc']:.3f}); its F1 at the default 50% threshold was {best['f1']:.3f}.",
        f"The strongest model's held-out recall was {best['recall']:.1%}; "
        "a lower review threshold or a fixed daily review capacity changes the number flagged.",
    ]
    if not logistic.empty:
        row = logistic.iloc[0]
        findings.append(
            f"Logistic regression reached {row['accuracy']:.1%} accuracy, "
            f"{row['precision']:.1%} precision and {row['recall']:.1%} recall on the same holdout."
        )
    findings.append(
        "These are retrospective results on historical labels. They do not show that a reminder "
        "caused attendance, and they are not clinical validation."
    )
    return findings


def threshold_curve(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.2, 0.81, 0.1):
        summary = simulate_capacity(
            y_true, probabilities, mode="Risk threshold", threshold=float(threshold)
        )
        rows.append(
            {
                "threshold": float(threshold),
                "flagged": int(summary["flagged"]),
                "precision": float(summary["precision"]),
                "recall": float(summary["recall"]),
            }
        )
    return pd.DataFrame(rows)

