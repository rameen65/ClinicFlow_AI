"""Dataset loading, leakage-safe feature engineering, and quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATED_PATH = PROJECT_ROOT / "data" / "simulated" / "appointments_demo.csv"
PUBLIC_PATH = PROJECT_ROOT / "data" / "public" / "KaggleV2-May-2016.csv"


@dataclass
class PreparedDataset:
    key: str
    label: str
    source: str
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    references: pd.Series
    audit: dict[str, Any]
    sensitive: pd.DataFrame


def _quality_summary(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    target: pd.Series,
    *,
    duplicate_ids: int,
    invalid_rows: int,
    group_name: str,
    target_column: str,
) -> dict[str, Any]:
    return {
        "rows_received": int(len(raw)),
        "rows_modelled": int(len(clean)),
        "rows_removed": int(len(raw) - len(clean)),
        "duplicate_ids": int(duplicate_ids),
        "invalid_rows": int(invalid_rows),
        "missing_cells_raw": int(raw.isna().sum().sum()),
        "target_missing_raw": int(raw[target_column].isna().sum()),
        "target_missing_after_validation": int(target.isna().sum()),
        "no_show_rate": float(target.mean()) if len(target) else 0.0,
        "unique_groups": int(clean[group_name].nunique()),
        "columns_received": int(raw.shape[1]),
    }


def _read_simulated() -> PreparedDataset:
    if not SIMULATED_PATH.is_file():
        raise FileNotFoundError(f"Simulated dataset not found: {SIMULATED_PATH}")
    raw = pd.read_csv(SIMULATED_PATH)
    required = {
        "appointment_id", "age_group", "lead_time_days", "previous_no_shows",
        "reminder_sent", "weekday", "appointment_hour", "booking_channel",
        "distance_km", "no_show",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Simulated CSV is missing required columns: {', '.join(sorted(missing))}")

    numeric = ["lead_time_days", "previous_no_shows", "appointment_hour", "distance_km"]
    clean = raw.copy()
    for column in numeric:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean["no_show"] = pd.to_numeric(clean["no_show"], errors="coerce")
    duplicate_ids = int(clean["appointment_id"].duplicated(keep="first").sum())
    invalid = clean[numeric + ["no_show"]].isna().any(axis=1)
    invalid |= ~clean["no_show"].isin([0, 1])
    invalid |= (clean["lead_time_days"] < 0) | (clean["previous_no_shows"] < 0)
    invalid |= (clean["appointment_hour"] < 0) | (clean["appointment_hour"] > 23)
    invalid |= clean["distance_km"] < 0
    invalid |= clean["appointment_id"].duplicated(keep="first")
    invalid_rows = int(invalid.sum())
    clean = clean.loc[~invalid].reset_index(drop=True)
    if len(clean) < 20 or clean["no_show"].nunique() != 2:
        raise ValueError("Need at least 20 valid appointments and both outcome classes.")

    X = clean[
        [
            "age_group", "lead_time_days", "previous_no_shows", "reminder_sent",
            "weekday", "appointment_hour", "booking_channel", "distance_km",
        ]
    ].copy()
    X["reminder_sent"] = X["reminder_sent"].astype(str).str.strip().str.lower().map(
        {"yes": "Yes", "no": "No", "true": "Yes", "false": "No", "1": "Yes", "0": "No"}
    ).fillna("Unknown")
    groups = clean["appointment_id"].astype(str).rename("split_group")
    refs = clean["appointment_id"].astype(str).rename("appointment_reference")
    y = clean["no_show"].astype(int).rename("no_show")
    sensitive = clean[["age_group", "booking_channel", "reminder_sent"]].copy()
    audit = _quality_summary(
        raw, clean, y, duplicate_ids=duplicate_ids, invalid_rows=invalid_rows,
        group_name="appointment_id", target_column="no_show",
    )
    return PreparedDataset(
        key="simulated",
        label="Simulated clinic demo",
        source="Included generated sample; 1,200 synthetic appointments, not patient records.",
        X=X.reset_index(drop=True), y=y.reset_index(drop=True), groups=groups.reset_index(drop=True),
        references=refs.reset_index(drop=True), audit=audit,
        sensitive=sensitive.reset_index(drop=True),
    )


def _read_public() -> PreparedDataset:
    if not PUBLIC_PATH.is_file():
        raise FileNotFoundError(f"Public dataset not found: {PUBLIC_PATH}")
    raw = pd.read_csv(PUBLIC_PATH, low_memory=False)
    required = {
        "PatientId", "AppointmentID", "Gender", "ScheduledDay", "AppointmentDay",
        "Age", "Neighbourhood", "Scholarship", "Hipertension", "Diabetes",
        "Alcoholism", "Handcap", "SMS_received", "No-show",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Public CSV is missing required columns: {', '.join(sorted(missing))}")

    clean = raw.copy()
    clean["Age"] = pd.to_numeric(clean["Age"], errors="coerce")
    clean["ScheduledDay_parsed"] = pd.to_datetime(clean["ScheduledDay"], errors="coerce", utc=True)
    clean["AppointmentDay_parsed"] = pd.to_datetime(clean["AppointmentDay"], errors="coerce", utc=True)
    clean["No-show"] = clean["No-show"].astype("string").str.strip().str.lower().map(
        {"yes": 1, "no": 0}
    )
    duplicate_ids = int(clean["AppointmentID"].duplicated(keep="first").sum())
    invalid = clean["Age"].isna() | (clean["Age"] < 0) | (clean["Age"] > 120)
    invalid |= clean["ScheduledDay_parsed"].isna() | clean["AppointmentDay_parsed"].isna()
    invalid |= clean["No-show"].isna()
    invalid |= clean["AppointmentID"].duplicated(keep="first")
    invalid |= clean["PatientId"].isna()
    invalid_rows = int(invalid.sum())
    clean = clean.loc[~invalid].reset_index(drop=True)
    if len(clean) < 20 or clean["No-show"].nunique() != 2:
        raise ValueError("Need at least 20 valid appointments and both outcome classes.")

    lead_days = (
        clean["AppointmentDay_parsed"] - clean["ScheduledDay_parsed"]
    ).dt.total_seconds() / 86400.0
    X = pd.DataFrame(
        {
            "lead_time_days": lead_days.clip(lower=0).astype(float),
            "appointment_weekday": clean["AppointmentDay_parsed"].dt.day_name(),
            "scheduled_hour": clean["ScheduledDay_parsed"].dt.hour.astype(float),
            "age": clean["Age"].astype(float),
            "gender": clean["Gender"].astype("string").fillna("Unknown"),
            "neighbourhood": clean["Neighbourhood"].astype("string").fillna("Unknown"),
            "scholarship": pd.to_numeric(clean["Scholarship"], errors="coerce"),
            "hypertension": pd.to_numeric(clean["Hipertension"], errors="coerce"),
            "diabetes": pd.to_numeric(clean["Diabetes"], errors="coerce"),
            "alcoholism": pd.to_numeric(clean["Alcoholism"], errors="coerce"),
            "handicap": pd.to_numeric(clean["Handcap"], errors="coerce"),
            "sms_received": pd.to_numeric(clean["SMS_received"], errors="coerce"),
        }
    )
    # The IDs and raw timestamps are deliberately excluded from model inputs.
    groups = clean["PatientId"].astype(str).rename("split_group")
    refs = pd.Series(
        [f"BRA-{i:06d}" for i in range(1, len(clean) + 1)], name="appointment_reference"
    )
    y = clean["No-show"].astype(int).rename("no_show")
    age_band = pd.cut(
        clean["Age"], bins=[-0.1, 17, 29, 44, 59, 120],
        labels=["0–17", "18–29", "30–44", "45–59", "60+"],
    ).astype("string")
    sensitive = pd.DataFrame(
        {
            "age_band": age_band,
            "gender": clean["Gender"].astype("string").fillna("Unknown"),
            "sms_received": clean["SMS_received"].astype("string").fillna("Unknown"),
        }
    )
    audit = _quality_summary(
        raw, clean, y, duplicate_ids=duplicate_ids, invalid_rows=invalid_rows,
        group_name="PatientId", target_column="No-show",
    )
    # Invalid negative booking intervals are measurable data quality issues. Keep the rows
    # after setting their lead time to zero, and report the count instead of silently losing them.
    audit["negative_lead_intervals"] = int((lead_days < 0).sum())
    audit["patient_identifier_used_for_split_only"] = True
    audit["model_input_columns"] = int(X.shape[1])
    return PreparedDataset(
        key="public_brazil_2016",
        label="Brazil public appointments (2016)",
        source="Joni Hoppen, Kaggle Medical Appointment No Shows; CC BY-NC-SA 4.0.",
        X=X.reset_index(drop=True), y=y.reset_index(drop=True), groups=groups.reset_index(drop=True),
        references=refs.reset_index(drop=True), audit=audit,
        sensitive=sensitive.reset_index(drop=True),
    )


def load_dataset(key: str) -> PreparedDataset:
    """Load one source only. Simulated and public rows are never concatenated."""
    if key == "simulated":
        return _read_simulated()
    if key == "public_brazil_2016":
        return _read_public()
    raise ValueError(f"Unknown dataset key: {key}")


def prepare_prediction_features(
    raw: pd.DataFrame, key: str
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Prepare a new unlabeled CSV with the same predictor schema used for training.

    IDs and target labels are ignored. Invalid rows are excluded; missing predictor values
    that remain are handled by the already-fitted training pipeline.
    """
    if raw.empty:
        raise ValueError("The uploaded CSV has no records.")

    if key == "simulated":
        columns = [
            "age_group", "lead_time_days", "previous_no_shows", "reminder_sent",
            "weekday", "appointment_hour", "booking_channel", "distance_km",
        ]
        missing = set(columns).difference(raw.columns)
        if missing:
            raise ValueError(f"CSV is missing simulated predictors: {', '.join(sorted(missing))}")
        features = raw[columns].copy()
        invalid = pd.Series(False, index=raw.index)
        numeric = ["lead_time_days", "previous_no_shows", "appointment_hour", "distance_km"]
        for column in numeric:
            original = features[column]
            converted = pd.to_numeric(original, errors="coerce")
            invalid |= (original.notna() & converted.isna())
            features[column] = converted
        invalid |= features["lead_time_days"].lt(0).fillna(False)
        invalid |= features["previous_no_shows"].lt(0).fillna(False)
        invalid |= features["appointment_hour"].lt(0).fillna(False) | features["appointment_hour"].gt(23).fillna(False)
        invalid |= features["distance_km"].lt(0).fillna(False)
        features["reminder_sent"] = features["reminder_sent"].astype("string").str.strip().str.lower().map(
            {"yes": "Yes", "no": "No", "true": "Yes", "false": "No", "1": "Yes", "0": "No"}
        ).fillna("Unknown")
        for column in ["age_group", "weekday", "booking_channel"]:
            features[column] = features[column].astype("string").fillna("Unknown").astype(str)
    elif key == "public_brazil_2016":
        columns = [
            "Gender", "ScheduledDay", "AppointmentDay", "Age", "Neighbourhood",
            "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap", "SMS_received",
        ]
        missing = set(columns).difference(raw.columns)
        if missing:
            raise ValueError(f"CSV is missing public-dataset predictors: {', '.join(sorted(missing))}")
        source = raw[columns].copy()
        scheduled = pd.to_datetime(source["ScheduledDay"], errors="coerce", utc=True)
        appointment = pd.to_datetime(source["AppointmentDay"], errors="coerce", utc=True)
        raw_age = pd.to_numeric(source["Age"], errors="coerce")
        invalid = scheduled.isna() | appointment.isna()
        invalid |= raw_age.isna() | raw_age.lt(0) | raw_age.gt(120)
        numeric_features: dict[str, pd.Series] = {}
        binary_fields = {
            "scholarship": "Scholarship", "hypertension": "Hipertension",
            "diabetes": "Diabetes", "alcoholism": "Alcoholism", "sms_received": "SMS_received",
        }
        for output, source_column in binary_fields.items():
            converted = pd.to_numeric(source[source_column], errors="coerce")
            invalid |= converted.isna() | ~converted.isin([0, 1])
            numeric_features[output] = converted
        handicap = pd.to_numeric(source["Handcap"], errors="coerce")
        invalid |= handicap.isna() | handicap.lt(0)
        lead_days = (appointment - scheduled).dt.total_seconds() / 86400.0
        features = pd.DataFrame(
            {
                "lead_time_days": lead_days.clip(lower=0),
                "appointment_weekday": appointment.dt.day_name().astype("string").fillna("Unknown"),
                "scheduled_hour": scheduled.dt.hour.astype(float),
                "age": raw_age,
                "gender": source["Gender"].astype("string").fillna("Unknown"),
                "neighbourhood": source["Neighbourhood"].astype("string").fillna("Unknown"),
                **numeric_features,
                "handicap": handicap,
            },
            index=raw.index,
        )
    else:
        raise ValueError(f"Unknown dataset key: {key}")

    invalid = invalid.fillna(True)
    references = pd.Series(
        [f"Uploaded row {position + 1}" for position in range(len(raw))],
        index=raw.index,
        name="appointment_reference",
    )
    target_present = any(column in raw.columns for column in ("no_show", "No-show"))
    audit = {
        "rows_received": int(len(raw)),
        "rows_scored": int((~invalid).sum()),
        "invalid_rows_excluded": int(invalid.sum()),
        "missing_predictor_cells": int(features.isna().sum().sum()),
        "target_label_ignored": bool(target_present),
        "identifier_fields_used_for_scoring": False,
    }
    features = features.loc[~invalid].reset_index(drop=True)
    references = references.loc[~invalid].reset_index(drop=True)
    if features.empty:
        raise ValueError("No valid rows remain after predictor validation.")
    return features, references, audit

