from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The project also supports Python's embeddable Windows distribution, whose
# python._pth intentionally omits the current directory from sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinicflow.data import PUBLIC_PATH, SIMULATED_PATH, load_dataset, prepare_prediction_features
from clinicflow.modeling import evaluate_all, local_feature_impacts, model_names, simulate_capacity
from streamlit.testing.v1 import AppTest


class DatasetTests(unittest.TestCase):
    def test_simulated_source_is_separate_and_has_no_target_or_ids_in_features(self):
        data = load_dataset("simulated")
        self.assertEqual(data.audit["rows_received"], 1200)
        self.assertEqual(len(data.X), len(data.y))
        self.assertNotIn("no_show", data.X.columns)
        self.assertNotIn("appointment_id", data.X.columns)
        self.assertTrue(set(data.y.unique()).issubset({0, 1}))

    def test_public_dataset_schema_and_patient_split_group(self):
        data = load_dataset("public_brazil_2016")
        self.assertEqual(data.audit["rows_received"], 110527)
        self.assertGreater(data.audit["rows_modelled"], 110000)
        self.assertNotIn("PatientId", data.X.columns)
        self.assertNotIn("AppointmentID", data.X.columns)
        self.assertNotIn("No-show", data.X.columns)
        self.assertGreater(data.groups.nunique(), 50000)
        self.assertEqual(data.label, "Brazil public appointments (2016)")

    def test_ten_requested_classifier_families_are_registered(self):
        self.assertEqual(len(model_names()), 10)
        self.assertIn("Support Vector Machine", model_names())
        self.assertIn("Gaussian Naive Bayes", model_names())

    def test_simulated_csv_can_be_prepared_for_unlabeled_scoring(self):
        raw = pd.read_csv(SIMULATED_PATH).head(8)
        features, references, audit = prepare_prediction_features(raw, "simulated")
        self.assertEqual(len(features), 8)
        self.assertEqual(len(references), 8)
        self.assertNotIn("no_show", features.columns)
        self.assertNotIn("appointment_id", features.columns)
        self.assertTrue(audit["target_label_ignored"])
        self.assertFalse(audit["identifier_fields_used_for_scoring"])

    def test_public_csv_can_be_prepared_without_using_ids_or_target(self):
        raw = pd.read_csv(PUBLIC_PATH, nrows=12)
        features, references, audit = prepare_prediction_features(raw, "public_brazil_2016")
        self.assertEqual(len(features), 12)
        self.assertEqual(len(references), 12)
        self.assertNotIn("No-show", features.columns)
        self.assertNotIn("PatientId", features.columns)
        self.assertNotIn("AppointmentID", features.columns)
        self.assertTrue(audit["target_label_ignored"])


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = load_dataset("simulated")
        sample = dataset.X.sample(n=360, random_state=73).sort_index()
        cls.data = dataset
        cls.X = sample
        cls.y = dataset.y.loc[sample.index]
        cls.groups = dataset.groups.loc[sample.index]
        cls.evaluation = evaluate_all(cls.X, cls.y, cls.groups)

    def test_all_models_train_and_report_real_holdout_metrics(self):
        result = self.evaluation
        self.assertEqual(len(result.models), 10, result.failures)
        self.assertEqual(len(result.metrics), 10)
        self.assertTrue(result.metadata["groups_disjoint"])
        self.assertEqual(result.metadata["train_rows"] + result.metadata["test_rows"], len(self.X))
        self.assertTrue(np.isfinite(result.metrics[["accuracy", "precision", "recall", "f1", "roc_auc"]].to_numpy()).all())
        for name, probabilities in result.probabilities.items():
            self.assertEqual(name in result.models, True)
            self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_capacity_top_k_flags_exactly_the_limit(self):
        result = self.evaluation
        summary = simulate_capacity(
            result.y_test, result.probabilities["Logistic Regression"],
            mode="Top-K capacity", capacity=7,
        )
        self.assertEqual(summary["flagged"], 7)
        self.assertLessEqual(summary["capacity_utilization"], 1.0)

    def test_local_explanation_is_finite_and_has_one_row_per_feature(self):
        result = self.evaluation
        impacts = local_feature_impacts(
            result.models["Logistic Regression"], result.X_test.iloc[0], result.X_train
        )
        self.assertEqual(len(impacts), result.X_train.shape[1])
        self.assertTrue(np.isfinite(impacts["risk_change_pp"].to_numpy()).all())

    def test_fitted_model_scores_new_unlabeled_csv_rows(self):
        raw = pd.read_csv(SIMULATED_PATH).head(5)
        features, _, audit = prepare_prediction_features(raw, "simulated")
        model = self.evaluation.models["Logistic Regression"]
        probabilities = model.predict_proba(features)[:, list(model.classes_).index(1)]
        self.assertEqual(len(probabilities), 5)
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertTrue(audit["target_label_ignored"])


class AppSmokeTests(unittest.TestCase):
    def test_seven_section_app_renders_trains_and_opens_scoring_page(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        result = AppTest.from_file(str(app_path), default_timeout=120).run()
        self.assertEqual(len(result.exception), 0, [item.message for item in result.exception])
        self.assertEqual(
            result.radio[0].value,
            "Executive Dashboard",
        )
        self.assertEqual(
            list(result.radio[0].options),
            [
                "Executive Dashboard",
                "Dataset and Data-Quality Intelligence",
                "AI Model Laboratory",
                "Individual Prediction Explanations",
                "AI Copilot",
                "Decision Simulator",
                "Responsible AI and Evaluation",
            ],
        )
        self.assertTrue(any("Train and evaluate 10 models" in b.label for b in result.button))
        result.button[0].click()
        result.run(timeout=120)
        self.assertEqual(len(result.exception), 0, [item.message for item in result.exception])
        result.radio[0].set_value("Individual Prediction Explanations")
        result.run(timeout=120)
        self.assertEqual(len(result.exception), 0, [item.message for item in result.exception])
        self.assertTrue(result.file_uploader)


if __name__ == "__main__":
    unittest.main(verbosity=2)
