"""Positivity, fit isolation, dimension budgets and responsive synthetic signal."""
import unittest

import numpy as np
import pandas as pd

from ua_alerts.models import HISTORY_COLUMNS, NEWS_COLUMNS, ModelConfig, fit_predict_poisson


def synthetic_data(n=90):
    rng = np.random.default_rng(27)
    days = pd.date_range("2026-01-01", periods=n, tz="UTC")
    frame = pd.DataFrame({c: rng.uniform(.5, 4, n) for c in HISTORY_COLUMNS + NEWS_COLUMNS})
    frame["origin"] = days - pd.Timedelta(hours=12)
    frame["target_day"] = days
    frame["target_end"] = days + pd.Timedelta(days=1)
    frame["label_available_at"] = days + pd.Timedelta(days=1, hours=8)
    frame["y_train_vintage"] = rng.poisson(np.exp(.6 + .35 * frame.kyiv_lag1))
    frame["y_final"] = 999999  # Never a model input.
    frame["titles"] = ["civilian shelter" if i % 2 == 0 else "city support" for i in range(n)]
    row = frame.iloc[[-1]].copy()
    row["origin"] = days[-1] + pd.Timedelta(days=3, hours=12)
    row["target_day"] = row.origin.dt.floor("D") + pd.Timedelta(days=1)
    row["titles"] = "civilian shelter"
    return frame, row


class CountModelContract(unittest.TestCase):
    def test_positive_mean_and_not_constant_when_signal_varies(self):
        train, row = synthetic_data()
        config = ModelConfig(False, .01, None)
        low, high = row.copy(), row.copy()
        low["kyiv_lag1"], high["kyiv_lag1"] = .2, 6.
        a, _ = fit_predict_poisson(train, low, config)
        b, _ = fit_predict_poisson(train, high, config)
        self.assertGreater(a, 0)
        self.assertGreater(b, a * 1.5)

    def test_budget_tracks_eligible_window_not_full_frame(self):
        train, row = synthetic_data(180)
        _, audit = fit_predict_poisson(train, row, ModelConfig(True, .1, 56))
        self.assertEqual(audit["training_rows"], 56)
        self.assertLessEqual(audit["text_feature_count"], min(50, 56 // 3))
        self.assertLessEqual(audit["total_feature_count"], 13 + 56 // 3)

    def test_future_labels_and_text_cannot_affect_estimator(self):
        train, row = synthetic_data()
        config = ModelConfig(True, .1, None)
        expected, trace = fit_predict_poisson(train, row, config)
        future = train.tail(10).copy()
        future["origin"] = row.iloc[0].origin + pd.Timedelta(days=1)
        future["target_end"] = row.iloc[0].origin + pd.Timedelta(days=3)
        future["label_available_at"] = row.iloc[0].origin + pd.Timedelta(days=4)
        future["y_train_vintage"] = 9999999
        future["titles"] = "futurevocabulary " * 100
        augmented = pd.concat([train, future], ignore_index=True)
        actual, audit = fit_predict_poisson(augmented, row, config)
        self.assertAlmostEqual(expected, actual)
        self.assertEqual(trace["vocabulary"], audit["vocabulary"])
        self.assertEqual(trace["training_rows"], audit["training_rows"])

    def test_final_revised_truth_not_used_as_input_or_fit_target(self):
        train, row = synthetic_data()
        config = ModelConfig(False, .1, None)
        expected, _ = fit_predict_poisson(train, row, config)
        train["y_final"] = np.arange(len(train)) * 123456
        actual, _ = fit_predict_poisson(train, row, config)
        self.assertAlmostEqual(expected, actual)

    def test_missing_text_falls_back_to_numeric_poisson(self):
        train, row = synthetic_data()
        train["titles"], row["titles"] = "", "futureonlyword"
        prediction, audit = fit_predict_poisson(train, row, ModelConfig(True, .1, None))
        self.assertGreater(prediction, 0)
        self.assertEqual(audit["text_feature_count"], 0)

    def test_recent_level_offset_matches_intercept_only_likelihood(self):
        train, row = synthetic_data()
        for column in HISTORY_COLUMNS:
            train[column] = 2.
            row[column] = 2.
        train["y_train_vintage"] = np.tile([2, 3, 4], len(train) // 3)
        row["kyiv_mean7"] = 8.
        actual, audit = fit_predict_poisson(train, row, ModelConfig(False, .1, None, True))
        expected = (8. + .5 / 7) * train.y_train_vintage.sum() / (len(train) * (2. + .5 / 7))
        self.assertAlmostEqual(actual, expected, places=5)
        self.assertEqual(audit["numeric_feature_count"], 0)

    def test_future_labels_cannot_affect_recent_level_offset(self):
        train, row = synthetic_data()
        config = ModelConfig(False, .1, 56, True)
        expected, _ = fit_predict_poisson(train, row, config)
        extra = train.tail(5).copy()
        extra["label_available_at"] = row.iloc[0].origin + pd.Timedelta(days=1)
        extra["y_train_vintage"] = 1000000
        actual, _ = fit_predict_poisson(pd.concat([train, extra], ignore_index=True), row, config)
        self.assertAlmostEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
