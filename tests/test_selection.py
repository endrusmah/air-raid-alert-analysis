"""Physical holdout separation and locked-selection integrity."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ua_alerts.features import load_data, eligible_training
from ua_alerts.models import ModelConfig
from ua_alerts.validation import select_models, read_selection, pipeline_hash, split_schedule


class SelectionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.snapshots, cls.batches = load_data(cls.root / "data")

    def test_future_test_snapshots_and_articles_cannot_change_selection(self):
        start, _, cutoff = split_schedule(self.snapshots, 28)
        def sentinel_model(frame, row, config):
            # Detect accidental test-row exposure even if the model's own fit
            # guard would otherwise mask it.
            self.assertTrue((frame.target_day < start).all())
            self.assertNotIn("y_final", frame.columns)
            eligible = eligible_training(frame, row.iloc[0].origin)
            value = float(eligible.y_train_vintage.mean() + .01 * row.iloc[0].kyiv_lag1)
            if config.with_news:
                value += .001 * row.iloc[0].news_titles3
            return value, {"total_feature_count": 2, "training_rows": len(eligible)}
        candidates = [ModelConfig(False, .1, 56), ModelConfig(True, .1, 56)]
        with patch("ua_alerts.validation.fit_predict_poisson", side_effect=sentinel_model):
            expected = select_models(self.snapshots, self.batches, candidates=candidates)[0]
            corrupted_snapshots, corrupted_batches = copy.deepcopy(self.snapshots), copy.deepcopy(self.batches)
            for snapshot in corrupted_snapshots:
                if pd.Timestamp(snapshot["available_at"]) >= cutoff:
                    for day in snapshot["counts"].values():
                        for region in day:
                            day[region] = 999999
            for batch in corrupted_batches:
                if pd.Timestamp(batch["available_at"]) >= cutoff:
                    for record in batch["records"]:
                        record["title"] = "future_test_only_vocab " * 200
                        record["tone"] = 999999
                        record["themes"] = "WEAPONS;"
            actual = select_models(corrupted_snapshots, corrupted_batches, candidates=candidates)[0]
        self.assertEqual(expected, actual)
        self.assertEqual(actual["test_rows_seen_during_selection"], 0)

    def test_modified_selection_cannot_be_evaluated(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            folder = Path(temporary)
            body = json.dumps({"pipeline_sha256": pipeline_hash(), "selected_model": "first"})
            (folder / "selection.json").write_text(body)
            (folder / "selection.sha256").write_text(hashlib.sha256(body.encode()).hexdigest())
            self.assertEqual(read_selection(folder)["selected_model"], "first")
            (folder / "selection.json").write_text(body.replace("first", "other"))
            with self.assertRaisesRegex(ValueError, "changed after locking"):
                read_selection(folder)

    def test_changed_pipeline_cannot_reuse_selection(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            folder = Path(temporary)
            body = json.dumps({"pipeline_sha256": "different", "selected_model": "first"})
            (folder / "selection.json").write_text(body)
            (folder / "selection.sha256").write_text(hashlib.sha256(body.encode()).hexdigest())
            with self.assertRaisesRegex(ValueError, "Pipeline changed"):
                read_selection(folder)


if __name__ == "__main__":
    unittest.main()
