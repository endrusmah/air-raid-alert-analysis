"""Run the temporal contract against every real row in the bundled dataset."""
import json
from pathlib import Path
import unittest

import pandas as pd

from ua_alerts.features import load_data, build_dataset, eligible_training


class FrozenDataContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.snapshots, cls.batches = load_data(cls.root / "data")
        cls.frame, cls.audit = build_dataset(cls.snapshots, cls.batches)

    def test_every_actual_source_precedes_its_origin(self):
        for row in self.audit:
            origin = pd.Timestamp(row["origin"])
            for key in ["alert_available_at", "max_news_published_at", "max_news_available_at",
                        "max_news_indexed_at", "max_batch_available_at"]:
                if row[key] is not None:
                    self.assertLess(pd.Timestamp(row[key]), origin, (origin, key))

    def test_every_training_label_precedes_fit(self):
        for origin in self.frame.origin:
            train = eligible_training(self.frame, origin)
            self.assertTrue((train.target_end < origin).all())
            self.assertTrue((train.label_available_at < origin).all())

    def test_every_snapshot_has_server_recorded_publisher_proof(self):
        for snapshot in self.snapshots:
            proof = snapshot["workflow_proof"]
            self.assertEqual(proof["conclusion"], "success")
            self.assertLessEqual(pd.Timestamp(proof["created_at"]), pd.Timestamp(snapshot["commit_at"]))
            self.assertLessEqual(pd.Timestamp(snapshot["commit_at"]), pd.Timestamp(proof["updated_at"]))
            self.assertGreater(pd.Timestamp(snapshot["available_at"]), pd.Timestamp(proof["updated_at"]))

    def test_news_sample_has_no_download_failures(self):
        acquisition = json.loads((self.root / "data" / "acquisition.json").read_text())
        self.assertFalse(acquisition["errors"])
        days = sorted(pd.Timestamp(b["batch_at"]) for b in self.batches)
        self.assertEqual(days, list(pd.date_range(days[0], days[-1], freq="D")))

    def test_model_selection_is_before_first_test_origin(self):
        metrics = json.loads((self.root / "reports" / "selection.json").read_text())
        selection = pd.Timestamp(metrics["selection_at"])
        validation = self.frame[(self.frame.target_day >= pd.Timestamp(metrics["validation"]["start"]))
                                & (self.frame.target_day <= pd.Timestamp(metrics["validation"]["end"]))]
        self.assertTrue((validation.label_available_at < selection).all())
        self.assertTrue((validation.target_end < selection).all())


if __name__ == "__main__":
    unittest.main()
