"""Adversarial tests for timing, vintages, deduplication and fit boundaries."""
import copy
import unittest

import pandas as pd

from ua_alerts.acquire import REGIONS, tag
from ua_alerts.features import asof_features, eligible_training, latest_snapshot
from ua_alerts.models import ModelConfig, fit_predict_poisson


def fixture():
    origin = pd.Timestamp("2026-09-10T12:00Z")
    counts = {str(d): {region: 1 for region in REGIONS}
              for d in pd.date_range("2026-07-01", "2026-09-10", tz="UTC")}
    snapshot = {"commit": "past", "commit_at": "2026-09-10T04:00Z",
                "available_at": "2026-09-10T10:00Z", "max_start": "2026-09-10T03:00Z", "counts": counts}
    record = {"id": "past-title", "url": "https://example.org/past", "title": "Ukraine civilian shelters",
              "published_at": "2026-09-09T20:00Z", "indexed_at": "2026-09-10T00:00Z",
              "available_at": "2026-09-10T02:00Z", "tone": -2., "themes": "ARMEDCONFLICT;"}
    batch = {"batch_at": "2026-09-10T00:00Z", "available_at": "2026-09-10T02:00Z", "records": [record]}
    return origin, [snapshot], [batch]


class TemporalContract(unittest.TestCase):
    def test_future_alert_snapshot_cannot_change_features(self):
        origin, snapshots, batches = fixture()
        expected = asof_features(origin, snapshots, batches)
        future = copy.deepcopy(snapshots[0])
        future.update(commit="future", available_at="2026-09-10T13:00Z")
        for day in future["counts"].values():
            for region in REGIONS:
                day[region] = 99999
        self.assertEqual(expected, asof_features(origin, snapshots + [future], batches))

    def test_future_publication_ingestion_and_indexing_excluded(self):
        origin, snapshots, batches = fixture()
        expected = asof_features(origin, snapshots, batches)
        for field in ["published_at", "available_at", "indexed_at"]:
            bad = copy.deepcopy(batches[0]["records"][0])
            bad.update(id="future", title="future missile retrospective")
            bad[field] = "2026-09-10T13:00Z"
            augmented = copy.deepcopy(batches)
            augmented[0]["records"].append(bad)
            self.assertEqual(expected, asof_features(origin, snapshots, augmented))

    def test_equal_origin_is_excluded(self):
        origin, snapshots, batches = fixture()
        snapshots[0]["available_at"] = str(origin)
        with self.assertRaises(ValueError):
            latest_snapshot(snapshots, origin)

    def test_batch_not_available_even_if_article_claims_earlier(self):
        origin, snapshots, batches = fixture()
        batches[0]["available_at"] = "2026-09-10T13:00Z"
        features, titles, _ = asof_features(origin, snapshots, batches)
        self.assertEqual(features["news_titles3"], 0)
        self.assertEqual(titles, "")

    def test_syndication_deduplication_is_within_origin(self):
        origin, snapshots, batches = fixture()
        duplicate = copy.deepcopy(batches[0]["records"][0])
        duplicate["id"] = "duplicate"
        batches[0]["records"].append(duplicate)
        features, titles, _ = asof_features(origin, snapshots, batches)
        self.assertEqual(features["news_titles3"], 1)

    def test_target_day_and_delayed_label_are_purged(self):
        origin = pd.Timestamp("2026-09-10T12:00Z")
        frame = pd.DataFrame({
            "origin": pd.to_datetime(["2026-09-07T12:00Z"] * 4, utc=True),
            "target_end": pd.to_datetime(["2026-09-09T00:00Z", "2026-09-11T00:00Z",
                                           "2026-09-09T00:00Z", "2026-09-10T12:00Z"], utc=True),
            "label_available_at": pd.to_datetime(["2026-09-09T08:00Z", "2026-09-09T08:00Z",
                                                   "2026-09-11T08:00Z", "2026-09-09T08:00Z"], utc=True),
        })
        self.assertEqual(eligible_training(frame, origin).index.tolist(), [0])

    def test_news_only_in_test_does_not_fit_vocabulary(self):
        from test_models import synthetic_data
        train, row = synthetic_data()
        config = ModelConfig(True, .1, None)
        expected, _ = fit_predict_poisson(train, row, config)
        row["titles"] = "civilian shelter " + "futureunseenword " * 100
        actual, audit = fit_predict_poisson(train, row, config)
        self.assertNotIn("futureunseenword", audit["vocabulary"])
        self.assertAlmostEqual(expected, actual)

    def test_unknown_publication_is_not_inferred_from_url(self):
        self.assertIsNone(tag("<PAGE_TITLE>News</PAGE_TITLE>", "PAGE_PRECISEPUBTIMESTAMP"))
        self.assertEqual(tag("<PAGE_TITLE>Russia &amp; Ukraine</PAGE_TITLE>", "PAGE_TITLE"), "Russia & Ukraine")

    def test_future_counts_in_same_snapshot_do_not_change_features(self):
        origin, snapshots, batches = fixture()
        expected = asof_features(origin, snapshots, batches)
        snapshots[0]["counts"][str(origin.floor("D"))] = {region: 9999 for region in REGIONS}
        self.assertEqual(expected, asof_features(origin, snapshots, batches))

    def test_naive_origin_rejected(self):
        origin, snapshots, batches = fixture()
        with self.assertRaises(ValueError):
            asof_features(origin.tz_localize(None), snapshots, batches)


if __name__ == "__main__":
    unittest.main()
