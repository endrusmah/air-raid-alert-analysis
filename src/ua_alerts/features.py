"""All features are built against a strict, timezone-aware forecast origin."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .acquire import REGIONS


def load_data(data_dir: Path):
    snapshots = [json.loads(p.read_text()) for p in sorted((data_dir / "snapshots").glob("*.json"))]
    snapshots.sort(key=lambda s: pd.Timestamp(s["available_at"]))
    batches = [json.loads(p.read_text()) for p in sorted((data_dir / "news").glob("*.json"))]
    return snapshots, batches


def latest_snapshot(snapshots, origin):
    eligible = [s for s in snapshots if pd.Timestamp(s["available_at"]) < origin]
    if not eligible:
        raise ValueError("No alert snapshot available before forecast origin")
    return eligible[-1]


def day_count(snapshot, day, region="Kyiv City"):
    key = str(pd.Timestamp(day))
    if key not in snapshot["counts"]:
        raise ValueError(f"Day {key} outside retained snapshot coverage")
    return snapshot["counts"][key][region]


def asof_features(origin: pd.Timestamp, snapshots, batches):
    if origin.tzinfo is None:
        raise ValueError("Forecast origin must be timezone-aware")
    origin = origin.tz_convert("UTC")
    snapshot = latest_snapshot(snapshots, origin)
    complete_day = min(origin.floor("D") - pd.Timedelta(days=1),
                       pd.Timestamp(snapshot["commit_at"]).floor("D") - pd.Timedelta(days=1),
                       pd.Timestamp(snapshot["max_start"]).floor("D") - pd.Timedelta(days=1))
    features = {"history_age_days": (origin.floor("D") - complete_day).days}
    histories = {}
    for region in REGIONS:
        values = np.array([day_count(snapshot, complete_day - pd.Timedelta(days=i), region)
                           for i in range(28)], dtype=float)
        histories[region] = values
    city = histories["Kyiv City"]
    neighbors = np.mean([histories[r] for r in REGIONS if r != "Kyiv City"], axis=0)
    features.update(kyiv_lag1=city[0], kyiv_lag7=city[6], kyiv_mean3=city[:3].mean(),
                    kyiv_mean7=city[:7].mean(), kyiv_mean28=city.mean(),
                    neighbors_lag1=neighbors[0], neighbors_mean7=neighbors[:7].mean())
    target_day = origin.floor("D") + pd.Timedelta(days=1)
    # Calendar values of the target day are known at origin (UTC accounting).
    features["weekday_sin"] = np.sin(2 * np.pi * target_day.dayofweek / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * target_day.dayofweek / 7)

    eligible_batches = [b for b in batches if
                        origin - pd.Timedelta(days=7) <= pd.Timestamp(b["batch_at"]) < origin
                        and pd.Timestamp(b["available_at"]) < origin]
    records = []
    for batch in eligible_batches:
        for record in batch["records"]:
            pub, avail = pd.Timestamp(record["published_at"]), pd.Timestamp(record["available_at"])
            if (origin - pd.Timedelta(days=3) <= pub < origin and avail < origin
                    and pd.Timestamp(record["indexed_at"]) < origin):
                records.append(record)
    records.sort(key=lambda r: (pd.Timestamp(r["available_at"]), r["id"]))
    unique, seen = [], set()
    for record in records:
        # Within-origin deduplication avoids using future syndication/revisions.
        key = re.sub(r"\s+", " ", record["title"].casefold()).strip()
        if key not in seen:
            seen.add(key)
            unique.append(record)
    features["news_titles3"] = len(unique)
    features["news_tone3"] = float(np.mean([r["tone"] for r in unique])) if unique else 0.
    conflict_count = sum(any(t in r["themes"] for t in ("ARMEDCONFLICT", "MILITARY", "WEAPONS"))
                         for r in unique)
    features["news_conflict_fraction3"] = conflict_count / len(unique) if unique else 0.
    titles = " . ".join(r["title"] for r in unique)
    audit = {"alert_snapshot_sha": snapshot["commit"],
             "alert_available_at": snapshot["available_at"],
             "last_complete_history_day": str(complete_day),
             "max_news_published_at": max((r["published_at"] for r in unique), default=None),
             "max_news_available_at": max((r["available_at"] for r in unique), default=None),
             "max_news_indexed_at": max((r["indexed_at"] for r in unique), default=None),
             "max_batch_available_at": max((b["available_at"] for b in eligible_batches), default=None),
             "news_ids": [r["id"] for r in unique]}
    for field in ["alert_available_at", "max_news_published_at", "max_news_available_at",
                  "max_news_indexed_at", "max_batch_available_at"]:
        if audit[field] is not None:
            assert pd.Timestamp(audit[field]) < origin, field
    return features, titles, audit


def build_dataset(snapshots, batches):
    if not snapshots or not batches:
        raise ValueError("Real snapshots and news data are required")
    first = pd.Timestamp(snapshots[0]["available_at"]).floor("D") + pd.Timedelta(days=2)
    last_snapshot = snapshots[-1]
    # Never assume the last event marks a complete day; only use the prior day.
    last = min(pd.Timestamp(last_snapshot["commit_at"]).floor("D"),
               pd.Timestamp(last_snapshot["max_start"]).floor("D")) - pd.Timedelta(days=1)
    rows, audits = [], []
    for target_day in pd.date_range(first, last, freq="D"):
        origin = target_day - pd.Timedelta(hours=12)
        features, titles, audit = asof_features(origin, snapshots, batches)
        target_end = target_day + pd.Timedelta(days=1)
        # A training label is frozen in the first snapshot that covers the whole
        # target day; delayed labels cannot enter a fit until their availability.
        label_snapshots = [s for s in snapshots if pd.Timestamp(s["available_at"]) > target_end
                           and pd.Timestamp(s["max_start"]) >= target_end
                           and str(target_day) in s["counts"]]
        if not label_snapshots:
            continue
        label_snapshot = label_snapshots[0]
        final_count = day_count(last_snapshot, target_day) if str(target_day) in last_snapshot["counts"] else np.nan
        rows.append({"origin": origin, "target_day": target_day, "target_end": target_end,
                     "label_available_at": pd.Timestamp(label_snapshot["available_at"]),
                     "y_train_vintage": day_count(label_snapshot, target_day),
                     "y_final": final_count, "titles": titles, **features})
        audits.append({"origin": str(origin), "target_day": str(target_day),
                       "label_available_at": label_snapshot["available_at"],
                       "label_snapshot_sha": label_snapshot["commit"], **audit})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No fully observed target days")
    return frame, audits


def eligible_training(frame, fit_at):
    # Both checks are necessary: a completed outcome may still be unpublished.
    return frame[(frame.origin < fit_at) & (frame.target_end < fit_at)
                 & (frame.label_available_at < fit_at)]
