"""Select a compact Poisson configuration without exposing test rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance

from .features import build_dataset, eligible_training
from .models import ModelConfig, fit_predict_poisson, baseline_predictions

ALPHAS = (.01, .1, 1.)
WINDOWS = (56, 112, None)


def pipeline_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def scores(y, prediction):
    return {"MAE": float(mean_absolute_error(y, prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
            "Poisson_deviance": float(mean_poisson_deviance(y, prediction))}


def split_schedule(snapshots, test_days):
    last = snapshots[-1]
    last_day = min(pd.Timestamp(last["commit_at"]).floor("D"),
                   pd.Timestamp(last["max_start"]).floor("D")) - pd.Timedelta(days=1)
    test_start = last_day - pd.Timedelta(days=test_days - 1)
    return test_start, last_day, test_start - pd.Timedelta(hours=12)


def select_models(snapshots, batches, *, test_days=28, validation_days=28, candidates=None):
    test_start, test_end, selection_at = split_schedule(snapshots, test_days)
    # Physical separation: no test feature rows, test articles, test outcomes,
    # or later revisions are given to feature construction or selection.
    visible_snapshots = [s for s in snapshots if pd.Timestamp(s["available_at"]) < selection_at]
    visible_batches = [b for b in batches if pd.Timestamp(b["available_at"]) < selection_at]
    frame, feature_audit = build_dataset(visible_snapshots, visible_batches)
    frame = eligible_training(frame, selection_at).drop(columns=["y_final"]).reset_index(drop=True)
    if (frame.target_day >= test_start).any():
        raise AssertionError("Test row reached selection")
    validation_indices = frame.index[-validation_days:]
    if len(validation_indices) != validation_days or len(frame) < validation_days + 50:
        raise ValueError("Insufficient available train/validation history")
    y_val = frame.loc[validation_indices, "y_train_vintage"].to_numpy()
    if candidates is None:
        candidates = [ModelConfig(news, alpha, window, offset) for news in (False, True)
                      for offset in (False, True) for alpha in ALPHAS for window in WINDOWS]
    predictions = frame.loc[validation_indices, ["origin", "target_day", "label_available_at", "y_train_vintage"]].copy()
    trials, audits = {}, {}
    for config in candidates:
        values, trace = [], []
        for index in validation_indices:
            row = frame.loc[[index]]
            value, audit = fit_predict_poisson(frame, row, config)
            values.append(value)
            trace.append({"origin": str(row.iloc[0].origin), **audit})
        predictions[config.name] = values
        stats = scores(y_val, np.array(values))
        trials[config.name] = {"config": config.to_dict(), "scores": stats,
                              "features_min": min(t["total_feature_count"] for t in trace),
                              "features_max": max(t["total_feature_count"] for t in trace),
                              "training_rows_min": min(t["training_rows"] for t in trace),
                              "prediction_std": float(np.std(values))}
        audits[config.name] = trace
        print(f"Validation {config.name}: MAE={stats['MAE']:.4f}, features={trials[config.name]['features_max']}", flush=True)
    benchmarks = {}
    for name in ("rolling28", "rolling7", "weekday8"):
        predictions[name] = [baseline_predictions(frame, frame.loc[[i]])[name] for i in validation_indices]
        benchmarks[name] = scores(y_val, predictions[name].to_numpy())
    selected = min(trials, key=lambda name: trials[name]["scores"]["MAE"])
    families = {}
    for use_news, family in [(False, "history_poisson"), (True, "history_news_poisson")]:
        names = [name for name in trials if trials[name]["config"]["with_news"] == use_news]
        if names:
            families[family] = min(names, key=lambda name: trials[name]["scores"]["MAE"])
    selection = {
        "schema_version": 2, "pipeline_sha256": pipeline_hash(),
        "selection_at": str(selection_at), "test": {"start": str(test_start), "end": str(test_end), "days": test_days},
        "validation": {"start": str(frame.loc[validation_indices].target_day.min()),
                       "end": str(frame.loc[validation_indices].target_day.max()), "days": validation_days,
                       "labels": "first usable vintage, published before selection"},
        "selection_pool": "compact Poisson models, as requested; averages are separate benchmarks",
        "selection_metric": "validation MAE only", "selected_model": selected,
        "selected_config": trials[selected]["config"], "family_winners": families,
        "trials": trials, "benchmarks": benchmarks,
        "visible_snapshots": [s["commit"] for s in visible_snapshots],
        "visible_news_batches": [b["raw_sha256"] for b in visible_batches],
        "max_selection_target_end": str(frame.target_end.max()),
        "max_selection_label_available_at": str(frame.label_available_at.max()),
        "test_rows_seen_during_selection": 0,
        "historical_holdout_reused": True,
        "holdout_note": "Version 1 already scored this historical period; version 2 excludes it from tuning but does not make it a new unseen holdout",
    }
    assert pd.Timestamp(selection["max_selection_target_end"]) < selection_at
    assert pd.Timestamp(selection["max_selection_label_available_at"]) < selection_at
    return selection, frame, predictions, feature_audit, audits[selected]


def save_selection(out, result):
    selection, frame, predictions, feature_audit, fit_audit = result
    out.mkdir(parents=True, exist_ok=True)
    body = json.dumps(selection, indent=2)
    (out / "selection.json").write_text(body)
    (out / "selection.sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "\n")
    frame.to_csv(out / "selection_features.csv", index=False)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    (out / "selection_feature_audit.json").write_text(json.dumps(feature_audit, indent=2))
    (out / "validation_fit_audit.json").write_text(json.dumps(fit_audit, indent=2))


def read_selection(out):
    body = (out / "selection.json").read_bytes()
    expected = (out / "selection.sha256").read_text().strip()
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("Selection artifact changed after locking")
    selection = json.loads(body)
    if selection["pipeline_sha256"] != pipeline_hash():
        raise ValueError("Pipeline changed after model selection; select and validate again before testing")
    return selection
