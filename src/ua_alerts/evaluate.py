"""Separate validation-only selection from locked, final historical scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .features import load_data, build_dataset
from .models import ModelConfig, fit_predict_poisson, baseline_predictions
from .validation import select_models, save_selection, read_selection, scores, split_schedule


def config_from_dict(config):
    return ModelConfig(config["with_news"], config["alpha"], config["train_window"], config.get("level_offset", False))


def block_interval(y, a, b, seed=42):
    differences = np.abs(y - a) - np.abs(y - b)
    rng = np.random.default_rng(seed)
    n, block = len(y), 7
    values = []
    for _ in range(3000):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        indices = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        values.append(differences[indices].mean())
    return [float(x) for x in np.quantile(values, [.025, .975])]


def evaluate_locked(snapshots, batches, acquisition, out, selection):
    frozen_hash = hashlib.sha256((out / "selection.json").read_bytes()).hexdigest()
    frame, feature_audit = build_dataset(snapshots, batches)
    start, end = pd.Timestamp(selection["test"]["start"]), pd.Timestamp(selection["test"]["end"])
    test = frame.index[(frame.target_day >= start) & (frame.target_day <= end)]
    if len(test) != selection["test"]["days"] or frame.loc[test, "y_final"].isna().any():
        raise ValueError("Incomplete final test outcomes")
    if frame.loc[test[0], "origin"] != pd.Timestamp(selection["selection_at"]):
        raise ValueError("Test boundary changed since selection")
    frame.to_csv(out / "features.csv", index=False)
    (out / "feature_audit.json").write_text(json.dumps(feature_audit, indent=2))
    predictions = frame.loc[test, ["origin", "target_day", "y_final"]].copy()
    for name in ("rolling28", "rolling7", "weekday8"):
        predictions[name] = [baseline_predictions(frame, frame.loc[[i]])[name] for i in test]
    traces = {}
    for family, winner in selection["family_winners"].items():
        config = config_from_dict(selection["trials"][winner]["config"])
        values, trace = [], []
        for index in test:
            row = frame.loc[[index]]
            p, audit = fit_predict_poisson(frame, row, config)
            values.append(p)
            trace.append({"origin": str(row.iloc[0].origin), "hyperparameters_selected_at": selection["selection_at"], **audit})
        predictions[family] = values
        traces[family] = trace
    selected_family = "history_news_poisson" if selection["selected_config"]["with_news"] else "history_poisson"
    predictions["selected_prediction"] = predictions[selected_family]
    predictions["selected_model"] = selection["selected_model"]
    # Prequential residual bands use only already published vintage outcomes.
    validation = pd.read_csv(out / "validation_predictions.csv", parse_dates=["origin", "target_day", "label_available_at"])
    previous = [(r["label_available_at"], abs(r["y_train_vintage"] - r[selection["selected_model"]]))
                for _, r in validation.iterrows()]
    lower, upper = [], []
    for i in test:
        origin = frame.loc[i, "origin"]
        residuals = [error for stamp, error in previous if stamp < origin][-56:]
        q = float(np.quantile(residuals, .9, method="higher"))
        p = predictions.loc[i, "selected_prediction"]
        lower.append(max(0., np.floor(p - q)))
        upper.append(np.ceil(p + q))
        previous.append((frame.loc[i, "label_available_at"], abs(frame.loc[i, "y_train_vintage"] - p)))
    predictions["interval90_lower"], predictions["interval90_upper"] = lower, upper
    predictions.to_csv(out / "predictions.csv", index=False)
    (out / "training_audit.json").write_text(json.dumps(traces[selected_family], indent=2))
    (out / "all_test_fit_audit.json").write_text(json.dumps(traces, indent=2))
    names = ["rolling28", "rolling7", "weekday8", *selection["family_winners"]]
    y = predictions.y_final.to_numpy()
    test_scores = {name: scores(y, predictions[name].to_numpy()) for name in names}
    val_frame = pd.read_csv(out / "selection_features.csv", parse_dates=["origin", "target_day", "label_available_at", "target_end"])
    val_start = pd.Timestamp(selection["validation"]["start"])
    first_val_origin = val_start - pd.Timedelta(hours=12)
    initial_train = val_frame[(val_frame.target_end < first_val_origin) & (val_frame.label_available_at < first_val_origin)]
    val_period = frame[(frame.target_day >= val_start) & (frame.target_day <= pd.Timestamp(selection["validation"]["end"]))]
    summary = {
        "schema_version": 2, "task": "Kyiv City next UTC calendar-day alert-start count; issued at preceding day 12:00 UTC",
        "data_as_of": acquisition["as_of"], "last_alert_snapshot": snapshots[-1]["commit_at"],
        "latest_scored_target": str(end), "data_rows": len(frame), "alert_snapshots": len(snapshots),
        "news_batches": len(batches), "news_articles_with_publication_and_title": sum(len(b["records"]) for b in batches),
        "origins_with_text_fraction": float((frame.news_titles3 > 0).mean()),
        "train_initial_target_start": str(frame.iloc[0].target_day),
        "training_rows_at_first_validation_origin": len(initial_train),
        "validation": selection["validation"], "validation_trials": selection["trials"],
        "validation_benchmarks": selection["benchmarks"], "test": selection["test"],
        "selected_model": selection["selected_model"], "selected_config": selection["selected_config"],
        "selection_at": selection["selection_at"], "selection_metric": selection["selection_metric"],
        "selection_pool": selection["selection_pool"], "test_metrics": test_scores,
        "prediction_range": {"min": float(predictions.selected_prediction.min()), "max": float(predictions.selected_prediction.max()),
                             "std": float(predictions.selected_prediction.std(ddof=0)),
                             "distinct_rounded_2dp": int(predictions.selected_prediction.round(2).nunique())},
        "feature_budget": {"history_numeric_cap": 10, "news_numeric_addition": 3, "text_cap": 50,
                           "text_per_training_row_cap": "floor(n_train/3)",
                           "selected_total_min": min(t["total_feature_count"] for t in traces[selected_family]),
                           "selected_total_max": max(t["total_feature_count"] for t in traces[selected_family])},
        "test_label_revision_days": int((frame.loc[test, "y_train_vintage"] != frame.loc[test, "y_final"]).sum()),
        "validation_label_revision_days": int((val_period.y_train_vintage != val_period.y_final).sum()),
        "interval90": {"empirical_coverage": float(np.mean((y >= lower) & (y <= upper))),
                       "mean_width": float(np.mean(np.array(upper) - lower)), "method": "past available absolute residuals; exploratory"},
        "selection_unchanged_during_test": frozen_hash == hashlib.sha256((out / "selection.json").read_bytes()).hexdigest(),
        "historical_holdout_reused": True, "holdout_note": selection["holdout_note"],
        "versions": {"python": __import__("sys").version.split()[0], "pandas": pd.__version__, "numpy": np.__version__, "sklearn": __import__("sklearn").__version__},
    }
    if "history_news_poisson" in predictions and "history_poisson" in predictions:
        summary["news_minus_history_mae"] = test_scores["history_news_poisson"]["MAE"] - test_scores["history_poisson"]["MAE"]
        summary["news_minus_history_mae_block_bootstrap95"] = block_interval(y, predictions.history_news_poisson.to_numpy(), predictions.history_poisson.to_numpy())
    assert summary["selection_unchanged_during_test"]
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    plot_predictions(predictions, test_scores, selected_family, out)
    print(json.dumps({k: summary[k] for k in ["selected_model", "test_metrics", "prediction_range", "feature_budget", "historical_holdout_reused"]}, indent=2), flush=True)
    return summary


def plot_predictions(predictions, test_scores, selected_family, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    dates = predictions.target_day.dt.tz_localize(None)
    axes[0].plot(dates, predictions.y_final, color="#172b4d", lw=2, marker="o", ms=4, label="Observed")
    axes[0].plot(dates, predictions.rolling28, color="#9da6ae", label="28-day benchmark")
    axes[0].plot(dates, predictions[selected_family], color="#d5911c", lw=2, label="Validation-selected Poisson")
    axes[0].set(title="Kyiv city • locked final evaluation (reused historical holdout)", ylabel="Alert starts / UTC day")
    axes[0].legend(ncol=2, frameon=False)
    axes[0].grid(alpha=.15)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    names = list(test_scores)
    axes[1].barh([n.replace("_", " ") for n in names], [test_scores[n]["MAE"] for n in names], color="#3274a1")
    axes[1].set(xlabel="Mean absolute error • lower is better", xlim=(0, max(test_scores[n]["MAE"] for n in names) * 1.12))
    axes[1].grid(axis="x", alpha=.15)
    fig.savefig(out / "evaluation.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--test-days", type=int, default=28)
    parser.add_argument("--validation-days", type=int, default=28)
    parser.add_argument("--stage", choices=["select", "test", "all"], default="all")
    args = parser.parse_args()
    acquisition = json.loads((args.data_dir / "acquisition.json").read_text())
    if acquisition["errors"]:
        raise ValueError("Resolve acquisition failures first")
    snapshots, batches = load_data(args.data_dir)
    if args.stage in ("select", "all"):
        result = select_models(snapshots, batches, test_days=args.test_days, validation_days=args.validation_days)
        save_selection(args.out_dir, result)
        print("Locked validation-only choice:", result[0]["selected_model"], flush=True)
    if args.stage in ("test", "all"):
        selection = read_selection(args.out_dir)
        if selection["test"]["days"] != args.test_days or selection["validation"]["days"] != args.validation_days:
            raise ValueError("Split settings differ from locked selection")
        scheduled_start, scheduled_end, _ = split_schedule(snapshots, args.test_days)
        if scheduled_start != pd.Timestamp(selection["test"]["start"]) or scheduled_end != pd.Timestamp(selection["test"]["end"]):
            raise ValueError("Data vintage changed since selection")
        evaluate_locked(snapshots, batches, acquisition, args.out_dir, selection)


if __name__ == "__main__":
    main()
