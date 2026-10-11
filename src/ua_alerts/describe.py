"""Descriptive analysis restricted to the initial training period."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import eligible_training


def main():
    out = Path("reports")
    frame = pd.read_csv(out / "features.csv", parse_dates=["origin", "target_day", "target_end", "label_available_at"])
    metrics = json.loads((out / "metrics.json").read_text())
    cutoff = pd.Timestamp(metrics["validation"]["start"]) - pd.Timedelta(hours=12)
    train = eligible_training(frame, cutoff)
    series = train.set_index("target_day").y_train_vintage
    weekdays = train.assign(weekday=train.target_day.dt.dayofweek).groupby("weekday").y_train_vintage.agg(["mean", "count"])
    acf = [series.autocorr(lag) for lag in range(1, 15)]
    description = {"description_as_of": str(cutoff), "rows": len(train),
                   "daily_counts": series.describe().to_dict(), "autocorrelation_lags1to14": acf,
                   "weekday_mean_and_observations": weekdays.to_dict("index")}
    (out / "training_description.json").write_text(json.dumps(description, indent=2))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    dates = series.index.tz_localize(None)
    axes[0].plot(dates, series.values, color="#bbc7d4", label="Recorded daily starts")
    axes[0].plot(dates, series.rolling(7).mean(), color="#256191", lw=2, label="7-day mean")
    axes[0].set(title="Initial training period • descriptive analysis", ylabel="Alert starts / UTC day")
    axes[0].legend(frameon=False, ncol=2)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[1].bar(np.arange(7), weekdays.reindex(range(7))["mean"], color="#256191")
    axes[1].set(xticks=np.arange(7), xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], ylabel="Mean starts")
    axes[2].bar(np.arange(1, 15), acf, color="#d5911c")
    axes[2].axhline(0, color="#666", lw=.8)
    axes[2].set(xlabel="Lag in days", ylabel="Autocorrelation", xticks=range(1, 15))
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    fig.savefig(out / "training_analysis.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
