"""Issue an auditable forecast at an explicit origin in the frozen vintage."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .models import ModelConfig, fit_predict_poisson
from .validation import read_selection
from .features import load_data, asof_features, build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", help="Timezone-aware instant; default is latest noon before data cutoff")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--out", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    acquisition = json.loads((args.data_dir / "acquisition.json").read_text())
    if acquisition["errors"]:
        raise ValueError("Incomplete acquisition")
    cutoff = pd.Timestamp(acquisition["as_of"])
    origin = pd.Timestamp(args.origin) if args.origin else cutoff.floor("D") + pd.Timedelta(hours=12)
    if not args.origin and origin > cutoff:
        origin -= pd.Timedelta(days=1)
    if origin.tzinfo is None:
        raise ValueError("Specify a timezone, for example 2026-10-09T12:00:00Z")
    origin = origin.tz_convert("UTC")
    if origin > cutoff:
        raise ValueError("Origin exceeds the frozen vintage; acquire newer data first")
    if origin.hour != 12 or origin.minute or origin.second or origin.microsecond:
        raise ValueError("This target definition requires an origin at 12:00 UTC")
    selection = read_selection(args.report_dir)
    if pd.Timestamp(selection["selection_at"]) > origin:
        raise ValueError("This model selection was not available at the requested origin")
    snapshots, batches = load_data(args.data_dir)
    f, titles, audit = asof_features(origin, snapshots, batches)
    row = pd.DataFrame([{**f, "titles": titles, "origin": origin,
                         "target_day": origin.floor("D") + pd.Timedelta(days=1)}])
    frame, _ = build_dataset(snapshots, batches)
    chosen = selection["selected_config"]
    config = ModelConfig(chosen["with_news"], chosen["alpha"], chosen["train_window"], chosen.get("level_offset", False))
    p, fit_audit = fit_predict_poisson(frame, row, config)
    result = {"origin": str(origin), "target_day": str(row.iloc[0].target_day),
              "predicted_count": p, "model": selection["selected_model"], "training_rows": fit_audit["training_rows"],
              "max_train_label_available_at": fit_audit["max_train_label_available_at"],
              "model_selected_at": selection["selection_at"], "audit": audit, "fit_audit": fit_audit,
              "note": "Expected recorded alert starts for civilian planning; official warnings determine protective action"}
    output = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
