"""
Runs all 5 heuristics on every instance of every provided dataset,
computes bins-used, lower bound, and %gap-over-lower-bound, and
writes:
  - results_detailed.csv   (one row per instance x heuristic)
  - results_summary.csv    (one row per dataset x heuristic, mean %gap)
"""

import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from heuristics import HEURISTICS
from simulator import lower_bound, online_bin_packing, falkenauer_utility

CURRENT_DIR = Path(__file__).resolve().parent
FINAL_DIR = CURRENT_DIR.parent

DATA_DIR = FINAL_DIR / "data"          
OUTPUT_DIR = CURRENT_DIR              

DATASETS = {}
for file_path in glob.glob(os.path.join(DATA_DIR, "*.json")):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    DATASETS[dataset_name] = file_path


def main():
    detailed_rows = []

    if not DATASETS:
        print(f"No JSON datasets found in {DATA_DIR}!")
        return

    print(f"Found {len(DATASETS)} datasets in {DATA_DIR}: {list(DATASETS.keys())}")

    for dataset_name, path in DATASETS.items():
        with open(path) as f:
            data = json.load(f)

        for inst_name, inst in data.items():
            capacity = inst["capacity"]
            items = inst["items"]
            lb = lower_bound(items, capacity)

            for heur_name, heur_fn in HEURISTICS.items():
                t0 = time.time()
                should_sort = heur_name not in ["First-Fit", "Best-Fit"]
                bins_used, bins_remain = online_bin_packing(items, capacity, heur_fn, sort_items=should_sort)
                elapsed = time.time() - t0

                gap_pct = (bins_used - lb) / lb * 100.0
                utility = falkenauer_utility(bins_remain, capacity, k=2.0)

                detailed_rows.append(
                    {
                        "dataset": dataset_name,
                        "instance": inst_name,
                        "num_items": inst["num_items"],
                        "capacity": capacity,
                        "heuristic": heur_name,
                        "bins_used": bins_used,
                        "falkenauer_utility": round(utility, 4),
                        "lower_bound": round(lb, 3),
                        "gap_pct": round(gap_pct, 4),
                        "time_sec": round(elapsed, 4),
                    }
                )

            print(f"done: {dataset_name} / {inst_name}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    detailed = pd.DataFrame(detailed_rows)
    detailed.to_csv(
        os.path.join(OUTPUT_DIR, "results_detailed.csv"), index=False
    )

    summary = (
        detailed.groupby(["dataset", "heuristic"])
        .agg(
            mean_gap_pct=("gap_pct", "mean"),
            std_gap_pct=("gap_pct", "std"),
            mean_utility=("falkenauer_utility", "mean"),
            mean_bins_used=("bins_used", "mean"),
            mean_time_sec=("time_sec", "mean"),
            n_instances=("instance", "count"),
        )
        .reset_index()
        .sort_values(["dataset", "mean_gap_pct"])
    )

    summary.to_csv(os.path.join(OUTPUT_DIR, "results_summary.csv"), index=False)

    print("\n=== SUMMARY (lower gap_pct = better, higher utility = better) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()