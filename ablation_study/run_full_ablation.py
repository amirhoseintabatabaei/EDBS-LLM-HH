"""
Comprehensive ablation study runner for EDBS-LLM-HH.

Extends the single built-in ablation mode in main.py (which only tests 4
hand-picked variants) into a full one-factor-at-a-time (OFAT) sensitivity
study over every independently controllable design choice in the paper,
run across multiple datasets and multiple random seeds, with a paired
Wilcoxon signed-rank significance test against the "Full" baseline for
every variant.

See ABLATION_PLAN.md in this folder for the full research design and the
justification for each factor/level choice.

Usage:
    cd EDBS-LLM-HH
    python ablation_study/run_full_ablation.py \
        --datasets OR2 Weibull_10k Shift_Abrupt \
        --seeds 7 13 21 \
        --batch-size 50 \
        --initial-pool-size 10 \
        --output-dir ablation_study/results

Notes:
    - Each variant differs from the "Full" baseline in exactly ONE factor
      (classic OFAT design), so any change in a metric can be attributed
      to that single factor rather than a confound.
    - Every variant is run with the SAME set of seeds as the baseline, so
      the per-(instance, seed) results are paired -- this lets us run a
      paired Wilcoxon signed-rank test on the gap% differences instead of
      an unpaired test, which is considerably more powerful for the small
      sample sizes typical of expensive LLM-in-the-loop runs.
    - This script calls the real engine and therefore consumes real LLM
      API quota. Start with 1-2 datasets and 3 seeds to sanity check
      before running the full grid.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, replace
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (  # noqa: E402
    EngineConfig,
    load_dataset,
    logger,
    run_on_dataset_multi_seed,
)

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Factor / level definitions
#
# Every entry is (variant_label, overrides) where `overrides` is applied on
# top of the "Full" baseline EngineConfig. Group names are informational
# only (used for grouping the report); the runner treats every variant as
# an independent, single-factor experiment against the baseline.
# ---------------------------------------------------------------------------

BASELINE_LABEL = "Full (baseline)"

FACTOR_GROUPS: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
    "A. Portfolio size (K_H)": [
        ("K_H=3", {"portfolio_size": 3}),
        ("K_H=4 (default)", {"portfolio_size": 4}),
        ("K_H=5", {"portfolio_size": 5}),
        ("K_H=6", {"portfolio_size": 6}),
    ],
    "B. Batch ordering": [
        ("Descending sort (default)", {"sort_batch_descending": True}),
        ("No sort (arrival order)", {"sort_batch_descending": False}),
    ],
    "C. Diversity-preserving crossover": [
        ("Enabled, p=0.3 (default)", {"use_diversity_preservation_crossover": True, "diversity_preservation_prob": 0.3}),
        ("Disabled", {"use_diversity_preservation_crossover": False}),
        ("Enabled, p=0.15", {"use_diversity_preservation_crossover": True, "diversity_preservation_prob": 0.15}),
        ("Enabled, p=0.5", {"use_diversity_preservation_crossover": True, "diversity_preservation_prob": 0.5}),
    ],
    "D. Medium-severity crossover pairing": [
        ("Leader pairing (default)", {"medium_crossover_leader_pairing": True}),
        ("Random pairing", {"medium_crossover_leader_pairing": False}),
    ],
    "E. Event-driven vs. periodic triggering": [
        ("Event-driven (default)", {"use_trigger_monitor": True}),
        ("Periodic, every 3 batches", {"use_trigger_monitor": False, "periodic_trigger_interval": 3}),
        ("Periodic, every 5 batches", {"use_trigger_monitor": False, "periodic_trigger_interval": 5}),
        ("Periodic, every 10 batches", {"use_trigger_monitor": False, "periodic_trigger_interval": 10}),
    ],
    "F. Trigger EWMA smoothing (alpha)": [
        ("alpha=0.3 (slower)", {"trigger_alpha": 0.3}),
        ("alpha=0.5 (default)", {"trigger_alpha": 0.5}),
        ("alpha=0.7 (faster)", {"trigger_alpha": 0.7}),
    ],
    "G. Quality-degradation sensitivity (lambda)": [
        ("lambda=0.25 (more sensitive)", {"trigger_threshold_mult": 0.25}),
        ("lambda=0.5 (default)", {"trigger_threshold_mult": 0.5}),
        ("lambda=1.0 (less sensitive)", {"trigger_threshold_mult": 1.0}),
    ],
    "H. Distribution-shift sensitivity (tau_z)": [
        ("tau_z=0.5 (more sensitive)", {"trigger_shift_z": 0.5}),
        ("tau_z=0.8 (default)", {"trigger_shift_z": 0.8}),
        ("tau_z=1.2 (less sensitive)", {"trigger_shift_z": 1.2}),
    ],
    "I. Trigger cooldown (batches)": [
        ("cooldown=3", {"trigger_cooldown_batches": 3}),
        ("cooldown=6 (default)", {"trigger_cooldown_batches": 6}),
        ("cooldown=10", {"trigger_cooldown_batches": 10}),
    ],
    "J. Self-healing attempts": [
        ("1 attempt", {"max_step_fix_attempts": 1}),
        ("3 attempts (default)", {"max_step_fix_attempts": 3}),
        ("5 attempts", {"max_step_fix_attempts": 5}),
    ],
    "K. Cumulative (leave-everything-out)": [
        (
            "Bare-bones (no diversity crossover, periodic trigger @5, no leader pairing)",
            {
                "use_diversity_preservation_crossover": False,
                "medium_crossover_leader_pairing": False,
                "use_trigger_monitor": False,
                "periodic_trigger_interval": 5,
                "sort_batch_descending": False,
            },
        ),
    ],
}

# Factor "E. Initial pool size" is handled separately below since it's a
# run_on_dataset_multi_seed argument, not an EngineConfig field.
INITIAL_POOL_SIZE_LEVELS = [5, 10, 15]


def build_baseline_config() -> EngineConfig:
    return EngineConfig()  # every field at its paper-default value


def gap_pct(gap: float, lower_bound: float) -> float:
    return (gap / lower_bound) * 100.0 if lower_bound > 0 else 0.0


def run_variant(
    dataset: Dict[str, Dict],
    dataset_name: str,
    variant_label: str,
    config: EngineConfig,
    seeds: List[int],
    batch_size: int,
    initial_pool_size: int,
    output_dir: str,
) -> Dict[int, Dict[str, Dict]]:
    variant_dir = os.path.join(
        output_dir, dataset_name, variant_label.replace(" ", "_").replace("/", "_").replace(",", "")
    )
    logger.info(f"\n>>> [{dataset_name}] Running variant: {variant_label}")
    result = run_on_dataset_multi_seed(
        dataset=dataset,
        seeds=seeds,
        batch_size=batch_size,
        initial_pool_size=initial_pool_size,
        sleep_seconds=0,
        output_dir=variant_dir,
        config=config,
    )
    return result["per_seed"]


def flatten_rows(
    dataset_name: str,
    factor_group: str,
    variant_label: str,
    per_seed: Dict[int, Dict[str, Dict]],
) -> List[Dict[str, Any]]:
    rows = []
    for seed, instances in per_seed.items():
        for inst_name, m in instances.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "factor_group": factor_group,
                    "variant": variant_label,
                    "seed": seed,
                    "instance": inst_name,
                    "bins": m["bins"],
                    "gap": m["gap"],
                    "gap_pct": gap_pct(m["gap"], m["lower_bound"]),
                    "utilization": m["utilization"],
                    "api_calls": m.get("api_calls", 0),
                }
            )
    return rows


def paired_wilcoxon_p(baseline_rows: List[Dict], variant_rows: List[Dict], metric: str) -> Any:
    """
    Pairs baseline and variant rows on (seed, instance) and runs a paired
    Wilcoxon signed-rank test on the given metric. Returns the p-value, or
    None if scipy is unavailable, there are too few pairs, or all
    differences are zero.
    """
    if not _HAVE_SCIPY:
        return None

    baseline_map = {(r["seed"], r["instance"]): r[metric] for r in baseline_rows}
    variant_map = {(r["seed"], r["instance"]): r[metric] for r in variant_rows}
    common_keys = sorted(set(baseline_map) & set(variant_map))

    if len(common_keys) < 5:
        return None

    a = [baseline_map[k] for k in common_keys]
    b = [variant_map[k] for k in common_keys]
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diffs):
        return 1.0

    try:
        _, p = wilcoxon(a, b)
        return float(p)
    except ValueError:
        return None


def summarize(rows: List[Dict[str, Any]], baseline_rows_by_dataset: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    import statistics

    grouped: Dict[Tuple[str, str, str], List[Dict]] = {}
    for r in rows:
        key = (r["dataset"], r["factor_group"], r["variant"])
        grouped.setdefault(key, []).append(r)

    summary_rows = []
    for (dataset_name, factor_group, variant_label), group_rows in grouped.items():
        gap_vals = [r["gap_pct"] for r in group_rows]
        util_vals = [r["utilization"] for r in group_rows]
        api_vals = [r["api_calls"] for r in group_rows]

        baseline_rows = baseline_rows_by_dataset.get(dataset_name, [])
        p_gap = paired_wilcoxon_p(baseline_rows, group_rows, "gap_pct") if variant_label != BASELINE_LABEL else None

        summary_rows.append(
            {
                "dataset": dataset_name,
                "factor_group": factor_group,
                "variant": variant_label,
                "n_obs": len(group_rows),
                "mean_gap_pct": statistics.mean(gap_vals) if gap_vals else float("nan"),
                "std_gap_pct": statistics.stdev(gap_vals) if len(gap_vals) > 1 else 0.0,
                "mean_utilization": statistics.mean(util_vals) if util_vals else float("nan"),
                "std_utilization": statistics.stdev(util_vals) if len(util_vals) > 1 else 0.0,
                "mean_api_calls": statistics.mean(api_vals) if api_vals else float("nan"),
                "std_api_calls": statistics.stdev(api_vals) if len(api_vals) > 1 else 0.0,
                "wilcoxon_p_vs_baseline": p_gap,
                "significant_at_0.05": (p_gap is not None and p_gap < 0.05),
            }
        )
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True,
                         help="Dataset names (without .json), matching files in data/, e.g. OR2 Weibull_10k Shift_Abrupt")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21],
                         help="Random seeds to run every variant with (paired across variants).")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--initial-pool-size", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    parser.add_argument("--factor-groups", nargs="+", default=None,
                         help="Subset of factor group names to run (default: all). Use quotes, e.g. --factor-groups \"C. Diversity-preserving crossover\"")
    parser.add_argument("--skip-pool-size-sweep", action="store_true",
                         help="Skip the separate initial_pool_size sweep (it re-runs initialization multiple times and is the most expensive factor).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    baseline_config = build_baseline_config()

    all_rows: List[Dict[str, Any]] = []
    baseline_rows_by_dataset: Dict[str, List[Dict]] = {}

    groups_to_run = args.factor_groups or list(FACTOR_GROUPS.keys())

    for dataset_name in args.datasets:
        dataset_path = f"{dataset_name}.json"
        if not os.path.exists(os.path.join(args.data_dir, dataset_path)):
            logger.warning(f"Dataset file not found, skipping: {dataset_path}")
            continue
        dataset = load_dataset(dataset_path)

        # --- Baseline (Full) run, once per dataset ---
        baseline_per_seed = run_variant(
            dataset, dataset_name, BASELINE_LABEL, baseline_config,
            args.seeds, args.batch_size, args.initial_pool_size, args.output_dir,
        )
        baseline_rows = flatten_rows(dataset_name, "Baseline", BASELINE_LABEL, baseline_per_seed)
        all_rows.extend(baseline_rows)
        baseline_rows_by_dataset[dataset_name] = baseline_rows

        # --- Every OFAT variant ---
        for group_name in groups_to_run:
            for variant_label, overrides in FACTOR_GROUPS[group_name]:
                if variant_label == "K_H=4 (default)" or variant_label.endswith("(default)"):
                    # Same as baseline for every field except the one being
                    # highlighted; still run it explicitly so the report
                    # shows the full level range for that factor.
                    pass
                config = replace(baseline_config, **overrides)
                per_seed = run_variant(
                    dataset, dataset_name, variant_label, config,
                    args.seeds, args.batch_size, args.initial_pool_size, args.output_dir,
                )
                all_rows.extend(flatten_rows(dataset_name, group_name, variant_label, per_seed))

        # --- Initial pool size sweep (separate: not an EngineConfig field) ---
        if not args.skip_pool_size_sweep:
            for pool_size in INITIAL_POOL_SIZE_LEVELS:
                label = f"initial_pool_size={pool_size}" + (" (default)" if pool_size == args.initial_pool_size else "")
                per_seed = run_variant(
                    dataset, dataset_name, label, baseline_config,
                    args.seeds, args.batch_size, pool_size, args.output_dir,
                )
                all_rows.extend(flatten_rows(dataset_name, "L. Initial pool size", label, per_seed))

    # --- Write raw per-(seed, instance) rows ---
    raw_path = os.path.join(args.output_dir, "ablation_raw.csv")
    with open(raw_path, "w", newline="") as f:
        if all_rows:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    logger.info(f"[Raw results saved]: {raw_path}")

    # --- Summarize + significance test ---
    summary_rows = summarize(all_rows, baseline_rows_by_dataset)
    summary_path = os.path.join(args.output_dir, "ablation_summary.csv")
    with open(summary_path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    logger.info(f"[Summary saved]: {summary_path}")

    json_path = os.path.join(args.output_dir, "ablation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    if not _HAVE_SCIPY:
        logger.warning(
            "scipy is not installed -- significance testing was skipped. "
            "Install it with `pip install scipy` and re-run to get Wilcoxon p-values."
        )

    print("\n" + "=" * 100)
    print("ABLATION STUDY COMPLETE")
    print("=" * 100)
    print(f"Raw rows:      {raw_path}")
    print(f"Summary (csv): {summary_path}")
    print(f"Summary (json):{json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
