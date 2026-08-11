# Ablation Study Plan — EDBS-LLM-HH

This document specifies a comprehensive ablation study for the framework
described in the paper, grounded directly in the configurable knobs that
exist in `main.py` (`EngineConfig`, `TriggerMonitor`, and the
`initial_pool_size` argument). It extends the single built-in ablation
(`run_ablation_study`, 4 hand-picked variants) into a full factorial-style
sensitivity study, runnable with `ablation_study/run_full_ablation.py`.

## 1. Research questions

The paper's four contributions map directly onto four groups of questions:

| Paper contribution | Ablation question |
|---|---|
| Diversity-Aware Beam Search | How much does the fixed 4-champion lexicographic beam (Quality/Utilization/Speed/Diversity) depend on portfolio size `K_H` and batch ordering? |
| Dynamic Behavioral Diversity Metric | Does diversity-preserving crossover (and its probability `p_div`) meaningfully change portfolio quality/diversity, or is it a marginal add-on? |
| Event-Driven Adaptation | Does the *event-driven* trigger monitor actually beat a naive *periodic* (fixed-interval) LLM-calling schedule at matched or lower API cost? How sensitive is it to its own thresholds? |
| Self-Healing | How many self-correction attempts are actually needed before returns diminish? |

A fifth, cross-cutting question: **which single factor, when removed or
mis-tuned, hurts solution quality (gap%) or explodes API cost the most?**
This is answered by the "K. Cumulative" bare-bones variant and by ranking
all variants by effect size in the final report.

## 2. Factors and levels

All factors are varied **one at a time (OFAT)** against the same "Full"
baseline (`EngineConfig()` with every field at its paper-default value).
This isolates the effect of each design choice — a full factorial grid
across 12 factors would require thousands of expensive LLM-in-the-loop
runs and is not practical; OFAT is the standard, tractable design for this
regime and is what `run_full_ablation.py` implements.

| Group | Factor | Code field | Default | Levels tested |
|---|---|---|---|---|
| A | Portfolio size | `EngineConfig.portfolio_size` (newly exposed; was a hardcoded constant `K_H=4`) | 4 | 3, 4, 5, 6 |
| B | Batch ordering | `EngineConfig.sort_batch_descending` | True | True, False |
| C | Diversity-preserving crossover | `use_diversity_preservation_crossover`, `diversity_preservation_prob` | True, 0.3 | off; on @ 0.15 / 0.3 / 0.5 |
| D | Medium-severity crossover pairing | `medium_crossover_leader_pairing` | True | True, False |
| E | Event-driven vs. periodic triggering | `use_trigger_monitor`, `periodic_trigger_interval` | True | event-driven; periodic @ 3 / 5 / 10 |
| F | Trigger EWMA smoothing | `trigger_alpha` (newly exposed; was hardcoded in `TriggerMonitor()`) | 0.5 | 0.3, 0.5, 0.7 |
| G | Quality-degradation sensitivity (λ) | `trigger_threshold_mult` (newly exposed) | 0.5 | 0.25, 0.5, 1.0 |
| H | Distribution-shift sensitivity (τz) | `trigger_shift_z` (newly exposed) | 0.8 | 0.5, 0.8, 1.2 |
| I | Trigger cooldown | `trigger_cooldown_batches` (newly exposed) | 6 | 3, 6, 10 |
| J | Self-healing attempts | `max_step_fix_attempts` | 3 | 1, 3, 5 |
| K | Cumulative ("bare-bones") | combination of B, C, D, E | — | all disabled/periodic at once |
| L | Initial pool size | `run_on_dataset_multi_seed(initial_pool_size=...)` | 10 | 5, 10, 15 |

**Note on `EngineConfig` extensions:** `trigger_alpha`, `trigger_threshold_mult`,
`trigger_shift_z`, `trigger_cooldown_batches`, and `portfolio_size` did not
previously exist as `EngineConfig` fields — the trigger thresholds were
hardcoded inside `TriggerMonitor.__init__`'s default arguments, and the
portfolio size was a module-level constant `K_H` used directly inside
`HyperHeuristicEngine`. Both have been threaded through `EngineConfig` so
they can be varied per-run without editing `main.py` for every experiment.
**The added defaults are identical to the previous hardcoded values**, so
`EngineConfig()` reproduces the exact original behavior; only the ablation
runner takes advantage of the new flexibility.

### Factors intentionally *not* ablated (architectural constants)

- **Beam width / number of champions.** The lexicographic selection
  (`champion_selection`) picks exactly 4 champions (Quality, Utilization,
  Speed, Diversity) by construction — this is tied 1:1 to the four scoring
  criteria in the paper (`f1`, `f2`, `f3`, and the diversity score) and
  cannot be varied independently without adding new criteria. This is
  reported as a fixed architectural choice rather than a tunable factor.
- **`max_per_parent` (branching cap)** — a minor implementation constant
  (`MAX_PER_PARENT = 2`); left out to keep the study within a reasonable
  scope, flagged here for future work.

## 3. Datasets

To keep LLM API cost bounded while still covering the three regimes the
paper distinguishes, the ablation study runs on **three representative
datasets** rather than all eight:

| Dataset | Regime represented |
|---|---|
| `OR2` | Classical, stationary, small-scale (500 items) |
| `Weibull_10k` | Large-scale, stationary, skewed distribution |
| `Shift_Abrupt` | Non-stationary, distribution-shift stress test |

This directly targets the paper's own claim that distribution-shift
robustness and large-scale stationary performance are where the framework
differentiates itself (Section IV.C). Running the full grid on all eight
datasets is possible by passing more `--datasets` values, at proportionally
higher API cost.

## 4. Seeds and statistical protocol

- Every variant (baseline included) is run with the **same set of seeds**
  (default: `7, 13, 21` — override with `--seeds`), so results are
  **paired** at the `(dataset, instance, seed)` level.
- Because LLM outputs are stochastic even at fixed seed (temperature,
  routing between the two Gemini models, API non-determinism per the
  paper's own limitations section), 3 seeds is a practical minimum; 5
  seeds is preferable if API budget allows (`--seeds 7 13 21 42 99`).
- For each variant, `run_full_ablation.py` reports **mean ± std** of
  gap%, utilization, and API calls per run.
- For each variant, a **paired two-sided Wilcoxon signed-rank test**
  (`scipy.stats.wilcoxon`) is run against the baseline's gap% values,
  paired by `(seed, instance)`. Wilcoxon is used instead of a paired
  t-test because gap% is not guaranteed to be normally distributed at
  this sample size, and it is the standard non-parametric choice for
  paired algorithm-comparison data. Requires `scipy`
  (`pip install scipy`); the script degrades gracefully (skips
  significance testing, still reports means) if scipy isn't installed.
- Effects are flagged **significant at α = 0.05**. With only 3 seeds ×
  5 instances per dataset (15 pairs), the test is under-powered for
  small effects — this is stated explicitly in the report rather than
  overinterpreting non-significant results as "no effect."

## 5. Metrics reported per variant

| Metric | Why it matters |
|---|---|
| Mean gap % (`(bins − LB) / LB × 100`) | Primary solution-quality metric, matches Table I of the paper |
| Mean Falkenauer-style utilization | Secondary quality metric, matches Table I |
| Mean LLM API calls per run | The paper's efficiency claim (Table II) — an ablation that improves gap% but explodes API calls is not a free win |
| Wilcoxon p-value vs. baseline (gap%) | Statistical significance of the change |

Wall-clock runtime is intentionally *not* used as a primary comparison
metric across variants, since it is dominated by external LLM API latency
and network conditions rather than the algorithmic factor being studied;
`f3` (per-batch elapsed time) remains available in the raw per-run logs
for anyone who wants to analyze it separately.

## 6. Running it

```bash
cd EDBS-LLM-HH
pip install scipy   # optional, enables significance testing

python ablation_study/run_full_ablation.py \
    --datasets OR2 Weibull_10k Shift_Abrupt \
    --seeds 7 13 21 \
    --batch-size 50 \
    --initial-pool-size 10 \
    --output-dir ablation_study/results
```

Useful flags:

- `--factor-groups "C. Diversity-preserving crossover" "E. Event-driven vs. periodic triggering"`
  — run only a subset of factor groups (cheaper, for iterating on one
  question at a time).
- `--skip-pool-size-sweep` — skip the initial-pool-size sweep (Group L),
  which is the most expensive factor since it re-runs the zero-shot
  initialization phase from scratch for each level.

**Cost warning:** the full grid (12 factor groups × ~3 levels average ×
3 datasets × 3 seeds, plus the Group L pool-size sweep) is roughly
**100+ engine runs**. Even at the paper's reported 5.4–7.2 LLM calls per
run, this is on the order of 600–800 LLM API calls. Start with
`--factor-groups` on a single dataset and 1-2 seeds to sanity-check the
pipeline before committing to the full grid.

## 7. Outputs

Running the script produces, under `--output-dir`:

- `ablation_raw.csv` — one row per `(dataset, factor_group, variant, seed,
  instance)`, with `bins`, `gap`, `gap_pct`, `utilization`, `api_calls`.
  This is the ground truth for any further custom analysis (e.g. in a
  notebook or Excel).
- `ablation_summary.csv` / `ablation_summary.json` — one row per
  `(dataset, factor_group, variant)` with mean/std of each metric and the
  Wilcoxon p-value vs. the baseline.
- Full per-run logs, tree-search plots, and checkpoints under
  `<output-dir>/<dataset>/<variant>/` (same structure as a normal
  `run_on_dataset_multi_seed` call), for deep-diving into any specific
  variant that looks surprising.
