# EDBS-LLM-HH

**Event-Driven Beam Search LLM Hyper-Heuristic for Online Bin Packing**

[![Paper](https://img.shields.io/badge/ICISE%202026-Paper-orange)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of the paper *"Event-Driven Beam Search Large
Language Model Hyper-Heuristic for Online Bin Packing"*, presented at the
12th International Conference on Industrial and Systems Engineering (ICISE
2026), Ferdowsi University of Mashhad.

> Amirhosein Tabatabaei, Farid Khoshalhan — Department of Industrial
> Engineering, K. N. Toosi University of Technology, Tehran, Iran.

## Overview

Large Language Model (LLM) hyper-heuristics can discover strong heuristics
for the Online Bin Packing Problem (OBPP), but existing approaches typically
require thousands to millions of LLM queries per run and freeze the
discovered heuristic before deployment. **EDBS-LLM-HH** addresses both
limitations with a generation-selection hyper-heuristic that:

- Maintains a portfolio of candidate heuristics and expands a **diversity-
  aware multi-criteria beam search** (quality, utilization, runtime, and
  behavioral diversity champions) at every incoming batch.
- Measures **behavioral diversity** with a pairwise decision-distance metric
  computed over a dynamic probe set, rather than surface-level code
  similarity.
- Runs an **event-driven trigger monitor** that watches for quality
  degradation, diversity collapse, and input distribution shifts, and only
  then invokes severity-matched LLM evolution (mutation, crossover, or
  zero-shot generation) plus a self-healing sandbox loop for runtime
  crashes.

Across eight synthetic benchmarks the method matches or outperforms
classical heuristics (Best-Fit, First-Fit) and LLM-based baselines
(FunSearch, EoH, MCTS-AHD, HSEvo) while requiring under 10 LLM API
calls per run — orders of magnitude fewer than prior LLM
hyper-heuristics.

## Repository layout

```
.
├── main.py                     # Core engine: search, diversity metric,
│                                # trigger monitor, LLM evolution, CLI runner
├── prompts/                    # LLM prompt templates used by main.py
│   ├── initial_heuristics.txt  #   zero-shot pool generation
│   ├── mutate.txt              #   low/medium/high mutation
│   ├── crossover.txt           #   recombining two heuristics
│   ├── generate_new.txt        #   zero-shot generation on trigger
│   ├── self_correct.txt        #   fixing a runtime crash (first attempt)
│   └── heavy_mutation.txt      #   fallback rewrite after repeated crashes
├── data/
│   └── datasetcreator.py       # Regenerates the benchmark datasets
├── baselines/                  # Independent evaluation of published
│   │                            # baseline heuristics, batched/online
│   ├── heuristics.py           #   First-Fit, Best-Fit, EoH, HSEvo,
│   │                            #   MCTS-AHD, FunSearch (OR & Weibull variants)
│   ├── simulator.py            #   shared online bin-packing simulator
│   └── run_benchmark.py        #   runs every heuristic on every dataset,
│                                #   writes results_detailed.csv / results_summary.csv
├── ablation_study/              # Comprehensive ablation study
│   ├── ABLATION_PLAN.md        #   full research design: factors, levels,
│   │                            #   datasets, seeds, statistical protocol
│   └── run_full_ablation.py    #   runner: OFAT sweep over every
│                                #   EngineConfig / TriggerMonitor knob,
│                                #   multi-seed, with paired Wilcoxon tests
├── requirements.txt
├── .env.example                 # Template for your Gemini API key(s)
└── README.md
```

Running the pipeline creates two additional (git-ignored) directories:
`bin_packing_outputs/` for standard runs and `ablation_results/` for the
ablation study mode. Running `baselines/run_benchmark.py` writes
`results_detailed.csv` and `results_summary.csv` into `baselines/`.

## Installation

```bash
git clone https://github.com/amirhoseintabatabaei/EDBS-LLM-HH.git
cd EDBS-LLM-HH
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### API keys

The engine calls Google Gemini (`gemini-3.1-flash-lite` and
`gemini-3.5-flash-lite`, used interchangeably at runtime) for heuristic
generation, mutation, crossover, and self-healing. Copy the template and
add your key(s):

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY_1=...
```

You can list multiple keys (`GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, ...);
`main.py` rotates across all account/model combinations automatically and
backs off with a cooldown if every combination is temporarily out of quota.

## Preparing datasets

Datasets are **not** committed to the repository by default (see
`.gitignore`) to keep it lightweight and to make regeneration explicit and
reproducible. Generate them locally with:

```bash
cd data
python datasetcreator.py
```

This writes one `data/<name>.json` file per dataset in the schema expected
by `main.py` and by `baselines/run_benchmark.py`:

```json
{
  "instance_key": {
    "capacity": 100,
    "num_items": 500,
    "items": [23, 87, ...]
  }
}
```

By default this produces:

- `OR1`–`OR4`: 5 instances x 500 items, capacity 100, item sizes ~
  DiscreteUniform[20, 100].
- `Weibull_5k` / `10k` / `20k` / `50k`: 5 instances of the given size,
  capacity 150, item sizes ~ Weibull(shape=4.5, scale=45), rounded and
  clipped to `[1, capacity]`.
- `Shift_Abrupt`, `Shift_Gradual`, `Shift_Abrupt_Large`: segmented
  distribution-shift datasets alternating between a low-size and a
  high-size range, either with an abrupt segment boundary or a gradual
  linear transition. Segment boundaries (`shift_points`) are stored under
  each instance's `shift_metadata` key **for offline analysis only** — the
  online engine never reads this field while packing items.

Edit the calls at the bottom of `data/datasetcreator.py` to change sizes,
seeds, or add/remove datasets. If you'd rather supply your own instances,
drop any `*.json` file following the same schema into `data/`.

## Running the hyper-heuristic

```bash
python main.py
```

You'll be prompted to choose:

1. **Standard Run** — runs every dataset found in `data/` sequentially,
   asking for an initial heuristic pool size, whether to sort each batch
   largest-to-smallest before packing, and a per-dataset batch size.
2. **Automated Ablation Study** — runs a single dataset across several
   internally-defined engine configurations for comparison.

You can also provide comma-separated random seeds (e.g. `7,13,42`) to
average results over multiple runs. If `data/` is empty, `main.py` falls
back to a small in-memory demo run.

Per-run outputs (packed states, metrics, checkpoints, and — if matplotlib
is available — plots) are written under `bin_packing_outputs/<dataset>/` or
`ablation_results/<dataset>/`.

## Running the baseline comparison

`baselines/` independently re-evaluates the published, best-performing
heuristics from Best-Fit, First-Fit, EoH, HSEvo, MCTS-AHD, and FunSearch
(both its OR-discovered and Weibull-discovered variants) against every
dataset in `data/`, using the exact same batched/online interaction
pattern as `main.py` (see `simulator.py`) for a fair comparison:

```bash
cd baselines
python run_benchmark.py
```

This writes `results_detailed.csv` (one row per instance x heuristic) and
`results_summary.csv` (one row per dataset x heuristic, with mean gap %,
mean Falkenauer utility, mean bins used, and mean runtime) into
`baselines/`.

## Running the ablation study

`main.py` ships a small built-in ablation mode (menu option 2, 4 hand-picked
variants). `ablation_study/` extends this into a full one-factor-at-a-time
sensitivity study over every independently tunable design choice —
portfolio size, batch ordering, diversity-preserving crossover (and its
probability), event-driven vs. periodic triggering, every trigger
threshold (EWMA smoothing, quality-degradation sensitivity,
distribution-shift sensitivity, cooldown), self-healing attempt budget,
and initial pool size — run across multiple datasets and seeds, with a
paired Wilcoxon signed-rank significance test against the "Full" baseline
for every variant. See [`ablation_study/ABLATION_PLAN.md`](ablation_study/ABLATION_PLAN.md)
for the full research design and rationale.

```bash
pip install scipy  # optional, enables significance testing

python ablation_study/run_full_ablation.py \
    --datasets OR2 Weibull_10k Shift_Abrupt \
    --seeds 7 13 21 \
    --output-dir ablation_study/results
```

This writes `ablation_raw.csv` (every `(dataset, variant, seed, instance)`
result), `ablation_summary.csv` / `.json` (mean ± std per variant, plus
Wilcoxon p-values vs. the baseline), and full per-run logs/plots under
`ablation_study/results/<dataset>/<variant>/`.

**Cost note:** the full grid is on the order of 100+ engine runs (~600–800
LLM calls at the paper's reported rate). Use `--factor-groups` to run one
question at a time, and `--skip-pool-size-sweep` to skip the most
expensive factor, while iterating.

## How it works (brief)

At each online step:

1. Every beam node is expanded with every heuristic in the active
   portfolio (parallelized via a thread pool), producing a child pool.
2. The pairwise behavioral diversity matrix over the active portfolio is
   (re)computed from a dynamic probe set.
3. The next-generation beam is built via **Lexicographic Champion-Based
   Selection**: a Quality Champion, Utilization Champion, Speed Champion,
   and Behavioral Diversity Champion are picked in that order from the
   pool.
4. The **Trigger Monitor** evaluates quality-degradation (EWMA over the
   optimality-gap delta), diversity-collapse (EWMA over pairwise
   dissimilarity), and input-distribution-shift (Welch's t-test / z-score)
   signals.
5. If a statistically significant event fires (and the cooldown has
   elapsed), a structured reflection report — including specific
   unaccommodated item sizes and free-space distribution — is sent to the
   LLM, which performs severity-matched mutation, crossover, or zero-shot
   generation (`prompts/mutate.txt`, `prompts/crossover.txt`,
   `prompts/generate_new.txt`). Any runtime crash during evaluation is fed
   back to the LLM as a stack trace: the first attempt uses
   `prompts/self_correct.txt`; if it keeps failing, `main.py` escalates to
   `prompts/heavy_mutation.txt` for a heavier rewrite before falling back
   to a heavy mutation and, ultimately, halting.
6. The active portfolio is refreshed via Top-K deduplicated selection;
   evicted heuristics are archived with per-identifier version caps.

See Figure 1 and Section III of the paper for the full flowchart and
formal definitions.

## Results

Full numbers (mean optimality gap / mean utility) and LLM API-call counts
for all eight datasets reported in the paper (OR1–OR3, Weibull_5k/10k/50k,
Shift_Abrupt, Shift_Abrupt_Large) and every baseline are given in Tables I
and II of the paper. Summary: our method attains the lowest or
tied-lowest optimality gap and highest space utilization on most datasets,
while using around 10 LLM API calls per run versus ~2,000–10⁶ for
LLM-based baselines.

## Reproducibility notes

- All datasets are generated with fixed, independent random seeds (see
  `data/datasetcreator.py`).
- Baselines are re-evaluated in the same batched, online fashion as our
  method (`baselines/simulator.py`) for a fair, matched input-output
  interaction pattern; exact baseline LLM query counts (FunSearch, EoH,
  MCTS-AHD) reported in the paper are order-of-magnitude estimates from
  the original papers, since those baselines were not re-run end-to-end
  (see Section IV.B–D of the paper for details and limitations).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{tabatabaei2026edbs,
  title     = {Event-Driven Beam Search Large Language Model Hyper-Heuristic for Online Bin Packing},
  author    = {Tabatabaei, Amirhosein and Khoshalhan, Farid},
  booktitle = {Proceedings of the 12th International Conference on Industrial and Systems Engineering (ICISE)},
  year      = {2026},
  address   = {Ferdowsi University of Mashhad, Mashhad, Iran}
}
```



## License

Released under the [MIT License](LICENSE).
