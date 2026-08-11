"""
Unified library of the 5 online bin-packing priority heuristics.

Convention (matches FunSearch / EoH / HSEvo / MCTSAHD papers):
    The simulator filters bins to those with remaining capacity >= item
    BEFORE calling the heuristic. So every `bins` / `bins_remain_cap`
    array received here only contains *feasible* bins. The heuristic
    returns one score per feasible bin; the bin with the highest score
    is chosen. If no bin is feasible, a new bin is opened (handled by
    the simulator, not by the heuristics).
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1) EoH
# ---------------------------------------------------------------------------
def priority_eoh(item, bins: np.ndarray) -> np.ndarray:
    bins = bins.astype(float)
    item_f = float(item)

    slack = bins - item_f
    eps = 1e-9

    tight = np.exp(-0.75 * slack) / (1.0 + slack)

    scale = np.maximum(np.percentile(bins, 60), 1.0)
    target = 0.42 * scale + 0.18 * item_f
    width = 0.20 * scale + 1.0
    reuse = np.exp(-0.5 * ((slack - target) / width) ** 2)

    cap_scale = np.maximum(np.max(bins), 1.0)
    preserve = 1.0 - (bins / (cap_scale + eps))
    preserve = np.clip(preserve, 0.0, 1.0)

    frag = np.log1p(slack) / np.log1p(cap_scale + 1.0)

    slack_med = np.maximum(np.median(slack), eps)
    harmony = 1.0 / (1.0 + np.abs(slack - slack_med) / (slack_med + 1.0))

    scores = (
        2.20 * tight
        + 1.05 * reuse
        + 0.45 * harmony
        + 0.30 * preserve
        - 0.60 * frag
    )
    return scores


# ---------------------------------------------------------------------------
# 2) HSEvo
# ---------------------------------------------------------------------------
def priority_hsevo(item, bins_remain_cap: np.ndarray) -> np.ndarray:
    ratios = item / bins_remain_cap
    log_ratios = np.log(ratios)
    priorities = -log_ratios
    return priorities


# ---------------------------------------------------------------------------
# 3) MCTS-AHD
# ---------------------------------------------------------------------------
def priority_mctsahd(item, bins_remain_cap: np.ndarray):
    priority = []
    total_capacity = sum(bins_remain_cap)
    total_bins = len(bins_remain_cap)
    average_capacity = total_capacity / total_bins if total_bins > 0 else 1
    gap_penalty_threshold = average_capacity * 0.5

    for cap in bins_remain_cap:
        if cap >= item:
            remaining_space = cap - item
            gap_penalty = max(0, abs(remaining_space - gap_penalty_threshold))
            proximity_score = -gap_penalty + (item / (remaining_space + 1e-7))

            utilization_ratio = (cap - remaining_space) / cap
            proximity_score += (cap / (remaining_space + 1e-7)) * 0.1
            if 0.5 < utilization_ratio <= 0.8:
                proximity_score += 0.7
            elif utilization_ratio > 0.8:
                proximity_score -= 1.5
            elif utilization_ratio < 0.5:
                proximity_score -= 0.3

            priority.append(proximity_score)
        else:
            priority.append(0)

    return np.array(priority)


# ---------------------------------------------------------------------------
# 4) FunSearch (Weibull-discovered)
# ---------------------------------------------------------------------------
def priority_funsearch_weibull(item: float, bins: np.ndarray) -> np.ndarray:
    max_bin_cap = max(bins)

    score = (bins - max_bin_cap) ** 2 / item + bins ** 2 / (item ** 2)
    score = score + bins ** 2 / item ** 3

    score[bins > item] = -score[bins > item]
    score[1:] -= score[:-1]

    return score


# ---------------------------------------------------------------------------
# 5) FunSearch (OR-discovered)
# ---------------------------------------------------------------------------
def priority_funsearch_or(item: float, bins: np.ndarray) -> np.ndarray:
    def s(bin_cap, item):
        d = bin_cap - item
        if d <= 2:
            return 4
        elif d <= 3:
            return 3
        elif d <= 5:
            return 2
        elif d <= 7:
            return 1
        elif d <= 9:
            return 0.9
        elif d <= 12:
            return 0.95
        elif d <= 15:
            return 0.97
        elif d <= 18:
            return 0.98
        elif d <= 20:
            return 0.98
        elif d <= 21:
            return 0.98
        else:
            return 0.99

    return np.array([s(b, item) for b in bins])


# ---------------------------------------------------------------------------
# 6) First-Fit (FF)
# ---------------------------------------------------------------------------
def priority_first_fit(item: float, bins: np.ndarray) -> np.ndarray:
    """
    First-Fit strategy: Assigns the highest score to the first feasible bin.
    Since bins are already filtered to feasible ones by the simulator,
    we can assign decreasing scores based on index: [len, len-1, ..., 1].
    """
    n = len(bins)
    return np.arange(n, 0, -1, dtype=float)


# ---------------------------------------------------------------------------
# 7) Best-Fit (BF)
# ---------------------------------------------------------------------------
def priority_best_fit(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Best-Fit strategy: Chooses the feasible bin with the SMALLEST remaining capacity
    (i.e., the tightest fit after adding the item).
    Returning negative capacity ensures np.argmax selects the bin with min capacity.
    """
    bins = bins.astype(float)
    return -bins


# ---------------------------------------------------------------------------
# HEURISTICS DICTIONARY
# ---------------------------------------------------------------------------
HEURISTICS = {
    "First-Fit": priority_first_fit,
    "Best-Fit": priority_best_fit,
    "EoH": priority_eoh,
    "HSEvo": priority_hsevo,
    "MCTS-AHD": priority_mctsahd,
    "FunSearch-Weibull": priority_funsearch_weibull,
    "FunSearch-OR": priority_funsearch_or,
}
