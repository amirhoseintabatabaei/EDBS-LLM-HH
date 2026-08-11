"""
Online bin-packing simulator, shared by all heuristics.

Standard protocol (identical to FunSearch / EoH / HSEvo / MCTS-AHD papers):
  - Items arrive one at a time, in the given order.
  - For each item, look at currently OPEN bins whose remaining capacity
    is >= item (feasible bins).
  - If no open bin is feasible -> open a brand-new bin for the item.
  - Otherwise, call the heuristic's priority function on the feasible
    bins' remaining capacities; place the item in the bin with the
    highest returned score (ties -> first).
"""

from typing import Callable, Sequence, Tuple
import numpy as np


def falkenauer_utility(bins_remain: np.ndarray, capacity: float, k: float = 2.0) -> float:
    """
    Computes Falkenauer's Fitness Utility:
    U = sum((fill_i / capacity)^k) / N
    """
    if len(bins_remain) == 0:
        return 0.0
    fill_ratios = (capacity - bins_remain) / capacity
    return float(np.sum(fill_ratios ** k) / len(bins_remain))


def online_bin_packing(items: Sequence[int], capacity: int,
                        priority_fn: Callable[[float, np.ndarray], np.ndarray],
                        sort_items: bool = True) -> Tuple[int, np.ndarray]:
    bins_remain = np.empty(0, dtype=float)
    
    n = len(items)
    batch_size = max(1, n // 20)
    

    processed_items = []
    for i in range(0, n, batch_size):
        batch = list(items[i:i + batch_size])
        if sort_items:
            batch.sort(reverse=True)
        processed_items.extend(batch)

    for item in processed_items:
        item = float(item)
        feasible_mask = bins_remain >= item

        if not np.any(feasible_mask):
            bins_remain = np.append(bins_remain, capacity - item)
            continue

        feasible_idx = np.flatnonzero(feasible_mask)
        feasible_caps = bins_remain[feasible_idx]

        scores = np.asarray(priority_fn(item, feasible_caps), dtype=float)
        best_local = int(np.argmax(scores))
        best_bin = feasible_idx[best_local]
        bins_remain[best_bin] -= item

    return len(bins_remain), bins_remain


def lower_bound(items: Sequence[int], capacity: int) -> float:
    """Continuous (L1) lower bound: sum(items)/capacity, no ceiling."""
    return sum(items) / capacity
