from dotenv import load_dotenv
load_dotenv()
import os
import sys
import time
import math
import random
import re
import platform
import threading
import signal
import json
import pickle
import logging
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Tuple, Set, Any
from collections import defaultdict, deque
from google import genai
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    
except ImportError:
    genai = None

try:

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hyperheuristic")


BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
DATA_DIR = BASE_DIR / "data"

def load_prompt(filename: str) -> str:
    filepath = PROMPTS_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def load_dataset(filename: str) -> Dict[str, Dict]:
    filepath = DATA_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_gemini_keys() -> List[str]:
    found_keys = []
    for key, val in os.environ.items():
        if key.startswith("GEMINI_API_KEY") and val.strip():
            match = re.search(r'\d+$', key)
            order = int(match.group()) if match else 0
            found_keys.append((order, val.strip()))
    
    found_keys.sort(key=lambda x: x[0])
    return [key_val for _, key_val in found_keys]

ACCOUNT_KEYS: List[str] = _load_gemini_keys()

MODEL_NAMES: List[str] = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]

GEMINI_AVAILABLE = genai is not None and len(ACCOUNT_KEYS) > 0

class GeminiRotator:
    def __init__(self, account_keys: List[str], model_names: List[str]):
        self.account_keys = account_keys
        self.model_names = model_names
        self.rotation: List[Tuple[int, int]] = [
            (a, m) for a in range(len(account_keys)) for m in range(len(model_names))
        ]
        self.pos = 0
        self.total_calls = 0

    @staticmethod
    def _is_capacity_error(e: Exception) -> bool:
        msg = str(e).lower()
        keywords = ("quota", "rate limit", "resource_exhausted", "429", "capacity", "exhausted", "503")
        return any(k in msg for k in keywords)

    def call(self, prompt: str) -> str:
        n = len(self.rotation)
        while True:
            last_error: Optional[Exception] = None
            for step in range(n):
                idx = (self.pos + step) % n
                acc_idx, model_idx = self.rotation[idx]
                key = self.account_keys[acc_idx]
                model_name = self.model_names[model_idx]
                if not key.strip():
                    continue

                try:
                    client = genai.Client(api_key=key)
                    response = client.models.generate_content(model=model_name, contents=prompt)
                    self.pos = idx
                    self.total_calls += 1
                    return response.text
                except Exception as e:
                    last_error = e
                    reason = "capacity/quota exhausted" if self._is_capacity_error(e) else f"error ({e})"
                    logger.info(f"[Gemini] Account {acc_idx + 1} / {model_name}: {reason} — moving to next")
                    continue

            self.pos = (self.pos + 1) % n
            logger.warning(
                f"[Gemini] All {n} Account/Model combinations exhausted ({last_error}). "
                f"Entering 3-minute cooldown (180 seconds) before retrying..."
            )
            time.sleep(180)


_rotator: Optional[GeminiRotator] = GeminiRotator(ACCOUNT_KEYS, MODEL_NAMES) if GEMINI_AVAILABLE else None


def call_llm(prompt: str) -> str:
    if not GEMINI_AVAILABLE or _rotator is None:
        raise RuntimeError("Gemini API not configured (no keys in the environment).")
    return _rotator.call(prompt)


def extract_code_blocks(text: str) -> List[str]:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return blocks if blocks else [text.strip()]


HEURISTIC_SIGNATURE_DOC = """
def score_bin(bin_remaining, item, capacity, avg_bins_rem, n_open_bins):
    ...
    return score
"""


def llm_generate_initial_heuristics(n: int, rng: Optional[random.Random] = None) -> List[str]:
    rng = rng or random
    random_seed_words = ["variance_minimization", "probabilistic_routing", "bin_density_entropy", "bimodal_split", "nonlinear_scaling"]
    selected_seed = rng.sample(random_seed_words, 2)

    template = load_prompt("initial_heuristics.txt")
    prompt = template.format(
        n=n,
        seed_0=selected_seed[0],
        seed_1=selected_seed[1],
        signature_doc=HEURISTIC_SIGNATURE_DOC,
        HEURISTIC_SIGNATURE_DOC=HEURISTIC_SIGNATURE_DOC
    )

    response = call_llm(prompt)
    return extract_code_blocks(response)


def llm_mutate(code: str, reflection: str, level: str) -> str:
    template = load_prompt("mutate.txt")
    prompt = template.format(code=code, reflection=reflection, level=level)

    response = call_llm(prompt)
    return extract_code_blocks(response)[0]


def llm_crossover(code1: str, code2: str, reflection: str) -> str:
    template = load_prompt("crossover.txt")
    prompt = template.format(code1=code1, code2=code2, reflection=reflection)

    response = call_llm(prompt)
    return extract_code_blocks(response)[0]


def llm_generate_new(reflection: str, n: int = 2) -> List[str]:
    template = load_prompt("generate_new.txt")
    prompt = template.format(reflection=reflection, n=n)

    response = call_llm(prompt)
    return extract_code_blocks(response)


DEFAULT_CAPACITY = 1.0


def make_random_batch(batch_size: int, rng: random.Random) -> List[float]:
    return [round(rng.uniform(0.05, 0.8), 3) for _ in range(batch_size)]


def chunk_items(items: List[float], batch_size: int) -> List[List[float]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def iter_dataset_instances(dataset: Dict[str, Dict]) -> List[Tuple[str, List[float], float]]:
    result = []
    for test_name, instance in dataset.items():
        capacity = instance["capacity"]
        items = instance["items"]
        result.append((test_name, items, capacity))
    return result


def compute_lower_bound(items: List[float], capacity: float) -> int:
    return math.ceil(sum(items) / capacity)


@dataclass
class PackingState:
    bins: List[float] = field(default_factory=list)
    capacity: float = DEFAULT_CAPACITY

    def copy(self) -> "PackingState":
        return PackingState(bins=list(self.bins), capacity=self.capacity)

    @property
    def n_bins(self) -> int:
        return len(self.bins)

    @property
    def utilization(self) -> float:
        if not self.bins:
            return 0.0
        
        return sum(((self.capacity - b) / self.capacity) ** 2 for b in self.bins) / len(self.bins)

class HeuristicRuntimeCrash(Exception):
    def __init__(self, error_msg, full_id):
        self.error_msg = error_msg
        self.full_id = full_id
        super().__init__(self.error_msg)


_SCORE_EPS = 1e-9


def _select_bin(
    score_fn: Callable,
    bins: List[float],
    item: float,
    capacity: float
) -> int:
    n_open = len(bins)
    avg_rem = (sum(bins) / n_open) if n_open > 0 else capacity

    best_idx = -1
    best_score = -float('inf')

    for idx, bin_rem in enumerate(bins):
        if bin_rem >= item:
            try:
                score = score_fn(
                    bin_remaining=bin_rem,
                    item=item,
                    capacity=capacity,
                    avg_bins_rem=avg_rem,
                    n_open_bins=n_open
                )
            except Exception:
                score = -float('inf')

            if score > best_score:
                best_score = score
                best_idx = idx

    return best_idx


def apply_heuristic(
    heuristic_fn: Callable,
    state: PackingState,
    batch: List[float],
    heuristic_id: str = "unknown",
    sort_descending: bool = False,
) -> Tuple[PackingState, float, List[float]]:
    new_state = state.copy()
    capacity = state.capacity
    start = time.perf_counter()

    items = sorted(batch, reverse=True) if sort_descending else list(batch)
    forced_new_bin_items: List[float] = []

    for item in items:
        try:
            idx = _select_bin(heuristic_fn, new_state.bins, item, capacity)
        except Exception as e:
            import traceback
            err_details = (
                f"Crash on item {item}. Bins state: {new_state.bins}. "
                f"Error: {str(e)}\n{traceback.format_exc()}"
            )
            raise HeuristicRuntimeCrash(err_details, heuristic_id)

        if idx == -1:
            forced_new_bin_items.append(item)
            new_state.bins.append(capacity - item)
        else:
            new_state.bins[idx] -= item

    runtime = time.perf_counter() - start
    return new_state, runtime, forced_new_bin_items


SAFE_BUILTINS = {
    "len": len, "range": range, "min": min, "max": max, "abs": abs,
    "enumerate": enumerate, "sum": sum, "sorted": sorted,
    "float": float, "int": int, "True": True, "False": False, "None": None,
    "round": round, "list": list, "dict": dict, "tuple": tuple, "set": set,
    "str": str, "bool": bool, "isinstance": isinstance, "zip": zip,
    "map": map, "filter": filter, "any": any, "all": all, "divmod": divmod,
    "reversed": reversed,
    "pow": pow,
    "log": math.log, "exp": math.exp, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "floor": math.floor, "ceil": math.ceil
}


class HeuristicValidationError(Exception):
    pass


class _HeuristicTimeout(Exception):
    pass


_SUPPORTS_SIGALRM = platform.system() != "Windows" and threading.current_thread() is threading.main_thread()


def _alarm_handler(signum, frame):
    raise _HeuristicTimeout("Timeout during execution (possible infinite loop)")


@dataclass
class Heuristic:
    hid: str
    version: int
    code: str
    fn: Callable = field(repr=False)

    @property
    def full_id(self) -> str:
        return f"{self.hid}_v{self.version}"


def compile_heuristic(code: str, hid: str, version: int, timeout: float = 0.5) -> Heuristic:
    namespace: Dict = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(code, namespace)
    except Exception as e:
        raise HeuristicValidationError(f"Compile error: {e}")

    fn = namespace.get("score_bin")
    if not callable(fn):
        raise HeuristicValidationError(
            "score_bin(bin_remaining, item, capacity, avg_bins_rem, n_open_bins) function not found"
        )

    probes = [
        (0.5 * DEFAULT_CAPACITY, 0.4 * DEFAULT_CAPACITY, DEFAULT_CAPACITY, 0.6 * DEFAULT_CAPACITY, 3),
        (1.0 * DEFAULT_CAPACITY, 0.9 * DEFAULT_CAPACITY, DEFAULT_CAPACITY, 0.5 * DEFAULT_CAPACITY, 1),
        (0.15 * DEFAULT_CAPACITY, 0.1 * DEFAULT_CAPACITY, DEFAULT_CAPACITY, 0.3 * DEFAULT_CAPACITY, 5),
        (0.05 * DEFAULT_CAPACITY, 0.05 * DEFAULT_CAPACITY, DEFAULT_CAPACITY, 0.2 * DEFAULT_CAPACITY, 2),
    ]

    for probe_args in probes:
        if _SUPPORTS_SIGALRM:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                val = fn(*probe_args)
            except _HeuristicTimeout as e:
                raise HeuristicValidationError(str(e))
            except Exception as e:
                raise HeuristicValidationError(f"Runtime error: {e}")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            result_holder: Dict = {}

            def _probe():
                try:
                    result_holder["value"] = fn(*probe_args)
                except Exception as e:
                    result_holder["error"] = e

            t = threading.Thread(target=_probe, daemon=True)
            t.start()
            t.join(timeout)
            if t.is_alive():
                raise HeuristicValidationError("Timeout during execution (possible infinite loop)")
            if "error" in result_holder:
                raise HeuristicValidationError(f"Runtime error: {result_holder['error']}")
            val = result_holder.get("value")

        if val is not None:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise HeuristicValidationError("score_bin output must be a number (int/float) or None")
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                raise HeuristicValidationError("score_bin returned a non-finite value (NaN/Inf)")

    return Heuristic(hid=hid, version=version, code=code, fn=fn)

def behavioral_diversity(
    h1: Heuristic, h2: Heuristic, probe_bins: List[float], probe_items: List[float], capacity: float
) -> float:
    if not probe_items:
        return 0.0

    n_bins = max(1, len(probe_bins))
    NEW_BIN_PENALTY = 1.0
    total_distance = 0.0

    for item in probe_items:
        try:
            d1 = _select_bin(h1.fn, list(probe_bins), item, capacity)
        except Exception:
            d1 = -999
        try:
            d2 = _select_bin(h2.fn, list(probe_bins), item, capacity)
        except Exception:
            d2 = -999

        if d1 == d2:
            distance = 0.0
        elif d1 == -999 or d2 == -999:
            distance = 1.5
        elif d1 == -1 or d2 == -1:
            distance = NEW_BIN_PENALTY
        else:
            distance = abs(d1 - d2) / n_bins

        total_distance += distance

    return total_distance / len(probe_items)


def compute_diversity_matrix(
    heuristics: List[Heuristic], probe_bins: List[float], probe_items: List[float], capacity: float
) -> Dict[Tuple[str, str], float]:
    matrix: Dict[Tuple[str, str], float] = {}
    for i, hi in enumerate(heuristics):
        matrix[(hi.full_id, hi.full_id)] = 0.0
        for hj in heuristics[i + 1:]:
            d = behavioral_diversity(hi, hj, probe_bins, probe_items, capacity)
            matrix[(hi.full_id, hj.full_id)] = d
            matrix[(hj.full_id, hi.full_id)] = d
    return matrix


class _NodeIdGen:
    _next = 0

    @classmethod
    def next(cls) -> int:
        cls._next += 1
        return cls._next


@dataclass
class Node:
    state: PackingState
    parent: Optional["Node"] = None
    heuristic_used: Optional[str] = None
    f1: float = 0.0
    f2: float = 0.0
    f3: float = 0.0
    no_fit_items: List[float] = field(default_factory=list)
    node_id: int = field(default_factory=_NodeIdGen.next)


def node_distance(n1: Node, n2: Node, diversity_matrix: Dict) -> float:
    return diversity_matrix.get((n1.heuristic_used, n2.heuristic_used), 0.5)


K_H = 4
MAX_PER_PARENT = 2


def select_top_k_with_dedup(
    candidates: List[Heuristic],
    evaluations: Dict[str, Tuple[float, float, float]],
    diversity_matrix: Dict,
    k: int = K_H,
) -> List[Heuristic]:
    remaining = list(candidates)
    selected: List[Heuristic] = []

    def pick(key_fn, reverse=False):
        chosen = sorted(remaining, key=key_fn, reverse=reverse)[0]
        remaining.remove(chosen)
        selected.append(chosen)

    pick(lambda h: evaluations[h.full_id][0])
    if remaining:
        pick(lambda h: evaluations[h.full_id][1], reverse=True)
    if remaining:
        pick(lambda h: evaluations[h.full_id][2])
    if remaining:
        def div_score(h):
            return min(
                diversity_matrix.get((h.full_id, s.full_id), 0.5) for s in selected
            )
        best_div = sorted(remaining, key=div_score, reverse=True)[0]
        selected.append(best_div)

    return selected[:k]


def champion_selection(
    pool: List[Node],
    diversity_matrix: Dict,
    max_per_parent: int = 2,
    k: int = K_H,
    rng: Optional[random.Random] = None
) -> List[Node]:
    rng = rng or random.Random()
    candidates_pool = list(pool)
    selected: List[Node] = []
    selected_ids: Set[int] = set()
    parent_count: Dict[int, int] = defaultdict(int)

    def eligible(node: Node) -> bool:
        return node.node_id not in selected_ids and parent_count[id(node.parent)] < max_per_parent

    def pick(key_fn) -> Optional[Node]:
        candidates = [n for n in candidates_pool if eligible(n)]
        if not candidates:
            candidates = [n for n in candidates_pool if n.node_id not in selected_ids]
        if not candidates:
            candidates = candidates_pool

        if not candidates:
            return None

        best = sorted(candidates, key=key_fn)[0]
        selected.append(best)
        selected_ids.add(best.node_id)
        parent_count[id(best.parent)] += 1
        return best

    pick(lambda n: (n.f1, -n.f2, n.f3, rng.random()))
    pick(lambda n: (-n.f2, n.f1, n.f3, rng.random()))
    pick(lambda n: (n.f3, n.f1, -n.f2, rng.random()))

    def div_key(n: Node):
        d_val = min(
            (diversity_matrix.get((n.heuristic_used, s.heuristic_used), 0.5) for s in selected),
            default=0.0
        )
        return (-d_val, n.f1, -n.f2, n.f3, rng.random())

    pick(div_key)  

    unique_selected: List[Node] = []
    seen_states: List[List[float]] = []

    for node in selected:
        norm_bins = sorted([round(b, 3) for b in node.state.bins])
        if norm_bins not in seen_states:
            seen_states.append(norm_bins)
            unique_selected.append(node)

    if len(unique_selected) < k:
        remaining_pool = sorted(candidates_pool, key=lambda n: (n.f1, -n.f2, n.f3))
        for node in remaining_pool:
            norm_bins = sorted([round(b, 3) for b in node.state.bins])
            if norm_bins not in seen_states:
                seen_states.append(norm_bins)
                unique_selected.append(node)
            if len(unique_selected) == k:
                break

    return unique_selected[:k] if unique_selected else selected[:k]


class TriggerMonitor:
    def __init__(self, alpha: float = 0.5, threshold_mult: float = 0.5,
                 shift_z: float = 0.8, cooldown_batches: int = 6):
        self.alpha = alpha
        self.threshold_mult = threshold_mult
        self.shift_z = shift_z
        self.cooldown_batches = cooldown_batches
        self._batches_since_trigger = cooldown_batches

        self.ewma_eff: Optional[float] = None
        self.ewma_div: Optional[float] = None
        self.history_eff: deque = deque(maxlen=30)
        self.batch_history: deque = deque(maxlen=8)

    def _update_ewma(self, val: float, prev: Optional[float]) -> float:
        return val if prev is None else self.alpha * val + (1.0 - self.alpha) * prev

    def check_distribution_shift(self, batch: List[float]) -> Tuple[bool, str]:
        shift = False
        severity = "medium"

        if len(self.batch_history) >= 3:
            recent_means = [sum(b) / len(b) for b in self.batch_history]
            recent_mean = statistics.mean(recent_means)
            recent_std = statistics.stdev(recent_means) if len(recent_means) > 1 else 0.0

            current_mean = sum(batch) / len(batch)

            within_std = statistics.stdev(batch) if len(batch) > 1 else 0.0
            se = math.sqrt(
                (within_std ** 2) / max(1, len(batch))
                + (recent_std ** 2)
            )
            se = max(se, 1e-6)

            z = abs(current_mean - recent_mean) / se

            if z > self.shift_z:
                shift = True
                severity = "high" if z > self.shift_z * 1.6 else "medium"

        self.batch_history.append(batch)
        return shift, severity

    def update_and_check(self, beam: List[Node], pool: List[Node], batch: List[float], capacity: float, diversity_matrix: Dict) -> Optional[Dict]:
        self._batches_since_trigger += 1

        if not beam or len(beam) < 2:
            return None

        pair_distances = []
        for i in range(len(beam)):
            for j in range(i + 1, len(beam)):
                h_i = beam[i].heuristic_used
                h_j = beam[j].heuristic_used
                dist = diversity_matrix.get((h_i, h_j), 0.5)
                pair_distances.append(dist)

        div = sum(pair_distances) / len(pair_distances) if pair_distances else 1.0
        
        alpha_div = 0.2
        self.ewma_div = div if self.ewma_div is None else alpha_div * div + (1.0 - alpha_div) * self.ewma_div

        batch_volume = sum(batch)
        best_node = max(beam, key=lambda n: n.f2)
        parent = best_node.parent
        bins_before = parent.state.n_bins if parent is not None else 0
        bins_after = best_node.state.n_bins
        new_bins = max(0, bins_after - bins_before)
        lb = math.ceil(batch_volume / capacity) if capacity > 0 else 1

        local_eff = (max(self.history_eff) * 1.1 if self.history_eff else 1.0) if new_bins == 0 else (lb / new_bins)
        self.history_eff.append(local_eff)
        self.ewma_eff = self._update_ewma(local_eff, self.ewma_eff)

        shifted, shift_severity = self.check_distribution_shift(batch)

        if self.ewma_div < 0.005:
            self._batches_since_trigger = 0
            return {"type": "diversity_collapse", "severity": "high"}

        if self._batches_since_trigger < self.cooldown_batches:
            return None

        mean_eff = statistics.mean(self.history_eff) if len(self.history_eff) >= 3 else 1.0
        std_eff = statistics.stdev(self.history_eff) if len(self.history_eff) >= 3 else 0.01

        trigger = None
        
        if self.ewma_div < 0.10:
            trigger = {"type": "diversity_collapse", "severity": "high"}
        elif self.ewma_div < 0.20:
            trigger = {"type": "diversity_collapse", "severity": "medium"}
        elif self.ewma_eff < (mean_eff - self.threshold_mult * std_eff):
            trigger = {"type": "quality_degradation", "severity": "medium"}
        elif shifted:
            trigger = {"type": "distribution_shift", "severity": shift_severity}

        if trigger is not None:
            self._batches_since_trigger = 0

        return trigger


class Archive:
    def __init__(self, max_versions_per_hid: int = 20):
        self.store: Dict[str, List[Heuristic]] = defaultdict(list)
        self.max_versions_per_hid = max_versions_per_hid

    def add(self, heuristics: List[Heuristic]):
        for h in heuristics:
            versions = self.store[h.hid]
            versions.append(h)
            if len(versions) > self.max_versions_per_hid:
                del versions[: len(versions) - self.max_versions_per_hid]

    def latest_versions(self) -> List[Heuristic]:
        return [versions[-1] for versions in self.store.values() if versions]


def format_beam_profile(label: str, beam: List[Node]) -> str:
    lines = [f"--- {label} ---"]
    for i, n in enumerate(beam, start=1):
        lines.append(
            f"  Node {i}: heuristic={n.heuristic_used or '-'} | bins={n.f1:4d} | "
            f"utilization={n.f2:6.2%} | runtime={n.f3:.5f}s"
        )
    return "\n".join(lines)


def beam_profile_record(label: str, beam: List[Node], trigger: Optional[Dict]) -> Dict[str, Any]:
    return {
        "label": label,
        "trigger": trigger,
        "beam": [
            {
                "heuristic": n.heuristic_used,
                "bins": n.f1,
                "utilization": n.f2,
                "runtime": n.f3,
                "no_fit_items": n.no_fit_items,
            }
            for n in beam
        ],
    }


def export_heuristic_sequence(
    final_node: Node,
    path: List[str],
    all_heuristics: Dict[str, "Heuristic"],
    filepath: str,
    instance_name: str = "",
) -> None:
    with open(filepath, "w") as f:
        if instance_name:
            f.write(f"Instance: {instance_name}\n")
        f.write(f"Final bin count: {final_node.f1}\n")
        f.write(f"Final utilization: {final_node.f2:.2%}\n")
        f.write(f"Heuristic sequence (versioned): {path}\n\n")
        f.write("Final problem state (remaining capacity of each open bin):\n")
        f.write(f"{final_node.state.bins}\n")
        f.write("\n" + "=" * 70 + "\n")
        for step_idx, full_id in enumerate(path, start=1):
            h = all_heuristics.get(full_id)
            f.write(f"\n=== Step {step_idx}: {full_id} ===\n")
            if h is not None:
                f.write(h.code.strip() + "\n")
            else:
                f.write("<code not found — heuristic was pruned from the archive>\n")


def plot_tree_search(history: List[List[Node]], final_node: Node, output_path: str, title: str = "Tree-Search Progression") -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.info("[Visualization skipped] matplotlib not installed.")
        return
    if not history:
        return

    champion_ids = set()
    curr = final_node
    while curr is not None:
        champion_ids.add(curr.node_id)
        curr = curr.parent

    fig, ax = plt.subplots(figsize=(15, 8))

    node_pos: Dict[int, Tuple[int, float]] = {}
    for batch_index, nodes in enumerate(history, start=1):
        for n in nodes:
            node_pos[n.node_id] = (batch_index, n.f1)
            if n.parent and n.parent.node_id not in node_pos:
                node_pos[n.parent.node_id] = (batch_index - 1, n.parent.f1)

    for batch_index, nodes in enumerate(history, start=1):
        for n in nodes:
            if n.parent and n.parent.node_id in node_pos:
                x0, y0 = node_pos[n.parent.node_id]
                x1, y1 = node_pos[n.node_id]
                if not (n.node_id in champion_ids and n.parent.node_id in champion_ids):
                    ax.plot([x0, x1], [y0, y1], color="gainsboro", linewidth=0.8, alpha=0.5, zorder=1)

    for batch_index, nodes in enumerate(history, start=1):
        xs = [batch_index for n in nodes if n.node_id not in champion_ids]
        ys = [n.f1 for n in nodes if n.node_id not in champion_ids]
        if xs:
            ax.scatter(xs, ys, color="lightgray", s=15, alpha=0.5, zorder=2)

    colors = {
        'Quality': '#1f77b4',
        'Utilization': '#2ca02c',
        'Speed': '#ff7f0e',
        'Diversity': '#9467bd'
    }

    criterion_nodes: Dict[str, List[Tuple[int, float]]] = {
        'Quality': [],
        'Utilization': [],
        'Speed': [],
        'Diversity': []
    }

    for batch_index, nodes in enumerate(history, start=1):
        beam_members = nodes[:4]
        roles = ['Quality', 'Utilization', 'Speed', 'Diversity']
        for i, node in enumerate(beam_members):
            role = roles[i] if i < len(roles) else 'Diversity'
            criterion_nodes[role].append((batch_index, node.f1))

    for role, points in criterion_nodes.items():
        if points:
            pxs, pys = zip(*points)
            ax.plot(pxs, pys, color=colors[role], linestyle='--', linewidth=1.2, alpha=0.7, zorder=3, label=f'Branch: {role}')
            ax.scatter(pxs, pys, color=colors[role], s=35, zorder=4)

    curr = final_node
    c_xs, c_ys = [], []
    while curr:
        if curr.node_id in node_pos:
            x, y = node_pos[curr.node_id]
            c_xs.append(x)
            c_ys.append(y)
        if curr.parent and curr.parent.node_id in node_pos:
            x0, y0 = node_pos[curr.parent.node_id]
            x1, y1 = node_pos[curr.node_id]
            ax.plot([x0, x1], [y0, y1], color="crimson", linewidth=2.8, zorder=5)
        curr = curr.parent

    if c_xs:
        ax.scatter(c_xs, c_ys, color="crimson", s=70, edgecolor="maroon", zorder=6, label="Final Winner Path")

    ax.set_xlabel("Batch Number (Depth)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Objective Value (Total Bins)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"[Visualization saved]: {output_path}")


@dataclass
class EngineConfig:
    medium_crossover_leader_pairing: bool = True
    use_diversity_preservation_crossover: bool = True
    diversity_preservation_prob: float = 0.3
    max_workers: int = 8
    max_step_fix_attempts: int = 3
    sort_batch_descending: bool = True
    use_trigger_monitor: bool = True
    periodic_trigger_interval: int = 5
    # Trigger-monitor thresholds, exposed here (instead of only inside
    # TriggerMonitor's own defaults) so ablation studies can vary them
    # per-run without editing TriggerMonitor itself. Defaults match
    # TriggerMonitor's original hardcoded defaults, so leaving these
    # untouched reproduces the exact previous behavior.
    trigger_alpha: float = 0.5
    trigger_threshold_mult: float = 0.5
    trigger_shift_z: float = 0.8
    trigger_cooldown_batches: int = 6
    portfolio_size: int = K_H


MAX_INIT_LLM_ATTEMPTS = 5


class HyperHeuristicEngine:
    def __init__(self, seed: int = 42, config: Optional[EngineConfig] = None):
        self.rng = random.Random(seed)
        self.config = config or EngineConfig()
        self._hid_counter = 0
        self.portfolio: List[Heuristic] = []
        self.beam: List[Node] = []
        self.archive = Archive()
        self.trigger_monitor = TriggerMonitor(
            alpha=self.config.trigger_alpha,
            threshold_mult=self.config.trigger_threshold_mult,
            shift_z=self.config.trigger_shift_z,
            cooldown_batches=self.config.trigger_cooldown_batches,
        )
        self.evaluations: Dict[str, Tuple[float, float, float]] = {}
        self.diversity_matrix: Dict = {}

        self._probe_bins: List[float] = []
        self._probe_items: List[float] = []
        self.capacity: float = DEFAULT_CAPACITY

        self.all_heuristics: Dict[str, Heuristic] = {}
        self.history: List[List[Node]] = []
        self._batch_log: List[Dict[str, Any]] = []
        self._executor: Optional[ThreadPoolExecutor] = None
        if self.config.max_workers > 1:
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

    def __del__(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    def _new_hid(self) -> str:
        self._hid_counter += 1
        return f"h{self._hid_counter}"

    def _next_version(self, hid: str) -> int:
        existing_versions = [
            h.version for full_id, h in self.all_heuristics.items() if h.hid == hid
        ]
        return (max(existing_versions) + 1) if existing_versions else 1

    def _log_batch_record(self, record: Dict[str, Any]) -> None:
        self._batch_log.append(record)

    def flush_batch_log(self, path: str) -> None:
        with open(path, "w") as f:
            for record in self._batch_log:
                f.write(json.dumps(record) + "\n")

    def update_probes_dynamically(self, batch: List[float], capacity: float, num_probes: int = 12):
        if not batch:
            return
        min_item = min(batch)
        max_item = max(batch)

        self._probe_items = [
            round(self.rng.uniform(min_item, max_item), 3)
            for _ in range(num_probes)
        ]

        self._probe_bins = [
            round(capacity * self.rng.uniform(0.1, 0.9), 3)
            for _ in range(num_probes)
        ]

    def _map_apply_heuristic(
        self, jobs: List[Tuple[Node, Heuristic, List[float]]]
    ) -> List[Tuple[Node, Heuristic, PackingState, float, List[float]]]:
        results: List[Tuple[Node, Heuristic, PackingState, float, List[float]]] = []
        sort_desc = self.config.sort_batch_descending

        if self._executor is None:
            for parent, h, batch in jobs:
                state, rt, trace = apply_heuristic(
                    h.fn, parent.state, batch, heuristic_id=h.full_id, sort_descending=sort_desc
                )
                results.append((parent, h, state, rt, trace))
            return results

        futures = {
            self._executor.submit(
                apply_heuristic, h.fn, parent.state, batch, h.full_id, sort_desc
            ): (parent, h)
            for parent, h, batch in jobs
        }
        first_crash: Optional[HeuristicRuntimeCrash] = None
        for future in as_completed(futures):
            parent, h = futures[future]
            try:
                state, rt, trace = future.result()
                results.append((parent, h, state, rt, trace))
            except HeuristicRuntimeCrash as crash:
                if first_crash is None:
                    first_crash = crash
        if first_crash is not None:
            raise first_crash
        return results

    def initialize(self, first_batch: List[float], capacity: float = DEFAULT_CAPACITY, initial_pool_size: int = 10):
        self.capacity = capacity
        self.update_probes_dynamically(first_batch, capacity)

        def init_llm_self_correct(code: str, crash_report: str) -> str:
            prompt = load_prompt("self_correct.txt").format(code=code, crash_report=crash_report)
            return extract_code_blocks(call_llm(prompt))[0]

        valid_candidates: List[Heuristic] = []

        if not GEMINI_AVAILABLE:
            raise RuntimeError("LLM unavailable. Cannot initialize portfolio without API keys.")

        logger.info(f"[Init] Requesting a large pool of {initial_pool_size} candidate heuristics from Gemini...")
        attempt = 0
        while len(valid_candidates) < initial_pool_size and attempt < MAX_INIT_LLM_ATTEMPTS:
            attempt += 1
            needed = initial_pool_size - len(valid_candidates)
            logger.info(f"  [Attempt {attempt}/{MAX_INIT_LLM_ATTEMPTS}] Requesting {needed} more candidates...")

            try:
                codes = llm_generate_initial_heuristics(needed, rng=self.rng)
                for code in codes:
                    if len(valid_candidates) >= initial_pool_size:
                        break
                    try:
                        fresh_hid = self._new_hid()
                        compiled = compile_heuristic(code, hid=fresh_hid, version=1)

                        valid_candidates.append(compiled)
                        logger.info(f"    -> Compiled and added to initial pool: {compiled.full_id}")

                    except HeuristicValidationError as e:
                        logger.info(f"    -> Sandbox validation failed: {e}")
            except Exception as e:
                logger.warning(f"    -> LLM Connection error: {e}")
                time.sleep(2)

        if not valid_candidates or len(valid_candidates) < self.config.portfolio_size:
            raise RuntimeError("LLM failed to generate enough valid heuristics. Engine stopped.")

        logger.info(f"[Init] Success! Generated raw pool of {len(valid_candidates)} unique heuristics.")

        for h in valid_candidates:
            self.all_heuristics[h.full_id] = h

        root_state = PackingState(capacity=capacity)

        evals = {}
        for idx_h, h in enumerate(valid_candidates):
            current_heuristic = h
            while True:
                try:
                    state, rt, _trace = apply_heuristic(
                        current_heuristic.fn, root_state, first_batch,
                        heuristic_id=current_heuristic.full_id,
                        sort_descending=self.config.sort_batch_descending,
                    )
                    evals[current_heuristic.full_id] = (state.n_bins, state.utilization, rt)
                    valid_candidates[idx_h] = current_heuristic
                    break
                except HeuristicRuntimeCrash as crash:
                    logger.warning(f"    -> [INITIALIZATION CRASH DETECTED] Heuristic {crash.full_id} failed on first batch.")
                    logger.info("       Initiating synchronous Self-Correction loop...")

                    corrected_code = init_llm_self_correct(current_heuristic.code, crash.error_msg)
                    new_version = self._next_version(current_heuristic.hid)

                    current_heuristic = compile_heuristic(corrected_code, current_heuristic.hid, new_version)
                    self.all_heuristics[current_heuristic.full_id] = current_heuristic

        self.evaluations = evals
        self.diversity_matrix = compute_diversity_matrix(valid_candidates, self._probe_bins, self._probe_items, capacity)

        self.portfolio = select_top_k_with_dedup(valid_candidates, evals, self.diversity_matrix, self.config.portfolio_size)
        logger.info(f"[Portfolio Selected] Filtered down to {self.config.portfolio_size} elite heuristics: {[h.full_id for h in self.portfolio]}")

        root = Node(state=root_state, parent=None, heuristic_used=None)
        jobs = [(root, h, first_batch) for h in self.portfolio]
        results = self._map_apply_heuristic(jobs)
        self.beam = [
            Node(
                state=state, parent=root, heuristic_used=h.full_id,
                f1=state.n_bins, f2=state.utilization, f3=rt, no_fit_items=trace,
            )
            for (_, h, state, rt, trace) in results
        ]

        self.history.append(list(self.beam))

    def step(self, batch: List[float]) -> Optional[Dict]:
        self.update_probes_dynamically(batch, self.capacity)

        def fix_broken_code(code: str, crash_report: str, fallback: bool = False) -> str:
            if fallback:
                prompt = load_prompt("heavy_mutation.txt").format(code=code, crash_report=crash_report)
            else:
                prompt = load_prompt("self_correct.txt").format(code=code, crash_report=crash_report)
            return extract_code_blocks(call_llm(prompt))[0]

        fix_attempts: Dict[str, int] = defaultdict(int)

        while True:
            pool: List[Node] = []
            try:
                jobs = [(parent, h, batch) for parent in self.beam for h in self.portfolio]
                results = self._map_apply_heuristic(jobs)
                for parent, h, state, rt, trace in results:
                    pool.append(Node(
                        state=state, parent=parent, heuristic_used=h.full_id,
                        f1=state.n_bins, f2=state.utilization, f3=rt, no_fit_items=trace,
                    ))
                break

            except HeuristicRuntimeCrash as crash:
                logger.warning(f"[EXECUTION CRASH DETECTED] Heuristic {crash.full_id} failed mid-batch.")

                target_idx = next(i for i, h in enumerate(self.portfolio) if h.full_id == crash.full_id)
                broken_heuristic = self.portfolio[target_idx]
                fix_attempts[broken_heuristic.hid] += 1

                if fix_attempts[broken_heuristic.hid] > self.config.max_step_fix_attempts:
                    raise RuntimeError(
                        f"Exceeded {self.config.max_step_fix_attempts} self-correction attempts for "
                        f"{broken_heuristic.hid}. Engine crashed."
                    )

                logger.info(" -> Initiating synchronous Self-Correction loop via Gemini...")
                try:
                    corrected_code = fix_broken_code(broken_heuristic.code, crash.error_msg, fallback=False)
                    new_version = self._next_version(broken_heuristic.hid)
                    compiled = compile_heuristic(corrected_code, broken_heuristic.hid, new_version)
                    logger.info(f" -> Successfully fixed! Upgraded to {compiled.full_id}. Retrying batch expansion...")

                    self.portfolio[target_idx] = compiled
                    self.all_heuristics[compiled.full_id] = compiled
                except Exception as compile_err:
                    logger.warning(f" -> LLM failed to fix the code cleanly ({compile_err}). Forcing backup heavy mutation...")
                    try:
                        corrected_code = fix_broken_code(broken_heuristic.code, crash.error_msg, fallback=True)
                        new_version = self._next_version(broken_heuristic.hid)
                        compiled = compile_heuristic(corrected_code, broken_heuristic.hid, new_version)
                        self.portfolio[target_idx] = compiled
                        self.all_heuristics[compiled.full_id] = compiled
                    except Exception as second_err:
                        logger.warning(f" -> Backup heavy mutation also failed to compile ({second_err}).")

        self.diversity_matrix = compute_diversity_matrix(self.portfolio, self._probe_bins, self._probe_items, self.capacity)
        self.beam = champion_selection(pool, self.diversity_matrix, MAX_PER_PARENT)

        self.history.append(list(pool))
        if self.config.use_trigger_monitor:
            trigger = self.trigger_monitor.update_and_check(self.beam, pool, batch, self.capacity, self.diversity_matrix)

        else:
            if len(self.history) % self.config.periodic_trigger_interval == 0:
                trigger = {"type": "periodic_intervention", "severity": "medium"}
            else:
                trigger = None

        if trigger:
            self._handle_trigger(trigger, batch)

        return trigger

    def _handle_trigger(self, trigger: Dict, current_batch: List[float]):
        if not GEMINI_AVAILABLE:
            logger.info(f"[Trigger] {trigger['type']} ({trigger['severity']}) — no LLM configured, keeping current Portfolio.")
            return

        reflection = self._build_reflection(trigger)
        severity = trigger["severity"]

        logger.info(f"[Trigger Activated] Type: {trigger['type']} | Severity: {severity}")
        logger.info("    -> Pausing engine stream. Requesting urgent LLM intervention from Gemini...")

        plans: List[Tuple] = []

        if severity == "low":
            target = self._weakest_heuristic()
            plans.append(("mutate", target.hid, target.code, "small (Low)"))

        elif severity == "medium":
            if self.rng.random() < 0.5:
                target = self._weakest_heuristic()
                plans.append(("mutate", target.hid, target.code, "medium (Medium)"))
            else:
                if self.config.medium_crossover_leader_pairing:
                    h_quality = self._best_by_criterion(0, reverse=False)
                    h_util = self._best_by_criterion(1, reverse=True)
                    if h_quality.hid == h_util.hid:
                        alt = [h for h in self.portfolio if h.hid != h_quality.hid]
                        if alt:
                            h_util = alt[0]
                else:
                    pair = self.rng.sample(self.portfolio, 2) if len(self.portfolio) >= 2 else list(self.portfolio) * 2
                    h_quality, h_util = pair[0], pair[1]
                plans.append(("crossover", self._new_hid(), h_quality.code, h_util.code))

            if self.config.use_diversity_preservation_crossover and self.rng.random() < self.config.diversity_preservation_prob:
                h_div = self._most_diverse_heuristic()
                h_quality = self._best_by_criterion(0, reverse=False)
                if h_div.hid != h_quality.hid:
                    plans.append(("crossover", self._new_hid(), h_div.code, h_quality.code))

        else:
            target = self._weakest_heuristic()
            plans.append(("heavy_mutate", target.hid, target.code))
            plans.append(("new", self._new_hid()))

        payload: List[Tuple[str, str]] = []
        try:
            for plan in plans:
                if plan[0] == "mutate":
                    _, hid, code, level = plan
                    new_code = llm_mutate(code, reflection, level=level)
                    payload.append((hid, new_code))
                elif plan[0] == "heavy_mutate":
                    _, hid, code = plan
                    new_code = llm_mutate(code, reflection, level="heavy (High)")
                    payload.append((hid, new_code))
                elif plan[0] == "crossover":
                    _, hid, code1, code2 = plan
                    new_code = llm_crossover(code1, code2, reflection)
                    payload.append((hid, new_code))
                elif plan[0] == "new":
                    _, hid = plan
                    codes = llm_generate_new(reflection, n=2)
                    if codes:
                        payload.append((hid, codes[0]))
                    for extra_code in codes[1:]:
                        payload.append((self._new_hid(), extra_code))

            self._apply_llm_payload(payload, current_batch)

        except Exception as e:
            logger.warning(f"    -> [LLM Intervention Failed]: {e} — Continuing with existing portfolio.")

    def _apply_llm_payload(self, payload: List[Tuple[str, str]], current_batch: List[float]):
        new_heuristics: List[Heuristic] = []
        for hid, code in payload:
            version = self._next_version(hid)
            try:
                compiled = compile_heuristic(code, hid, version)
                new_heuristics.append(compiled)
                self.all_heuristics[compiled.full_id] = compiled
            except HeuristicValidationError as e:
                logger.info(f"    -> [LLM Compilation Refused]: {e}")

        if not new_heuristics:
            return

        reference_state = self.beam[0].state
        evals = dict(self.evaluations)
        surviving_heuristics: List[Heuristic] = []
        for h in new_heuristics:
            try:
                state, rt, _trace = apply_heuristic(
                    h.fn, reference_state, current_batch,
                    heuristic_id=h.full_id, sort_descending=self.config.sort_batch_descending,
                )
                evals[h.full_id] = (state.n_bins, state.utilization, rt)
                surviving_heuristics.append(h)
            except HeuristicRuntimeCrash as crash:
                logger.info(f"    -> [Candidate {crash.full_id} crashed during evaluation, dropping it only]: {crash.error_msg.splitlines()[0]}")
        self.evaluations = evals
        new_heuristics = surviving_heuristics
        if not new_heuristics:
            logger.info("    -> [LLM Intervention] All candidates crashed on evaluation; keeping existing portfolio.")
            return

        all_candidates = list(self.portfolio) + new_heuristics
        diversity_matrix = compute_diversity_matrix(all_candidates, self._probe_bins, self._probe_items, self.capacity)
        new_portfolio = select_top_k_with_dedup(all_candidates, evals, diversity_matrix, self.config.portfolio_size)

        kept_ids = {p.full_id for p in new_portfolio}
        removed = [h for h in self.portfolio if h.full_id not in kept_ids]
        self.archive.add(removed)

        self.portfolio = new_portfolio
        self.diversity_matrix = diversity_matrix
        logger.info(f"    -> [Elite Portfolio Mutated Successfully]: {[h.full_id for h in self.portfolio]}")

    def _weakest_heuristic(self) -> Heuristic:
        return max(self.portfolio, key=lambda h: self.evaluations.get(h.full_id, (0, 0, 0))[0])

    def _best_by_criterion(self, idx: int, reverse: bool = False) -> Heuristic:
        return sorted(
            self.portfolio,
            key=lambda h: self.evaluations.get(h.full_id, (0.0, 0.0, 0.0))[idx],
            reverse=reverse,
        )[0]

    def _most_diverse_heuristic(self) -> Heuristic:
        def avg_div(h: Heuristic) -> float:
            others = [o for o in self.portfolio if o.hid != h.hid]
            if not others:
                return 0.0
            return sum(
                self.diversity_matrix.get(
                    (h.full_id, o.full_id),
                    self.diversity_matrix.get((o.full_id, h.full_id), 0.5),
                )
                for o in others
            ) / len(others)
        return max(self.portfolio, key=avg_div)

    def _build_reflection(self, trigger: Dict) -> str:
        lines = [f"Trigger type: {trigger['type']} | severity: {trigger['severity']}"]

        for n in self.beam:
            lines.append(
                f"  Beam member: bins={n.f1}, utilization={n.f2:.2%}, "
                f"runtime={n.f3:.5f}s, heuristic={n.heuristic_used}"
            )

            if n.no_fit_items:
                sizes_str = ", ".join(f"{s:.3f}" for s in n.no_fit_items)
                lines.append(
                    f"    -> Items that did NOT fit any existing bin this batch "
                    f"(forced a new bin to open): [{sizes_str}]"
                )
            else:
                lines.append(
                    "    -> Every item in this batch fit into an already-open bin "
                    "(no new bin was needed)."
                )

            free_spaces = sorted(n.state.bins, reverse=True)
            if free_spaces:
                free_str = ", ".join(f"{f:.3f}" for f in free_spaces)
                mean_free = statistics.mean(free_spaces)
                lines.append(
                    f"    -> Free space across {len(free_spaces)} open bins "
                    f"(sorted desc): [{free_str}]  "
                    f"(min={min(free_spaces):.3f}, mean={mean_free:.3f}, max={max(free_spaces):.3f})"
                )
            else:
                lines.append("    -> No bins are currently open.")

        return "\n".join(lines)

    def finalize(self) -> Tuple[Node, List[str]]:
        final_node = min(self.beam, key=lambda n: n.f1)
        path: List[str] = []
        node = final_node
        while node.parent is not None:
            path.append(node.heuristic_used)
            node = node.parent
        path.reverse()
        return final_node, path

    def save_checkpoint(self, path: str) -> None:
        data = {
            "hid_counter": self._hid_counter,
            "rng_state": self.rng.getstate(),
            "config": self.config,
            "capacity": self.capacity,
            "portfolio": [(h.hid, h.version, h.code) for h in self.portfolio],
            "archive": {
                hid: [(h.version, h.code) for h in versions]
                for hid, versions in self.archive.store.items()
            },
            "all_heuristics": {
                fid: (h.hid, h.version, h.code) for fid, h in self.all_heuristics.items()
            },
            "evaluations": self.evaluations,
            "beam": self.beam,
            "history": self.history,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"[Checkpoint saved]: {path}")

    @classmethod
    def load_checkpoint(cls, path: str, config: Optional[EngineConfig] = None) -> "HyperHeuristicEngine":
        with open(path, "rb") as f:
            data = pickle.load(f)

        engine = cls(seed=0, config=config or data.get("config"))
        engine.rng.setstate(data["rng_state"])
        engine._hid_counter = data["hid_counter"]
        engine.capacity = data.get("capacity", DEFAULT_CAPACITY)

        def _recompile(hid: str, version: int, code: str) -> Heuristic:
            return compile_heuristic(code, hid, version)

        engine.portfolio = [_recompile(hid, v, c) for hid, v, c in data["portfolio"]]

        engine.archive = Archive()
        for hid, versions in data["archive"].items():
            engine.archive.store[hid] = [_recompile(hid, v, c) for v, c in versions]

        engine.all_heuristics = {
            fid: _recompile(hid, v, c) for fid, (hid, v, c) in data["all_heuristics"].items()
        }
        engine.evaluations = data["evaluations"]
        engine.beam = data["beam"]
        engine.history = data["history"]

        engine._probe_bins = []
        engine._probe_items = []
        engine.diversity_matrix = {}

        logger.info(f"[Checkpoint loaded]: {path} (resuming at batch {len(engine.history)})")
        return engine


def run_demo(n_batches: int = 15, batch_size: int = 20, seed: int = 7, output_dir: str = ".", config: Optional[EngineConfig] = None, capacity: float = DEFAULT_CAPACITY):
    rng = random.Random(seed)
    engine = HyperHeuristicEngine(seed=seed, config=config)

    first_batch = make_random_batch(batch_size, rng)
    engine.initialize(first_batch, capacity=capacity)
    logger.info(f"Initial Portfolio: {[h.full_id for h in engine.portfolio]}")
    logger.info("\n" + format_beam_profile("Batch 01", engine.beam))
    engine._log_batch_record(beam_profile_record("Batch 01", engine.beam, None))

    for t in range(2, n_batches + 1):
        batch = make_random_batch(batch_size, rng)
        trigger = engine.step(batch)
        status = f"Trigger: {trigger['type']}({trigger['severity']})" if trigger else "No Trigger"
        label = f"Batch {t:02d} | {status}"
        logger.info("\n" + format_beam_profile(label, engine.beam))
        engine._log_batch_record(beam_profile_record(label, engine.beam, trigger))

    final_node, path = engine.finalize()
    logger.info("=== Final Result ===")
    logger.info(f"Final bin count: {final_node.f1}")
    logger.info(f"Final utilization: {final_node.f2:.2%}")
    logger.info(f"Heuristic sequence (versioned): {path}")
    logger.info(f"Final problem state (remaining bin capacities): {final_node.state.bins}")

    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, "demo_best_sequence.txt")
    export_heuristic_sequence(final_node, path, engine.all_heuristics, txt_path, instance_name="demo")
    logger.info(f"[Best sequence saved]: {txt_path}")

    png_path = os.path.join(output_dir, "demo_tree_search.png")
    plot_tree_search(engine.history, final_node, png_path, title="Demo Run — Tree-Search Progression")

    log_path = os.path.join(output_dir, "demo_run_log.jsonl")
    engine.flush_batch_log(log_path)
    logger.info(f"[Structured run log saved]: {log_path}")


def run_on_dataset(
    dataset: Dict[str, Dict],
    batch_size: int,
    dataset_name: str,
    initial_pool_size: int = 10,
    sleep_seconds: int = 300,
    seed: int = 7,
    output_dir: str = "bin_packing_outputs",
    config: Optional[EngineConfig] = None,
    checkpoint_every: int = 0,
    resume_from: Optional[str] = None,
) -> Dict[str, Dict]:
    instances = iter_dataset_instances(dataset)
    all_results: Dict[str, Dict] = {}
    os.makedirs(output_dir, exist_ok=True)

    tree_search_dir = os.path.join(output_dir, "tree_search")
    best_sequence_dir = os.path.join(output_dir, "best_sequence")
    run_log_dir = os.path.join(output_dir, "run_log")
    os.makedirs(tree_search_dir, exist_ok=True)
    os.makedirs(best_sequence_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    for i, (test_name, items, instance_capacity) in enumerate(instances):
        instance_no = i + 1
        lower_bound = compute_lower_bound(items, instance_capacity)
        logger.info(
            f"=== Solving instance: {test_name} "
            f"({len(items)} items, capacity={instance_capacity}, batch_size={batch_size}) ==="
        )
        logger.info(f"Lower bound (ceil(sum(items)/capacity)): {lower_bound} bins")

        batches = chunk_items(items, batch_size)
        if not batches:
            logger.info(f"  Instance {test_name} has no items — skipped.")
            continue

        checkpoint_path = os.path.join(output_dir, f"{test_name}_checkpoint.pkl")
        start_batch_idx = 1

        calls_before = _rotator.total_calls if _rotator is not None else 0

        if resume_from and i == 0 and os.path.exists(resume_from):
            engine = HyperHeuristicEngine.load_checkpoint(resume_from, config=config)
            start_batch_idx = len(engine.history) + 1
            logger.info(f"Resumed {test_name} from {resume_from} at batch {start_batch_idx}")
        else:
            engine = HyperHeuristicEngine(seed=seed, config=config)
            engine.initialize(batches[0], capacity=instance_capacity, initial_pool_size=initial_pool_size)
            start_batch_idx = 2
            logger.info(f"Initial Portfolio: {[h.full_id for h in engine.portfolio]}")
            label = f"[{test_name}] Batch 001/{len(batches):03d}"
            logger.info("\n" + format_beam_profile(label, engine.beam))
            engine._log_batch_record(beam_profile_record(label, engine.beam, None))

        for t, batch in enumerate(batches[1:], start=2):
            if t < start_batch_idx:
                continue
            trigger = engine.step(batch)
            status = f"Trigger: {trigger['type']}({trigger['severity']})" if trigger else "No Trigger"
            label = f"[{test_name}] Batch {t:03d}/{len(batches):03d} | {status}"
            logger.info("\n" + format_beam_profile(label, engine.beam))
            engine._log_batch_record(beam_profile_record(label, engine.beam, trigger))

            if checkpoint_every > 0 and t % checkpoint_every == 0:
                engine.save_checkpoint(checkpoint_path)

        final_node, path = engine.finalize()
        gap = final_node.f1 - lower_bound

        calls_after = _rotator.total_calls if _rotator is not None else 0
        api_calls_used = calls_after - calls_before

        logger.info(f"--- Result for {test_name} ---")
        logger.info(f"Lower bound: {lower_bound} bins")
        logger.info(f"Final bin count: {final_node.f1}  (gap to lower bound: +{gap})")
        logger.info(f"Final utilization: {final_node.f2:.2%}")
        logger.info(f"API calls used: {api_calls_used}")
        logger.info(f"Heuristic sequence (versioned): {path}")
        logger.info(f"Final problem state (remaining bin capacities): {final_node.state.bins}")

        file_prefix = f"{dataset_name}_{instance_no}"

        txt_path = os.path.join(best_sequence_dir, f"{file_prefix}_best_sequence.txt")
        export_heuristic_sequence(final_node, path, engine.all_heuristics, txt_path, instance_name=test_name)
        logger.info(f"[Best sequence saved]: {txt_path}")

        png_path = os.path.join(tree_search_dir, f"{file_prefix}_tree_search.png")
        plot_tree_search(engine.history, final_node, png_path, title=f"{test_name} — Tree-Search Progression")

        log_path = os.path.join(run_log_dir, f"{file_prefix}_run_log.jsonl")
        engine.flush_batch_log(log_path)
        logger.info(f"[Structured run log saved]: {log_path}")

        all_results[test_name] = {
            "capacity": instance_capacity,
            "lower_bound": lower_bound,
            "bins": final_node.f1,
            "gap": gap,
            "utilization": final_node.f2,
            "api_calls": api_calls_used,
            "path": path,
            "final_state": list(final_node.state.bins),
            "sequence_file": txt_path,
            "plot_file": png_path,
            "log_file": log_path,
        }

        if checkpoint_every > 0 and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        is_last = (i == len(instances) - 1)
        if not is_last and sleep_seconds > 0:
            logger.info(f"... Resting for {sleep_seconds} seconds before the next instance ...")
            time.sleep(sleep_seconds)

    if all_results:
        gap_pct_list = [
            (r["gap"] / r["lower_bound"]) * 100 if r["lower_bound"] > 0 else 0.0
            for r in all_results.values()
        ]
        util_list = [r["utilization"] for r in all_results.values()]
        api_calls_list = [r["api_calls"] for r in all_results.values()]

        dataset_summary = {
            "n_instances": len(all_results),
            "avg_gap_pct": statistics.mean(gap_pct_list),
            "avg_utilization": statistics.mean(util_list),
            "avg_api_calls": statistics.mean(api_calls_list),
        }

        logger.info("\n" + "=" * 70)
        logger.info("           DATASET-LEVEL SUMMARY (averaged over all instances)")
        logger.info("=" * 70)
        logger.info(f"  Instances solved     : {dataset_summary['n_instances']}")
        logger.info(f"  Avg optimality gap % : {dataset_summary['avg_gap_pct']:.2f}%")
        logger.info(f"  Avg utilization      : {dataset_summary['avg_utilization']:.2%}")
        logger.info(f"  Avg API calls        : {dataset_summary['avg_api_calls']:.2f}")
        logger.info("=" * 70)

        local_summary_path = os.path.join(output_dir, "dataset_summary.json")
        with open(local_summary_path, "w") as f:
            json.dump(dataset_summary, f, indent=2)
        logger.info(f"[Dataset summary saved]: {local_summary_path}")

        global_summary_path = os.path.join(BASE_DIR, f"summary_{dataset_name}.json")
        with open(global_summary_path, "w") as f:
            json.dump(dataset_summary, f, indent=2)
        logger.info(f"[Dataset summary also saved]: {global_summary_path}")

    return all_results

def run_on_dataset_multi_seed(
    dataset: Dict[str, Dict],
    seeds: List[int],
    batch_size: int,
    initial_pool_size: int = 10,
    sleep_seconds: int = 0,
    output_dir: str = "bin_packing_outputs",
    config: Optional[EngineConfig] = None,
    checkpoint_every: int = 0,
) -> Dict[str, Any]:
    per_seed: Dict[int, Dict[str, Dict]] = {}
    for seed in seeds:
        seed_output_dir = os.path.join(output_dir, f"seed_{seed}")
        logger.info(f"===== Multi-seed run: seed={seed} =====")
        per_seed[seed] = run_on_dataset(
            dataset,
            batch_size=batch_size,
            initial_pool_size=initial_pool_size,
            sleep_seconds=sleep_seconds,
            seed=seed,
            output_dir=seed_output_dir,
            config=config,
            checkpoint_every=checkpoint_every,
        )

    instance_names = set()
    for results in per_seed.values():
        instance_names.update(results.keys())

    aggregated: Dict[str, Dict[str, Any]] = {}
    for name in sorted(instance_names):
        bins_vals = [per_seed[s][name]["bins"] for s in seeds if name in per_seed[s]]
        gap_vals = [per_seed[s][name]["gap"] for s in seeds if name in per_seed[s]]
        util_vals = [per_seed[s][name]["utilization"] for s in seeds if name in per_seed[s]]
        api_calls_vals = [per_seed[s][name]["api_calls"] for s in seeds if name in per_seed[s]]

        def _mean_std(vals: List[float]) -> Tuple[float, float]:
            if not vals:
                return (float("nan"), float("nan"))
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return (mean, std)

        bins_mean, bins_std = _mean_std(bins_vals)
        gap_mean, gap_std = _mean_std(gap_vals)
        util_mean, util_std = _mean_std(util_vals)
        api_calls_mean, api_calls_std = _mean_std(api_calls_vals)

        aggregated[name] = {
            "bins_mean": bins_mean,
            "bins_stdev": bins_std,
            "gap_mean": gap_mean,
            "gap_stdev": gap_std,
            "utilization_mean": util_mean,
            "utilization_stdev": util_std,
            "api_calls_mean": api_calls_mean,
            "api_calls_stdev": api_calls_std,
            "n_seeds": len(bins_vals),
        }
        logger.info(
            f"[{name}] bins={bins_mean:.2f}±{bins_std:.2f}  "
            f"gap={gap_mean:.2f}±{gap_std:.2f}  "
            f"utilization={util_mean:.2%}±{util_std:.2%}  (n={len(bins_vals)} seeds)"
            f"api_calls={api_calls_mean:.1f}±{api_calls_std:.1f}  (n={len(bins_vals)} seeds)"
        )

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "multi_seed_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"seeds": seeds, "aggregated": aggregated}, f, indent=2)
    logger.info(f"[Multi-seed summary saved]: {summary_path}")

    return {"per_seed": per_seed, "aggregated": aggregated}

def run_ablation_study(
    dataset: Dict[str, Dict],
    dataset_name: str,
    batch_size: int = 50,
    seed: int = 7,
    output_dir: str = "ablation_results"
) -> None:
    experiments = {
        "Proposed (Full)": EngineConfig(
            sort_batch_descending=True,
            use_diversity_preservation_crossover=True,
            use_trigger_monitor=True
        ),
        "w/o Descending Sort": EngineConfig(
            sort_batch_descending=False,
            use_diversity_preservation_crossover=True,
            use_trigger_monitor=True
        ),
        "w/o Diversity Preservation": EngineConfig(
            sort_batch_descending=True,
            use_diversity_preservation_crossover=False,
            use_trigger_monitor=True
        ),
        "w/o EWMA Monitor (Fixed Steps)": EngineConfig(
            sort_batch_descending=True,
            use_diversity_preservation_crossover=True,
            use_trigger_monitor=False,
            periodic_trigger_interval=5
        )
    }

    config_results: Dict[str, Dict[str, Any]] = {}

    for exp_name, config in experiments.items():
        logger.info(f"\n==========================================")
        logger.info(f" Starting Ablation Variant: {exp_name}")
        logger.info(f"==========================================")

        exp_dir = os.path.join(output_dir, exp_name.replace(" ", "_").replace("/", "_"))
        results = run_on_dataset(
            dataset=dataset,
            batch_size=batch_size,
            dataset_name=dataset_name,
            initial_pool_size=10,
            seed=seed,
            output_dir=exp_dir,
            config=config,
            sleep_seconds=0
        )
        config_results[exp_name] = results

    print("\n" + "=" * 115)
    print("                              ABLATION STUDY COMPARISON RESULTS")
    print("=" * 115)

    all_instances = set()
    for results in config_results.values():
        all_instances.update(results.keys())

    for instance_name in sorted(all_instances):
        print(f"\n{'Instance: ' + instance_name:<60}")
        print(f"{'Config Name':<30} | {'Bins':<6} | {'Gap':<14} | {'Utilization':<12} | {'API Calls':<10}")
        print("-" * 115)

        for exp_name in experiments.keys():
            if instance_name in config_results[exp_name]:
                m = config_results[exp_name][instance_name]
                gap_pct = (m['gap'] / m['lower_bound']) * 100 if m['lower_bound'] > 0 else 0
                print(
                    f"{exp_name:<30} | {m['bins']:<6} | "
                    f"+{m['gap']} ({gap_pct:>5.1f}%) | {m['utilization']:>10.2%} | "
                    f"{m.get('api_calls', 0):<10}"
                )

    print("\n" + "=" * 115)
    print("                              AVERAGE RESULTS ACROSS ALL INSTANCES")
    print("=" * 115)
    print(
        f"{'Config Name':<30} | {'Avg Bins':<10} | {'Avg Gap':<10} | "
        f"{'Avg Gap %':<10} | {'Avg Util':<10} | {'Avg API Calls':<14}"
    )
    print("-" * 115)

    summary_data: Dict[str, Any] = {}
    for exp_name in experiments.keys():
        results = config_results[exp_name]
        if not results:
            continue

        bins_list = [m['bins'] for m in results.values()]
        gap_list = [m['gap'] for m in results.values()]
        util_list = [m['utilization'] for m in results.values()]
        api_calls_list = [m.get('api_calls', 0) for m in results.values()]
        gap_pct_list = [
            (m['gap'] / m['lower_bound']) * 100 if m['lower_bound'] > 0 else 0
            for m in results.values()
        ]

        avg_bins = statistics.mean(bins_list)
        avg_gap = statistics.mean(gap_list)
        avg_gap_pct = statistics.mean(gap_pct_list)
        avg_util = statistics.mean(util_list)
        avg_api_calls = statistics.mean(api_calls_list)

        print(
            f"{exp_name:<30} | {avg_bins:<10.2f} | {avg_gap:<10.2f} | "
            f"{avg_gap_pct:<9.2f}% | {avg_util:<9.2%} | {avg_api_calls:<14.2f}"
        )

        summary_data[exp_name] = {
            "avg_bins": avg_bins,
            "avg_gap": avg_gap,
            "avg_gap_pct": avg_gap_pct,
            "avg_utilization": avg_util,
            "avg_api_calls": avg_api_calls,
            "instances": len(results),
            "details": {
                inst_name: {
                    "bins": m['bins'],
                    "gap": m['gap'],
                    "gap_pct": (m['gap'] / m['lower_bound']) * 100 if m['lower_bound'] > 0 else 0,
                    "utilization": f"{m['utilization']:.2%}",
                    "api_calls": m.get('api_calls', 0),
                }
                for inst_name, m in results.items()
            }
        }

    print("=" * 115 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"[Ablation Summary saved]: {summary_path}")

if __name__ == "__main__":

    data_files = list(DATA_DIR.glob("*.json"))

    if not data_files:
        logger.warning(
            f"No JSON files found in directory '{DATA_DIR}'. Running demo instead."
        )
        run_demo(output_dir="bin_packing_outputs", config=EngineConfig())
    else:
        print("\n" + "-" * 50)
        print(" Select Execution Mode:")
        print(" [1] Standard Run (auto-runs ALL datasets in 'data/' sequentially)")
        print(" [2] Automated Ablation Study (choose ONE dataset)")
        print("-" * 50)
        run_mode = input("Select mode (1 or 2 - default 1): ").strip()

        seeds_raw = input(
            "Seeds (comma-separated, blank = single seed 7): "
        ).strip()
        seeds = (
            [int(s) for s in seeds_raw.split(",")] if seeds_raw else [7]
        )

        DATASET_REST_SECONDS = 300

        if run_mode == "2":
            print("\n" + "=" * 50)
            print(" Available Datasets in 'data/' directory:")
            for idx, file_path in enumerate(data_files, start=1):
                print(f" [{idx}] {file_path.name}")
            print("=" * 50)

            selected_idx = input(
                f"\nSelect dataset number (1 to {len(data_files)} - default 1): "
            ).strip()
            try:
                choice_num = int(selected_idx) - 1
                if choice_num < 0 or choice_num >= len(data_files):
                    choice_num = 0
            except ValueError:
                choice_num = 0

            chosen_file = data_files[choice_num]
            print(f"\nSelected dataset: {chosen_file.name}")

            dataset_dict = load_dataset(chosen_file.name)
            dataset_key = chosen_file.stem

            user_batch_size = int(
                input("Batch size (default 50): ").strip() or "50"
            )

            logger.info("Starting Automated Ablation Study Pipeline...")
            run_ablation_study(
                dataset=dataset_dict,
                dataset_name=dataset_key,
                batch_size=user_batch_size,
                seed=seeds[0],
                output_dir=os.path.join("ablation_results", dataset_key),
            )

        else:
            print("\n" + "=" * 50)
            print(" Datasets found in 'data/' directory (will run ALL automatically):")
            for idx, file_path in enumerate(data_files, start=1):
                print(f" [{idx}] {file_path.name}")
            print("=" * 50)

            user_initial_pool = int(
                input("Enter Initial Pool Size (default 10): ").strip() or "10"
            )
            sort_raw = input(
                "Sort each batch largest-to-smallest before packing? (Y/n): "
            ).strip().lower()
            sort_batch_descending = sort_raw not in ("n", "no")

            engine_config = EngineConfig(sort_batch_descending=sort_batch_descending)

            print("\n" + "-" * 50)
            print(" Set batch size for each dataset:")
            print("-" * 50)
            batch_sizes: Dict[str, int] = {}
            for file_path in data_files:
                raw = input(f"  Batch size for '{file_path.name}' (default 50): ").strip()
                try:
                    batch_sizes[file_path.stem] = int(raw) if raw else 50
                except ValueError:
                    batch_sizes[file_path.stem] = 50

            for d_idx, file_path in enumerate(data_files, start=1):
                dataset_key = file_path.stem
                dataset_batch_size = batch_sizes[dataset_key]
                logger.info(
                    f"\n########## Dataset {d_idx}/{len(data_files)}: {file_path.name} "
                    f"(batch_size={dataset_batch_size}) ##########"
                )
                dataset_dict = load_dataset(file_path.name)
                dataset_output_dir = os.path.join("bin_packing_outputs", dataset_key)

                if len(seeds) == 1:
                    run_on_dataset(
                        dataset_dict,
                        batch_size=dataset_batch_size,
                        dataset_name=dataset_key,
                        initial_pool_size=user_initial_pool,
                        sleep_seconds=0,
                        seed=seeds[0],
                        output_dir=dataset_output_dir,
                        config=engine_config,
                        checkpoint_every=50,
                    )
                else:
                    run_on_dataset_multi_seed(
                        dataset_dict,
                        seeds=seeds,
                        batch_size=dataset_batch_size,
                        initial_pool_size=user_initial_pool,
                        sleep_seconds=0,
                        output_dir=dataset_output_dir,
                        config=engine_config,
                        checkpoint_every=50,
                    )

                is_last_dataset = (d_idx == len(data_files))
                if not is_last_dataset:
                    logger.info(
                        f"... Resting for {DATASET_REST_SECONDS} seconds before the next dataset ..."
                    )
                    time.sleep(DATASET_REST_SECONDS)

            logger.info("\n########## All datasets processed. ##########")