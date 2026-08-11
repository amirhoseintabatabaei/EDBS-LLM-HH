import os
import json
import numpy as np
folder_path = os.path.dirname(os.path.abspath(__file__))

def generate_or_dataset(dataset_name, num_instances=5, num_items=500, capacity=100, min_w=20, max_w=100, seed=42, folder_path=folder_path):
    np.random.seed(seed)
    dataset = {}
    for i in range(num_instances):
        inst_key = f"u{num_items}_{i:02d}"
        items = np.random.randint(min_w, max_w + 1, size=num_items).tolist()
        dataset[inst_key] = {
            "capacity": capacity,
            "num_items": num_items,
            "items": items
        }
    
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, f"{dataset_name}.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)

def generate_weibull_dataset(size_label, num_items, num_instances=5, capacity=150, alpha=4.5, beta=45.0, seed=100, folder_path=folder_path):
    np.random.seed(seed)
    dataset = {}
    for i in range(num_instances):
        inst_key = f"test_{i}"
        raw_samples = np.random.weibull(alpha, size=num_items) * beta
        items = np.clip(np.round(raw_samples), 1, capacity).astype(int).tolist()
        dataset[inst_key] = {
            "capacity": capacity,
            "num_items": num_items,
            "items": items
        }
    
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, f"Weibull_{size_label}.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)


def generate_shift_dataset(
    dataset_name,
    num_instances=5,
    num_items=500,
    capacity=100,
    num_segments=4,
    shift_mode="abrupt",        
    segment_ranges=None,        
    gradual_transition_frac=0.15, 
    seed=200,
    folder_path=folder_path,
    record_shift_points=True,     
):
   
    rng = np.random.RandomState(seed)
    dataset = {}


    if segment_ranges is None:
        low_range = (max(1, int(capacity * 0.05)), int(capacity * 0.25))
        high_range = (int(capacity * 0.45), int(capacity * 0.80))
        segment_ranges = [
            low_range if s % 2 == 0 else high_range
            for s in range(num_segments)
        ]

    base_len = num_items // num_segments
    seg_lengths = [base_len] * num_segments
    seg_lengths[-1] += num_items - base_len * num_segments  

    for i in range(num_instances):
        inst_key = f"shift_{shift_mode}_{i:02d}"
        items = []
        shift_points = []  
        cursor = 0

        for seg_idx, (min_w, max_w) in enumerate(segment_ranges):
            seg_len = seg_lengths[seg_idx]
            shift_points.append(cursor)

            if shift_mode == "abrupt" or seg_idx == 0:
                seg_items = rng.randint(min_w, max_w + 1, size=seg_len).tolist()

            else:
                prev_min_w, prev_max_w = segment_ranges[seg_idx - 1]
                n_transition = max(1, int(seg_len * gradual_transition_frac))
                seg_items = []
                for pos in range(seg_len):
                    if pos < n_transition:
                        p_prev = 1.0 - (pos / n_transition)
                        if rng.random() < p_prev:
                            val = rng.randint(prev_min_w, prev_max_w + 1)
                        else:
                            val = rng.randint(min_w, max_w + 1)
                    else:
                        val = rng.randint(min_w, max_w + 1)
                    seg_items.append(int(val))

            items.extend(seg_items)
            cursor += seg_len

        dataset[inst_key] = {
            "capacity": capacity,
            "num_items": num_items,
            "items": items,
        }
        if record_shift_points:
            dataset[inst_key]["shift_metadata"] = {
                "mode": shift_mode,
                "num_segments": num_segments,
                "segment_ranges": segment_ranges,
                "shift_points": shift_points,
            }

    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, f"{dataset_name}.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)


for i in range(1, 5):
    generate_or_dataset(
        dataset_name=f"OR{i}",
        num_instances=5,
        num_items=500,
        capacity=100,
        seed=40 + i
    )

weibull_sizes = {
    "5k": 5000,
    "10k": 10000,
    "20k": 20000,
    "50k": 50000
}

for label, n_items in weibull_sizes.items():
    generate_weibull_dataset(
        size_label=label,
        num_items=n_items,
        num_instances=5,
        capacity=150,
        seed=100 + n_items // 1000
    )


generate_shift_dataset(
    dataset_name="Shift_Abrupt",
    num_instances=5,
    num_items=500,
    capacity=100,
    num_segments=4,
    shift_mode="abrupt",
    seed=201,
)

generate_shift_dataset(
    dataset_name="Shift_Gradual",
    num_instances=5,
    num_items=500,
    capacity=100,
    num_segments=4,
    shift_mode="gradual",
    gradual_transition_frac=0.15,
    seed=202,
)

generate_shift_dataset(
    dataset_name="Shift_Abrupt_Large",
    num_instances=5,
    num_items=5000,
    capacity=150,
    num_segments=6,
    shift_mode="abrupt",
    seed=203,
)