# ============================================================
# SURROGATE_DATA_GENERATOR.PY
# Phase A: Data Generation & Sequence-Level Feature Engineering
#
# Generates diverse problem instances (cached in dataset/raw/),
# extracts 28 order-dependent features for sequence permutations,
# computes exact combined_cost(), and saves train/test splits.
# ============================================================

import os
import time
import math
import random
import pickle
import csv
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.core.scheduler import generate_problem, combined_cost, MACHINES, VEHICLES
    from src.core.hill_climbing import hill_climbing
except ImportError:
    from scheduler import generate_problem, combined_cost, MACHINES, VEHICLES
    from hill_climbing import hill_climbing

# Paths
DIR_DATASET_RAW = os.path.join(PROJECT_ROOT, "dataset", "raw")
DIR_DATASET_TRAIN = os.path.join(PROJECT_ROOT, "dataset", "training")
PKL_PROBLEMS_CACHE = os.path.join(DIR_DATASET_RAW, "problems_cache.pkl")
CSV_ALL_FEATURES = os.path.join(DIR_DATASET_TRAIN, "surrogate_features.csv")
CSV_TRAIN_SPLIT = os.path.join(DIR_DATASET_TRAIN, "train_split.csv")
CSV_TEST_SPLIT = os.path.join(DIR_DATASET_TRAIN, "test_split.csv")

FEATURE_NAMES = [
    "f01_num_ops",
    "f02_num_jobs",
    "f03_mean_process",
    "f04_mean_travel",
    "f05_load_m1",
    "f06_load_m2",
    "f07_load_m3",
    "f08_load_m4",
    "f09_load_std",
    "f10_load_max",
    "f11_load_min",
    "f12_load_range",
    "f13_same_machine_adj",
    "f14_same_vehicle_adj",
    "f15_same_job_adj",
    "f16_travel_q1",
    "f17_travel_q4",
    "f18_process_q1",
    "f19_process_q4",
    "f20_pos_weighted_process",
    "f21_machine_entropy",
    "f22_max_same_machine_run",
    "f23_vehicle_switches",
    "f24_avg_job_gap",
    "f25_contention_x_maxload",
    "f26_norm_pos_weighted_process",
    "f27_relative_imbalance",
    "f28_run_x_travel",
]


# ------------------------------------------------------------
# 1. FAST SEQUENCE FEATURE EXTRACTION (Target < 0.1ms per sample)
# ------------------------------------------------------------
def extract_sequence_features(ops):
    """
    Extracts 28 fixed-length scalar features from an operation sequence.
    Captures both problem-level properties and order-dependent sequence dynamics.
    """
    n = len(ops)
    if n == 0:
        return np.zeros(28, dtype=np.float32)

    # Group 1: Problem-level properties
    p_times = [op["processing_time"] for op in ops]
    t_times = [op["travel_time"] for op in ops]
    jobs = [op["job"] for op in ops]
    machines = [op["machine"] for op in ops]
    vehicles = [op["vehicle"] for op in ops]

    unique_jobs = set(jobs)
    num_jobs = len(unique_jobs)
    mean_p = sum(p_times) / n
    mean_t = sum(t_times) / n

    # Group 2: Machine load distribution
    m_loads = {"M1": 0.0, "M2": 0.0, "M3": 0.0, "M4": 0.0}
    for m, p in zip(machines, p_times):
        m_loads[m] = m_loads.get(m, 0.0) + p

    loads_list = [m_loads["M1"], m_loads["M2"], m_loads["M3"], m_loads["M4"]]
    load_mean = sum(loads_list) / 4.0
    load_variance = sum((x - load_mean) ** 2 for x in loads_list) / 4.0
    load_std = math.sqrt(load_variance)
    load_max = max(loads_list)
    load_min = min(loads_list)
    load_range = load_max - load_min

    # Group 3: Order-dependent sequence dynamics
    same_machine_adj = 0
    same_vehicle_adj = 0
    same_job_adj = 0
    vehicle_switches = 0
    max_same_m_run = 1
    curr_same_m_run = 1

    # Machine transition matrix counts for entropy
    trans_counts = {}

    for i in range(n - 1):
        m_curr, m_next = machines[i], machines[i + 1]
        v_curr, v_next = vehicles[i], vehicles[i + 1]
        j_curr, j_next = jobs[i], jobs[i + 1]

        if m_curr == m_next:
            same_machine_adj += 1
            curr_same_m_run += 1
            if curr_same_m_run > max_same_m_run:
                max_same_m_run = curr_same_m_run
        else:
            curr_same_m_run = 1

        if v_curr == v_next:
            same_vehicle_adj += 1
        else:
            vehicle_switches += 1

        if j_curr == j_next:
            same_job_adj += 1

        pair = (m_curr, m_next)
        trans_counts[pair] = trans_counts.get(pair, 0) + 1

    # Shannon entropy of machine transitions
    total_trans = n - 1 if n > 1 else 1
    entropy = 0.0
    for cnt in trans_counts.values():
        p_trans = cnt / total_trans
        if p_trans > 0:
            entropy -= p_trans * math.log2(p_trans)

    # Positional quarter aggregations
    q_len = max(1, n // 4)
    travel_q1 = sum(t_times[:q_len]) / q_len
    travel_q4 = sum(t_times[-q_len:]) / q_len
    process_q1 = sum(p_times[:q_len]) / q_len
    process_q4 = sum(p_times[-q_len:]) / q_len

    # Position-weighted processing time sum
    pos_weighted_process = 0.0
    for i in range(n):
        weight = (n - i) / n
        pos_weighted_process += p_times[i] * weight

    # Average positional gap between operations of the same job
    job_indices = {}
    for i, j in enumerate(jobs):
        if j not in job_indices:
            job_indices[j] = []
        job_indices[j].append(i)

    gaps = []
    for idx_list in job_indices.values():
        if len(idx_list) > 1:
            for k in range(len(idx_list) - 1):
                gaps.append(idx_list[k + 1] - idx_list[k])
    avg_job_gap = (sum(gaps) / len(gaps)) if gaps else 1.0

    # Group 4: Interaction features
    contention_x_maxload = same_machine_adj * load_max
    norm_pos_weighted = pos_weighted_process / n
    rel_imbalance = (load_std / load_max) if load_max > 0 else 0.0
    run_x_travel = max_same_m_run * mean_t

    features = [
        float(n),
        float(num_jobs),
        float(mean_p),
        float(mean_t),
        float(loads_list[0]),
        float(loads_list[1]),
        float(loads_list[2]),
        float(loads_list[3]),
        float(load_std),
        float(load_max),
        float(load_min),
        float(load_range),
        float(same_machine_adj),
        float(same_vehicle_adj),
        float(same_job_adj),
        float(travel_q1),
        float(travel_q4),
        float(process_q1),
        float(process_q4),
        float(pos_weighted_process),
        float(entropy),
        float(max_same_m_run),
        float(vehicle_switches),
        float(avg_job_gap),
        float(contention_x_maxload),
        float(norm_pos_weighted),
        float(rel_imbalance),
        float(run_x_travel),
    ]

    return np.array(features, dtype=np.float32)


# ------------------------------------------------------------
# 2. PROBLEM INSTANCE GENERATION & CACHING
# ------------------------------------------------------------
def get_or_create_problems(n_problems=300):
    os.makedirs(DIR_DATASET_RAW, exist_ok=True)

    if os.path.exists(PKL_PROBLEMS_CACHE):
        print(f"Loading cached problems from {PKL_PROBLEMS_CACHE}...")
        with open(PKL_PROBLEMS_CACHE, "rb") as f:
            problems_dict = pickle.load(f)
        print(f"Loaded {len(problems_dict)} problem instances from cache.")
        return problems_dict

    print(f"Generating {n_problems} diverse Ulusoy-compliant problem instances...")
    problems_dict = {}
    part_sizes = [5, 8, 10, 12, 15, 18]

    for seed in range(1, n_problems + 1):
        num_parts = part_sizes[(seed - 1) % len(part_sizes)]
        pool_num = ((seed - 1) % 2) + 1
        layout_num = ((seed - 1) % 3) + 1

        ops = generate_problem(
            num_parts=num_parts,
            pool_num=pool_num,
            layout_num=layout_num,
            seed=seed
        )
        key = f"P_{seed}_np{num_parts}_pool{pool_num}_lay{layout_num}"
        problems_dict[key] = {
            "problem_id": key,
            "num_parts": num_parts,
            "pool_num": pool_num,
            "layout_num": layout_num,
            "seed": seed,
            "ops": ops,
        }

    with open(PKL_PROBLEMS_CACHE, "wb") as f:
        pickle.dump(problems_dict, f)

    print(f"Saved {len(problems_dict)} problem instances to {PKL_PROBLEMS_CACHE}")
    return problems_dict


# ------------------------------------------------------------
# 3. SEQUENCE PERMUTATION & DATASET BUILDING
# ------------------------------------------------------------
def build_surrogate_dataset(problems_dict, samples_per_problem=150):
    """
    Generates permutations per problem covering random, heuristic, and local-search spaces.
    Target: ~50,000+ samples.
    """
    os.makedirs(DIR_DATASET_TRAIN, exist_ok=True)

    print("=" * 80)
    print("PHASE A: GENERATING SURROGATE TRAINING DATASET")
    print("=" * 80)
    print(f"Problems: {len(problems_dict)} | Samples per problem: ~{samples_per_problem}")

    records = []
    t_start = time.perf_counter()

    for idx, (prob_key, prob_data) in enumerate(problems_dict.items(), start=1):
        base_ops = prob_data["ops"]
        problem_id = prob_data["problem_id"]

        # 1. Base order
        cost_base = combined_cost(base_ops)
        feat_base = extract_sequence_features(base_ops)
        records.append((problem_id, feat_base, cost_base))

        # 2. Heuristic order (LPT: longest processing time first)
        heur_ops = list(base_ops)
        heur_ops.sort(key=lambda x: x["processing_time"], reverse=True)
        cost_heur = combined_cost(heur_ops)
        feat_heur = extract_sequence_features(heur_ops)
        records.append((problem_id, feat_heur, cost_heur))

        # 3. SPT order (shortest processing time first)
        spt_ops = list(base_ops)
        spt_ops.sort(key=lambda x: x["processing_time"])
        cost_spt = combined_cost(spt_ops)
        feat_spt = extract_sequence_features(spt_ops)
        records.append((problem_id, feat_spt, cost_spt))

        # 4. Hill-climbing optimized sample (to sample high-fitness neighborhood)
        hc_seq, hc_cost = hill_climbing(base_ops, iterations=200)
        feat_hc = extract_sequence_features(hc_seq)
        records.append((problem_id, feat_hc, hc_cost))

        # 5. Random permutations
        remaining = samples_per_problem - 4
        for _ in range(remaining):
            perm = list(base_ops)
            random.shuffle(perm)
            cost_perm = combined_cost(perm)
            feat_perm = extract_sequence_features(perm)
            records.append((problem_id, feat_perm, cost_perm))

        if idx % 50 == 0 or idx == len(problems_dict):
            elapsed = time.perf_counter() - t_start
            print(f"  Processed {idx}/{len(problems_dict)} problems ({len(records)} samples, {elapsed:.1f}s)")

    print("-" * 80)
    print(f"Total samples generated: {len(records)}")

    # 4. Train / Test Split by Problem ID (prevent data leakage across problems)
    problem_keys = list(problems_dict.keys())
    random.seed(42)
    random.shuffle(problem_keys)

    n_train_probs = int(0.8 * len(problem_keys))
    train_prob_set = set(problem_keys[:n_train_probs])
    test_prob_set = set(problem_keys[n_train_probs:])

    header = ["problem_id"] + FEATURE_NAMES + ["true_cost"]

    # Write All, Train, and Test CSVs
    with open(CSV_ALL_FEATURES, "w", newline="", encoding="utf-8") as f_all, \
         open(CSV_TRAIN_SPLIT, "w", newline="", encoding="utf-8") as f_train, \
         open(CSV_TEST_SPLIT, "w", newline="", encoding="utf-8") as f_test:

        w_all = csv.writer(f_all)
        w_train = csv.writer(f_train)
        w_test = csv.writer(f_test)

        w_all.writerow(header)
        w_train.writerow(header)
        w_test.writerow(header)

        n_train_samples = 0
        n_test_samples = 0

        for prob_id, feat, cost in records:
            row = [prob_id] + list(feat) + [round(cost, 2)]
            w_all.writerow(row)

            if prob_id in train_prob_set:
                w_train.writerow(row)
                n_train_samples += 1
            else:
                w_test.writerow(row)
                n_test_samples += 1

    print(f"Saved complete dataset to {CSV_ALL_FEATURES}")
    print(f"Train split: {n_train_samples} samples ({len(train_prob_set)} problems) -> {CSV_TRAIN_SPLIT}")
    print(f"Test split:  {n_test_samples} samples ({len(test_prob_set)} problems) -> {CSV_TEST_SPLIT}")

    # Benchmark feature extraction speed
    sample_ops = list(problems_dict[problem_keys[0]]["ops"])
    n_runs = 2000
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = extract_sequence_features(sample_ops)
    t_feat = (time.perf_counter() - t0) / n_runs * 1e6

    print("-" * 80)
    print(f"Speed Check: Feature extraction takes {t_feat:.2f} microseconds per sequence")
    if t_feat < 100.0:
        print("[CHECKPOINT A2 PASSED] Extraction speed is well below 100 microseconds (0.1ms).")
    else:
        print("[WARNING] Extraction speed exceeds 100 microseconds target.")
    print("=" * 80)


if __name__ == "__main__":
    problems = get_or_create_problems(n_problems=300)
    build_surrogate_dataset(problems, samples_per_problem=170)
