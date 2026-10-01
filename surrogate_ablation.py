# ============================================================
# SURROGATE_ABLATION.PY
# Phase F: Ablation Studies & Robustness Verification for Surrogate-GHCA
#
# 1. Generation Scaling Curve (Diminishing Returns of Search Budget)
# 2. Re-calibration Interval Sensitivity (Surrogate Drift vs Speed)
# 3. Cost Weight Stability (0.5/0.5 vs 0.7/0.3 vs 0.9/0.1)
# ============================================================

import time
import numpy as np
from scheduler import generate_problem, combined_cost, calculate_oct, calculate_load_balance
from genetic_algorithm import genetic_algorithm, genetic_algorithm_surrogate


def run_generation_scaling_ablation():
    print("=" * 80)
    print("1. GENERATION SCALING ABLATION: SEARCH BUDGET VS SCHEDULE COST")
    print("=" * 80)

    # 6 representative test problems (3 small, 3 large)
    test_specs = [
        (8, 1, 1, 1),
        (8, 2, 2, 21),
        (8, 1, 3, 5),
        (18, 1, 1, 11),
        (18, 2, 2, 31),
        (18, 1, 3, 15),
    ]

    gen_budgets = [200, 500, 1000, 1500]

    print(f"{'Problem':<15} {'Std GA (200)':<14} " + " ".join([f"Surr-{g:<8}" for g in gen_budgets]))
    print("-" * 80)

    avg_costs = {g: [] for g in gen_budgets}
    avg_std = []

    for num_parts, pool, layout, seed in test_specs:
        ops = generate_problem(num_parts=num_parts, pool_num=pool, layout_num=layout, seed=seed)
        label = f"{num_parts}p-seed{seed}"

        _, c_std = genetic_algorithm(ops)
        avg_std.append(c_std)

        row_str = f"{label:<15} {c_std:<14.2f}"
        for g in gen_budgets:
            _, c_sur = genetic_algorithm_surrogate(
                ops,
                generations=g,
                pop_size=20,
                true_eval_interval=max(20, g // 25),
                true_eval_top_k=3,
            )
            avg_costs[g].append(c_sur)
            row_str += f" {c_sur:<12.2f}"
        print(row_str)

    print("-" * 80)
    mean_std = float(np.mean(avg_std))
    summary_str = f"{'MEAN COST':<15} {mean_std:<14.2f}"
    for g in gen_budgets:
        m_g = float(np.mean(avg_costs[g]))
        summary_str += f" {m_g:<12.2f}"
    print(summary_str)
    print("=" * 80 + "\n")


def run_interval_sensitivity_ablation():
    print("=" * 80)
    print("2. RE-EVALUATION INTERVAL SENSITIVITY (True Cost Calibration)")
    print("=" * 80)

    intervals = [25, 50, 100, 200]
    ops = generate_problem(num_parts=18, pool_num=1, layout_num=1, seed=12)

    print(f"{'Interval':<15} {'Final Cost':<15} {'Elapsed Time (s)':<18}")
    print("-" * 80)

    for interval in intervals:
        t0 = time.perf_counter()
        _, cost = genetic_algorithm_surrogate(
            ops,
            generations=1000,
            pop_size=20,
            true_eval_interval=interval,
            true_eval_top_k=3,
        )
        elapsed = time.perf_counter() - t0
        print(f"Every {interval:<9} {cost:<15.2f} {elapsed:<18.2f}")

    print("=" * 80 + "\n")


def run_weight_sensitivity_check():
    print("=" * 80)
    print("3. COST WEIGHT CONFIGURATION STABILITY")
    print("=" * 80)

    configs = [
        (0.5, 0.5, "Equal (0.5 / 0.5)"),
        (0.7, 0.3, "Default (0.7 / 0.3)"),
        (0.9, 0.1, "OCT-dominant (0.9 / 0.1)"),
    ]

    ops = generate_problem(num_parts=18, pool_num=1, layout_num=2, seed=14)

    seq_std, _ = genetic_algorithm(ops)
    seq_sur, _ = genetic_algorithm_surrogate(ops, generations=1000, true_eval_interval=50)

    oct_std = calculate_oct(seq_std)
    lb_std = calculate_load_balance(seq_std)

    oct_sur = calculate_oct(seq_sur)
    lb_sur = calculate_load_balance(seq_sur)

    print(f"{'Weight Config':<25} {'Std GA Cost':<15} {'Surrogate Cost':<15} {'Diff':<10}")
    print("-" * 80)

    for w1, w2, name in configs:
        cost_std = w1 * oct_std + w2 * lb_std
        cost_sur = w1 * oct_sur + w2 * lb_sur
        diff = cost_std - cost_sur
        print(f"{name:<25} {cost_std:<15.2f} {cost_sur:<15.2f} {diff:>+8.2f}")

    print("=" * 80)


if __name__ == "__main__":
    run_generation_scaling_ablation()
    run_interval_sensitivity_ablation()
    run_weight_sensitivity_check()
