# ============================================================
# SURROGATE_BENCHMARK.PY
# Phase D: Full 40-Problem Benchmark Evaluation & Statistical Analysis
#
# Evaluates Baseline, GHCA, Heuristic-GHCA, Linear-ML-GHCA, and
# Surrogate-GHCA across all 40 dual-scale benchmark problems.
# Computes paired Wilcoxon signed-rank tests and rank-biserial effect sizes.
# ============================================================

import copy
import time
import csv
import os
import numpy as np
from scipy import stats

from scheduler import generate_problem, calculate_oct, calculate_load_balance, combined_cost
from genetic_algorithm import genetic_algorithm, genetic_algorithm_surrogate
from ml_layer import train_model, ml_ghca

DIR_BENCHMARKS = "dataset/benchmarks"
CSV_SURROGATE_RESULTS = os.path.join(DIR_BENCHMARKS, "surrogate_benchmark_results.csv")
CSV_STATISTICAL_TESTS = os.path.join(DIR_BENCHMARKS, "surrogate_statistical_tests.csv")


def rank_biserial_correlation(x, y):
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    w_pos = np.sum(ranks[diff > 0])
    w_neg = np.sum(ranks[diff < 0])
    tot = w_pos + w_neg
    return float((w_pos - w_neg) / tot) if tot > 0 else 0.0


def run_surrogate_benchmarks():
    os.makedirs(DIR_BENCHMARKS, exist_ok=True)

    print("=" * 80)
    print("PHASE D: RUNNING 40 DUAL-SCALE BENCHMARK EVALUATIONS WITH SURROGATE-GHCA")
    print("=" * 80)

    # 1. Train Linear ML model once for comparison baseline
    print("Training standard Linear ML model for comparison...")
    linear_model = train_model()

    # 40 Benchmarks identical to Phases 1-3
    benchmarks = []
    # Set 1: pool 1, small (P1-P10)
    for i in range(1, 11):
        benchmarks.append((f"P{i}", 8, 1, (i % 3) + 1, i))
    # Set 2: pool 1, large (P11-P20)
    for i in range(11, 21):
        benchmarks.append((f"P{i}", 18, 1, (i % 3) + 1, i))
    # Set 3: pool 2, small (P21-P30)
    for i in range(21, 31):
        benchmarks.append((f"P{i}", 8, 2, (i % 3) + 1, i))
    # Set 4: pool 2, large (P31-P40)
    for i in range(31, 41):
        benchmarks.append((f"P{i}", 18, 2, (i % 3) + 1, i))

    results = []

    for prob_id, num_parts, pool_num, layout_num, seed in benchmarks:
        ops = generate_problem(num_parts=num_parts, pool_num=pool_num, layout_num=layout_num, seed=seed)
        regime = "Small" if num_parts == 8 else "Large"

        # 1. Baseline
        base_oct = calculate_oct(ops)
        base_lb = calculate_load_balance(ops)
        base_cost = combined_cost(ops)

        # 2. Plain GHCA (200 gen)
        t0 = time.perf_counter()
        ghca_seq, ghca_cost = genetic_algorithm(ops)
        t_ghca = time.perf_counter() - t0
        ghca_oct = calculate_oct(ghca_seq)
        ghca_lb = calculate_load_balance(ghca_seq)

        # 3. Heuristic-GHCA (LPT sort + 200 gen)
        heur_ops = copy.deepcopy(ops)
        heur_ops.sort(key=lambda x: x["processing_time"], reverse=True)
        heur_seq, heur_cost = genetic_algorithm(heur_ops)
        heur_oct = calculate_oct(heur_seq)
        heur_lb = calculate_load_balance(heur_seq)

        # 4. Linear-ML-GHCA
        lin_seq, lin_cost = ml_ghca(ops, linear_model)
        lin_oct = calculate_oct(lin_seq)
        lin_lb = calculate_load_balance(lin_seq)

        # 5. Surrogate-GHCA (surrogate accelerated)
        t0 = time.perf_counter()
        sur_seq, sur_cost = genetic_algorithm_surrogate(
            ops,
            generations=1200,
            pop_size=20,
            true_eval_interval=40,
            true_eval_top_k=3,
        )
        t_sur = time.perf_counter() - t0
        sur_oct = calculate_oct(sur_seq)
        sur_lb = calculate_load_balance(sur_seq)

        pct_sur_vs_ghca = round((ghca_cost - sur_cost) / ghca_cost * 100, 2)
        pct_sur_vs_heur = round((heur_cost - sur_cost) / heur_cost * 100, 2)
        pct_sur_vs_lin = round((lin_cost - sur_cost) / lin_cost * 100, 2)
        pct_sur_vs_base = round((base_cost - sur_cost) / base_cost * 100, 2)

        results.append({
            "Problem": prob_id,
            "Regime": regime,
            "Jobs": num_parts,
            "Ops": len(ops),
            "Baseline_Cost": base_cost,
            "GHCA_Cost": ghca_cost,
            "Heuristic_Cost": heur_cost,
            "Linear_ML_Cost": lin_cost,
            "Surrogate_Cost": sur_cost,
            "Baseline_OCT": base_oct,
            "GHCA_OCT": ghca_oct,
            "Heuristic_OCT": heur_oct,
            "Linear_OCT": lin_oct,
            "Surrogate_OCT": sur_oct,
            "Baseline_LB": base_lb,
            "GHCA_LB": ghca_lb,
            "Heuristic_LB": heur_lb,
            "Linear_LB": lin_lb,
            "Surrogate_LB": sur_lb,
            "Time_GHCA_s": round(t_ghca, 3),
            "Time_Surrogate_s": round(t_sur, 3),
            "Pct_vs_Baseline": pct_sur_vs_base,
            "Pct_vs_GHCA": pct_sur_vs_ghca,
            "Pct_vs_Heuristic": pct_sur_vs_heur,
            "Pct_vs_Linear_ML": pct_sur_vs_lin,
        })

        print(
            f"{prob_id:<5} ({regime:<5}) | GHCA: {ghca_cost:<7.2f} | Heur: {heur_cost:<7.2f} | "
            f"LinML: {lin_cost:<7.2f} | Surr: {sur_cost:<7.2f} | Surr vs GHCA: {pct_sur_vs_ghca:>+6.2f}%"
        )

    # Save to CSV
    with open(CSV_SURROGATE_RESULTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved raw benchmark results to {CSV_SURROGATE_RESULTS}")

    # ------------------------------------------------------------
    # STATISTICAL HYPOTHESIS TESTING
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS & PAIRED WILCOXON SIGNED-RANK TESTS")
    print("=" * 80)

    test_summaries = []

    for scope in ["Small", "Large", "Pooled"]:
        if scope == "Pooled":
            subset = results
            n_items = len(subset)
            scope_title = f"POOLED AGGREGATE (N={n_items} Problems)"
        else:
            subset = [r for r in results if r["Regime"] == scope]
            n_items = len(subset)
            scope_title = f"REGIME: {scope.upper()} SCALE (N={n_items} Problems)"

        base_costs = [r["Baseline_Cost"] for r in subset]
        ghca_costs = [r["GHCA_Cost"] for r in subset]
        heur_costs = [r["Heuristic_Cost"] for r in subset]
        lin_costs = [r["Linear_ML_Cost"] for r in subset]
        sur_costs = [r["Surrogate_Cost"] for r in subset]

        print(f"\n--- {scope_title} ---")
        print(f"Mean Costs: Baseline={np.mean(base_costs):.2f}, GHCA={np.mean(ghca_costs):.2f}, "
              f"Heuristic={np.mean(heur_costs):.2f}, Linear-ML={np.mean(lin_costs):.2f}, Surrogate={np.mean(sur_costs):.2f}")

        comparisons = [
            ("Baseline vs Surrogate", base_costs, sur_costs),
            ("GHCA vs Surrogate", ghca_costs, sur_costs),
            ("Heuristic vs Surrogate", heur_costs, sur_costs),
            ("Linear-ML vs Surrogate", lin_costs, sur_costs),
        ]

        for comp_name, ctrl, exp in comparisons:
            diff = np.array(ctrl) - np.array(exp)
            wins = int(np.sum(diff > 0))
            ties = int(np.sum(diff == 0))
            losses = int(np.sum(diff < 0))

            try:
                w_stat, p_val = stats.wilcoxon(ctrl, exp, alternative="greater")
            except Exception:
                w_stat, p_val = np.nan, np.nan

            r_effect = rank_biserial_correlation(ctrl, exp)
            mean_diff = float(np.mean(diff))
            mean_pct = float(np.mean((diff / np.array(ctrl)) * 100))

            verdict = "Significant Advantage (p < 0.05)" if p_val < 0.05 else (
                "Marginal (0.05 <= p < 0.10)" if p_val < 0.10 else "Null / No Significant Difference"
            )

            print(f"  * {comp_name:<25}: Diff = {mean_diff:>+6.2f} ({mean_pct:>+6.2f}%) | "
                  f"W/T/L = {wins:>2}/{ties:>2}/{losses:>2} | Wilcoxon W = {w_stat:<6.1f}, p = {p_val:.5f} | "
                  f"Rank-biserial r = {r_effect:>+6.3f} | {verdict}")

            test_summaries.append({
                "Scope": scope,
                "Comparison": comp_name,
                "Control_Mean": round(float(np.mean(ctrl)), 2),
                "Surrogate_Mean": round(float(np.mean(exp)), 2),
                "Mean_Diff": round(mean_diff, 2),
                "Mean_Pct_Gain": round(mean_pct, 2),
                "Wins": wins,
                "Ties": ties,
                "Losses": losses,
                "Wilcoxon_W": round(float(w_stat), 2) if not np.isnan(w_stat) else "",
                "P_Value": round(float(p_val), 5) if not np.isnan(p_val) else "",
                "Rank_Biserial_r": round(r_effect, 3),
                "Verdict": verdict,
            })

    # Save test summaries
    with open(CSV_STATISTICAL_TESTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=test_summaries[0].keys())
        writer.writeheader()
        writer.writerows(test_summaries)

    print(f"\nSaved statistical test results to {CSV_STATISTICAL_TESTS}")
    print("=" * 80)

    return results, test_summaries


if __name__ == "__main__":
    run_surrogate_benchmarks()
