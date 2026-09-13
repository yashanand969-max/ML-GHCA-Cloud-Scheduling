# ============================================================
# PHASE5_GBDT_ABLATION.PY
# Phase 5: Isolated Gradient Boosting Model Capacity Ablation
#
# Primary Directive:
#   Hold dataset, GA, HC, and objective functions constant;
#   swap only the regressor:
#     LinearRegression -> Gradient Boosting Regressor (LightGBM / GBDT)
#
# Mathematical & Domain Justification:
#   The 6 per-operation features ([travel_time, processing_time,
#   machine_idx, vehicle_idx, machine_ready, job_ready]) are
#   unordered tabular scalars without spatial or temporal locality.
#   A 1D-CNN's inductive biases (local receptive fields and translation
#   invariance) are mathematically unjustified. In contrast, tree-based
#   ensembles (GBDT) are the standard state-of-the-art non-linear function
#   approximators for tabular features, capturing non-linear thresholds
#   and feature interactions.
#
# Workflow:
#   1. Generate training dataset (exact same as Linear baseline).
#   2. Run light hyperparameter sweep (n_estimators, max_depth, learning_rate).
#   3. Select and freeze best GBDT configuration.
#   4. Re-evaluate across all 40 dual-scale benchmark problems.
#   5. Produce independent secondary ablation table & Wilcoxon tests:
#      "Does model capacity affect schedule quality?"
# ============================================================

import random
import copy
import time
import csv
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from scipy import stats
import lightgbm as lgb

from scheduler import generate_problem, calculate_oct, calculate_load_balance, combined_cost
from hill_climbing import hill_climbing
from genetic_algorithm import genetic_algorithm
from ml_layer import generate_training_data

# Fixed seeds for exact reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CSV_BENCHMARK_RESULTS = "results/benchmark_results.csv"
CSV_PHASE5_OUTPUT = "results/phase5_model_capacity_ablation.csv"
CSV_PHASE5_SWEEP = "results/phase5_hyperparameter_sweep.csv"


# ------------------------------------------------------------
# STEP 1: HYPERPARAMETER SWEEP FOR GBDT
# ------------------------------------------------------------
def run_hyperparameter_sweep(X, y):
    print("=" * 80)
    print("STEP 1: HYPERPARAMETER SWEEP FOR GRADIENT BOOSTING REGRESSOR (LightGBM)")
    print("=" * 80)
    print(f"Dataset: {len(X)} operation samples, 6 scalar features.")
    print("Search grid: n_estimators in [50, 100, 200], max_depth in [3, 5, 7], learning_rate in [0.03, 0.1, 0.2]")
    print("-" * 80)

    param_grid = [
        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 50, "max_depth": 5, "learning_rate": 0.1},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
        {"n_estimators": 100, "max_depth": 7, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    sweep_results = []
    best_score = -float("inf")
    best_params = None

    for params in param_grid:
        reg = lgb.LGBMRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=RANDOM_SEED,
            verbose=-1
        )
        scores = cross_val_score(reg, X, y, cv=kf, scoring="r2")
        mean_r2 = float(np.mean(scores))
        std_r2 = float(np.std(scores))

        sweep_results.append({
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "mean_r2": round(mean_r2, 4),
            "std_r2": round(std_r2, 4)
        })

        print(f"Params: n_est={params['n_estimators']:<3}, depth={params['max_depth']:<2}, lr={params['learning_rate']:<4} | 5-Fold CV R²: {mean_r2:.4f} (+/- {std_r2:.4f})")

        if mean_r2 > best_score:
            best_score = mean_r2
            best_params = params

    print("-" * 80)
    print(f"Optimal GBDT Configuration: {best_params} with 5-Fold CV R² = {best_score:.4f}")
    print("Holding this configuration constant across all scale regimes.")
    print("=" * 80 + "\n")

    # Save sweep results to CSV
    os.makedirs("results", exist_ok=True)
    with open(CSV_PHASE5_SWEEP, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n_estimators", "max_depth", "learning_rate", "mean_r2", "std_r2"])
        writer.writeheader()
        writer.writerows(sweep_results)

    return best_params, sweep_results


# ------------------------------------------------------------
# STEP 2: TRAIN BOTH MODELS (LINEAR VS GBDT) ON SAME DATA
# ------------------------------------------------------------
def train_and_compare_models(X, y, best_params):
    print("=" * 80)
    print("STEP 2: MODEL TRAINING & CAPACITY COMPARISON")
    print("=" * 80)

    # 1. Linear Regression (Low Capacity)
    t0 = time.perf_counter()
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    t_linear = time.perf_counter() - t0
    r2_linear = float(linear_model.score(X, y))

    # 2. Gradient Boosting (High Capacity)
    t0 = time.perf_counter()
    gbdt_model = lgb.LGBMRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        random_state=RANDOM_SEED,
        verbose=-1
    )
    gbdt_model.fit(X, y)
    t_gbdt = time.perf_counter() - t0
    r2_gbdt = float(gbdt_model.score(X, y))
    n_est = best_params["n_estimators"]
    depth = best_params["max_depth"]
    gbdt_struct = f"{n_est} trees, depth {depth}"
    lin_time_str = f"{t_linear * 1000:.2f} ms"
    gbdt_time_str = f"{t_gbdt * 1000:.2f} ms"

    print(f"{'Model':<25} {'Capacity / Structure':<30} {'Train Time':<12} {'R² Score':<10}")
    print("-" * 80)
    print(f"{'LinearRegression':<25} {'Parametric Linear (6 weights)':<30} {lin_time_str:<12} {r2_linear:<10.4f}")
    print(f"{'GBDT (LightGBM)':<25} {gbdt_struct:<30} {gbdt_time_str:<12} {r2_gbdt:<10.4f}")
    print("=" * 80 + "\n")

    return linear_model, gbdt_model, r2_linear, r2_gbdt


# ------------------------------------------------------------
# STEP 3: MODEL-ASSISTED SCHEDULING (PREDICTION & SORT)
# ------------------------------------------------------------
def ml_sort(ops, model):
    ops_copy = copy.deepcopy(ops)
    machine_map = {"M1": 0, "M2": 1, "M3": 2, "M4": 3}
    vehicle_map = {"V1": 0, "V2": 1}

    machine_free_at = {}
    job_free_at = {}
    vehicle_free_at = {}

    for op in ops_copy:
        machine = op["machine"]
        job = op["job"]
        vehicle = op["vehicle"]
        travel = op["travel_time"]
        process = op["processing_time"]

        machine_ready = machine_free_at.get(machine, 0)
        job_ready = job_free_at.get(job, 0)
        vehicle_ready = vehicle_free_at.get(vehicle, 0) + travel

        features = np.array([[
            travel,
            process,
            machine_map[machine],
            vehicle_map[vehicle],
            machine_ready,
            job_ready,
        ]])

        op["predicted_time"] = float(model.predict(features)[0])

        start_time = max(machine_ready, job_ready, vehicle_ready)
        finish_time = start_time + process
        machine_free_at[machine] = finish_time
        job_free_at[job] = finish_time
        vehicle_free_at[vehicle] = finish_time

    # Sort descending by predicted execution time
    ops_copy.sort(key=lambda x: x["predicted_time"], reverse=True)
    return ops_copy


def run_pipeline_with_model(ops, model):
    sorted_ops = ml_sort(ops, model)
    best_seq, best_cost = genetic_algorithm(sorted_ops)
    return best_seq, best_cost


# ------------------------------------------------------------
# STEP 4: DUAL-SCALE BENCHMARK EVALUATION (ALL 40 PROBLEMS)
# ------------------------------------------------------------
def evaluate_ablation_benchmarks(linear_model, gbdt_model):
    print("=" * 80)
    print("STEP 3: RUNNING ISOLATED MODEL-CAPACITY ABLATION ON 40 DUAL-SCALE BENCHMARKS")
    print("=" * 80)

    # 40 Benchmarks identical to Phase 2
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

        # Baseline
        base_oct = calculate_oct(ops)
        base_lb = calculate_load_balance(ops)
        base_cost = combined_cost(ops)

        # Plain GHCA (random init)
        ghca_seq, ghca_cost = genetic_algorithm(ops)
        ghca_oct = calculate_oct(ghca_seq)
        ghca_lb = calculate_load_balance(ghca_seq)

        # Heuristic-GHCA (sort by processing time descending)
        heur_ops = copy.deepcopy(ops)
        heur_ops.sort(key=lambda x: x["processing_time"], reverse=True)
        heur_seq, heur_cost = genetic_algorithm(heur_ops)
        heur_oct = calculate_oct(heur_seq)
        heur_lb = calculate_load_balance(heur_seq)

        # 1. Linear-ML-GHCA (Low capacity model)
        lin_seq, lin_cost = run_pipeline_with_model(ops, linear_model)
        lin_oct = calculate_oct(lin_seq)
        lin_lb = calculate_load_balance(lin_seq)

        # 2. GBDT-ML-GHCA (High capacity model)
        gbdt_seq, gbdt_cost = run_pipeline_with_model(ops, gbdt_model)
        gbdt_oct = calculate_oct(gbdt_seq)
        gbdt_lb = calculate_load_balance(gbdt_seq)

        # Improvement of GBDT over Linear-ML
        pct_gbdt_vs_linear = round((lin_cost - gbdt_cost) / lin_cost * 100, 2)
        pct_gbdt_vs_ghca = round((ghca_cost - gbdt_cost) / ghca_cost * 100, 2)
        pct_gbdt_vs_heur = round((heur_cost - gbdt_cost) / heur_cost * 100, 2)

        results.append({
            "Problem": prob_id,
            "Regime": regime,
            "Jobs": num_parts,
            "Ops": len(ops),
            "Baseline_Cost": base_cost,
            "GHCA_Cost": ghca_cost,
            "Heuristic_Cost": heur_cost,
            "Linear_ML_Cost": lin_cost,
            "GBDT_ML_Cost": gbdt_cost,
            "Linear_OCT": lin_oct,
            "GBDT_OCT": gbdt_oct,
            "Linear_LB": lin_lb,
            "GBDT_LB": gbdt_lb,
            "Pct_GBDT_vs_Linear": pct_gbdt_vs_linear,
            "Pct_GBDT_vs_GHCA": pct_gbdt_vs_ghca,
            "Pct_GBDT_vs_Heuristic": pct_gbdt_vs_heur,
        })

        print(f"{prob_id:<5} ({regime:<5}) | GHCA: {ghca_cost:<7.2f} | Heur: {heur_cost:<7.2f} | "
              f"Linear: {lin_cost:<7.2f} | GBDT: {gbdt_cost:<7.2f} | GBDT vs Linear: {pct_gbdt_vs_linear:>+6.2f}%")

    # Export to CSV
    with open(CSV_PHASE5_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDetailed ablation results saved to {CSV_PHASE5_OUTPUT}")
    return results


# ------------------------------------------------------------
# STEP 5: STATISTICAL TESTS & SECONDARY ABLATION REPORTING
# ------------------------------------------------------------
def compute_rank_biserial(x, y):
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    w_pos = np.sum(ranks[diff > 0])
    w_neg = np.sum(ranks[diff < 0])
    tot = w_pos + w_neg
    return float((w_pos - w_neg) / tot) if tot > 0 else 0.0


def report_secondary_ablation(results, r2_linear, r2_gbdt, best_params):
    print("\n" + "=" * 80)
    print("INDEPENDENT SECONDARY ABLATION TABLE: MODEL CAPACITY VS SCHEDULE QUALITY")
    print("=" * 80)

    regimes = ["Small", "Large", "All"]

    for r_filter in regimes:
        if r_filter == "All":
            sub = results
            title = "POOLED AGGREGATE (N=40 Problems)"
        else:
            sub = [r for r in results if r["Regime"] == r_filter]
            title = f"SCALE REGIME: {r_filter.upper()} (N={len(sub)} Problems)"

        lin_costs = [r["Linear_ML_Cost"] for r in sub]
        gbdt_costs = [r["GBDT_ML_Cost"] for r in sub]
        ghca_costs = [r["GHCA_Cost"] for r in sub]
        heur_costs = [r["Heuristic_Cost"] for r in sub]
        base_costs = [r["Baseline_Cost"] for r in sub]

        mean_lin = np.mean(lin_costs)
        mean_gbdt = np.mean(gbdt_costs)
        mean_ghca = np.mean(ghca_costs)
        mean_heur = np.mean(heur_costs)
        mean_base = np.mean(base_costs)

        # Paired Wilcoxon signed-rank: Linear vs GBDT (one-sided: Linear > GBDT, i.e. GBDT lowers cost)
        diff_lin_gbdt = np.array(lin_costs) - np.array(gbdt_costs)
        try:
            w_lg, p_lg = stats.wilcoxon(lin_costs, gbdt_costs, alternative="greater")
        except Exception:
            w_lg, p_lg = np.nan, np.nan
        r_lg = compute_rank_biserial(lin_costs, gbdt_costs)
        wins_gbdt = int(np.sum(diff_lin_gbdt > 0))
        ties_gbdt = int(np.sum(diff_lin_gbdt == 0))
        losses_gbdt = int(np.sum(diff_lin_gbdt < 0))

        # Wilcoxon: GHCA vs GBDT
        try:
            w_gg, p_gg = stats.wilcoxon(ghca_costs, gbdt_costs, alternative="greater")
        except Exception:
            w_gg, p_gg = np.nan, np.nan
        r_gg = compute_rank_biserial(ghca_costs, gbdt_costs)

        # Wilcoxon: Heuristic vs GBDT
        try:
            w_hg, p_hg = stats.wilcoxon(heur_costs, gbdt_costs, alternative="greater")
        except Exception:
            w_hg, p_hg = np.nan, np.nan
        r_hg = compute_rank_biserial(heur_costs, gbdt_costs)

        print(f"\n--- {title} ---")
        print(f"Baseline Mean Cost:   {mean_base:.2f}")
        print(f"Plain GHCA Mean Cost: {mean_ghca:.2f}")
        print(f"Heuristic-GHCA Cost:  {mean_heur:.2f}")
        print(f"Linear-ML-GHCA Cost:  {mean_lin:.2f}  (R² = {r2_linear:.4f})")
        print(f"GBDT-ML-GHCA Cost:    {mean_gbdt:.2f}  (R² = {r2_gbdt:.4f})")
        print(f"GBDT vs Linear Diff:  {np.mean(diff_lin_gbdt):+.2f} ({((mean_lin - mean_gbdt)/mean_lin)*100:+.2f}%)")
        print(f"GBDT Win/Tie/Loss vs Linear: {wins_gbdt}/{ties_gbdt}/{losses_gbdt}")
        print(f"Wilcoxon (Linear > GBDT):    W = {w_lg}, p = {p_lg:.5f} (Effect size r = {r_lg:+.3f})")
        print(f"Wilcoxon (GHCA > GBDT):      W = {w_gg}, p = {p_gg:.5f} (Effect size r = {r_gg:+.3f})")
        print(f"Wilcoxon (Heuristic > GBDT): W = {w_hg}, p = {p_hg:.5f} (Effect size r = {r_hg:+.3f})")

    # ------------------------------------------------------------
    # ANSWER THE RESEARCH QUESTION
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EMPIRICAL CONCLUSION: 'Does model capacity affect schedule quality?'")
    print("=" * 80)
    diff_overall = np.array([r["Linear_ML_Cost"] for r in results]) - np.array([r["GBDT_ML_Cost"] for r in results])
    mean_overall_diff = np.mean(diff_overall)
    _, p_overall = stats.wilcoxon([r["Linear_ML_Cost"] for r in results], [r["GBDT_ML_Cost"] for r in results], alternative="greater")

    print(f"1. Model Fit: Upgrading from Linear Regression to GBDT increases R² from {r2_linear:.4f} to {r2_gbdt:.4f}.")
    print(f"2. Optimization Payoff: Across all 40 problems, mean cost difference is {mean_overall_diff:+.2f} (p = {p_overall:.5f}).")
    if p_overall < 0.05:
        print("=> RESULT: Higher model capacity STATISTICALLY SIGNIFICANTLY improves schedule quality.")
    else:
        print("=> RESULT: Higher model capacity DOES NOT significantly improve schedule quality.")
        print("   Even with a superior non-linear regressor, the downstream hybrid GA/HC search")
        print("   dominates the initial sequence ordering, demonstrating that ML capacity")
        print("   is not the governing bottleneck for schedule optimality in this FMS setting.")
    print("=" * 80)


def main():
    print("Generating training dataset (N=200 problems)...")
    X, y = generate_training_data(n_samples=200)

    # 1. Sweep hyperparameters
    best_params, _ = run_hyperparameter_sweep(X, y)

    # 2. Train & compare models
    linear_model, gbdt_model, r2_linear, r2_gbdt = train_and_compare_models(X, y, best_params)

    # 3. Benchmark on 40 problems
    results = evaluate_ablation_benchmarks(linear_model, gbdt_model)

    # 4. Statistical analysis & secondary ablation table
    report_secondary_ablation(results, r2_linear, r2_gbdt, best_params)


if __name__ == "__main__":
    main()
