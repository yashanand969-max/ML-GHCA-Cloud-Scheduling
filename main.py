# ============================================================
# MAIN.PY
# Runs all 40 benchmark problems
# Compares Baseline, GHCA, Heuristic-GHCA (ablation), and ML-GHCA
#
# IMPORTANT: The meaningful comparison for this project is
# ML-GHCA vs plain GHCA (does the ML layer help?) and
# ML-GHCA vs Heuristic-GHCA (does the ML layer beat a cheap,
# non-ML sort-by-processing-time heuristic?). Baseline is only
# a sanity check — any optimizer beats no optimization, so that
# comparison alone does NOT validate the ML contribution.
#
# Saves results to CSV for paper
# ============================================================

import copy
import csv
from scheduler import generate_problem, calculate_oct, calculate_load_balance, combined_cost
from genetic_algorithm import genetic_algorithm
from ml_layer import train_model, ml_ghca

# --- 40 BENCHMARK PROBLEMS ---
# Varying job counts and operation counts
# Similar to paper's Table 7 structure
# Seeds are fixed so results are reproducible

benchmarks = []
# Set 1: pool 1, small
for i in range(1, 11):
    benchmarks.append((f"P{i}", 8, 1, (i % 3) + 1, i))
# Set 2: pool 1, large
for i in range(11, 21):
    benchmarks.append((f"P{i}", 18, 1, (i % 3) + 1, i))
# Set 3: pool 2, small
for i in range(21, 31):
    benchmarks.append((f"P{i}", 8, 2, (i % 3) + 1, i))
# Set 4: pool 2, large
for i in range(31, 41):
    benchmarks.append((f"P{i}", 18, 2, (i % 3) + 1, i))


# ============================================================
# ABLATION: heuristic sort, no ML involved
# Sorts operations by processing_time descending (longest job
# first — a classic non-ML dispatch heuristic), then hands the
# sorted sequence to the SAME GHCA pipeline used for ML-GHCA.
# This isolates whether the ML layer specifically is adding
# value, versus any smart-looking initial ordering.
# ============================================================

def heuristic_sort(ops):
    ops_copy = copy.deepcopy(ops)
    ops_copy.sort(key=lambda op: op["processing_time"], reverse=True)
    return ops_copy


def heuristic_ghca(ops):
    sorted_ops = heuristic_sort(ops)
    best_seq, best_cost = genetic_algorithm(sorted_ops)
    return best_seq, best_cost


def run_all_benchmarks():
    # train ML model once before running all problems
    print("Training ML model...")
    model = train_model()
    print("ML model ready.\n")

    results = []

    for prob_id, num_parts, pool_num, layout_num, seed in benchmarks:
        ops = generate_problem(num_parts=num_parts, pool_num=pool_num, layout_num=layout_num, seed=seed)

        # baseline — no algorithm (sanity check only, not the main comparison)
        baseline_oct = calculate_oct(ops)
        baseline_lb = calculate_load_balance(ops)
        baseline_cost = combined_cost(ops)

        # GHCA — HC + GA, random initial ordering
        ghca_seq, ghca_cost = genetic_algorithm(ops)
        ghca_oct = calculate_oct(ghca_seq)
        ghca_lb = calculate_load_balance(ghca_seq)

        # Heuristic-GHCA — ablation: non-ML sort + same GHCA pipeline
        heur_seq, heur_cost = heuristic_ghca(ops)
        heur_oct = calculate_oct(heur_seq)
        heur_lb = calculate_load_balance(heur_seq)

        # ML-GHCA — our method: ML sort + same GHCA pipeline
        ml_seq, ml_cost = ml_ghca(ops, model)
        ml_oct = calculate_oct(ml_seq)
        ml_lb = calculate_load_balance(ml_seq)

        # --- comparisons ---
        # vs baseline: sanity check only (any optimizer wins this)
        pct_vs_baseline = round((baseline_cost - ml_cost) / baseline_cost * 100, 2)

        # vs plain GHCA: THE comparison that tests the ML contribution
        pct_vs_ghca = round((ghca_cost - ml_cost) / ghca_cost * 100, 2)

        # vs heuristic-GHCA: does ML beat a cheap non-ML sort?
        pct_vs_heuristic = round((heur_cost - ml_cost) / heur_cost * 100, 2)

        results.append({
            "Problem": prob_id,
            "Jobs": num_parts,
            "Ops": len(ops),
            "Baseline_OCT": baseline_oct,
            "Baseline_LB": baseline_lb,
            "Baseline_Cost": baseline_cost,
            "GHCA_OCT": ghca_oct,
            "GHCA_LB": ghca_lb,
            "GHCA_Cost": ghca_cost,
            "Heuristic_OCT": heur_oct,
            "Heuristic_LB": heur_lb,
            "Heuristic_Cost": heur_cost,
            "MLGHCA_OCT": ml_oct,
            "MLGHCA_LB": ml_lb,
            "MLGHCA_Cost": ml_cost,
            "Pct_Improvement_vs_Baseline": pct_vs_baseline,
            "Pct_Improvement_vs_GHCA": pct_vs_ghca,
            "Pct_Improvement_vs_Heuristic": pct_vs_heuristic,
        })

        print(f"{prob_id:<6} | Baseline: {baseline_cost:<8} | GHCA: {round(ghca_cost,2):<8} | "
              f"Heuristic: {round(heur_cost,2):<8} | ML-GHCA: {round(ml_cost,2):<8} | "
              f"vs GHCA: {pct_vs_ghca:>6}% | vs Heuristic: {pct_vs_heuristic:>6}%")

    # save to CSV
    import os
    os.makedirs("results", exist_ok=True)

    with open("results/benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to results/benchmark_results.csv")

    # summary statistics
    avg_baseline = sum(r["Baseline_Cost"] for r in results) / len(results)
    avg_ghca = sum(r["GHCA_Cost"] for r in results) / len(results)
    avg_heuristic = sum(r["Heuristic_Cost"] for r in results) / len(results)
    avg_ml = sum(r["MLGHCA_Cost"] for r in results) / len(results)
    avg_vs_baseline = sum(r["Pct_Improvement_vs_Baseline"] for r in results) / len(results)
    avg_vs_ghca = sum(r["Pct_Improvement_vs_GHCA"] for r in results) / len(results)
    avg_vs_heuristic = sum(r["Pct_Improvement_vs_Heuristic"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"SUMMARY ACROSS 40 PROBLEMS")
    print(f"{'='*60}")
    print(f"Avg Baseline Cost:            {round(avg_baseline, 2)}")
    print(f"Avg GHCA Cost:                {round(avg_ghca, 2)}")
    print(f"Avg Heuristic-GHCA Cost:      {round(avg_heuristic, 2)}")
    print(f"Avg ML-GHCA Cost:             {round(avg_ml, 2)}")
    print(f"{'-'*60}")
    print(f"Avg Improvement vs Baseline:   {round(avg_vs_baseline, 2)}%  (sanity check only)")
    print(f"Avg Improvement vs GHCA:       {round(avg_vs_ghca, 2)}%  (does ML help over plain GHCA?)")
    print(f"Avg Improvement vs Heuristic:  {round(avg_vs_heuristic, 2)}%  (does ML beat a cheap non-ML sort?)")
    print(f"{'='*60}")
    if avg_vs_ghca <= 0:
        print("\nNOTE: ML-GHCA did not improve on plain GHCA on average.")
        print("Report this honestly as a limitation rather than omitting it.")
    if avg_vs_heuristic <= 0:
        print("\nNOTE: ML-GHCA did not beat the cheap heuristic sort on average.")
        print("This is an important, reportable finding about the ML layer's value.")


if __name__ == "__main__":
    run_all_benchmarks()