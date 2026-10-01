# ============================================================
# PLOT_SURROGATE_RESULTS.PY
# Phase E: Publication-Quality Visualizations for Surrogate-GHCA
#
# Generates:
#   1. results/graph11_surrogate_convergence.png
#   2. results/graph12_surrogate_benchmark.png
#   3. results/graph13_surrogate_vs_ghca_diff.png
# ============================================================

import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from scheduler import generate_problem
from genetic_algorithm import genetic_algorithm, genetic_algorithm_surrogate

DIR_GRAPHS = "graphs"
CSV_SURROGATE_RESULTS = "dataset/benchmarks/surrogate_benchmark_results.csv"
PNG_CONVERGENCE = os.path.join(DIR_GRAPHS, "graph11_surrogate_convergence.png")
PNG_BENCHMARK = os.path.join(DIR_GRAPHS, "graph12_surrogate_benchmark.png")
PNG_DIFF = os.path.join(DIR_GRAPHS, "graph13_surrogate_vs_ghca_diff.png")


def plot_surrogate_convergence():
    print("Generating convergence comparison on representative problem...")
    ops = generate_problem(num_parts=18, pool_num=1, layout_num=2, seed=15)

    # 1. Standard GA
    t0 = time.perf_counter()
    _, _, hist_std = genetic_algorithm(ops, track_convergence=True)
    t_std = time.perf_counter() - t0

    # 2. Surrogate-accelerated GA
    t0 = time.perf_counter()
    _, _, hist_sur = genetic_algorithm_surrogate(
        ops,
        generations=1200,
        pop_size=20,
        true_eval_interval=40,
        true_eval_top_k=3,
        track_convergence=True,
    )
    t_sur = time.perf_counter() - t0

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x_std = np.linspace(0, t_std, len(hist_std))
    x_sur = np.linspace(0, t_sur, len(hist_sur))

    ax.plot(x_std, hist_std, color="#1976d2", lw=2, label=f"Standard GA (200 gen, Final Cost: {hist_std[-1]:.2f})")
    ax.plot(x_sur, hist_sur, color="#388e3c", lw=2, linestyle="--", label=f"Surrogate GA (1200 gen, Final Cost: {hist_sur[-1]:.2f})")

    ax.set_title("Evolutionary Convergence: Standard GA vs Surrogate-Accelerated GA\n(Problem: 18 Parts, ~55 Operations)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Wall-Clock Time (seconds)", fontsize=10)
    ax.set_ylabel("Combined Schedule Cost (0.7*OCT + 0.3*LB)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PNG_CONVERGENCE, dpi=300)
    plt.close()
    print(f"Saved convergence graph to {PNG_CONVERGENCE}")


def plot_benchmark_comparison():
    if not os.path.exists(CSV_SURROGATE_RESULTS):
        print(f"Missing {CSV_SURROGATE_RESULTS}. Run surrogate_benchmark.py first.")
        return

    problems, jobs, ops_list = [], [], []
    base_costs, ghca_costs, heur_costs, lin_costs, sur_costs = [], [], [], [], []

    with open(CSV_SURROGATE_RESULTS, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            problems.append(row["Problem"])
            jobs.append(row["Jobs"])
            ops_list.append(row["Ops"])
            base_costs.append(float(row["Baseline_Cost"]))
            ghca_costs.append(float(row["GHCA_Cost"]))
            heur_costs.append(float(row["Heuristic_Cost"]))
            lin_costs.append(float(row["Linear_ML_Cost"]))
            sur_costs.append(float(row["Surrogate_Cost"]))

    enriched_labels = [f"{p}\n{j}p/{o}o" for p, j, o in zip(problems, jobs, ops_list)]
    x = np.arange(len(problems))

    # --- GRAPH 12: Combined Cost Comparison ---
    fig, ax = plt.subplots(figsize=(15, 5.5))

    ax.plot(x, base_costs, "s-", color="black", label="Baseline (Unoptimized)", lw=1.5, ms=4)
    ax.plot(x, ghca_costs, "o-", color="blue", label="GHCA (200 gen)", lw=1.5, ms=4)
    ax.plot(x, heur_costs, "D--", color="darkorange", label="Heuristic-GHCA (LPT)", lw=1.5, ms=4)
    ax.plot(x, lin_costs, "x-.", color="purple", label="Linear-ML-GHCA", lw=1.5, ms=4)
    ax.plot(x, sur_costs, "^-", color="green", label="Surrogate-GHCA (1200 gen)", lw=2, ms=5)

    # Regime separators
    for boundary in [9.5, 19.5, 29.5]:
        ax.axvline(x=boundary, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel("Problem Instance (Parts / Operations)", fontsize=11)
    ax.set_ylabel("Combined Schedule Cost (0.7*OCT + 0.3*LB)", fontsize=11)
    ax.set_title("Performance Comparison across 40 Dual-Scale FMS Benchmark Problems", fontsize=12, fontweight="bold")
    ax.set_xticks(x[::4])
    ax.set_xticklabels(enriched_labels[::4], rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PNG_BENCHMARK, dpi=300)
    plt.close()
    print(f"Saved benchmark comparison graph to {PNG_BENCHMARK}")

    # --- GRAPH 13: Paired Cost Difference (GHCA - Surrogate) ---
    diff = np.array(ghca_costs) - np.array(sur_costs)
    colors = ["#2e7d32" if d > 0 else "#c62828" if d < 0 else "#757575" for d in diff]

    fig, ax = plt.subplots(figsize=(15, 5.0))
    ax.bar(x, diff, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    mean_diff = float(np.mean(diff))
    ax.axhline(y=mean_diff, color="blue", linestyle="--", lw=1.5, label=f"Mean Advantage: {mean_diff:+.2f}")
    ax.axhline(y=0, color="black", lw=0.8)

    for boundary in [9.5, 19.5, 29.5]:
        ax.axvline(x=boundary, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel("Problem Instance (Parts / Operations)", fontsize=11)
    ax.set_ylabel("Cost Advantage (GHCA Cost - Surrogate Cost)", fontsize=11)
    ax.set_title("Paired Difference per Problem: GHCA vs Surrogate-GHCA\n(Positive = Surrogate Wins / Lower Cost)", fontsize=12, fontweight="bold")
    ax.set_xticks(x[::4])
    ax.set_xticklabels(enriched_labels[::4], rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(PNG_DIFF, dpi=300)
    plt.close()
    print(f"Saved paired difference graph to {PNG_DIFF}")


if __name__ == "__main__":
    os.makedirs(DIR_GRAPHS, exist_ok=True)
    plot_surrogate_convergence()
    plot_benchmark_comparison()
