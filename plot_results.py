# ============================================================
# PLOT_RESULTS.PY
# Generates all graphs needed for the paper
# 5 graphs total — aligned with main.py CSV schema
# ============================================================

import csv
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

REQUIRED_COLUMNS = [
    "Problem",
    "Baseline_OCT", "Baseline_LB", "Baseline_Cost",
    "GHCA_OCT", "GHCA_LB", "GHCA_Cost",
    "Heuristic_OCT", "Heuristic_LB", "Heuristic_Cost",
    "MLGHCA_OCT", "MLGHCA_LB", "MLGHCA_Cost",
    "Pct_Improvement_vs_Baseline",
    "Pct_Improvement_vs_GHCA",
    "Pct_Improvement_vs_Heuristic",
]


def load_results():
    with open("results/benchmark_results.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("benchmark_results.csv is empty — run main.py first.")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                "benchmark_results.csv is outdated (missing columns: "
                + ", ".join(missing)
                + "). Re-run main.py to regenerate results."
            )

        problems = []
        baseline_oct, ghca_oct, heur_oct, ml_oct = [], [], [], []
        baseline_lb, ghca_lb, heur_lb, ml_lb = [], [], [], []
        baseline_cost, ghca_cost, heur_cost, ml_cost = [], [], [], []
        pct_vs_baseline, pct_vs_ghca, pct_vs_heuristic = [], [], []

        for row in reader:
            problems.append(row["Problem"])
            baseline_oct.append(float(row["Baseline_OCT"]))
            ghca_oct.append(float(row["GHCA_OCT"]))
            heur_oct.append(float(row["Heuristic_OCT"]))
            ml_oct.append(float(row["MLGHCA_OCT"]))
            baseline_lb.append(float(row["Baseline_LB"]))
            ghca_lb.append(float(row["GHCA_LB"]))
            heur_lb.append(float(row["Heuristic_LB"]))
            ml_lb.append(float(row["MLGHCA_LB"]))
            baseline_cost.append(float(row["Baseline_Cost"]))
            ghca_cost.append(float(row["GHCA_Cost"]))
            heur_cost.append(float(row["Heuristic_Cost"]))
            ml_cost.append(float(row["MLGHCA_Cost"]))
            pct_vs_baseline.append(float(row["Pct_Improvement_vs_Baseline"]))
            pct_vs_ghca.append(float(row["Pct_Improvement_vs_GHCA"]))
            pct_vs_heuristic.append(float(row["Pct_Improvement_vs_Heuristic"]))

    return (problems,
            baseline_oct, ghca_oct, heur_oct, ml_oct,
            baseline_lb, ghca_lb, heur_lb, ml_lb,
            baseline_cost, ghca_cost, heur_cost, ml_cost,
            pct_vs_baseline, pct_vs_ghca, pct_vs_heuristic)


def _improvement_bar_colors(values):
    return ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in values]


def plot_improvement(ax, x, values, ylabel, title):
    colors = _improvement_bar_colors(values)
    ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    avg = np.mean(values)
    ax.axhline(y=avg, color='blue', linestyle='--', linewidth=2,
               label=f'Average: {round(avg, 2)}%')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')


def plot_all():
    (problems,
     baseline_oct, ghca_oct, heur_oct, ml_oct,
     baseline_lb, ghca_lb, heur_lb, ml_lb,
     baseline_cost, ghca_cost, heur_cost, ml_cost,
     pct_vs_baseline, pct_vs_ghca, pct_vs_heuristic) = load_results()

    x = np.arange(len(problems))

    # --------------------------------------------------------
    # GRAPH 1 — OCT Comparison across 40 problems
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(x, baseline_oct, 's-', color='black',
            label='Baseline', linewidth=1.5, markersize=4)
    ax.plot(x, ghca_oct, 'o-', color='blue',
            label='GHCA', linewidth=1.5, markersize=4)
    ax.plot(x, heur_oct, 'D--', color='darkorange',
            label='Heuristic-GHCA', linewidth=1.5, markersize=4)
    ax.plot(x, ml_oct, '^-', color='green',
            label='ML-GHCA (Ours)', linewidth=1.5, markersize=4)

    ax.set_xlabel("Problem Number", fontsize=12)
    ax.set_ylabel("Operational Completion Time (OCT)", fontsize=12)
    ax.set_title("OCT Comparison: Baseline vs GHCA vs Heuristic-GHCA vs ML-GHCA",
                 fontsize=13)
    ax.set_xticks(x[::4])
    ax.set_xticklabels(problems[::4], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/graph1_oct_comparison.png", dpi=150)
    plt.close()
    print("Graph 1 saved — OCT comparison")

    # --------------------------------------------------------
    # GRAPH 2 — Load Balance Score across 40 problems
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(x, baseline_lb, 's-', color='black',
            label='Baseline', linewidth=1.5, markersize=4)
    ax.plot(x, ghca_lb, 'o-', color='blue',
            label='GHCA', linewidth=1.5, markersize=4)
    ax.plot(x, heur_lb, 'D--', color='darkorange',
            label='Heuristic-GHCA', linewidth=1.5, markersize=4)
    ax.plot(x, ml_lb, '^-', color='green',
            label='ML-GHCA (Ours)', linewidth=1.5, markersize=4)

    ax.set_xlabel("Problem Number", fontsize=12)
    ax.set_ylabel("Load Balance Score (Std Dev)", fontsize=12)
    ax.set_title("Load Balance: Baseline vs GHCA vs Heuristic-GHCA vs ML-GHCA",
                 fontsize=13)
    ax.set_xticks(x[::4])
    ax.set_xticklabels(problems[::4], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/graph2_load_balance.png", dpi=150)
    plt.close()
    print("Graph 2 saved — Load balance comparison")

    # --------------------------------------------------------
    # GRAPH 3 — Combined Cost comparison (bar chart)
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    width = 0.2
    ax.bar(x - 1.5 * width, baseline_cost, width, label='Baseline',
           color='black', alpha=0.7)
    ax.bar(x - 0.5 * width, ghca_cost, width, label='GHCA',
           color='blue', alpha=0.7)
    ax.bar(x + 0.5 * width, heur_cost, width, label='Heuristic-GHCA',
           color='darkorange', alpha=0.7)
    ax.bar(x + 1.5 * width, ml_cost, width, label='ML-GHCA (Ours)',
           color='green', alpha=0.7)

    ax.set_xlabel("Problem Number", fontsize=12)
    ax.set_ylabel("Combined Cost (0.7×OCT + 0.3×LB)", fontsize=12)
    ax.set_title("Combined Cost Comparison across 40 Benchmark Problems",
                 fontsize=13)
    ax.set_xticks(x[::4])
    ax.set_xticklabels(problems[::4], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("results/graph3_combined_cost.png", dpi=150)
    plt.close()
    print("Graph 3 saved — Combined cost bar chart")

    # --------------------------------------------------------
    # GRAPH 4 — ML-GHCA improvement vs GHCA (primary comparison)
    # Positive = ML-GHCA is better; negative = GHCA wins
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    plot_improvement(
        ax, x, pct_vs_ghca,
        ylabel="Improvement over GHCA (%)",
        title="ML-GHCA vs GHCA — Does the ML Layer Help?",
    )
    ax.set_xlabel("Problem Number", fontsize=12)
    ax.set_xticks(x[::4])
    ax.set_xticklabels(problems[::4], rotation=45)
    plt.tight_layout()
    plt.savefig("results/graph4_improvement_vs_ghca.png", dpi=150)
    plt.close()
    print("Graph 4 saved — Improvement vs GHCA")

    # --------------------------------------------------------
    # GRAPH 5 — ML-GHCA improvement vs Heuristic-GHCA (ablation)
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    plot_improvement(
        ax, x, pct_vs_heuristic,
        ylabel="Improvement over Heuristic-GHCA (%)",
        title="ML-GHCA vs Heuristic-GHCA — Does ML Beat a Cheap Non-ML Sort?",
    )
    ax.set_xlabel("Problem Number", fontsize=12)
    ax.set_xticks(x[::4])
    ax.set_xticklabels(problems[::4], rotation=45)
    plt.tight_layout()
    plt.savefig("results/graph5_improvement_vs_heuristic.png", dpi=150)
    plt.close()
    print("Graph 5 saved — Improvement vs Heuristic-GHCA")

    print("\nAll graphs saved to results/ folder")
    print("Files:")
    print("  results/graph1_oct_comparison.png")
    print("  results/graph2_load_balance.png")
    print("  results/graph3_combined_cost.png")
    print("  results/graph4_improvement_vs_ghca.png  (primary ML comparison)")
    print("  results/graph5_improvement_vs_heuristic.png  (ablation)")
    print(f"\nSanity check — avg improvement vs baseline: "
          f"{round(np.mean(pct_vs_baseline), 2)}% (not the main ML claim)")


if __name__ == "__main__":
    plot_all()
