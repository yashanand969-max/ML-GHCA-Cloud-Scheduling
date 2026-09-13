# ============================================================
# PLOT_PHASE5_ABLATION.PY
# Visualizes Phase 5 Model Capacity Ablation results
# ============================================================

import csv
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)
CSV_PATH = "results/phase5_model_capacity_ablation.csv"


def plot_phase5_figures():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run phase5_gbdt_ablation.py first.")

    small_base, small_ghca, small_heur, small_lin, small_gbdt = [], [], [], [], []
    large_base, large_ghca, large_heur, large_lin, large_gbdt = [], [], [], [], []
    all_lin, all_gbdt = [], []
    problems = []

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            regime = row["Regime"]
            p = row["Problem"]
            problems.append(p)
            base = float(row["Baseline_Cost"])
            ghca = float(row["GHCA_Cost"])
            heur = float(row["Heuristic_Cost"])
            lin = float(row["Linear_ML_Cost"])
            gbdt = float(row["GBDT_ML_Cost"])

            all_lin.append(lin)
            all_gbdt.append(gbdt)

            if regime == "Small":
                small_base.append(base)
                small_ghca.append(ghca)
                small_heur.append(heur)
                small_lin.append(lin)
                small_gbdt.append(gbdt)
            else:
                large_base.append(base)
                large_ghca.append(ghca)
                large_heur.append(heur)
                large_lin.append(lin)
                large_gbdt.append(gbdt)

    # ------------------------------------------------------------
    # FIGURE: Multi-Method Schedule Comparison (Small vs Large)
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Regime A (Small Scale)
    labels_small = ["Baseline", "GHCA", "Heuristic", "Linear-ML", "GBDT-ML"]
    data_small = [small_base, small_ghca, small_heur, small_lin, small_gbdt]
    colors_small = ["#90caf9", "#a5d6a7", "#ce93d8", "#ffcc80", "#ef5350"]

    bp1 = axes[0].boxplot(data_small, tick_labels=labels_small, patch_artist=True,
                          medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp1["boxes"], colors_small):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    for i, dist in enumerate(data_small, start=1):
        jitter = np.random.normal(0, 0.04, size=len(dist))
        axes[0].scatter(np.full_like(dist, i) + jitter, dist, color="#212121", alpha=0.5, s=20, zorder=3)
    axes[0].set_title("Regime A: Small Scale (N=20)\nModel Capacity vs Schedule Cost", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Combined Schedule Cost", fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Panel 2: Regime B (Large Scale)
    labels_large = ["Baseline", "GHCA", "Heuristic", "Linear-ML", "GBDT-ML"]
    data_large = [large_base, large_ghca, large_heur, large_lin, large_gbdt]
    colors_large = ["#90caf9", "#a5d6a7", "#ce93d8", "#ffcc80", "#ef5350"]

    bp2 = axes[1].boxplot(data_large, tick_labels=labels_large, patch_artist=True,
                          medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp2["boxes"], colors_large):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    for i, dist in enumerate(data_large, start=1):
        jitter = np.random.normal(0, 0.04, size=len(dist))
        axes[1].scatter(np.full_like(dist, i) + jitter, dist, color="#212121", alpha=0.5, s=20, zorder=3)
    axes[1].set_title("Regime B: Large Scale (N=20)\nModel Capacity vs Schedule Cost", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Combined Schedule Cost", fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Panel 3: Paired Cost Difference (Linear-ML - GBDT-ML) per Problem
    diff = np.array(all_lin) - np.array(all_gbdt)
    bar_colors = ["#2e7d32" if d > 0 else "#c62828" if d < 0 else "#757575" for d in diff]
    x_idx = np.arange(len(diff))
    axes[2].bar(x_idx, diff, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    mean_diff = np.mean(diff)
    axes[2].axhline(y=mean_diff, color="blue", linestyle="--", linewidth=1.5,
                    label=f"Mean Diff: {mean_diff:+.2f}")
    axes[2].axhline(y=0, color="black", linewidth=0.8)
    axes[2].set_title("Paired Difference: Linear - GBDT Cost\n(Positive = GBDT Wins)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Problem Index (P1-P40)", fontsize=10)
    axes[2].set_ylabel("Cost Advantage (Linear - GBDT)", fontsize=10)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_fig = "results/graph9_model_capacity_schedule_comparison.png"
    plt.savefig(output_fig, dpi=300)
    plt.close()
    print(f"Saved: {output_fig}")


if __name__ == "__main__":
    plot_phase5_figures()
