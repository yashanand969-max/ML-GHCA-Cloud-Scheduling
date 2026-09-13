# ============================================================
# PLOT_DUAL_SCALE.PY
# Visualizes the 4 outcome distributions and dual-scale ablation
# ============================================================

import csv
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)
CSV_PATH = "results/benchmark_results.csv"


def load_dual_scale_data():
    small_base, small_ghca, small_heur, small_ml = [], [], [], []
    large_base, large_ghca, large_heur, large_ml = [], [], [], []
    small_pct_ghca, large_pct_ghca = [], []
    small_pct_heur, large_pct_heur = [], []
    small_pct_base, large_pct_base = [], []

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prob = row["Problem"]
            num = int(prob[1:])
            base = float(row["Baseline_Cost"])
            ghca = float(row["GHCA_Cost"])
            heur = float(row["Heuristic_Cost"])
            ml = float(row["MLGHCA_Cost"])
            p_base = float(row["Pct_Improvement_vs_Baseline"])
            p_ghca = float(row["Pct_Improvement_vs_GHCA"])
            p_heur = float(row["Pct_Improvement_vs_Heuristic"])

            if (1 <= num <= 10) or (21 <= num <= 30):
                small_base.append(base)
                small_ghca.append(ghca)
                small_heur.append(heur)
                small_ml.append(ml)
                small_pct_base.append(p_base)
                small_pct_ghca.append(p_ghca)
                small_pct_heur.append(p_heur)
            else:
                large_base.append(base)
                large_ghca.append(ghca)
                large_heur.append(heur)
                large_ml.append(ml)
                large_pct_base.append(p_base)
                large_pct_ghca.append(p_ghca)
                large_pct_heur.append(p_heur)

    return {
        "small": {
            "base": small_base, "ghca": small_ghca, "heur": small_heur, "ml": small_ml,
            "pct_base": small_pct_base, "pct_ghca": small_pct_ghca, "pct_heur": small_pct_heur
        },
        "large": {
            "base": large_base, "ghca": large_ghca, "heur": large_heur, "ml": large_ml,
            "pct_base": large_pct_base, "pct_ghca": large_pct_ghca, "pct_heur": large_pct_heur
        }
    }


def plot_distributions():
    data = load_dual_scale_data()

    # --- GRAPH 6: The 4 Primary Outcome Distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Small Regime (Baseline vs ML)
    bp1 = axes[0].boxplot(
        [data["small"]["base"], data["small"]["ml"]],
        tick_labels=["Small-Baseline", "Small-ML"],
        patch_artist=True,
        boxprops=dict(facecolor="#90caf9", color="#1565c0", linewidth=1.5),
        medianprops=dict(color="#b71c1c", linewidth=2),
        whiskerprops=dict(color="#1565c0", linewidth=1.5),
        capprops=dict(color="#1565c0", linewidth=1.5)
    )
    # overlay individual points
    for i, dist in enumerate([data["small"]["base"], data["small"]["ml"]], start=1):
        jitter = np.random.normal(0, 0.04, size=len(dist))
        axes[0].scatter(np.full_like(dist, i) + jitter, dist, color="#0d47a1", alpha=0.6, s=35, zorder=3)
    axes[0].set_title("Regime A: Small Scale (<= 12 Jobs)\nBaseline vs ML-GHCA", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Combined Cost (OCT & LB)", fontsize=11)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Panel 2: Large Regime (Baseline vs ML)
    bp2 = axes[1].boxplot(
        [data["large"]["base"], data["large"]["ml"]],
        tick_labels=["Large-Baseline", "Large-ML"],
        patch_artist=True,
        boxprops=dict(facecolor="#ffcc80", color="#e65100", linewidth=1.5),
        medianprops=dict(color="#b71c1c", linewidth=2),
        whiskerprops=dict(color="#e65100", linewidth=1.5),
        capprops=dict(color="#e65100", linewidth=1.5)
    )
    for i, dist in enumerate([data["large"]["base"], data["large"]["ml"]], start=1):
        jitter = np.random.normal(0, 0.04, size=len(dist))
        axes[1].scatter(np.full_like(dist, i) + jitter, dist, color="#bf360c", alpha=0.6, s=35, zorder=3)
    axes[1].set_title("Regime B: Large Scale (Job Sets 2 & 4)\nBaseline vs ML-GHCA", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Combined Cost (OCT & LB)", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Panel 3: Improvement percentage vs GHCA & Heuristic across scales
    labels = ["vs GHCA\n(Small)", "vs GHCA\n(Large)", "vs Heuristic\n(Small)", "vs Heuristic\n(Large)"]
    pct_distributions = [
        data["small"]["pct_ghca"], data["large"]["pct_ghca"],
        data["small"]["pct_heur"], data["large"]["pct_heur"]
    ]
    colors = ["#81c784", "#388e3c", "#ba68c8", "#6a1b9a"]
    bp3 = axes[2].boxplot(
        pct_distributions,
        tick_labels=labels,
        patch_artist=True,
        medianprops=dict(color="#d32f2f", linewidth=2)
    )
    for patch, c in zip(bp3['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for i, dist in enumerate(pct_distributions, start=1):
        jitter = np.random.normal(0, 0.04, size=len(dist))
        axes[2].scatter(np.full_like(dist, i) + jitter, dist, color="#212121", alpha=0.5, s=25, zorder=3)
    axes[2].axhline(y=0, color="black", linestyle="--", linewidth=1.2)
    axes[2].set_title("ML Improvement Distribution vs Controls\n(Small vs Large Regimes)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("% Improvement by ML-GHCA", fontsize=11)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_fig6 = "results/graph6_dual_scale_distributions.png"
    plt.savefig(output_fig6, dpi=300)
    plt.close()
    print(f"Saved: {output_fig6}")

    # --- GRAPH 7: Multi-Method Comparison per Scale Regime ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Small Scale Multi-method
    axes[0].boxplot(
        [data["small"]["base"], data["small"]["ghca"], data["small"]["heur"], data["small"]["ml"]],
        tick_labels=["Baseline", "GHCA", "Heuristic-GHCA", "ML-GHCA"],
        patch_artist=True,
        boxprops=dict(facecolor="#bbdefb", color="#0d47a1"),
        medianprops=dict(color="red", linewidth=2)
    )
    axes[0].set_title("Scale Regime A (Small <= 12 Jobs): All 4 Methods\n(N=20 problems, ~20-27 ops)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Combined Cost", fontsize=11)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Large Scale Multi-method
    axes[1].boxplot(
        [data["large"]["base"], data["large"]["ghca"], data["large"]["heur"], data["large"]["ml"]],
        tick_labels=["Baseline", "GHCA", "Heuristic-GHCA", "ML-GHCA"],
        patch_artist=True,
        boxprops=dict(facecolor="#ffe0b2", color="#e65100"),
        medianprops=dict(color="red", linewidth=2)
    )
    axes[1].set_title("Scale Regime B (Large Scale): All 4 Methods\n(N=20 problems, ~44-60 ops)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Combined Cost", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_fig7 = "results/graph7_regime_comparison_boxplots.png"
    plt.savefig(output_fig7, dpi=300)
    plt.close()
    print(f"Saved: {output_fig7}")


if __name__ == "__main__":
    plot_distributions()
