# ============================================================
# PLOT_CONVERGENCE.PY
# Generates convergence graph — Graph 5 for the paper
# Shows GHCA vs ML-GHCA: how fast each reaches best cost
# Run this AFTER updating genetic_algorithm.py
# ============================================================

import matplotlib.pyplot as plt
import os

from scheduler import generate_problem, combined_cost
from genetic_algorithm import genetic_algorithm
from ml_layer import train_model, ml_sort

os.makedirs("results", exist_ok=True)


def run_ghca_with_convergence(ops):
    """Run plain GHCA and return generation-by-generation best cost."""
    _, _, history = genetic_algorithm(ops, track_convergence=True)
    return history


def run_ml_ghca_with_convergence(ops, model):
    """Run ML-GHCA and return generation-by-generation best cost."""
    sorted_ops = ml_sort(ops, model)
    _, _, history = genetic_algorithm(sorted_ops, track_convergence=True)
    return history


def plot_convergence():
    print("Training ML model...")
    model = train_model()
    print("ML model ready.\n")

    # Use a medium-sized problem so difference is visible
    # 8 parts, pool 1, layout 1
    ops = generate_problem(num_parts=8, pool_num=1, layout_num=1, seed=99)

    print("Running GHCA with convergence tracking...")
    ghca_history = run_ghca_with_convergence(ops)

    print("Running ML-GHCA with convergence tracking...")
    ml_history = run_ml_ghca_with_convergence(ops, model)

    generations = list(range(len(ghca_history)))  # 0 to 200

    # --------------------------------------------------------
    # GRAPH 5 — Convergence Speed Comparison
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(generations, ghca_history, color='blue',
            linewidth=2, label='GHCA (plain)', alpha=0.85)

    ax.plot(generations, ml_history, color='green',
            linewidth=2, label='ML-GHCA (ours)', alpha=0.85)

    # Mark the point where each method first comes within a small
    # tolerance of its final best value. Exact float equality is too
    # fragile here: ties or near-plateaus get misread as "instant
    # convergence" and can produce misleading claims like "converges
    # 100% earlier" when a run simply starts close to its own final
    # value. A relative tolerance band is more robust.
    CONVERGENCE_TOLERANCE = 0.01  # within 1% of final value counts as converged

    ghca_final = ghca_history[-1]
    ml_final   = ml_history[-1]

    ghca_threshold = ghca_final * (1 + CONVERGENCE_TOLERANCE)
    ml_threshold   = ml_final * (1 + CONVERGENCE_TOLERANCE)

    ghca_conv_gen = next(i for i, v in enumerate(ghca_history) if v <= ghca_threshold)
    ml_conv_gen   = next(i for i, v in enumerate(ml_history)   if v <= ml_threshold)

    ax.axvline(x=ghca_conv_gen, color='blue', linestyle='--',
               alpha=0.5, linewidth=1.2,
               label=f'GHCA converges at gen {ghca_conv_gen}')

    ax.axvline(x=ml_conv_gen, color='green', linestyle='--',
               alpha=0.5, linewidth=1.2,
               label=f'ML-GHCA converges at gen {ml_conv_gen}')

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Combined Cost (0.7×OCT + 0.3×LB)", fontsize=12)
    ax.set_title("Convergence Speed: GHCA vs ML-GHCA\n(Lower is better — ML-GHCA reaches optimal solution faster)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/graph5_convergence.png", dpi=150)
    plt.close()

    print("\nGraph 5 saved — results/graph5_convergence.png")
    print(f"\n--- CONVERGENCE SUMMARY ---")
    print(f"GHCA   starting cost : {round(ghca_history[0], 2)}")
    print(f"GHCA   final cost    : {round(ghca_final, 2)}")
    print(f"GHCA   converges at  : Generation {ghca_conv_gen}")
    print(f"")
    print(f"ML-GHCA starting cost : {round(ml_history[0], 2)}")
    print(f"ML-GHCA final cost    : {round(ml_final, 2)}")
    print(f"ML-GHCA converges at  : Generation {ml_conv_gen}")
    print(f"")
    if ghca_conv_gen == 0 and ml_conv_gen == 0:
        print("Note: Both methods were within tolerance of their final cost at")
        print("generation 0 (i.e. the HC warm-start already nearly solved this")
        print("problem). This problem size is too easy to show a convergence-speed")
        print("difference; try a larger problem (more jobs/ops) for a meaningful graph.")
    elif ml_conv_gen < ghca_conv_gen:
        faster = round((1 - ml_conv_gen / ghca_conv_gen) * 100, 1)
        print(f"ML-GHCA converges {faster}% earlier than GHCA")
    elif ml_conv_gen > ghca_conv_gen:
        slower = round((ml_conv_gen / ghca_conv_gen - 1) * 100, 1)
        print(f"Note: On this problem GHCA converged {slower}% earlier than ML-GHCA.")
        print("Report this honestly rather than only showing runs favorable to ML-GHCA.")
    else:
        print("Note: Both methods converged at the same generation on this problem.")


if __name__ == "__main__":
    plot_convergence()