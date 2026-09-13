# ============================================================
# WEIGHT_SENSITIVITY.PY
# Phase 4 Checkpoint 3: Hyperparameter Sensitivity Test
#
# Tests whether the method ranking (Baseline > GHCA ~ Heuristic
# ~ ML-GHCA) is stable across different cost weight configs:
#   - 0.7 / 0.3  (current default)
#   - 0.5 / 0.5  (equal weighting)
#   - 0.9 / 0.1  (OCT-dominant)
#
# Uses raw OCT and LB values already stored in the CSV, so
# no re-run of main.py is needed.
# ============================================================

import csv
import numpy as np
from scipy import stats


WEIGHT_CONFIGS = [
    (0.5, 0.5, "Equal (0.5/0.5)"),
    (0.7, 0.3, "Default (0.7/0.3)"),
    (0.9, 0.1, "OCT-dominant (0.9/0.1)"),
]


def load_oct_lb():
    """Load per-method OCT and LB arrays from the benchmark CSV."""
    methods = {
        "Baseline": {"oct": [], "lb": []},
        "GHCA":     {"oct": [], "lb": []},
        "Heuristic":{"oct": [], "lb": []},
        "MLGHCA":   {"oct": [], "lb": []},
    }

    with open("results/benchmark_results.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            methods["Baseline"]["oct"].append(float(row["Baseline_OCT"]))
            methods["Baseline"]["lb"].append(float(row["Baseline_LB"]))
            methods["GHCA"]["oct"].append(float(row["GHCA_OCT"]))
            methods["GHCA"]["lb"].append(float(row["GHCA_LB"]))
            methods["Heuristic"]["oct"].append(float(row["Heuristic_OCT"]))
            methods["Heuristic"]["lb"].append(float(row["Heuristic_LB"]))
            methods["MLGHCA"]["oct"].append(float(row["MLGHCA_OCT"]))
            methods["MLGHCA"]["lb"].append(float(row["MLGHCA_LB"]))

    # Convert to numpy arrays
    for m in methods:
        methods[m]["oct"] = np.array(methods[m]["oct"])
        methods[m]["lb"]  = np.array(methods[m]["lb"])

    return methods


def compute_combined(oct_arr, lb_arr, w1, w2):
    """Recompute combined cost under given weights."""
    return w1 * oct_arr + w2 * lb_arr


def rank_biserial_correlation(x, y):
    """Matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-)."""
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    w_pos = np.sum(ranks[diff > 0])
    w_neg = np.sum(ranks[diff < 0])
    total = w_pos + w_neg
    if total == 0:
        return 0.0
    return float((w_pos - w_neg) / total)


def run_sensitivity():
    methods = load_oct_lb()

    print("=" * 80)
    print("HYPERPARAMETER SENSITIVITY TEST: COST WEIGHT STABILITY")
    print("=" * 80)
    print("Tests whether method rankings change under different w_OCT / w_LB configs.")
    print(f"N = {len(methods['Baseline']['oct'])} problems\n")

    # Track conclusions across configs for stability check
    all_conclusions = []

    for w1, w2, label in WEIGHT_CONFIGS:
        print(f"\n{'=' * 70}")
        print(f"  WEIGHT CONFIG: w_OCT = {w1}, w_LB = {w2}  ({label})")
        print(f"{'=' * 70}")

        # Recompute costs
        costs = {}
        for m_name in ["Baseline", "GHCA", "Heuristic", "MLGHCA"]:
            costs[m_name] = compute_combined(
                methods[m_name]["oct"], methods[m_name]["lb"], w1, w2
            )

        # Descriptive stats
        print(f"\n  {'Method':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print(f"  {'-' * 55}")
        for m_name in ["Baseline", "GHCA", "Heuristic", "MLGHCA"]:
            c = costs[m_name]
            print(f"  {m_name:<15} {np.mean(c):<10.2f} {np.std(c):<10.2f} "
                  f"{np.min(c):<10.2f} {np.max(c):<10.2f}")

        # Mean % improvement of ML-GHCA
        pct_vs_ghca = np.mean((costs["GHCA"] - costs["MLGHCA"]) / costs["GHCA"] * 100)
        pct_vs_heur = np.mean((costs["Heuristic"] - costs["MLGHCA"]) / costs["Heuristic"] * 100)

        print(f"\n  ML-GHCA avg improvement vs GHCA:      {pct_vs_ghca:+.3f}%")
        print(f"  ML-GHCA avg improvement vs Heuristic: {pct_vs_heur:+.3f}%")

        # Wilcoxon tests
        comparisons = [
            ("GHCA vs ML-GHCA", costs["GHCA"], costs["MLGHCA"]),
            ("Heuristic vs ML-GHCA", costs["Heuristic"], costs["MLGHCA"]),
        ]

        config_conclusions = {}

        for comp_name, x, y in comparisons:
            diff = x - y
            if np.all(diff == 0):
                p_val = 1.0
                r_es = 0.0
            else:
                try:
                    _, p_val = stats.wilcoxon(x, y, alternative="greater")
                except ValueError:
                    p_val = 1.0
                r_es = rank_biserial_correlation(x, y)

            if p_val < 0.05:
                conclusion = "SIGNIFICANT"
            elif p_val < 0.10:
                conclusion = "MARGINAL"
            else:
                conclusion = "NULL"

            config_conclusions[comp_name] = conclusion

            print(f"\n  {comp_name}:")
            print(f"    Wilcoxon p = {p_val:.6f}, rank-biserial r = {r_es:+.3f}")
            print(f"    Conclusion: {conclusion}")

        all_conclusions.append((label, config_conclusions))

    # ----- STABILITY VERDICT -----
    print("\n" + "=" * 80)
    print("STABILITY VERDICT")
    print("=" * 80)

    print(f"\n  {'Config':<25} {'GHCA vs ML-GHCA':<20} {'Heur vs ML-GHCA':<20}")
    print(f"  {'-' * 65}")
    for label, conc in all_conclusions:
        print(f"  {label:<25} {conc['GHCA vs ML-GHCA']:<20} "
              f"{conc['Heuristic vs ML-GHCA']:<20}")

    # Check if all configs agree
    ghca_conclusions = set(c["GHCA vs ML-GHCA"] for _, c in all_conclusions)
    heur_conclusions = set(c["Heuristic vs ML-GHCA"] for _, c in all_conclusions)

    stable = len(ghca_conclusions) == 1 and len(heur_conclusions) == 1

    if stable:
        print("\n  RESULT: STABLE")
        print("  Method rankings and significance conclusions are IDENTICAL across")
        print("  all three weight configurations. The 0.7/0.3 default does not")
        print("  artificially inflate or deflate ML-GHCA's advantage.")
    else:
        print("\n  RESULT: UNSTABLE")
        print("  Method rankings or significance conclusions CHANGE across weight")
        print("  configurations. This should be reported as a sensitivity finding.")
        if len(ghca_conclusions) > 1:
            print(f"  - GHCA vs ML-GHCA conclusions vary: {ghca_conclusions}")
        if len(heur_conclusions) > 1:
            print(f"  - Heuristic vs ML-GHCA conclusions vary: {heur_conclusions}")


if __name__ == "__main__":
    run_sensitivity()
