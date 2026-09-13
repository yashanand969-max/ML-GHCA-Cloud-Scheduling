# ============================================================
# ANOVA_TEST.PY
# Statistical validation of results
#
# Structure:
#   1. Descriptive statistics (all 4 methods)
#   2. Normality check on differences (justifies test choice)
#   3. Friedman test (omnibus, all methods) — mirrors base paper
#   4. Two-way ANOVA (omnibus, all methods) — mirrors base paper
#   5. Paired Wilcoxon signed-rank tests — the tests that
#      actually matter for THIS project's claim:
#         GHCA vs ML-GHCA          -> does the ML layer help?
#         Heuristic-GHCA vs ML-GHCA -> does ML beat a cheap,
#                                      non-ML sort heuristic?
#
# NOTE: The omnibus Friedman/ANOVA tests include Baseline,
# which trivially differs from the optimized methods and will
# dominate the variance. They are kept here for completeness
# and comparability with the base paper, but the Wilcoxon tests
# in section 5 are the ones that actually validate this
# project's contribution and should be the ones reported first.
# ============================================================

import csv
import numpy as np
from scipy import stats

# --- LOAD RESULTS FROM CSV ---
def load_results():
    baseline_costs = []
    ghca_costs = []
    heuristic_costs = []
    ml_costs = []

    with open("results/benchmark_results.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            baseline_costs.append(float(row["Baseline_Cost"]))
            ghca_costs.append(float(row["GHCA_Cost"]))
            heuristic_costs.append(float(row["Heuristic_Cost"]))
            ml_costs.append(float(row["MLGHCA_Cost"]))

    return baseline_costs, ghca_costs, heuristic_costs, ml_costs


# --- MATCHED-PAIRS RANK-BISERIAL CORRELATION (EFFECT SIZE) ---
# r = (W+ - W-) / (W+ + W-)
# where diffs = x - y. r in [-1, 1]; r > 0 means x > y systematically.
# Interpretation:  |r| < 0.3 = small,  0.3 <= |r| < 0.5 = medium,  |r| >= 0.5 = large

def rank_biserial_correlation(x, y):
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    w_pos = np.sum(ranks[diff > 0])
    w_neg = np.sum(ranks[diff < 0])
    total_ranks = w_pos + w_neg
    if total_ranks == 0:
        return 0.0
    return float((w_pos - w_neg) / total_ranks)


def effect_size_label(r):
    """Return human-readable effect-size category."""
    ar = abs(r)
    if ar < 0.3:
        return "small"
    elif ar < 0.5:
        return "medium"
    else:
        return "large"


# --- PARTITION DATA BY SCALE REGIME ---
# Small: P1-P10 (indices 0-9) + P21-P30 (indices 20-29)
# Large: P11-P20 (indices 10-19) + P31-P40 (indices 30-39)

def partition_by_regime(baseline, ghca, heuristic, ml):
    small_idx = list(range(0, 10)) + list(range(20, 30))
    large_idx = list(range(10, 20)) + list(range(30, 40))

    def pick(arr, idx_list):
        return np.array([arr[i] for i in idx_list])

    return {
        "Small": {
            "Baseline": pick(baseline, small_idx),
            "GHCA": pick(ghca, small_idx),
            "Heuristic": pick(heuristic, small_idx),
            "ML": pick(ml, small_idx),
        },
        "Large": {
            "Baseline": pick(baseline, large_idx),
            "GHCA": pick(ghca, large_idx),
            "Heuristic": pick(heuristic, large_idx),
            "ML": pick(ml, large_idx),
        },
    }


# --- DESCRIPTIVE STATISTICS ---
def descriptive_stats(baseline, ghca, heuristic, ml):
    print("=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(f"\n{'Metric':<12} {'Baseline':<12} {'GHCA':<12} {'Heuristic':<12} {'ML-GHCA':<12}")
    print("-" * 60)
    print(f"{'Mean':<12} {round(np.mean(baseline),2):<12} {round(np.mean(ghca),2):<12} "
          f"{round(np.mean(heuristic),2):<12} {round(np.mean(ml),2):<12}")
    print(f"{'Std Dev':<12} {round(np.std(baseline),2):<12} {round(np.std(ghca),2):<12} "
          f"{round(np.std(heuristic),2):<12} {round(np.std(ml),2):<12}")
    print(f"{'Min':<12} {round(np.min(baseline),2):<12} {round(np.min(ghca),2):<12} "
          f"{round(np.min(heuristic),2):<12} {round(np.min(ml),2):<12}")
    print(f"{'Max':<12} {round(np.max(baseline),2):<12} {round(np.max(ghca),2):<12} "
          f"{round(np.max(heuristic),2):<12} {round(np.max(ml),2):<12}")
    print(f"{'Median':<12} {round(np.median(baseline),2):<12} {round(np.median(ghca),2):<12} "
          f"{round(np.median(heuristic),2):<12} {round(np.median(ml),2):<12}")


# --- NORMALITY CHECK ---
# Shapiro-Wilk test on the paired differences (GHCA - ML-GHCA).
# This is what actually justifies choosing a paired t-test vs
# a non-parametric Wilcoxon signed-rank test for the comparison
# that matters. We do NOT assume normality — we check it.

def normality_check(ghca, ml, heuristic):
    print("\n" + "=" * 70)
    print("NORMALITY CHECK ON PAIRED DIFFERENCES (Shapiro-Wilk)")
    print("=" * 70)
    print("H0: differences are normally distributed")
    print("If p < 0.05, normality is rejected -> use Wilcoxon, not a paired t-test\n")

    diff_ghca_ml = np.array(ghca) - np.array(ml)
    stat1, p1 = stats.shapiro(diff_ghca_ml)
    print(f"GHCA - ML-GHCA differences:      W = {round(stat1,4)}, p = {round(p1,6)}")
    if p1 < 0.05:
        print("  -> Normality rejected. Wilcoxon signed-rank is the appropriate test.")
    else:
        print("  -> Normality not rejected. A paired t-test would also be defensible,")
        print("     but Wilcoxon is reported below for consistency and robustness.")

    diff_heur_ml = np.array(heuristic) - np.array(ml)
    stat2, p2 = stats.shapiro(diff_heur_ml)
    print(f"\nHeuristic - ML-GHCA differences: W = {round(stat2,4)}, p = {round(p2,6)}")
    if p2 < 0.05:
        print("  -> Normality rejected. Wilcoxon signed-rank is the appropriate test.")
    else:
        print("  -> Normality not rejected. A paired t-test would also be defensible,")
        print("     but Wilcoxon is reported below for consistency and robustness.")


# --- FRIEDMAN TEST (OMNIBUS, all 4 methods) ---
# Non-parametric test — does not assume normal distribution
# Tests if at least one method is significantly different.
# NOTE: including Baseline here means a significant result may
# simply reflect "optimized beats unoptimized" and should NOT
# be read as evidence the ML layer specifically helps.

def friedman_test(baseline, ghca, heuristic, ml):
    print("\n" + "=" * 70)
    print("FRIEDMAN TEST (omnibus, all 4 methods incl. Baseline)")
    print("=" * 70)
    print("H0: All methods have the same distribution")
    print("H1: At least two methods differ significantly")
    print("Significance level: 0.05\n")

    stat, p_value = stats.friedmanchisquare(baseline, ghca, heuristic, ml)

    print(f"Friedman statistic: {round(stat, 4)}")
    print(f"P-value:            {round(p_value, 6)}")

    if p_value < 0.05:
        print(f"\nResult: REJECT null hypothesis (p < 0.05)")
        print(f"Conclusion: Significant difference exists among the four methods.")
        print(f"Caution: with Baseline included, this is likely driven mostly by")
        print(f"Baseline vs. the optimized methods, not by differences among")
        print(f"GHCA / Heuristic / ML-GHCA. See the Wilcoxon tests below for that.")
    else:
        print(f"\nResult: FAIL TO REJECT null hypothesis")
        print(f"Conclusion: No significant difference found.")

    return stat, p_value


# --- TWO WAY ANOVA (OMNIBUS, all 4 methods) ---
# Same structure as base paper Section 5.2, extended to 4 methods.
# Factor 1 (treatments) = algorithms
# Factor 2 (blocks) = benchmark problems (40 problems)
# NOTE: same caveat as Friedman above — Baseline inflates the
# treatment effect. Kept for comparability with the base paper.

def two_way_anova(baseline, ghca, heuristic, ml):
    print("\n" + "=" * 70)
    print("TWO-WAY ANOVA (omnibus, all 4 methods incl. Baseline)")
    print("=" * 70)

    n = len(baseline)   # number of problems = 40
    k = 4               # number of algorithms = 4

    all_data = np.array([baseline, ghca, heuristic, ml])  # shape (4, 40)

    G = np.sum(all_data)
    CF = (G ** 2) / (n * k)
    TSS = np.sum(all_data ** 2) - CF

    treatment_totals = np.sum(all_data, axis=1)
    SST = np.sum(treatment_totals ** 2) / n - CF

    block_totals = np.sum(all_data, axis=0)
    SSB = np.sum(block_totals ** 2) / k - CF

    SSE = TSS - SST - SSB

    df_treatment = k - 1
    df_block = n - 1
    df_error = df_treatment * df_block

    MST = SST / df_treatment
    MSB = SSB / df_block
    MSE = SSE / df_error

    F_treatment = MST / MSE
    F_block = MSB / MSE

    F_crit_treatment = stats.f.ppf(0.95, df_treatment, df_error)
    F_crit_block = stats.f.ppf(0.95, df_block, df_error)

    p_treatment = 1 - stats.f.cdf(F_treatment, df_treatment, df_error)
    p_block = 1 - stats.f.cdf(F_block, df_block, df_error)

    print(f"\n{'Source':<20} {'SS':<12} {'DF':<6} {'MS':<12} {'F':<10} {'F_crit':<10} {'p':<10}")
    print("-" * 80)
    print(f"{'Treatments(Algo)':<20} {round(SST,2):<12} {df_treatment:<6} {round(MST,2):<12} "
          f"{round(F_treatment,3):<10} {round(F_crit_treatment,3):<10} {round(p_treatment,4):<10}")
    print(f"{'Blocks(Problems)':<20} {round(SSB,2):<12} {df_block:<6} {round(MSB,2):<12} "
          f"{round(F_block,3):<10} {round(F_crit_block,3):<10} {round(p_block,4):<10}")
    print(f"{'Error':<20} {round(SSE,2):<12} {df_error:<6} {round(MSE,2):<12}")
    print(f"{'Total':<20} {round(TSS,2):<12} {n*k - 1:<6}")

    print(f"\n--- DECISIONS ---")

    if F_treatment > F_crit_treatment:
        print(f"Treatments: F({round(F_treatment,3)}) > F_crit({round(F_crit_treatment,3)})")
        print(f"REJECT H0 — significant difference among algorithms")
        print(f"(Caution: likely driven by Baseline. See Wilcoxon tests below.)")
    else:
        print(f"Treatments: F({round(F_treatment,3)}) < F_crit({round(F_crit_treatment,3)})")
        print(f"FAIL TO REJECT H0 — no significant difference among algorithms")

    if F_block > F_crit_block:
        print(f"\nBlocks: F({round(F_block,3)}) > F_crit({round(F_crit_block,3)})")
        print(f"REJECT H0 — significant difference between problems")
    else:
        print(f"\nBlocks: F({round(F_block,3)}) < F_crit({round(F_crit_block,3)})")
        print(f"FAIL TO REJECT H0 — no significant difference between problems")

    return F_treatment, F_block, p_treatment, p_block


# --- PAIRED WILCOXON SIGNED-RANK TESTS ---
# THE tests that matter for this project's actual claim.
# Paired (same 40 problems), non-parametric, appropriate
# regardless of what the normality check above found.

def wilcoxon_tests(ghca, heuristic, ml):
    print("\n" + "=" * 70)
    print("PAIRED WILCOXON SIGNED-RANK TESTS (the comparisons that matter)")
    print("=" * 70)

    # --- GHCA vs ML-GHCA: does the ML layer help over plain GHCA? ---
    print("\n--- GHCA vs ML-GHCA ---")
    print("H0: median difference between GHCA and ML-GHCA costs is zero")
    print("H1: ML-GHCA costs are systematically lower (one-sided)\n")

    ghca_arr = np.array(ghca)
    ml_arr = np.array(ml)
    diff = ghca_arr - ml_arr

    if np.all(diff == 0):
        print("All paired differences are zero — Wilcoxon test is undefined.")
    else:
        try:
            stat, p_value = stats.wilcoxon(ghca_arr, ml_arr, alternative="greater")
            r_es = rank_biserial_correlation(ghca_arr, ml_arr)
            print(f"Wilcoxon statistic: {round(stat, 4)}")
            print(f"P-value (one-sided, GHCA > ML-GHCA): {round(p_value, 6)}")
            print(f"Effect size (rank-biserial r): {r_es:+.3f} ({effect_size_label(r_es)})")
            if p_value < 0.05:
                print("Result: REJECT H0 (p < 0.05)")
                print("Conclusion: ML-GHCA produces significantly lower cost than")
                print("plain GHCA. This supports the claim that the ML layer helps.")
            else:
                print("Result: FAIL TO REJECT H0")
                print("Conclusion: No significant evidence that ML-GHCA improves on")
                print("plain GHCA. This should be reported as a limitation, not omitted.")
        except ValueError as e:
            print(f"Test could not be run: {e}")

    # --- Heuristic-GHCA vs ML-GHCA: does ML beat a cheap non-ML sort? ---
    print("\n--- Heuristic-GHCA (non-ML sort) vs ML-GHCA ---")
    print("H0: median difference between Heuristic and ML-GHCA costs is zero")
    print("H1: ML-GHCA costs are systematically lower (one-sided)\n")

    heur_arr = np.array(heuristic)
    diff2 = heur_arr - ml_arr

    if np.all(diff2 == 0):
        print("All paired differences are zero — Wilcoxon test is undefined.")
    else:
        try:
            stat2, p_value2 = stats.wilcoxon(heur_arr, ml_arr, alternative="greater")
            r_es2 = rank_biserial_correlation(heur_arr, ml_arr)
            print(f"Wilcoxon statistic: {round(stat2, 4)}")
            print(f"P-value (one-sided, Heuristic > ML-GHCA): {round(p_value2, 6)}")
            print(f"Effect size (rank-biserial r): {r_es2:+.3f} ({effect_size_label(r_es2)})")
            if p_value2 < 0.05:
                print("Result: REJECT H0 (p < 0.05)")
                print("Conclusion: ML-GHCA produces significantly lower cost than a")
                print("cheap non-ML heuristic sort. This is the strongest evidence")
                print("that the ML layer specifically (not just any reordering) helps.")
            else:
                print("Result: FAIL TO REJECT H0")
                print("Conclusion: No significant evidence that the ML sort beats a")
                print("simple processing-time heuristic. Report this honestly — it")
                print("means the ML layer's contribution over a cheap heuristic is")
                print("not demonstrated by this experiment.")
        except ValueError as e:
            print(f"Test could not be run: {e}")


# ============================================================
# 6. REGIME-DECOUPLED WILCOXON TESTS
# Reports GHCA vs ML-GHCA and Heuristic vs ML-GHCA separately
# for Small (N=20) and Large (N=20) scale regimes.
# This decouples potential regime-specific effects that are
# obscured when all 40 problems are pooled into one test.
# ============================================================

def regime_wilcoxon_tests(baseline, ghca, heuristic, ml):
    print("\n" + "=" * 70)
    print("REGIME-DECOUPLED PAIRED WILCOXON SIGNED-RANK TESTS")
    print("=" * 70)
    print("Small regime: P1-P10 + P21-P30 (8 parts, ~20-27 ops)")
    print("Large regime: P11-P20 + P31-P40 (18 parts, ~44-60 ops)")
    print("Each regime: N = 20 paired observations\n")

    regimes = partition_by_regime(baseline, ghca, heuristic, ml)

    results = {}

    for regime_name in ["Small", "Large"]:
        r = regimes[regime_name]
        ml_arr = r["ML"]
        results[regime_name] = {}

        print(f"\n{'-' * 70}")
        print(f"  REGIME: {regime_name.upper()} (N = {len(ml_arr)})")
        print(f"{'-' * 70}")

        comparisons = [
            ("GHCA vs ML-GHCA", r["GHCA"], ml_arr,
             "does the ML layer help over plain GHCA?"),
            ("Heuristic-GHCA vs ML-GHCA", r["Heuristic"], ml_arr,
             "does ML beat a cheap non-ML sort?"),
        ]

        for comp_name, x, y, question in comparisons:
            diff = x - y
            wins = int(np.sum(diff > 0))
            ties = int(np.sum(diff == 0))
            losses = int(np.sum(diff < 0))
            mean_diff = float(np.mean(diff))
            # percentage relative to the comparator (x)
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_arr = np.where(x != 0, (diff / x) * 100, 0.0)
            mean_pct = float(np.mean(pct_arr))

            print(f"\n  --- {comp_name} ({question}) ---")
            print(f"  H0: median paired difference = 0")
            print(f"  H1: {comp_name.split(' vs ')[0]} costs are systematically higher (one-sided)\n")

            if np.all(diff == 0):
                print("  All paired differences are zero — test is undefined.")
                results[regime_name][comp_name] = {
                    "p_value": 1.0, "r": 0.0, "mean_pct": 0.0,
                    "wins": 0, "ties": len(diff), "losses": 0
                }
                continue

            try:
                w_stat, p_val = stats.wilcoxon(x, y, alternative="greater")
            except ValueError as e:
                print(f"  Test could not be run: {e}")
                results[regime_name][comp_name] = {
                    "p_value": 1.0, "r": 0.0, "mean_pct": mean_pct,
                    "wins": wins, "ties": ties, "losses": losses
                }
                continue

            r_es = rank_biserial_correlation(x, y)

            results[regime_name][comp_name] = {
                "p_value": p_val, "r": r_es, "mean_pct": mean_pct,
                "wins": wins, "ties": ties, "losses": losses
            }

            print(f"  Mean difference: {mean_diff:+.2f}  ({mean_pct:+.2f}%)")
            print(f"  Win / Tie / Loss: {wins} / {ties} / {losses}")
            print(f"  Wilcoxon W: {w_stat:.4f}")
            print(f"  P-value (one-sided): {p_val:.6f}")
            print(f"  Effect size (rank-biserial r): {r_es:+.3f} ({effect_size_label(r_es)})")

            if p_val < 0.05:
                print(f"  => SIGNIFICANT (p < 0.05): ML-GHCA is significantly better")
                print(f"     in the {regime_name} regime.")
            elif p_val < 0.10:
                print(f"  => MARGINAL TREND (0.05 <= p < 0.10): suggestive but not")
                print(f"     significant at alpha = 0.05 in the {regime_name} regime.")
            else:
                print(f"  => NULL RESULT (p >= 0.10): no significant advantage for")
                print(f"     ML-GHCA in the {regime_name} regime.")

    return results


# ============================================================
# 7. REGIME-LEVEL HONEST-FAILURE REPORTING
# Flags null or negative gains PER SCALE REGIME rather than
# across the pooled 40-problem aggregate. A pooled average can
# mask regime-specific failures: e.g. ML-GHCA might win at
# Large scale but lose at Small scale, yet the pooled mean
# shows ~0% and hides both effects.
# ============================================================

def regime_honest_failure_report(baseline, ghca, heuristic, ml, regime_results=None):
    print("\n" + "=" * 70)
    print("REGIME-LEVEL HONEST-FAILURE REPORT")
    print("=" * 70)
    print("Flags null or negative ML-GHCA gains per regime.")
    print("A 'failure' here means either:")
    print("  (a) mean % gain <= 0  (ML-GHCA is not cheaper on average), OR")
    print("  (b) Wilcoxon p >= 0.05 (gain is not statistically significant).")
    print("Both conditions are checked independently.\n")

    regimes = partition_by_regime(baseline, ghca, heuristic, ml)

    # If regime_results weren't passed in, run the tests now
    if regime_results is None:
        regime_results = regime_wilcoxon_tests(baseline, ghca, heuristic, ml)

    any_flag = False

    for regime_name in ["Small", "Large"]:
        r = regimes[regime_name]
        ml_arr = r["ML"]

        print(f"\n{'-' * 70}")
        print(f"  REGIME: {regime_name.upper()}")
        print(f"{'-' * 70}")

        checks = [
            ("vs GHCA", r["GHCA"], ml_arr, "GHCA vs ML-GHCA"),
            ("vs Heuristic", r["Heuristic"], ml_arr, "Heuristic-GHCA vs ML-GHCA"),
        ]

        for label, comparator, ml_vals, result_key in checks:
            diff = comparator - ml_vals
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_arr = np.where(comparator != 0, (diff / comparator) * 100, 0.0)
            mean_pct = float(np.mean(pct_arr))

            # Retrieve test result
            if regime_name in regime_results and result_key in regime_results[regime_name]:
                p_val = regime_results[regime_name][result_key]["p_value"]
                r_es = regime_results[regime_name][result_key]["r"]
            else:
                p_val = 1.0
                r_es = 0.0

            flag_negative = mean_pct <= 0
            flag_nonsig = p_val >= 0.05

            if flag_negative or flag_nonsig:
                any_flag = True
                flags = []
                if flag_negative:
                    flags.append(f"mean gain = {mean_pct:+.2f}% (non-positive)")
                if flag_nonsig:
                    flags.append(f"p = {p_val:.5f} (not significant)")

                print(f"\n  [!] FLAG [{regime_name}] ML-GHCA {label}:")
                for f in flags:
                    print(f"      * {f}")
                print(f"      Effect size r = {r_es:+.3f} ({effect_size_label(r_es)})")
                print(f"      -> Report this as a regime-specific limitation.")
                print(f"        The ML layer does NOT demonstrate a significant advantage")
                print(f"        over {label.replace('vs ', '')} in the {regime_name} regime.")
            else:
                print(f"\n  [OK] PASS [{regime_name}] ML-GHCA {label}:")
                print(f"      mean gain = {mean_pct:+.2f}%, p = {p_val:.5f}, r = {r_es:+.3f}")

    if not any_flag:
        print("\n  All regime-level checks passed. No honest-failure flags.")
    else:
        print("\n  " + "-" * 60)
        print("  SUMMARY: One or more regime-level limitations were flagged above.")
        print("  These should be reported per-regime in the paper, not masked by")
        print("  pooled aggregates. A pooled 'no significant difference' can hide")
        print("  opposing regime-specific effects.")
        print("  " + "-" * 60)


# --- RUN ALL TESTS ---
if __name__ == "__main__":
    baseline, ghca, heuristic, ml = load_results()

    # Sections 1-4: original omnibus tests (preserved for base-paper comparability)
    descriptive_stats(baseline, ghca, heuristic, ml)
    normality_check(ghca, ml, heuristic)
    friedman_test(baseline, ghca, heuristic, ml)
    two_way_anova(
        np.array(baseline),
        np.array(ghca),
        np.array(heuristic),
        np.array(ml)
    )

    # Section 5: pooled Wilcoxon (now with effect sizes)
    wilcoxon_tests(ghca, heuristic, ml)

    # Section 6: regime-decoupled Wilcoxon tests (Phase 3 P1)
    regime_results = regime_wilcoxon_tests(baseline, ghca, heuristic, ml)

    # Section 7: per-regime honest-failure reporting (Phase 3 P1)
    regime_honest_failure_report(baseline, ghca, heuristic, ml, regime_results)