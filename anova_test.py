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
            print(f"Wilcoxon statistic: {round(stat, 4)}")
            print(f"P-value (one-sided, GHCA > ML-GHCA): {round(p_value, 6)}")
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
            print(f"Wilcoxon statistic: {round(stat2, 4)}")
            print(f"P-value (one-sided, Heuristic > ML-GHCA): {round(p_value2, 6)}")
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


# --- RUN ALL TESTS ---
if __name__ == "__main__":
    baseline, ghca, heuristic, ml = load_results()

    descriptive_stats(baseline, ghca, heuristic, ml)
    normality_check(ghca, ml, heuristic)
    friedman_test(baseline, ghca, heuristic, ml)
    two_way_anova(
        np.array(baseline),
        np.array(ghca),
        np.array(heuristic),
        np.array(ml)
    )
    wilcoxon_tests(ghca, heuristic, ml)