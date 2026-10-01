# Surrogate Fitness Function: Phase-wise Implementation Plan
# =========================================================
# 
# OBJECTIVE: Train an ML model to predict combined_cost(sequence) in 
# microseconds, enabling 50-100x more GA generations in the same 
# wall-clock time. The existing GHCA pipeline stays at FULL STRENGTH.
#
# TARGET OUTCOME: ML-Surrogate-GHCA achieves statistically significant 
# improvement over plain GHCA (p < 0.05 on paired Wilcoxon), because 
# the GA can now explore 10,000-20,000 generations instead of 200.
#
# ARCHITECTURE:
#   Current:  [ML Sort] -> HC(1000) -> GA(200 gen, 20 pop, combined_cost eval)
#   Proposed: HC(1000) -> GA(10000+ gen, 20 pop, SURROGATE eval, periodic true eval)
#
# The ML model's role changes from "initial sorter" (which gets erased) 
# to "fast evaluator" (which enables deeper search).


# ==================================================================
# PROJECT STRUCTURE & DATA ORGANIZATION
# ==================================================================
#
# CURRENT PROBLEM:
#   - results/ mixes CSVs, PNGs, and sweep outputs in one flat folder
#   - generate_training_data() in ml_layer.py regenerates data in-memory
#     every time it's called -- no persistence, no caching
#   - Three scripts (ml_layer.py, phase5_gbdt_ablation.py,
#     plot_phase5_ablation.py) each independently regenerate training
#     data with no shared source of truth
#
# NEW DIRECTORY LAYOUT:
#
#   dataset/
#     raw/                          # Cached problem instances (reusable)
#       problems_cache.pkl          # 500+ problem instances serialized
#                                   # Keys: (num_parts, pool_num, layout_num, seed)
#                                   # Values: list[dict] (the ops list)
#
#     training/                     # Surrogate ML training data
#       surrogate_features.csv      # Full dataset: f01..f28 + true_cost + problem_id
#       train_split.csv             # 80% of problems (split by problem_id, not row)
#       test_split.csv              # 20% held-out problems (never seen during training)
#
#     benchmarks/                   # Standardized benchmark result CSVs
#       benchmark_results.csv       # 40-problem main results (from main.py)
#       dual_scale_summary.csv      # Dual-scale summary (from dual_scale_analysis.py)
#       phase5_model_capacity_ablation.csv
#       phase5_hyperparameter_sweep.csv
#       surrogate_benchmark_results.csv   # NEW: surrogate method results
#       surrogate_statistical_tests.csv   # NEW: Wilcoxon test outputs
#
#   models/                         # Trained model artifacts (git-tracked)
#     surrogate_model.pkl           # Best surrogate model (joblib)
#     surrogate_scaler.pkl          # StandardScaler fitted on training data
#
#   results/                        # ONLY output figures/graphs (regenerable)
#     graph1_oct_comparison.png
#     graph2_load_balance.png
#     ... (existing graphs stay here)
#     graph10_surrogate_model_fit.png       # NEW
#     graph11_surrogate_convergence.png     # NEW
#     graph12_surrogate_benchmark.png       # NEW
#     graph13_surrogate_vs_ghca_diff.png    # NEW
#
# MIGRATION CHECKLIST (do BEFORE starting Phase A):
#   [ ] Create dataset/raw/, dataset/training/, dataset/benchmarks/
#   [ ] Create models/
#   [ ] Move benchmark_results.csv -> dataset/benchmarks/benchmark_results.csv
#   [ ] Move dual_scale_summary.csv -> dataset/benchmarks/dual_scale_summary.csv
#   [ ] Move phase5_hyperparameter_sweep.csv -> dataset/benchmarks/
#   [ ] Update CSV_PATH constants in: anova_test.py, dual_scale_analysis.py,
#       plot_results.py, plot_dual_scale.py, plot_phase5_ablation.py,
#       phase5_gbdt_ablation.py, weight_sensitivity.py
#   [ ] Update main.py to write to dataset/benchmarks/benchmark_results.csv
#   [ ] Add dataset/, models/ to .gitignore selectively:
#       - Track dataset/benchmarks/*.csv (small, important)
#       - Track models/*.pkl ONLY if < 10MB
#       - Ignore dataset/raw/ (large, regenerable)
#       - Ignore dataset/training/ (large, regenerable)
#
# PATH CONSTANTS (use across all new files):
#
#   DIR_DATASET_RAW      = "dataset/raw"
#   DIR_DATASET_TRAINING = "dataset/training"
#   DIR_DATASET_BENCH    = "dataset/benchmarks"
#   DIR_MODELS           = "models"
#   DIR_RESULTS          = "results"
#
#   CSV_BENCHMARK        = "dataset/benchmarks/benchmark_results.csv"
#   CSV_SURROGATE_TRAIN  = "dataset/training/surrogate_features.csv"
#   CSV_SURROGATE_BENCH  = "dataset/benchmarks/surrogate_benchmark_results.csv"
#   PKL_PROBLEMS_CACHE   = "dataset/raw/problems_cache.pkl"
#   PKL_SURROGATE_MODEL  = "models/surrogate_model.pkl"
#   PKL_SURROGATE_SCALER = "models/surrogate_scaler.pkl"


# ==================================================================
# PHASE A: DATA GENERATION & FEATURE ENGINEERING
# ==================================================================
# 
# GOAL: Generate a large labeled dataset of (sequence -> combined_cost)
# pairs and design features that capture ORDER-DEPENDENT information.
# All data saved under dataset/ with proper train/test splits.
#
# CHECKPOINT A1: surrogate_data_generator.py exists and produces
#   dataset/training/surrogate_features.csv with N >= 50,000 rows.
#
# CHECKPOINT A2: Feature extraction runs in < 0.1ms per sequence 
#   (measured via timeit). This is critical -- if feature extraction 
#   is slow, the surrogate provides no speedup.
#
# CHECKPOINT A3: dataset/raw/problems_cache.pkl exists with 500+
#   cached problem instances for reproducibility.
#
# --- Implementation Details ---
#
# FILE: surrogate_data_generator.py (NEW, ~200 lines)
#
# Step A1: Generate and cache diverse problem instances
#   - Use generate_problem() with varied parameters:
#     num_parts in [5, 8, 10, 12, 15, 18]
#     pool_num in [1, 2]
#     layout_num in [1, 2, 3]
#     seeds 1..500
#   - Cache all generated problem instances to dataset/raw/problems_cache.pkl
#     using pickle. Key = (num_parts, pool_num, layout_num, seed), value = ops list.
#   - On subsequent runs, load from cache instead of regenerating.
#   - Print: "Cached X problem instances to dataset/raw/problems_cache.pkl"
#
# Step A2: Generate sequence permutations and compute true costs
#   - For each cached problem, generate 100+ random permutations
#   - Also include HC-refined and GA-refined sequences (not just random)
#     to cover the "good solution" region of the feature space.
#   - Compute true combined_cost() for each permutation
#   - Target: 50,000-100,000 (features, cost) samples
#
# Step A3: Design and extract sequence-level features
#   The features must capture ORDER-DEPENDENT properties of the sequence.
#   They must be fixed-length regardless of sequence length (20 ops vs 60 ops).
#
#   FEATURE SET (28 features total):
#
#   Group 1: Problem-level (4 features) -- same for all permutations
#     f01: num_operations (int)
#     f02: num_unique_jobs (int)
#     f03: mean_processing_time (float)
#     f04: mean_travel_time (float)
#
#   Group 2: Machine load distribution (8 features) -- order-independent
#     f05-f08: total_processing_time_on_M1..M4 (float x4)
#     f09: std_dev_of_machine_loads (float) -- proxy for LB component
#     f10: max_machine_load (float) -- proxy for OCT component
#     f11: min_machine_load (float)
#     f12: max_load_minus_min_load (float) -- imbalance indicator
#
#   Group 3: Order-dependent sequence features (12 features) -- THE KEY
#     f13: num_same_machine_consecutive_pairs (int)
#         Count how many times ops[i] and ops[i+1] use the same machine.
#         High value = machine contention = higher OCT.
#     f14: num_same_vehicle_consecutive_pairs (int)
#         Same for vehicle contention.
#     f15: num_same_job_consecutive_pairs (int)
#         Adjacent ops from same job -- preserves job precedence.
#     f16: mean_travel_time_of_first_quarter (float)
#         Avg travel time of ops in positions [0, n/4).
#     f17: mean_travel_time_of_last_quarter (float)
#         Avg travel time of ops in positions [3n/4, n).
#     f18: mean_processing_time_of_first_quarter (float)
#     f19: mean_processing_time_of_last_quarter (float)
#     f20: sum_processing_time_weighted_by_position (float)
#         sum(ops[i].processing_time * (n - i) / n for all i)
#         High value = heavy ops scheduled early = likely lower OCT.
#     f21: machine_transition_entropy (float)
#         Shannon entropy of the machine-to-machine transition matrix.
#         High entropy = diverse transitions = less contention.
#     f22: max_consecutive_same_machine_run (int)
#         Longest streak of consecutive ops on the same machine.
#     f23: vehicle_switch_count (int)
#         Number of times the vehicle changes between consecutive ops.
#     f24: avg_positional_gap_same_job (float)
#         For ops of the same job, avg gap in sequence position.
#         Low = job ops clustered = job finishes faster.
#
#   Group 4: Interaction features (4 features)
#     f25: f13 * f10 (machine contention * max load -- interaction)
#     f26: f20 / f01 (position-weighted processing normalized by size)
#     f27: f09 / f10 (load std_dev / max load -- relative imbalance)
#     f28: f22 * f04 (max same-machine run * avg travel -- penalty proxy)
#
#   IMPLEMENTATION: Write as a single function:
#     def extract_sequence_features(ops: list[dict]) -> np.ndarray:
#         # Returns array of shape (28,)
#         # Must complete in < 0.1ms for 60 operations
#
# Step A4: Split and save
#   - Assign each row a problem_id based on which problem it came from
#   - Split by PROBLEM_ID (not by row) to prevent data leakage:
#     Train: 80% of unique problem_ids
#     Test:  20% of unique problem_ids
#   - Save to:
#     dataset/training/surrogate_features.csv  (full dataset)
#     dataset/training/train_split.csv         (training rows only)
#     dataset/training/test_split.csv          (test rows only)
#
# VERIFICATION:
#   python surrogate_data_generator.py
#   -> Prints: "Cached X problem instances to dataset/raw/problems_cache.pkl"
#   -> Prints: "Generated X samples from Y problems"
#   -> Prints: "Feature extraction speed: X.XX us/sample"
#   -> Prints: "Train: X samples from Y problems | Test: X samples from Y problems"
#   -> Saves CSVs to dataset/training/


# ==================================================================
# PHASE B: SURROGATE MODEL TRAINING & VALIDATION
# ==================================================================
#
# GOAL: Train a fast, accurate cost predictor. The model must:
#   1. Predict combined_cost with low error (MAPE < 5%)
#   2. PRESERVE RANKING of solutions (Spearman rho > 0.95)
#      Ranking accuracy matters more than absolute accuracy --
#      the GA only needs to know which individual is BETTER,
#      not the exact cost.
#   3. Run inference in < 0.01ms per sample
#
# CHECKPOINT B1: surrogate_model.py exists with train_surrogate()
#   and predict_cost() functions.
#
# CHECKPOINT B2: On held-out test problems (from dataset/training/test_split.csv),
#   Spearman rank correlation >= 0.90 between predicted and true costs.
#
# CHECKPOINT B3: Inference speed < 0.01ms per sample (batch of 20).
#
# CHECKPOINT B4: models/surrogate_model.pkl and models/surrogate_scaler.pkl exist.
#
# --- Implementation Details ---
#
# FILE: surrogate_model.py (NEW, ~200 lines)
#
# Step B1: Load and split data
#   - Load dataset/training/train_split.csv and dataset/training/test_split.csv
#   - Standardize features (StandardScaler, fit on train only)
#   - Save scaler to models/surrogate_scaler.pkl
#
# Step B2: Train multiple model types, select best
#   Candidates (in order of speed):
#     1. Ridge Regression (fastest inference, baseline)
#     2. LightGBM / XGBoost (fast inference, handles nonlinearity)
#     3. Small MLP (2 hidden layers, 64-32 units, ReLU)
#        -> Use sklearn MLPRegressor for simplicity
#
#   For each model:
#     - 5-fold cross-validation (folds split by problem within train set)
#     - Report: MAE, MAPE, R-squared, Spearman rank correlation
#     - Measure inference time (batch of 20 samples, averaged over 1000 runs)
#
# Step B3: Select winning model based on:
#   Priority 1: Spearman rho >= 0.90 (ranking accuracy)
#   Priority 2: Inference time < 0.01ms per sample
#   Priority 3: Lowest MAPE
#
# Step B4: Final evaluation on held-out test set
#   - Load dataset/training/test_split.csv
#   - Report: MAE, MAPE, R-squared, Spearman rho on TEST data
#   - This is the number that matters -- train metrics can overfit
#
# Step B5: Save trained model
#   - joblib.dump(model, 'models/surrogate_model.pkl')
#   - joblib.dump(scaler, 'models/surrogate_scaler.pkl')
#
# Step B6: Produce diagnostic report
#   - Predicted vs actual scatter plot
#   - Residual distribution
#   - Rank correlation plot (true rank vs predicted rank for test problems)
#   - Save to results/graph10_surrogate_model_fit.png
#
# VERIFICATION:
#   python surrogate_model.py
#   -> Prints model comparison table
#   -> Prints: "Selected: [model_name], Spearman rho = X.XX, MAPE = X.X%"
#   -> Prints: "Inference speed: X.XX us/sample"
#   -> Prints: "Test set: Spearman rho = X.XX, MAPE = X.X%"
#   -> Saves model to models/surrogate_model.pkl
#   -> Saves scaler to models/surrogate_scaler.pkl
#   -> Saves diagnostics to results/graph10_surrogate_model_fit.png


# ==================================================================
# PHASE C: INTEGRATE SURROGATE INTO GENETIC ALGORITHM
# ==================================================================
#
# GOAL: Modify genetic_algorithm.py to use the surrogate model for
# most fitness evaluations, with periodic true-cost re-evaluation.
# Increase generation count from 200 to 5,000-20,000.
#
# CHECKPOINT C1: genetic_algorithm.py has a new function
#   genetic_algorithm_surrogate(ops, surrogate, scaler) that uses
#   the surrogate for fitness evaluation.
#
# CHECKPOINT C2: The surrogate-GA produces valid schedules (true cost
#   is computed and verified at the end).
#
# CHECKPOINT C3: Wall-clock time of surrogate-GA(10,000 gen) is
#   comparable to or less than standard GA(200 gen).
#
# --- Implementation Details ---
#
# FILE: genetic_algorithm.py (MODIFY, add ~80 lines)
#
# Step C1: Import surrogate utilities
#   from surrogate_data_generator import extract_sequence_features
#   import joblib  # for loading saved model/scaler
#
# Step C2: Add surrogate-based fitness evaluation
#
#   def surrogate_fitness(individual, surrogate, scaler):
#       features = extract_sequence_features(individual)
#       features_scaled = scaler.transform(features.reshape(1, -1))
#       return float(surrogate.predict(features_scaled)[0])
#
# Step C3: Add surrogate-accelerated GA function
#
#   def genetic_algorithm_surrogate(
#       ops,
#       surrogate_model,
#       scaler,
#       generations=10000,        # 50x more than standard
#       pop_size=20,              # same population
#       true_eval_interval=500,   # re-evaluate with true cost every 500 gen
#       true_eval_top_k=5,        # only re-evaluate top 5 individuals
#   ):
#       # 1. Run HC normally (true cost) to get initial best
#       hc_best, hc_cost = hill_climbing(ops)
#       
#       # 2. Create population (same as current)
#       population = create_population(ops, hc_best)
#       
#       # 3. Main GA loop
#       for gen in range(generations):
#           # FAST PATH: Use surrogate for selection & fitness
#           parents = tournament_selection_surrogate(population, surrogate, scaler)
#           # ... crossover, mutation (same as current) ...
#           
#           # PERIODIC TRUE EVALUATION: Every true_eval_interval generations
#           if gen % true_eval_interval == 0:
#               # Sort population by surrogate fitness
#               # Re-evaluate top_k with true combined_cost()
#               # Replace surrogate scores with true scores for these
#               # Update global best if improved
#               # This prevents the surrogate from drifting
#       
#       # 4. Final: evaluate best individual with TRUE cost
#       best_cost = combined_cost(best_sequence)
#       return best_sequence, best_cost
#
# Step C4: Keep the original genetic_algorithm() UNCHANGED
#   The surrogate version is a SEPARATE function. The original 200-gen
#   GA remains as the control/baseline.
#
# VERIFICATION:
#   python -c "
#   from genetic_algorithm import genetic_algorithm, genetic_algorithm_surrogate
#   from scheduler import generate_problem
#   import joblib, time
#   ops = generate_problem(num_parts=8, pool_num=1, layout_num=1, seed=42)
#   surrogate = joblib.load('models/surrogate_model.pkl')
#   scaler = joblib.load('models/surrogate_scaler.pkl')
#   
#   t0 = time.time(); _, cost_std = genetic_algorithm(ops); t_std = time.time()-t0
#   t0 = time.time(); _, cost_sur = genetic_algorithm_surrogate(ops, surrogate, scaler); t_sur = time.time()-t0
#   
#   print(f'Standard GA (200 gen): cost={cost_std:.2f}, time={t_std:.2f}s')
#   print(f'Surrogate GA (10000 gen): cost={cost_sur:.2f}, time={t_sur:.2f}s')
#   "


# ==================================================================
# PHASE D: FULL BENCHMARK EVALUATION & STATISTICAL TESTING
# ==================================================================
#
# GOAL: Run the surrogate-GHCA across all 40 dual-scale benchmark
# problems and perform rigorous statistical comparison against all
# existing methods.
#
# CHECKPOINT D1: surrogate_benchmark.py runs all 40 problems and
#   exports results to dataset/benchmarks/surrogate_benchmark_results.csv
#
# CHECKPOINT D2: Paired Wilcoxon signed-rank tests show p-value
#   and effect size for Surrogate-GHCA vs every other method.
#
# CHECKPOINT D3: Per-regime (Small/Large) analysis matches Phase 3 format.
#
# --- Implementation Details ---
#
# FILE: surrogate_benchmark.py (NEW, ~250 lines)
#
# Step D1: Run benchmarks
#   For each of the 40 problems (same seeds, same parameters as main.py):
#     1. Baseline: combined_cost(ops)  [no optimization]
#     2. GHCA: genetic_algorithm(ops)  [200 gen, full strength]
#     3. Heuristic-GHCA: heuristic_sort -> genetic_algorithm  [200 gen]
#     4. ML-GHCA (Linear): ml_sort -> genetic_algorithm  [200 gen]
#     5. Surrogate-GHCA: genetic_algorithm_surrogate(ops, ...)  [10,000+ gen]
#
#   Record: Problem, Regime, Jobs, Ops, and Cost for each method.
#   Also record wall-clock time for methods 2 and 5 to confirm time parity.
#
#   OUTPUT: dataset/benchmarks/surrogate_benchmark_results.csv
#
# Step D2: Statistical tests
#   For each pair (Surrogate-GHCA vs X):
#     - Paired Wilcoxon signed-rank test (one-sided: X > Surrogate)
#     - Matched-pairs rank-biserial correlation r
#     - Mean cost difference and percentage
#   
#   Run tests POOLED (N=40) and PER-REGIME (Small N=20, Large N=20).
#
#   OUTPUT: dataset/benchmarks/surrogate_statistical_tests.csv
#
# Step D3: Honest failure reporting
#   If Surrogate-GHCA does NOT achieve p < 0.05 vs GHCA, report honestly.
#   Flag per-regime results where surrogate shows null or negative gains.
#
# VERIFICATION:
#   python surrogate_benchmark.py
#   -> Runs all 40 problems
#   -> Prints statistical comparison table
#   -> Saves CSVs to dataset/benchmarks/


# ==================================================================
# PHASE E: VISUALIZATION & REPORTING
# ==================================================================
#
# GOAL: Produce publication-quality figures and update the project
# walkthrough with surrogate results.
#
# CHECKPOINT E1: At least 3 new graphs in results/ folder.
# CHECKPOINT E2: walkthrough.md updated with surrogate findings.
#
# --- Implementation Details ---
#
# FILE: plot_surrogate_results.py (NEW, ~200 lines)
#
# Graph 1: Surrogate model diagnostics (from Phase B)
#   - Predicted vs actual cost scatter with R-squared annotation
#   - Rank correlation visualization
#   -> results/graph10_surrogate_model_fit.png
#
# Graph 2: GA convergence comparison
#   - Standard GA convergence (200 gen) vs Surrogate GA (10,000 gen)
#   - X-axis: wall-clock time (not generation number)
#   - Y-axis: best cost found so far
#   - Shows that surrogate reaches lower cost in same time budget
#   -> results/graph11_surrogate_convergence.png
#
# Graph 3: Method comparison across 40 problems
#   - Bar chart or line plot: Baseline, GHCA, Heuristic, ML-GHCA, Surrogate-GHCA
#   - With enriched x-axis labels (Xp/Yo format) and regime separators
#   -> results/graph12_surrogate_benchmark.png
#
# Graph 4: Paired cost difference (GHCA - Surrogate) per problem
#   - Green bars = surrogate wins, red = surrogate loses
#   - With mean difference line and significance annotation
#   -> results/graph13_surrogate_vs_ghca_diff.png
#
# DATA SOURCES (all from dataset/benchmarks/):
#   - surrogate_benchmark_results.csv
#   - benchmark_results.csv (for GHCA/Heuristic/ML-GHCA comparison)
#
# VERIFICATION:
#   python plot_surrogate_results.py
#   -> Saves all 4 graphs to results/


# ==================================================================
# PHASE F: ABLATION & ROBUSTNESS CHECKS
# ==================================================================
#
# GOAL: Confirm the surrogate's contribution is real and robust.
#
# CHECKPOINT F1: Ablation shows more generations = better cost
#   (diminishing returns curve).
#
# CHECKPOINT F2: Weight sensitivity (0.5/0.5, 0.7/0.3, 0.9/0.1)
#   produces same conclusion with surrogate method.
#
# --- Implementation Details ---
#
# FILE: surrogate_ablation.py (NEW, ~150 lines)
#
# Ablation 1: Generation scaling curve
#   Run surrogate-GA with gen = [200, 500, 1000, 2000, 5000, 10000, 20000]
#   on 10 representative problems (5 small, 5 large).
#   Plot mean cost vs generation count -> shows diminishing returns.
#   This PROVES the surrogate's value: more search budget = lower cost.
#
# Ablation 2: True-eval interval sensitivity
#   Test true_eval_interval = [100, 200, 500, 1000, 2000]
#   Too frequent = slow (no speedup). Too rare = drift (bad ranking).
#   Find the sweet spot.
#
# Ablation 3: Weight sensitivity (same as weight_sensitivity.py)
#   Re-run surrogate-GHCA with w1/w2 = 0.5/0.5, 0.7/0.3, 0.9/0.1.
#   Confirm rankings are stable.
#
# VERIFICATION:
#   python surrogate_ablation.py
#   -> Prints generation scaling table
#   -> Prints interval sensitivity table
#   -> Prints weight stability verdict


# ==================================================================
# FILE CREATION & MODIFICATION SUMMARY
# ==================================================================
#
# NEW DIRECTORIES:
#   1. dataset/raw/                 [Phase A]
#   2. dataset/training/            [Phase A]
#   3. dataset/benchmarks/          [Pre-Phase A migration]
#   4. models/                      [Phase B]
#
# NEW FILES:
#   1. surrogate_data_generator.py  (~200 lines)  [Phase A]
#   2. surrogate_model.py           (~200 lines)  [Phase B]
#   3. surrogate_benchmark.py       (~250 lines)  [Phase D]
#   4. plot_surrogate_results.py    (~200 lines)  [Phase E]
#   5. surrogate_ablation.py        (~150 lines)  [Phase F]
#
# MODIFIED FILES:
#   6. genetic_algorithm.py         (+80 lines)   [Phase C]
#   7. main.py                      (CSV path)    [Pre-Phase A migration]
#   8. anova_test.py                (CSV path)    [Pre-Phase A migration]
#   9. dual_scale_analysis.py       (CSV path)    [Pre-Phase A migration]
#  10. plot_results.py              (CSV path)    [Pre-Phase A migration]
#  11. plot_dual_scale.py           (CSV path)    [Pre-Phase A migration]
#  12. plot_phase5_ablation.py      (CSV path)    [Pre-Phase A migration]
#  13. phase5_gbdt_ablation.py      (CSV path)    [Pre-Phase A migration]
#  14. weight_sensitivity.py        (CSV path)    [Pre-Phase A migration]
#
# MIGRATED FILES (move, not copy):
#  15. results/benchmark_results.csv          -> dataset/benchmarks/
#  16. results/dual_scale_summary.csv         -> dataset/benchmarks/
#  17. results/phase5_hyperparameter_sweep.csv -> dataset/benchmarks/
#
# TOTAL NEW CODE: ~1,080 lines across 5 new files + 1 modification
# TOTAL MIGRATION: ~14 path constant updates across 8 existing files
#
# EXECUTION ORDER:
#   Pre-Phase A (migration) -> Phase A -> Phase B -> Phase C ->
#   Phase D -> Phase E -> Phase F
#
#   Each phase has clear checkpoints. No phase depends on a later phase.
#
# ESTIMATED TIMELINE:
#   Pre-Phase A: 1 hour (directory creation, file moves, path updates)
#   Phase A: 2-3 hours (data generation is slow, ~30 min compute)
#   Phase B: 1-2 hours (model training + diagnostics)
#   Phase C: 1-2 hours (GA integration + testing)
#   Phase D: 2-3 hours (benchmark run is slow, ~45 min compute)
#   Phase E: 1-2 hours (plotting)
#   Phase F: 2-3 hours (ablation runs)
#   TOTAL: ~11-16 hours of implementation
