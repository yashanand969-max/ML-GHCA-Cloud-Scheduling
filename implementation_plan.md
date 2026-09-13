# Implementation Plan — Phase 5: Isolated Gradient Boosting Model Capacity Ablation

## Background & Objectives
In the updated FMS Scheduling Optimization research roadmap, **Phase 5 (P2 · OPTIONAL ARCHITECTURE EXPLORATION)** investigates the effect of model capacity on schedule quality.

### Core Directive & Justification
- **Pre-condition**: Satisfied. Phase 2 delivered a finalized, stable linear baseline.
- **Architecture Selection**: Gradient Boosting (GBDT / XGBoost / LightGBM) replaces the previously contemplated CNN / MLP.
  - *Domain Justification*: The 6 per-operation features (`[travel_time, processing_time, machine_idx, vehicle_idx, machine_ready, job_ready]`) are unordered tabular scalars without spatial or temporal translation invariance. A 1D-CNN’s inductive biases (local filters, weight sharing across adjacent indices) are unmotivated on tabular feature vectors. In contrast, tree-based ensemble gradient boosting natively handles non-linear interactions, thresholds (e.g., machine contention transitions), and tabular scaling without geometric assumptions.
- **Single-Variable Control**: Hold dataset generation, genetic algorithm (GA), hill climbing (HC), objective function ($0.7 \times \text{OCT} + 0.3 \times \text{LB}$), and benchmark instances strictly constant. Swap **only** the regressor:
  $$\text{LinearRegression} \longrightarrow \text{Gradient Boosting Regressor}$$
- **Hyperparameter Sweep**: Perform a light grid sweep (`n_estimators`, `max_depth`, `learning_rate`) and freeze the best configuration across both regimes.
- **Independent Secondary Ablation Table**: Systematically report:
  *"Does model capacity affect schedule quality?"*

---

## User Review Required

> [!IMPORTANT]
> This plan isolates **model capacity** as the sole independent variable while preserving all problem instances (Regime A: Small $\le 12$ jobs, Regime B: Large 15–20 parts), HC/GA parameters, and objective metrics established in Phases 1 & 2.
> The primary research question answered is: *Does upgrading from a linear model to a non-linear gradient-boosted ensemble improve makespan/cost, or does the search bottleneck lie elsewhere?*

---

## Proposed Implementation Steps

### 1. Model Capacity Implementation & Hyperparameter Sweep (`phase5_gbdt_ablation.py`)
- Import existing environment and algorithms from [`scheduler.py`](file:///c:/Users/prath/OneDrive/Desktop/PHASE%202/scheduler.py), [`hill_climbing.py`](file:///c:/Users/prath/OneDrive/Desktop/PHASE%202/hill_climbing.py), and [`genetic_algorithm.py`](file:///c:/Users/prath/OneDrive/Desktop/PHASE%202/genetic_algorithm.py).
- Generate the exact same training dataset ($N = 200$ problem simulations, $\approx 4400$ operation instances with the 6 canonical features).
- Implement a systematic grid sweep:
  - `n_estimators`: $[50, 100, 200]$
  - `max_depth`: $[3, 5, 7]$
  - `learning_rate`: $[0.03, 0.1, 0.2]$
  - Cross-validation $R^2$ and RMSE tracking.
- Select the best configuration and train the GBDT model.
- Run all 40 dual-scale benchmark problems through GBDT-assisted GHCA (`gbdt_ghca`):
  - Predict completion times using GBDT.
  - Sort operations in descending order of predicted execution time.
  - Seed the exact same HC (1000 iter) + GA (200 gen) optimization pipeline.

### 2. Independent Secondary Ablation Table
Directly compare the two ML architectures alongside the controls across both regimes:
- **Model Metrics**: Model Capacity, $R^2$ Score, Training Time, Parameter Count/Depth.
- **Schedule Quality**:
  - Regime A (Small): Linear-ML Mean Cost vs GBDT-ML Mean Cost vs GHCA vs Heuristic.
  - Regime B (Large): Linear-ML Mean Cost vs GBDT-ML Mean Cost vs GHCA vs Heuristic.
- **Statistical Tests**:
  - Paired Wilcoxon signed-rank test between Linear-ML-GHCA and GBDT-ML-GHCA.
  - Paired Wilcoxon signed-rank test between Heuristic-GHCA and GBDT-ML-GHCA.
  - Matched-pairs rank-biserial correlation $r$ (effect size).
- Export complete results to [`results/phase5_model_capacity_ablation.csv`](file:///c:/Users/prath/OneDrive/Desktop/PHASE%202/results/phase5_model_capacity_ablation.csv).

### 3. Visualizations (`plot_phase5_ablation.py`)
- `graph8_model_capacity_fit.png`: Regression fit and residual error comparison (Linear vs GBDT).
- `graph9_model_capacity_schedule_comparison.png`: Side-by-side boxplots and paired difference charts of schedule cost under Linear-ML vs GBDT-ML vs Baseline/GHCA across Small and Large regimes.

---

## Verification Plan

### Automated Verification
1. Run `python phase5_gbdt_ablation.py` in `PHASE 2`:
   - Verify hyperparameter sweep finds optimal GBDT parameters.
   - Verify all 40 benchmark problems are evaluated without error.
   - Verify statistical tests and CSV export are produced.
2. Run `python plot_phase5_ablation.py`:
   - Verify generated plots are saved to `results/`.
3. Update [walkthrough.md](file:///C:/Users/prath/.gemini/antigravity/brain/231fcb56-7072-465e-bd2b-03c05e31964f/walkthrough.md) with comprehensive findings, tables, and embedded figures.
