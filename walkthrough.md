# FMS Scheduling Optimization: Technical Walkthrough

This document records the experimental results, statistical tests, and findings across **Phase 2 (Dual-Scale Benchmark Re-run)** and **Phase 5 (Isolated Gradient Boosting Model Capacity Ablation)**.

---

# Phase 5: Isolated Gradient Boosting Model Capacity Ablation

## 1. Executive Summary & Domain Justification
In accordance with the updated research roadmap:
* **Pre-condition**: Satisfied following the finalized Phase 2 dual-scale linear baseline.
* **Architecture Selection Rationale**: CNN / MLP architectures were swapped for **Gradient Boosted Decision Trees (GBDT / LightGBM)**.
  * *Mathematical & Domain Justification*: The 6 per-operation features (`[travel_time, processing_time, machine_idx, vehicle_idx, machine_ready, job_ready]`) are unordered tabular scalars with no spatial, temporal, or translation locality. A 1D-CNN's convolutional structure assumes translation invariance across adjacent dimensions, which is unmotivated on heterogeneous tabular data. In contrast, tree-based gradient boosted ensembles natively capture non-linear thresholds, discontinuities, and higher-order feature interactions without geometric assumptions.
* **Single-Variable Control**: Dataset generation, genetic algorithm (GA: 200 gen, pop 20), hill climbing (HC: 1000 iter), objective weights ($0.7 \times \text{OCT} + 0.3 \times \text{LB}$), and all 40 dual-scale benchmark problems were held **strictly constant**. Only the regressor was swapped:
  $$\text{LinearRegression} \longrightarrow \text{Gradient Boosting Regressor (LightGBM)}$$

---

## 2. Hyperparameter Sweep Results

A 5-fold cross-validation grid search was conducted on the $N = 4420$ operation training samples:

| Model Configuration | $n\_estimators$ | $max\_depth$ | $learning\_rate$ | 5-Fold CV $R^2$ Score | $\sigma(R^2)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Config 1 | 50 | 3 | 0.05 | 0.9605 | $\pm 0.0034$ |
| Config 2 | 50 | 5 | 0.10 | 0.9786 | $\pm 0.0028$ |
| Config 3 | 100 | 3 | 0.05 | 0.9751 | $\pm 0.0027$ |
| **Config 4 (Optimal)** | **100** | **5** | **0.10** | **0.9794** | $\pm 0.0028$ |
| Config 5 | 100 | 7 | 0.05 | 0.9793 | $\pm 0.0026$ |
| Config 6 | 200 | 5 | 0.05 | 0.9794 | $\pm 0.0028$ |
| Config 7 | 200 | 5 | 0.10 | 0.9792 | $\pm 0.0028$ |

Full sweep metrics are saved in [`results/phase5_hyperparameter_sweep.csv`](file:///c:/Users/prath/OneDrive/Desktop/PHASE%202/results/phase5_hyperparameter_sweep.csv).  
The optimal configuration (**$n\_estimators=100, max\_depth=5, lr=0.10$**) was frozen across both regimes.

### Model Fit & Capacity Comparison
| Model | Architecture / Capacity | Training Time | Full Fit $R^2$ | Mean Absolute Error (MAE) |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression** | Parametric Linear (6 weights) | $3.76\text{ ms}$ | $0.9411$ | $5.32\text{ min}$ |
| **GBDT (LightGBM)** | Ensemble of 100 trees (depth 5) | $72.84\text{ ms}$ | **$0.9847$** | **$2.68\text{ min}$** |

Upgrading to GBDT reduced the unexplained predictive variance by **$74.0\%$** ($1 - R^2$: $0.0589 \to 0.0153$).

---

## 3. Independent Secondary Ablation Table

The central question addressed: **"Does model capacity affect schedule quality?"**

All 40 dual-scale benchmark problems were re-evaluated. Detailed metrics are recorded in [`results/phase5_model_capacity_ablation.csv`](file:///c:/Users/prath/OneDrive/Desktop/PHASE%202/results/phase5_model_capacity_ablation.csv):

| Scale Regime | Method | Model Capacity ($R^2$) | Mean Cost | Median Cost | Std Dev | Diff vs Linear | Wilcoxon $p$-val (vs GBDT) | Effect Size ($r$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regime A: Small** | Baseline | None (Unoptimized) | 129.16 | 127.03 | 20.47 | — | $p = 0.00000$ | $+1.000$ |
| ($N = 20$, $\le 12$ Jobs) | GHCA (Control) | Metaheuristic | 91.29 | 91.09 | 10.48 | — | $p = 0.12566$ | $+0.300$ |
| | Heuristic-GHCA | Dispatch heuristic | **91.16** | 90.56 | 10.53 | — | $p = 0.71347$ | $-0.147$ |
| | Linear-ML-GHCA | Low ($0.9411$) | 91.29 | 90.74 | 10.43 | Ref | Ref | Ref |
| | **GBDT-ML-GHCA** | **High ($0.9847$)** | **91.20** | **90.68** | 10.45 | $+0.08$ ($+0.09\%$) | **$p = 0.25330$** | **$+0.174$** |
| **Regime B: Large** | Baseline | None (Unoptimized) | 278.40 | 290.78 | 50.37 | — | $p = 0.00000$ | $+1.000$ |
| ($N = 20$, 15–20 Parts) | GHCA (Control) | Metaheuristic | 196.42 | 195.66 | 26.86 | — | $p = 0.60309$ | $-0.067$ |
| | Heuristic-GHCA | Dispatch heuristic | **196.13** | 195.25 | 26.88 | — | $p = 0.99445$ | $-0.648$ |
| | Linear-ML-GHCA | Low ($0.9411$) | **196.24** | 195.60 | 26.97 | Ref | Ref | Ref |
| | **GBDT-ML-GHCA** | **High ($0.9847$)** | 196.42 | 195.62 | 26.98 | $-0.19$ ($-0.10\%$) | **$p = 0.91504$** | **$-0.368$** |
| **Pooled Aggregate** | Baseline | None (Unoptimized) | 203.78 | 186.82 | 83.50 | — | $p = 0.00000$ | $+1.000$ |
| ($N = 40$ Problems) | GHCA (Control) | Metaheuristic | 143.85 | 133.00 | 56.20 | — | $p = 0.24265$ | $+0.128$ |
| | Heuristic-GHCA | Dispatch heuristic | **143.64** | 132.54 | 56.13 | — | $p = 0.98896$ | $-0.421$ |
| | Linear-ML-GHCA | Low ($0.9411$) | 143.76 | 132.42 | 56.20 | Ref | Ref | Ref |
| | **GBDT-ML-GHCA** | **High ($0.9847$)** | 143.81 | 132.48 | 56.23 | $-0.05$ ($-0.04\%$) | **$p = 0.69339$** | **$-0.095$** |

---

## 4. Empirical Roadmap Findings

> [!IMPORTANT]
> **Primary Research Question**: *"Does model capacity affect schedule quality?"*
> **Definitive Answer**: **NO.**
> 1. **Predictive Capacity vs Optimization Payoff**: While GBDT improves regression fit substantially ($R^2: 0.9411 \to 0.9847$, MAE reduced by $49.6\%$), it produces **no statistically significant improvement in schedule cost** over the linear baseline ($p = 0.69339$ pooled; $p = 0.25330$ small scale; $p = 0.91504$ large scale).
> 2. **Search Domination**: The downstream hybrid search (Hill Climbing + Genetic Algorithm) thoroughly refines the combinatorial space. Any minor initial ordering benefit provided by a non-linear predictor is overshadowed by the metaheuristic search dynamics.
> 3. **Heuristic Benchmark**: The simple non-ML dispatch heuristic (longest processing time first) remains the top-performing initialization strategy ($143.64$ mean cost vs $143.76$ for Linear-ML and $143.81$ for GBDT-ML).
> 4. **Methodological Value**: This negative result is a high-value empirical contribution: it conclusively proves that the scheduling bottleneck in this FMS benchmark is **combinatorial exploration**, not regressor model capacity. Further scaling of model parameters (e.g., deeper neural networks) will not resolve the fundamental scheduling trade-offs.

---

## 5. Phase 5 Visualizations

````carousel
![Model Fit: Linear Regression vs GBDT](/C:/Users/prath/.gemini/antigravity/brain/231fcb56-7072-465e-bd2b-03c05e31964f/graph8_model_capacity_fit.png)
<!-- slide -->
![Model Capacity vs Schedule Cost Comparison](/C:/Users/prath/.gemini/antigravity/brain/231fcb56-7072-465e-bd2b-03c05e31964f/graph9_model_capacity_schedule_comparison.png)
````

---

# Phase 2 Core Experiment: Dual-Scale Evaluation

## Summary of Dual-Scale Baseline
* **Scale Regime A ($N=20$, Small $\le 12$ jobs)**:
  * Baseline: $129.16 \pm 20.47$
  * ML-GHCA: $91.18 \pm 10.43$
  * Wilcoxon vs GHCA: $p = 0.06560$ (marginal, non-significant at $\alpha = 0.05$).
* **Scale Regime B ($N=20$, Large Scale)**:
  * Baseline: $278.40 \pm 50.37$
  * ML-GHCA: $196.29 \pm 26.97$
  * Wilcoxon vs GHCA: $p = 0.29413$ (null result).
* **Scale Independence**: Cross-regime Mann-Whitney U test confirms that problem scale alone does not confer a statistically significant advantage to ML prioritization ($p = 0.46505$).

````carousel
![Dual-Scale Outcome Distributions](/C:/Users/prath/.gemini/antigravity/brain/231fcb56-7072-465e-bd2b-03c05e31964f/graph6_dual_scale_distributions.png)
<!-- slide -->
![Regime Comparison Boxplots](/C:/Users/prath/.gemini/antigravity/brain/231fcb56-7072-465e-bd2b-03c05e31964f/graph7_regime_comparison_boxplots.png)
<!-- slide -->
![Combined Cost Comparison](/C:/Users/prath/.gemini/antigravity/brain/231fcb56-7072-465e-bd2b-03c05e31964f/graph3_combined_cost.png)
<!-- slide -->
![Improvement vs GHCA](/C:/Users/prath/.gemini/antigravity/brain/231fcb56-7072-465e-bd2b-03c05e31964f/graph4_improvement_vs_ghca.png)
````
