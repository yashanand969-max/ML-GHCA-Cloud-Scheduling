# Surrogate Fitness Function Optimization: Technical Walkthrough

This document records the full implementation, dataset migration, statistical testing, and ablation studies for the **Surrogate Fitness Function Framework (Solution #1)**.

---

## 1. Executive Summary & Architecture

To address the fundamental bottleneck of initial sequence ordering being erased by downstream search, the ML layer was elevated from a pre-search sorter into a **surrogate fitness evaluator**. 

* **Directory Restructuring**: Project assets were migrated into a disciplined `dataset/` directory separating `dataset/raw/`, `dataset/training/`, `dataset/benchmarks/`, `models/`, and `results/`.
* **Full-Strength Search Retention**: Hill Climbing (1000 iter) and Genetic Algorithm population dynamics remain at full strength.
* **Microsecond Fitness Evaluation**: Replaces computationally expensive simulation loops ($\sim 1\text{ ms}$) with a 28-feature regression surrogate ($\sim 6.44\ \mu\text{s}$ per evaluation), enabling rapid multi-generation exploration with periodic true-cost calibration.

---

## 2. Phase A: Feature Engineering & Dataset Generation

A dataset of **51,000 sequence samples** across 300 Ulusoy benchmark problems was generated and stored under `dataset/training/`:
* **Feature Extraction Speed**: **$71.18\ \mu\text{s}$** per sequence (satisfying the $< 100\ \mu\text{s}$ checkpoint).
* **28 Engineered Features**:
  * *Problem Level (4)*: `num_ops`, `num_jobs`, `mean_process`, `mean_travel`
  * *Machine Loads (8)*: Loads on $M_1..M_4$, load std-dev, load max, load min, load range
  * *Order Dynamics (12)*: Same-machine adjacency count, vehicle switches, job clustering gap, positional-weighted processing time, transition entropy
  * *Interactions (4)*: Contention $\times$ peak load, relative imbalance, position-weighted ratios

---

## 3. Phase B: Surrogate Model Performance

Three regression architectures were benchmarked on held-out test problems:

| Model Architecture | Test $R^2$ | Test MAE | Global Rank Correlation ($\rho$) | Inference Latency | Selected Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | **$0.9579$** | **$10.04$** | **$0.9749$** | **$6.44\ \mu\text{s}$ / sample** | **SELECTED** |
| **LightGBM (Trees)** | $0.9518$ | $10.81$ | $0.9716$ | $74.26\ \mu\text{s}$ / sample | Alternate |
| **MLP (64x32)** | $0.9556$ | $10.40$ | $0.9735$ | $5.48\ \mu\text{s}$ / sample | Alternate |

The fitted Ridge model and `StandardScaler` were serialized to [`models/surrogate_model.pkl`](file:///c:/Users/tiyas/OneDrive/Desktop/PBL_project/models/surrogate_model.pkl) and [`models/surrogate_scaler.pkl`](file:///c:/Users/tiyas/OneDrive/Desktop/PBL_project/models/surrogate_scaler.pkl).

---

## 4. Phase D: 40-Problem Dual-Scale Benchmark Results

Across all 40 standard problems (P1–P40) evaluated with 1,200 generations:

| Scale Regime | Method | Mean Cost | vs Baseline Diff | Wilcoxon $p$-value | Effect Size ($r$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Small Scale** ($N=20$) | Baseline | $129.16$ | Ref | — | — |
| | Plain GHCA (200 gen) | $91.29$ | $+37.87$ ($+28.6\%$) | $p = 0.00000$ | $+1.000$ |
| | Heuristic-GHCA | $91.16$ | $+38.00$ ($+28.7\%$) | $p = 0.00000$ | $+1.000$ |
| | Linear-ML-GHCA | $91.29$ | $+37.87$ ($+28.6\%$) | $p = 0.00000$ | $+1.000$ |
| | **Surrogate-GHCA (1200 gen)** | **$91.40$** | $+37.77$ ($+28.6\%$) | $p = 0.00000$ | $+1.000$ |
| **Large Scale** ($N=20$) | Baseline | $278.40$ | Ref | — | — |
| | Plain GHCA (200 gen) | $196.42$ | $+81.98$ ($+28.8\%$) | $p = 0.00000$ | $+1.000$ |
| | Heuristic-GHCA | $196.13$ | $+82.27$ ($+28.9\%$) | $p = 0.00000$ | $+1.000$ |
| | Linear-ML-GHCA | $196.22$ | $+82.18$ ($+28.8\%$) | $p = 0.00000$ | $+1.000$ |
| | **Surrogate-GHCA (1200 gen)** | **$196.42$** | $+81.98$ ($+28.8\%$) | $p = 0.00000$ | $+1.000$ |
| **Pooled Aggregate** ($N=40$) | Baseline | $203.78$ | Ref | — | — |
| | Plain GHCA (200 gen) | $143.85$ | $+59.93$ ($+28.7\%$) | $p = 0.00000$ | $+1.000$ |
| | Heuristic-GHCA | $143.64$ | $+60.14$ ($+28.8\%$) | $p = 0.00000$ | $+1.000$ |
| | Linear-ML-GHCA | $143.76$ | $+60.02$ ($+28.7\%$) | $p = 0.00000$ | $+1.000$ |
| | **Surrogate-GHCA (1200 gen)** | **$143.91$** | $+59.87$ ($+28.7\%$) | $p = 0.00000$ | $+1.000$ |

---

## 5. Phase F: Ablation & Sensitivity Analysis

### 1. Generation Scaling Curve
Evaluating search depth on schedule cost:
* Standard GA (200 gen): Mean Cost = **$143.49$**
* Surrogate GA (200 gen): Mean Cost = **$143.86$**
* Surrogate GA (500 gen): Mean Cost = **$143.89$**
* Surrogate GA (1,000 gen): Mean Cost = **$143.81$**
* Surrogate GA (1,500 gen): Mean Cost = **$143.49$**

### 2. Re-Calibration Interval Sensitivity
Testing true-cost evaluation frequency:
* Interval = Every 25 gen: Cost = $194.07$ (Time: $10.63\text{s}$)
* Interval = Every 50 gen: Cost = **$193.62$** (Time: $10.69\text{s}$)  *(Optimal)*
* Interval = Every 100 gen: Cost = $195.26$ (Time: $10.61\text{s}$)
* Interval = Every 200 gen: Cost = $195.23$ (Time: $10.82\text{s}$)

### 3. Cost Weight Stability
* $0.5 / 0.5$ (Equal): Std GA = $156.74$ | Surrogate = $155.25$ ($\Delta = +1.49$)
* $0.7 / 0.3$ (Default): Std GA = $217.00$ | Surrogate = $216.11$ ($\Delta = +0.89$)
* $0.9 / 0.1$ (OCT dominant): Std GA = $277.27$ | Surrogate = $276.97$ ($\Delta = +0.30$)
