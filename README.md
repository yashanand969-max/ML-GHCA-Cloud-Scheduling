# ML-GHCA Cloud & FMS Scheduling Optimization

A research framework for **Flexible Manufacturing System (FMS) Scheduling Optimization** using a hybrid **Genetic Algorithm with Hill Climbing (GHCA)** and **Machine Learning Surrogate Acceleration**.

---

## 📁 Repository Structure

```text
ML-GHCA-Cloud-Scheduling/
│
├── dataset/                        # Benchmark datasets & training caches
│   ├── raw/                        # Serialized problem instances (problems_cache.pkl)
│   ├── training/                   # 51,000 sequence feature rows (train/test splits)
│   └── benchmarks/                 # Standardized 40-problem benchmark CSV records
│       ├── benchmark_results.csv
│       ├── dual_scale_summary.csv
│       ├── surrogate_benchmark_results.csv
│       └── surrogate_statistical_tests.csv
│
├── docs/                           # Documentation, reports & research roadmaps
│   ├── fms_scheduling_project_roadmap_updated.pdf
│   ├── pbl2 FINAL REPORT.doc
│   ├── surrogate_implementation_plan.py
│   └── walkthrough.md
│
├── graphs/                         # High-resolution (300 DPI) publication figures
│   ├── graph1_oct_comparison.png
│   ├── graph2_load_balance.png
│   ├── graph3_combined_cost.png
│   ├── graph4_improvement_vs_ghca.png
│   ├── graph5_convergence.png
│   ├── graph5_improvement_vs_heuristic.png
│   ├── graph6_dual_scale_distributions.png
│   ├── graph7_regime_comparison_boxplots.png
│   ├── graph8_model_capacity_fit.png
│   ├── graph9_model_capacity_schedule_comparison.png
│   ├── graph10_surrogate_model_fit.png
│   ├── graph11_surrogate_convergence.png
│   ├── graph12_surrogate_benchmark.png
│   └── graph13_surrogate_vs_ghca_diff.png
│
├── models/                         # Serialized ML model artifacts
│   ├── surrogate_model.pkl         # Trained Ridge regression surrogate
│   └── surrogate_scaler.pkl        # Fitted StandardScaler
│
├── scheduler.py                    # Ulusoy benchmark generator & cost functions
├── genetic_algorithm.py            # GHCA & Surrogate-accelerated GA optimizers
├── hill_climbing.py                # Local search (1000 iter swap neighborhood)
├── ml_layer.py                     # Linear regression pre-search prioritization
├── main.py                         # 40-problem dual-scale benchmark runner
├── anova_test.py                   # Decoupled Wilcoxon tests & effect size analysis
├── dual_scale_analysis.py          # Small vs Large scale comparative analysis
├── weight_sensitivity.py           # Objective weight stability analysis (0.5/0.7/0.9)
├── phase5_gbdt_ablation.py         # Model capacity ablation study (GBDT / LightGBM)
├── surrogate_data_generator.py     # 28-feature sequence dataset builder
├── surrogate_model.py              # Surrogate training, ranking validation & diagnostics
├── surrogate_benchmark.py          # 40-problem surrogate benchmark & hypothesis tests
├── surrogate_ablation.py           # Generation budget & interval sensitivity checks
│
├── requirements.txt                # Pinned dependencies
├── .gitignore                      # Repository hygiene rules
└── README.md                       # Project documentation & reproduction guide
```

---

## ⚡ 3-Command Reproduction Guide

To reproduce the entire benchmark and generate all statistical tables and figures from scratch:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data generation and train the surrogate fitness model
python surrogate_data_generator.py && python surrogate_model.py

# 3. Execute 40-problem dual-scale benchmarks and plot all publication graphs
python surrogate_benchmark.py && python plot_surrogate_results.py && python plot_results.py
```

---

## 🔬 Scientific Methodology & System Architecture

### 1. Ulusoy Ground-Truth Benchmark Specification
* **Physical Layout**: 4 machines ($M_1..M_4$), 2 Automated Guided Vehicles ($V_1, V_2$), and 3 physical layouts based on Ulusoy et al.
* **Deterministic Part Pools**: Parts drawn from 2 part pools with strict machine routings (e.g., $\{4 \to 2 \to 3\}$).
* **Processing Times**: $p_{ij} \sim \mathcal{U}[3, 15]$ minutes with automated rejection/resampling strictly enforcing the $[0.05, 0.20]$ travel-to-processing ratio ($\bar{t}/\bar{p}$).

### 2. Dual-Scale Evaluation Regimes
* **Regime A (Small Scale)**: 8 parts, $\sim 20\text{--}27$ operations ($N=20$ problems).
* **Regime B (Large Scale)**: 18 parts, $\sim 44\text{--}60$ operations ($N=20$ problems).

### 3. Optimization Hyperparameters

| Hyperparameter | Value | Scientific Rationale |
| :--- | :---: | :--- |
| `HC_ITERATIONS` | $1,000$ | Thoroughly explores the local 2-swap neighborhood to warm-start the genetic population. |
| `POPULATION_SIZE` | $20$ | Maintains diverse chromosome pool without excessive evaluation cost. |
| `GENERATIONS` (Standard) | $200$ | Baseline convergence point for exact simulation evaluations. |
| `MUTATION_RATE` | $0.30$ | Balances exploration vs exploitation; uses random and adjacent swap mutations. |
| `TOURNAMENT_SIZE` | $3$ | Imparts steady selection pressure towards lower combined cost. |
| `SURROGATE_GENERATIONS` | $1,200\text{--}1,500$ | Leverages microsecond surrogate evaluations ($6.44\ \mu\text{s}$) to scale search depth. |
| `TRUE_EVAL_INTERVAL` | $50$ | Periodically re-calibrates top candidates with true cost to prevent surrogate drift. |
| `OBJECTIVE_WEIGHTS` | $0.7 \times \text{OCT} + 0.3 \times \text{LB}$ | Reflects throughput priority (Operational Completion Time) alongside Machine Load Balance. |

---

## 📊 Summary of Experimental Findings

### 40-Problem Dual-Scale Benchmark Results

| Method | Small Regime ($N=20$) | Large Regime ($N=20$) | Pooled ($N=40$) | Wilcoxon $p$-val (vs Baseline) | Effect Size ($r$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** (Unoptimized) | $129.16$ | $278.40$ | $203.78$ | Ref | — |
| **GHCA** (200 gen) | $91.29$ | $196.42$ | $143.85$ | $p < 0.0001$ | $+1.000$ |
| **Heuristic-GHCA** (LPT Sort) | $91.16$ | $196.13$ | $143.64$ | $p < 0.0001$ | $+1.000$ |
| **Linear-ML-GHCA** | $91.29$ | $196.22$ | $143.76$ | $p < 0.0001$ | $+1.000$ |
| **GBDT-ML-GHCA** | $91.20$ | $196.42$ | $143.81$ | $p < 0.0001$ | $+1.000$ |
| **Surrogate-GHCA** (1200 gen) | **$91.40$** | **$196.42$** | **$143.91$** | $p < 0.0001$ | $+1.000$ |

---

## 🎓 Viva Defence & Research Insights

### Q1: Why does initial ML sorting not improve final schedule cost over plain GHCA?
> **Answer**: Downstream metaheuristic search (1,000 Hill Climbing iterations + 200 GA generations) explores thousands of combinatorial permutations. Any initial ordering signal is quickly overridden and refined by the aggressive search operators.

### Q2: Why is a CNN inappropriate for operation scheduling data?
> **Answer**: A Convolutional Neural Network assumes translation invariance and spatial/temporal locality across adjacent dimensions (e.g., pixel grids). Our 6 per-operation features (`[travel, process, machine_idx, vehicle_idx, machine_ready, job_ready]`) are unordered tabular scalars. Applying convolutional kernels over heterogeneous tabular dimensions lacks mathematical justification.

### Q3: How does the Surrogate Fitness Function improve the pipeline without weakening GHCA?
> **Answer**: Rather than trying to initialize the sequence, the surrogate acts as a **fast cost evaluator** ($\sim 6.44\ \mu\text{s}$ vs $\sim 1\text{ ms}$ for exact simulation). It achieves a ranking correlation of $\rho = 0.9749$, allowing the GA to explore $1,200+$ generations with periodic exact re-calibrations to eliminate drift.

---

## 📜 Citation & Academic Integrity

This codebase strictly adheres to honest empirical reporting:
* All null or marginal statistical findings are transparently disclosed.
* Baseline comparisons are distinguished from rigorous control comparisons (GHCA vs Heuristic vs ML).
* All tests utilize two-scale paired Wilcoxon signed-rank statistics and matched-pairs rank-biserial effect sizes ($r$).
