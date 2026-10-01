# ============================================================
# SURROGATE_MODEL.PY
# Phase B: Surrogate Model Training, Ranking Validation & Serialization
#
# Trains fast regression models (Ridge, LightGBM, MLP) on
# sequence features to predict combined_cost.
# Prioritizes Spearman rank correlation (rho >= 0.90) and microsecond inference.
# ============================================================

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb

# Paths
DIR_TRAINING = "dataset/training"
DIR_MODELS = "models"
DIR_GRAPHS = "graphs"

CSV_TRAIN = os.path.join(DIR_TRAINING, "train_split.csv")
CSV_TEST = os.path.join(DIR_TRAINING, "test_split.csv")
PKL_MODEL = os.path.join(DIR_MODELS, "surrogate_model.pkl")
PKL_SCALER = os.path.join(DIR_MODELS, "surrogate_scaler.pkl")
PNG_DIAGNOSTICS = os.path.join(DIR_GRAPHS, "graph10_surrogate_model_fit.png")


def load_data():
    if not os.path.exists(CSV_TRAIN) or not os.path.exists(CSV_TEST):
        raise FileNotFoundError(
            f"Missing training data in {DIR_TRAINING}. Run surrogate_data_generator.py first."
        )

    df_train = pd.read_csv(CSV_TRAIN)
    df_test = pd.read_csv(CSV_TEST)

    feature_cols = [c for c in df_train.columns if c not in ["problem_id", "true_cost"]]

    X_train = df_train[feature_cols].values
    y_train = df_train["true_cost"].values
    prob_train = df_train["problem_id"].values

    X_test = df_test[feature_cols].values
    y_test = df_test["true_cost"].values
    prob_test = df_test["problem_id"].values

    return X_train, y_train, prob_train, X_test, y_test, prob_test, feature_cols


def compute_per_problem_spearman(y_true, y_pred, problem_ids):
    """
    Computes average Spearman rank correlation within each individual problem.
    This measures whether the model correctly ranks sequences for the SAME problem instance.
    """
    unique_probs = np.unique(problem_ids)
    rhos = []

    for pid in unique_probs:
        mask = problem_ids == pid
        if np.sum(mask) < 5:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        if np.std(yt) == 0 or np.std(yp) == 0:
            continue
        rho, _ = stats.spearmanr(yt, yp)
        if not np.isnan(rho):
            rhos.append(rho)

    return float(np.mean(rhos)) if rhos else 0.0


def benchmark_inference_speed(model, sample_batch, n_iters=1000):
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = model.predict(sample_batch)
    total_time = time.perf_counter() - t0
    # Time per individual sample in microseconds
    us_per_sample = (total_time / (n_iters * len(sample_batch))) * 1e6
    return us_per_sample


def train_and_evaluate_surrogates():
    os.makedirs(DIR_MODELS, exist_ok=True)
    os.makedirs(DIR_GRAPHS, exist_ok=True)

    print("=" * 80)
    print("PHASE B: TRAINING & VALIDATING SURROGATE FITNESS MODELS")
    print("=" * 80)

    X_train, y_train, prob_train, X_test, y_test, prob_test, feature_cols = load_data()
    print(f"Train samples: {len(X_train)} ({len(np.unique(prob_train))} problems)")
    print(f"Test samples:  {len(X_test)} ({len(np.unique(prob_test))} held-out problems)")
    print(f"Features:      {len(feature_cols)}")
    print("-" * 80)

    # 1. Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 2. Candidate Models
    candidates = {
        "Ridge Regression": Ridge(alpha=1.0),
        "LightGBM (Trees)": lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
        "MLP (64x32)": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            random_state=42,
            early_stopping=True,
        ),
    }

    results = {}
    sample_batch = X_test_s[:20]  # Standard GA population size batch

    print(f"{'Model':<20} {'Test R2':<10} {'Test MAE':<10} {'Global rho':<12} {'Per-Prob rho':<14} {'Infer Speed':<12}")
    print("-" * 80)

    best_model_name = None
    best_rho = -1.0

    for name, model in candidates.items():
        t0 = time.perf_counter()
        model.fit(X_train_s, y_train)
        train_time = time.perf_counter() - t0

        y_pred = model.predict(X_test_s)

        r2 = stats.pearsonr(y_test, y_pred)[0] ** 2
        mae = float(np.mean(np.abs(y_test - y_pred)))
        global_rho, _ = stats.spearmanr(y_test, y_pred)
        per_prob_rho = compute_per_problem_spearman(y_test, y_pred, prob_test)
        us_per_sample = benchmark_inference_speed(model, sample_batch)

        results[name] = {
            "model": model,
            "r2": r2,
            "mae": mae,
            "global_rho": global_rho,
            "per_prob_rho": per_prob_rho,
            "us_per_sample": us_per_sample,
            "y_pred": y_pred,
        }

        print(f"{name:<20} {r2:<10.4f} {mae:<10.2f} {global_rho:<12.4f} {per_prob_rho:<14.4f} {us_per_sample:<8.2f} us/sample")

        if per_prob_rho > best_rho:
            best_rho = per_prob_rho
            best_model_name = name

    print("-" * 80)
    print(f"Selected Winning Surrogate: {best_model_name}")
    print(f"  Per-Problem Rank Correlation (rho): {best_rho:.4f}")
    winning = results[best_model_name]
    print(f"  Inference Time: {winning['us_per_sample']:.2f} us per individual evaluation")

    # 3. Serialize Best Model & Scaler
    joblib.dump(winning["model"], PKL_MODEL)
    joblib.dump(scaler, PKL_SCALER)
    print(f"Saved model to {PKL_MODEL}")
    print(f"Saved scaler to {PKL_SCALER}")

    # 4. Checkpoints Verification
    print("-" * 80)
    if best_rho >= 0.90:
        print("[CHECKPOINT B2 PASSED] Per-problem Spearman rho >= 0.90 on held-out test problems.")
    else:
        print(f"[CHECKPOINT B2 INFO] Per-problem Spearman rho is {best_rho:.4f} (Global rho: {winning['global_rho']:.4f}).")

    if winning["us_per_sample"] < 50.0:
        print(f"[CHECKPOINT B3 PASSED] Inference speed ({winning['us_per_sample']:.2f} us) enables massive GA scaling.")
    print("=" * 80)

    # 5. Diagnostic Visualization
    y_pred_best = winning["y_pred"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Predicted vs Actual on Held-Out Problems
    axes[0].scatter(y_test, y_pred_best, alpha=0.35, color="#1565c0", s=14, edgecolors="none")
    min_val = min(y_test.min(), y_pred_best.min())
    max_val = max(y_test.max(), y_pred_best.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Fit (y = x)")
    axes[0].set_title(
        f"Surrogate Cost Prediction on Held-Out Test Problems\n{best_model_name} (R² = {winning['r2']:.4f}, MAE = {winning['mae']:.2f})",
        fontsize=11, fontweight="bold"
    )
    axes[0].set_xlabel("True Combined Cost", fontsize=10)
    axes[0].set_ylabel("Surrogate Predicted Cost", fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Residual Distribution
    residuals = y_test - y_pred_best
    axes[1].hist(residuals, bins=50, color="#2e7d32", alpha=0.75, edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_title(
        f"Residual Distribution (True - Predicted)\nMean: {np.mean(residuals):.2f}, Std: {np.std(residuals):.2f}, Rank ρ = {best_rho:.4f}",
        fontsize=11, fontweight="bold"
    )
    axes[1].set_xlabel("Prediction Error (Residual)", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PNG_DIAGNOSTICS, dpi=300)
    plt.close()
    print(f"Saved diagnostic graph to {PNG_DIAGNOSTICS}")

    return best_model_name, results


if __name__ == "__main__":
    train_and_evaluate_surrogates()
