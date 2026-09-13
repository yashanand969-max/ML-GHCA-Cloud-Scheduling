# ============================================================
# ML_LAYER.PY
# Linear Regression to predict operation execution time
# Uses prediction to create smarter initial sequence for HC
# This is our contribution over the base paper
# ============================================================

import random
import copy
from sklearn.linear_model import LinearRegression
import numpy as np
from scheduler import generate_problem, calculate_oct, calculate_load_balance, combined_cost
from hill_climbing import hill_climbing
from genetic_algorithm import genetic_algorithm


# --- STEP 1: GENERATE TRAINING DATA ---
# Run GHCA on many different problems
# Record features and resulting execution times
# This teaches our model what makes an operation take long

def generate_training_data(n_samples=200):
    X = []
    y = []

    machine_map = {"M1": 0, "M2": 1, "M3": 2, "M4": 3}
    vehicle_map = {"V1": 0, "V2": 1}

    for i in range(n_samples):
        ops = generate_problem(
            num_parts=random.randint(5, 10),
            pool_num=random.choice([1, 2]),
            layout_num=random.choice([1, 2, 3]),
            seed=i
        )

        machine_free_at = {}
        job_free_at = {}
        vehicle_free_at = {}

        for op in ops:
            machine = op["machine"]
            job = op["job"]
            vehicle = op["vehicle"]
            travel = op["travel_time"]
            process = op["processing_time"]

            machine_ready = machine_free_at.get(machine, 0)
            job_ready = job_free_at.get(job, 0)
            vehicle_ready = vehicle_free_at.get(vehicle, 0) + travel

            start_time = max(machine_ready, job_ready, vehicle_ready)
            finish_time = start_time + process

            machine_free_at[machine] = finish_time
            job_free_at[job] = finish_time
            vehicle_free_at[vehicle] = finish_time

            # richer features now
            X.append([
                travel,
                process,
                machine_map[machine],   # which machine
                vehicle_map[vehicle],   # which vehicle
                machine_ready,          # how busy is this machine
                job_ready,              # how far along is this job
            ])
            y.append(finish_time)

    return np.array(X), np.array(y)

# --- STEP 2: TRAIN THE MODEL ---

def train_model():
    X, y = generate_training_data(n_samples=200)
    model = LinearRegression()
    model.fit(X, y)
    print(f"ML Model trained on {len(X)} operation samples")
    print(f"R² score: {round(model.score(X, y), 3)}")
    return model


# --- STEP 3: SORT OPERATIONS USING ML PREDICTIONS ---
# Predict execution time for each operation
# Sort by predicted time descending
# Longer predicted operations go first — keeps machines busy

def ml_sort(ops, model):
    ops_copy = copy.deepcopy(ops)
    machine_map = {"M1": 0, "M2": 1, "M3": 2, "M4": 3}
    vehicle_map = {"V1": 0, "V2": 1}

    # simulate to get current machine and job states
    machine_free_at = {}
    job_free_at = {}
    vehicle_free_at = {}

    for op in ops_copy:
        machine = op["machine"]
        job = op["job"]
        vehicle = op["vehicle"]
        travel = op["travel_time"]
        process = op["processing_time"]

        machine_ready = machine_free_at.get(machine, 0)
        job_ready = job_free_at.get(job, 0)
        vehicle_ready = vehicle_free_at.get(vehicle, 0) + travel

        features = np.array([[
            travel,
            process,
            machine_map[machine],
            vehicle_map[vehicle],
            machine_ready,
            job_ready,
        ]])

        op["predicted_time"] = model.predict(features)[0]

        start_time = max(machine_ready, job_ready, vehicle_ready)
        finish_time = start_time + process
        machine_free_at[machine] = finish_time
        job_free_at[job] = finish_time
        vehicle_free_at[vehicle] = finish_time

    ops_copy.sort(key=lambda x: x["predicted_time"], reverse=True)
    return ops_copy


# --- STEP 4: ML-ASSISTED GHCA ---
# Instead of random initial sequence, use ML-sorted sequence
# Then run HC + GA on top of it

def ml_ghca(ops, model):
    # get ML-sorted sequence as starting point
    sorted_ops = ml_sort(ops, model)

    # run HC on the smarter starting sequence
    hc_best, hc_cost = hill_climbing(sorted_ops)

    # run GA seeded with HC result
    best_seq, best_cost = genetic_algorithm(sorted_ops)

    return best_seq, best_cost


# --- RUN AND COMPARE ALL THREE ---
if __name__ == "__main__":
    # train model once
    model = train_model()

    # use same problem for fair comparison
    ops = generate_problem(num_parts=5, pool_num=1, layout_num=1, seed=42)

    baseline_cost = combined_cost(ops)

    # run plain GHCA
    ghca_seq, ghca_cost = genetic_algorithm(ops)

    # run ML-GHCA
    ml_seq, ml_cost = ml_ghca(ops, model)

    print(f"\n{'='*45}")
    print(f"{'Method':<20} {'OCT':<10} {'LB':<10} {'Combined':<10}")
    print(f"{'='*45}")
    print(f"{'Baseline':<20} {calculate_oct(ops):<10} {calculate_load_balance(ops):<10} {baseline_cost:<10}")
    print(f"{'GHCA':<20} {calculate_oct(ghca_seq):<10} {calculate_load_balance(ghca_seq):<10} {ghca_cost:<10}")
    print(f"{'ML-GHCA (ours)':<20} {calculate_oct(ml_seq):<10} {calculate_load_balance(ml_seq):<10} {ml_cost:<10}")
    print(f"{'='*45}")
    print(f"\nGHCA improvement over baseline:    {round(baseline_cost - ghca_cost, 2)}")
    print(f"ML-GHCA improvement over GHCA:     {round(ghca_cost - ml_cost, 2)}")
    print(f"ML-GHCA total improvement:         {round(baseline_cost - ml_cost, 2)}")
