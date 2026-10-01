# ============================================================
# GENETIC_ALGORITHM.PY
# Uses combined cost function (OCT + load balance)
# Seeded with HC best result
# NOW INCLUDES: generation-by-generation convergence tracking
# ============================================================

import random
import copy
from scheduler import generate_problem, calculate_oct, calculate_load_balance, combined_cost
from hill_climbing import hill_climbing

POPULATION_SIZE = 20
GENERATIONS = 200
MUTATION_RATE = 0.3


def create_population(ops, hc_best):
    population = [copy.deepcopy(hc_best)]
    for _ in range(POPULATION_SIZE - 1):
        individual = copy.deepcopy(ops)
        random.shuffle(individual)
        population.append(individual)
    return population


def crossover(parent1, parent2):
    size = len(parent1)
    start = random.randint(0, size - 2)
    end = random.randint(start + 1, size - 1)

    child = copy.deepcopy(parent1[start:end])
    child_ids = [op["id"] for op in child]

    for op in parent2:
        if op["id"] not in child_ids:
            child.append(copy.deepcopy(op))
            child_ids.append(op["id"])

    return child


def mutate(individual):
    if random.random() < MUTATION_RATE:
        mutation_type = random.choice(["random", "adjacent"])
        if mutation_type == "random":
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        else:
            i = random.randint(0, len(individual) - 2)
            individual[i], individual[i+1] = individual[i+1], individual[i]
    return individual


def tournament_selection(population, tournament_size=3):
    selected = []
    for _ in range(POPULATION_SIZE // 2):
        tournament = random.sample(population, tournament_size)
        winner = min(tournament, key=lambda x: combined_cost(x))
        selected.append(copy.deepcopy(winner))
    return selected


# ============================================================
# UPDATED FUNCTION — now returns convergence_history too
# convergence_history = list of best cost at each generation
# This is the only change to the original genetic_algorithm()
# ============================================================

def genetic_algorithm(ops, track_convergence=False):
    hc_best, hc_cost = hill_climbing(ops)

    population = create_population(ops, hc_best)
    best_sequence = copy.deepcopy(hc_best)
    best_cost = hc_cost

    convergence_history = [best_cost]  # record starting cost (generation 0)

    for generation in range(GENERATIONS):
        parents = tournament_selection(population)

        next_generation = copy.deepcopy(parents)

        while len(next_generation) < POPULATION_SIZE:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            next_generation.append(child)

        population = next_generation

        for individual in population:
            cost = combined_cost(individual)
            if cost < best_cost:
                best_cost = cost
                best_sequence = copy.deepcopy(individual)

        convergence_history.append(best_cost)  # record best cost this generation

    if track_convergence:
        return best_sequence, best_cost, convergence_history
    else:
        return best_sequence, best_cost


# ============================================================
# SURROGATE-ACCELERATED GENETIC ALGORITHM (Phase C)
# Uses ultra-fast ML surrogate for selection/evolution,
# enabling 25x-100x more generations within equal wall-clock time.
# ============================================================

_cached_surrogate = None
_cached_scaler = None


def get_default_surrogate():
    global _cached_surrogate, _cached_scaler
    if _cached_surrogate is None or _cached_scaler is None:
        import joblib
        import os
        model_path = "models/surrogate_model.pkl"
        scaler_path = "models/surrogate_scaler.pkl"
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                "Surrogate model/scaler not found in models/. Run surrogate_model.py first."
            )
        _cached_surrogate = joblib.load(model_path)
        _cached_scaler = joblib.load(scaler_path)
    return _cached_surrogate, _cached_scaler


def evaluate_population_surrogate(population, surrogate_model, scaler):
    """
    Extracts features for all individuals and batch-predicts fitness.
    """
    from surrogate_data_generator import extract_sequence_features
    import numpy as np

    feat_matrix = np.array([extract_sequence_features(ind) for ind in population], dtype=np.float32)
    feat_scaled = scaler.transform(feat_matrix)
    pred_costs = surrogate_model.predict(feat_scaled)
    return pred_costs


def tournament_selection_surrogate_scores(population, scores, tournament_size=3):
    selected = []
    n = len(population)
    for _ in range(n // 2):
        indices = random.sample(range(n), tournament_size)
        best_idx = min(indices, key=lambda idx: scores[idx])
        selected.append(copy.deepcopy(population[best_idx]))
    return selected


def genetic_algorithm_surrogate(
    ops,
    surrogate_model=None,
    scaler=None,
    generations=1500,
    pop_size=20,
    true_eval_interval=50,
    true_eval_top_k=3,
    track_convergence=False,
):
    """
    Surrogate-accelerated GA:
    1. Warm-starts with full-strength Hill Climbing (1000 iter).
    2. Runs fast generations using microsecond surrogate evaluations.
    3. Maintains elitism so the best found candidate is never lost.
    4. Periodically validates top candidates with true combined_cost().
    5. Guarantees true cost validation at output.
    """
    if surrogate_model is None or scaler is None:
        surrogate_model, scaler = get_default_surrogate()

    # 1. Full-strength Hill Climbing warm-start (1000 iter)
    hc_best, hc_cost = hill_climbing(ops)

    population = [copy.deepcopy(hc_best)]
    for _ in range(pop_size - 1):
        ind = copy.deepcopy(ops)
        random.shuffle(ind)
        population.append(ind)

    best_sequence = copy.deepcopy(hc_best)
    best_true_cost = hc_cost

    convergence_history = [best_true_cost]

    # Pre-evaluate initial population with surrogate
    surrogate_scores = evaluate_population_surrogate(population, surrogate_model, scaler)

    for gen in range(1, generations + 1):
        # Selection using fast surrogate scores
        parents = tournament_selection_surrogate_scores(population, surrogate_scores)

        # Elitism: carry over the best found individual
        next_gen = [copy.deepcopy(best_sequence)]
        for p in parents[: (pop_size // 2) - 1]:
            next_gen.append(copy.deepcopy(p))

        while len(next_gen) < pop_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            next_gen.append(child)

        population = next_gen
        surrogate_scores = evaluate_population_surrogate(population, surrogate_model, scaler)

        # Periodic True Cost Validation & Re-calibration
        if gen % true_eval_interval == 0 or gen == generations:
            ranked_indices = sorted(range(pop_size), key=lambda i: surrogate_scores[i])
            top_k_indices = ranked_indices[:true_eval_top_k]

            for idx in top_k_indices:
                cand = population[idx]
                c_true = combined_cost(cand)
                if c_true < best_true_cost:
                    best_true_cost = c_true
                    best_sequence = copy.deepcopy(cand)

        convergence_history.append(best_true_cost)

    final_true_cost = combined_cost(best_sequence)

    if track_convergence:
        return best_sequence, final_true_cost, convergence_history
    else:
        return best_sequence, final_true_cost


if __name__ == "__main__":
    ops = generate_problem(num_parts=5, pool_num=1, layout_num=1, seed=42)

    baseline_oct = calculate_oct(ops)
    baseline_lb = calculate_load_balance(ops)
    baseline_cost = combined_cost(ops)

    best_seq, best_cost = genetic_algorithm(ops)

    print(f"--- BASELINE ---")
    print(f"OCT:          {baseline_oct}")
    print(f"Load Balance: {baseline_lb}")
    print(f"Combined:     {baseline_cost}")

    print(f"\n--- AFTER GHCA ---")
    print(f"OCT:          {calculate_oct(best_seq)}")
    print(f"Load Balance: {calculate_load_balance(best_seq)}")
    print(f"Combined:     {round(best_cost, 2)}")

    print(f"\nImprovement:  {round(baseline_cost - best_cost, 2)}")