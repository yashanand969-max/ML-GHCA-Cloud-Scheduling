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