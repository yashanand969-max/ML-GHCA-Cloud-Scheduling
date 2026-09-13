# ============================================================
# HILL_CLIMBING.PY
# Uses combined cost function (OCT + load balance)
# ============================================================

import random
import copy
from scheduler import generate_problem, calculate_oct, calculate_load_balance, combined_cost


def hill_climbing(ops, iterations=1000):
    current = copy.deepcopy(ops)
    current_cost = combined_cost(current)

    for _ in range(iterations):
        neighbor = copy.deepcopy(current)

        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        neighbor_cost = combined_cost(neighbor)

        if neighbor_cost < current_cost:
            current = neighbor
            current_cost = neighbor_cost

    return current, current_cost


if __name__ == "__main__":
    ops = generate_problem(num_parts=5, pool_num=1, layout_num=1, seed=42)

    baseline_oct = calculate_oct(ops)
    baseline_lb = calculate_load_balance(ops)
    baseline_cost = combined_cost(ops)

    best_seq, best_cost = hill_climbing(ops)

    print(f"--- BASELINE ---")
    print(f"OCT:          {baseline_oct}")
    print(f"Load Balance: {baseline_lb}")
    print(f"Combined:     {baseline_cost}")

    print(f"\n--- AFTER HC ---")
    print(f"OCT:          {calculate_oct(best_seq)}")
    print(f"Load Balance: {calculate_load_balance(best_seq)}")
    print(f"Combined:     {round(best_cost, 2)}")

    print(f"\nImprovement:  {round(baseline_cost - best_cost, 2)}")