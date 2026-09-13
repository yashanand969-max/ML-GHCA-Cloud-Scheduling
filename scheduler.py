# ============================================================
# SCHEDULER.PY
# Fixed version — sequence order now actually matters
# ============================================================

import random
import copy

MACHINES = ["M1", "M2", "M3", "M4"]
VEHICLES = ["V1", "V2"]

PART_POOL_1 = [
    [4, 2, 3], [3, 1, 4], [2, 4, 3], [2, 1, 3], [3, 4, 2, 1], 
    [4, 2, 1], [1, 2, 4], [3, 2, 4], [1, 3, 2], [1, 2, 3, 4], [4, 3, 1]
]

PART_POOL_2 = [
    [3, 4], [1, 2, 3, 4], [1, 2, 4], [1, 2], [1, 3, 4], 
    [1, 3], [1, 2, 3], [2, 4], [1, 4], [2, 3, 4]
]

LAYOUTS = {
    1: [
        [0.0, 0.6, 0.8, 1.0, 1.2],
        [0.6, 0.0, 0.6, 0.8, 1.0],
        [0.8, 0.6, 0.0, 0.6, 0.8],
        [1.0, 0.8, 0.6, 0.0, 0.6],
        [1.2, 1.0, 0.8, 0.6, 0.0]
    ],
    2: [
        [0.0, 1.0, 1.2, 1.4, 1.6],
        [1.0, 0.0, 1.0, 1.2, 1.4],
        [1.2, 1.0, 0.0, 1.0, 1.2],
        [1.4, 1.2, 1.0, 0.0, 1.0],
        [1.6, 1.4, 1.2, 1.0, 0.0]
    ],
    3: [
        [0.0, 1.4, 1.6, 1.8, 2.0],
        [1.4, 0.0, 1.4, 1.6, 1.8],
        [1.6, 1.4, 0.0, 1.4, 1.6],
        [1.8, 1.6, 1.4, 0.0, 1.4],
        [2.0, 1.8, 1.6, 1.4, 0.0]
    ]
}

def generate_problem(num_parts=10, pool_num=1, layout_num=1, seed=None):
    if seed is not None:
        random.seed(seed)

    pool = PART_POOL_1 if pool_num == 1 else PART_POOL_2
    layout = LAYOUTS[layout_num]
    
    non_zero_travels = [t for row in layout for t in row if t > 0]
    t_bar = sum(non_zero_travels) / len(non_zero_travels) if non_zero_travels else 0

    while True:
        operations = []
        op_id = 1
        processing_times = []

        for part_idx in range(num_parts):
            job_name = f"J{part_idx + 1}"
            routing = random.choice(pool)
            prev_node = 0  # L/U station is index 0
            
            for m_num in routing:
                curr_node = m_num
                travel_time = layout[prev_node][curr_node]
                process_time = random.randint(3, 15)
                processing_times.append(process_time)
                
                op = {
                    "id": f"O{op_id}",
                    "job": job_name,
                    "machine": f"M{curr_node}",
                    "vehicle": VEHICLES[op_id % len(VEHICLES)],
                    "travel_time": travel_time,
                    "processing_time": process_time,
                }
                operations.append(op)
                prev_node = curr_node
                op_id += 1
                
        p_bar = sum(processing_times) / len(processing_times) if processing_times else 1
        ratio = t_bar / p_bar
        if 0.05 <= ratio <= 0.20:
            break

    return operations


def calculate_oct(ops):
    machine_free_at = {}
    job_free_at = {}
    vehicle_free_at = {}
    last_finish = 0

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
        last_finish = max(last_finish, finish_time)

    return last_finish


def calculate_load_balance(ops):
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

    # load balance = std dev of machine finish times
    # machines not used get finish time 0
    all_times = [machine_free_at.get(m, 0) for m in MACHINES]
    mean = sum(all_times) / len(all_times)
    variance = sum((t - mean) ** 2 for t in all_times) / len(all_times)
    std_dev = variance ** 0.5

    return round(std_dev, 2)


def combined_cost(ops, w1=0.7, w2=0.3):
    oct_val = calculate_oct(ops)
    lb_val = calculate_load_balance(ops)
    return round(w1 * oct_val + w2 * lb_val, 2)


if __name__ == "__main__":
    ops = generate_problem(num_parts=5, pool_num=1, layout_num=1, seed=42)
    print(f"Generated {len(ops)} operations")
    print(f"OCT:          {calculate_oct(ops)}")
    print(f"Load Balance: {calculate_load_balance(ops)}")
    print(f"Combined:     {combined_cost(ops)}")

    # verify that different orderings give different OCT
    # this confirms sequence order actually matters now
    import copy
    shuffled = copy.deepcopy(ops)
    random.shuffle(shuffled)
    print(f"\nShuffled OCT:          {calculate_oct(shuffled)}")
    print(f"Shuffled Load Balance: {calculate_load_balance(shuffled)}")
    print(f"Shuffled Combined:     {combined_cost(shuffled)}")
    print(f"\nAre they different? {combined_cost(ops) != combined_cost(shuffled)}")