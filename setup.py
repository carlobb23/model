import random


def generate_dicts(P_Nr, D_Nr, T_Nr, patient_factor, Max_P):
    P, T, D = list(range(1, P_Nr + 1)), list(range(1, T_Nr + 1)), list(range(1, D_Nr + 1))

    # Even distribution of entry dates
    entries_per_day = P_Nr // D_Nr
    remaining_entries = P_Nr % D_Nr

    # Create list of entry dates
    entry_dates = []
    for d in D:
        # Add the base number of entries for this day
        entry_dates.extend([d] * entries_per_day)
        # Add one more if we still have remaining entries to distribute
        if remaining_entries > 0:
            entry_dates.append(d)
            remaining_entries -= 1

    # Shuffle the entry dates and assign to patients
    random.shuffle(entry_dates)
    Entry_p = {p: entry_dates[i - 1] for i, p in enumerate(P, 1)}

    # Generate R_p with Max_P limit
    R_p = {}
    for p in P:
        max_rp = min(Max_P, len(D) - Entry_p[p])  # Use Min to ensure R_p doesn't exceed Max_P
        R_p[p] = random.randint(1, max(1, max_rp))  # Ensure minimum of 1

    total_needed_hours = len(T) * patient_factor
    values = [i for i in range(1, 5)] * (len(T) // 4) + [i for i in range(1, len(T) % 4 + 1)]
    random.shuffle(values)
    random_values_for_t = {t: (values[i], random.randint(1, 5)) for i, t in enumerate(T)}

    max_td = {
        (t, d): (0 if d % 7 in (v[1], (v[1] + 1) % 7) else v[0])
        for t, v in random_values_for_t.items()
        for d in D
    }

    w = {}
    for p, entry_day in Entry_p.items():
        for d in range(1, len(D) + 1):
            w[(p, d)] = 1 if d >= entry_day else 0

    W, W_min = 5, 3
    E_app, E_min = 0.5, 0.2

    return R_p, Entry_p, max_td, P, D, T, w, W, W_min, E_app, E_min

def generate_dicts2(P_Nr, D_Nr, T_Nr, patient_factor):
    P = list(range(1, P_Nr + 1))
    T = ['t0'] + [f't{i}' for i in range(1, T_Nr + 1)]  # Now includes t0 and t1 through tN
    D = list(range(1, D_Nr + 1))

    Entry_p = {p: random.randint(1, len(D)) for p in P}
    R_p = {}
    for p in P:
        max_rp = len(D) - Entry_p[p]
        R_p[p] = random.randint(1, max(1, max_rp))

    total_needed_hours = len(T[1:]) * patient_factor  # Exclude t0 from calculation
    Max_t = {f't{t}': random.randint(1, 3) for t in range(1, T_Nr + 1)}  # Initialize t1 through tN

    current_total_hours = sum(Max_t.values())
    scaling_factor = total_needed_hours / current_total_hours

    # Scale the values for t1 through tN
    Max_t = {t: int(round(Max_t[t] * scaling_factor)) for t in Max_t}
    # Add t0 with infinity
    Max_t['t0'] = float('inf')

    w = {(p, d): 1 if d >= entry_day else 0
         for p, entry_day in Entry_p.items()
         for d in range(1, len(D) + 1)}

    W, W_min = 5, 3
    E_app, E_min = 0.5, 0.2

    return R_p, Entry_p, Max_t, P, D, T, w, W, W_min, E_app, E_min

def plan_dict(start_values, patient, therapist, day):
    dict = {}
    index = 1
    if therapist is None and day is None:
        dict[f"Patient_{index}"] = [{(p): start_values[(p)] for p in patient}]
    elif day is None:
        dict[f"Patient_{index}"] = [{(p, t): start_values[(p, t)] for p in patient for t in therapist}]
    elif therapist is None:
        dict[f"Patient_{index}"] = [{(p, d): start_values[(p, d)] for p in patient for d in day}]
    else:
        dict[f"Patient_{index}"] = [
            {(p, t, d): start_values[(p, t, d)] for  p in patient for t in therapist for d in day}]
    return dict

def get_unique_combinations_and_list_with_dicts(R_p, Entry_p):
    combined = [(R_p[p], Entry_p[p]) for p in R_p.keys()]

    unique_combinations = {}
    for p, combo in enumerate(combined, start=1):
        if combo not in unique_combinations:
            unique_combinations[combo] = []
        unique_combinations[combo].append(p)

    num_unique_combinations = len(unique_combinations)

    N_c = list(range(1, num_unique_combinations + 1))

    R_p_c = {}
    Entry_p_c = {}
    Profile_patient_count = {}
    for idx, (combo, patients) in enumerate(unique_combinations.items(), start=1):
        R_p_c[idx] = combo[0]
        Entry_p_c[idx] = combo[1]
        Profile_patient_count[idx] = len(patients)

    return N_c, R_p_c, Entry_p_c, Profile_patient_count, unique_combinations, num_unique_combinations


def adjust_start_values(start_x, start_LOS, unique_combinations):
    values_to_remove = []
    for combo in unique_combinations.values():
        if len(combo) > 1:
            values_to_remove.extend(combo[1:])

    mapping = {}
    new_p = 1
    max_p = max(max(p for (p, _, _) in start_x.keys()), max(start_LOS.keys()))

    for p in range(1, max_p + 1):
        if p not in values_to_remove:
            mapping[p] = new_p
            new_p += 1

    new_start_x = {}
    for (p, t, d), value in start_x.items():
        if p not in values_to_remove:
            new_key = (mapping[p], t, d)
            new_start_x[new_key] = value

    new_start_LOS = {}
    for p, value in start_LOS.items():
        if p not in values_to_remove:
            new_key = mapping[p]
            new_start_LOS[new_key] = value

    return new_start_x, new_start_LOS