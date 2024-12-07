from masterproblem import *
from setup import *
from subproblem import *
import pandas as pd
import numpy as np
from compactmodel import *

random.seed(22236)
R_p, Entry_p, Max_t, P, D, T, w, W, W_min, E_app, E_min = generate_dicts(27, 14, 9, 4, 4)
N_c, R_p_c, Entry_p_c, Nr_c, unique_combinations, num_unique_combinations = get_unique_combinations_and_list_with_dicts(R_p, Entry_p)

# **** Prerequisites ****
# Create Dataframes
data = pd.DataFrame({
    'P': N_c + [np.nan] * (max(len(N_c), len(T), len(D)) - len(N_c)),
    'T': T + [np.nan] * (max(len(N_c), len(T), len(D)) - len(T)),
    'D': D + [np.nan] * (max(len(N_c), len(T), len(D)) - len(D))
})

data_comp = pd.DataFrame({
    'P': P + [np.nan] * (max(len(P), len(T), len(D)) - len(P)),
    'T': T + [np.nan] * (max(len(P), len(T), len(D)) - len(T)),
    'D': D + [np.nan] * (max(len(P), len(T), len(D)) - len(D))
})

# **** Column Generation ****
# Prerequisites
modelImprovable = True
reached_max_itr = False
time_Limit = 1800
max_itr = 22
threshold = 1e-7

# Get Starting Solutions
problem_start = Problem(data_comp, W, W_min, R_p, Entry_p, Max_t, E_app, E_min)
problem_start.buildModel()
problem_start.solveStart()

# Create
start_x = {(p, t, d): problem_start.x[p, t, d].x for p in P for t in T for d in D}
start_LOS = {(p): problem_start.LOS[p].x for p in P}

while True:
    # Initialize iterations
    itr = 0
    last_itr = 0

    # Mapping
    start_x_c, start_LOS_c = adjust_start_values(start_x, start_LOS, unique_combinations)

    master = MasterProblem(data, max_itr, itr, Max_t, Nr_c)
    master.buildModel()
    master.initCoeffs()
    master.startSol(start_x_c, start_LOS_c)

    # Initialize and solve relaxed model
    master.solRelModel()

    # Retrieve dual values
    duals_td0, duals_p0 = master.getDuals()

    while (modelImprovable) and itr < max_itr:
        # Start
        itr += 1

        # Solve RMP
        master.current_iteration = itr + 1
        master.solRelModel()
        current_obj = master.Model.objval
        current_bound = master.Model.objval

        # Get and Print Duals
        duals_td, duals_p = master.getDuals()

        # Solve SPs
        modelImprovable = False
        for index in N_c:

            # Build SP
            subproblem = Subproblem(duals_p, duals_td, data, index, itr, R_p_c, Entry_p_c, W, W_min, E_app, E_min)
            subproblem.buildModel()

            # Save time to solve SP
            subproblem.solModel()
            reducedCost = subproblem.Model.objval

            # Increase latest used iteration
            last_itr = itr + 1

            # Generate and add columns with reduced cost
            if reducedCost < -threshold:
                Schedules_x = subproblem.getOptVals('x')
                Schedules_LOS = subproblem.getOptVals('LOS')
                master.addCol(index, itr, Schedules_x, Schedules_LOS)
                modelImprovable = True

            master.Model.update()

    if not modelImprovable:
        break

    if modelImprovable and itr == max_itr:
        break

        # Update Model
        master.Model.update()

        if not modelImprovable:
            break

    if modelImprovable and itr == max_itr:
        break

master.finSol()