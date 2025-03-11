from gurobipy import Model, GRB
import pandas as pd
import numpy as np
from utils import read_data, calculate_distance_sq, calculate_distance

def optimize_water_sources(data_file, max_distance, cost_borehole, cost_standpipe, cost_per_meter):
    # Load data
    data = read_data(data_file)
    
    # Filter households and water sources
    households = data[data['Type'] == 'Household']
    existing_water_sources = data[data['Type'] == 'Hand Pump']

    # Create a model
    model = Model("Water Source Optimization")
    
    # Add decision variable - position
    S = model.addMVar(shape=(2), vtype=GRB.CONTINUOUS, name='Position of standpipe S1')  # [0] = Lat, [1] = Lon

    # Define Cost
    cost_expr = cost_borehole + cost_standpipe + cost_per_meter * model.addVar(vtype=GRB.CONTINUOUS, name='DistanceCost')
    model.setObjective(cost_expr, GRB.MINIMIZE)

    # Define Impact
    distances = np.zeros((len(households), len(existing_water_sources) + 1))
    for h_index, household in households.iterrows():
        for e_index in existing_water_sources.index:
            distances[h_index][e_index] = calculate_distance_sq(household['Lat'], household['Lon'], existing_water_sources.loc[e_index, 'Lat'], existing_water_sources.loc[e_index, 'Lon'])
        distances[h_index][-1] = calculate_distance_sq(household['Lat'], household['Lon'], S[0], S[1])

    min_distances_old = np.min(distances[:, :-1], axis=1)
    min_distances_new = np.min(distances, axis=1)

    impact_expr = sum(household['Nb capita'] * (min_distances_new[h_index] <= max_distance**2) for h_index, household in households.iterrows()) - \
                  sum(household['Nb capita'] * (min_distances_old[h_index] <= max_distance**2) for h_index, household in households.iterrows())

    impact = model.addVar(vtype=GRB.CONTINUOUS, name='Impact')
    model.addConstr(impact == impact_expr, name='Impact_Calculation')

    # Add Constraints
    bounds = np.array([
        [data['Lon'].min(), data['Lon'].max()],  # Longitude bounds
        [data['Lat'].min(), data['Lat'].max()],  # Latitude bounds
    ])
    
    model.addConstr(S <= bounds[:, 1], name='max_bound')
    model.addConstr(S >= bounds[:, 0], name='min_bound')

    positions = []
    optimised_impact = []
    optimised_costs = []

    # Optimize the model whilst increasing the minimum impact
    while True:
        model.optimize()
        
        # Retrieve and print solution
        if model.status == GRB.OPTIMAL:
            cost = cost_borehole + cost_standpipe + cost_per_meter * min(
                calculate_distance(S.X[0], S.X[1], existing_water_sources.loc[other_index, 'Lat'], existing_water_sources.loc[other_index, 'Lon'])
                for other_index in existing_water_sources.index
            )
            
            impact_min = impact.X

            positions.append(S.X.tolist())
            optimised_impact.append(impact_min)
            optimised_costs.append(cost)
            model.addConstr(impact <= impact_min, name='impact_limits')
        else:
            print("No more optimal solutions found.")
            break
        
    return positions, optimised_impact, optimised_costs


