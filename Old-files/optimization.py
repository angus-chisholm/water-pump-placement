from gurobipy import Model, GRB
import pandas as pd
import numpy as np
from utils import read_data, calculate_distance

def optimize_water_sources(data_file, potential_locations, max_distance, cost_borehole, cost_standpipe, cost_per_meter):
    # Load data
    data = read_data(data_file)
    
    # Filter households and water sources
    households = data[data['Type'] == 'Household']
    existing_water_sources = data[data['Type'] == 'Hand Pump']

    # Create a model
    model = Model("Water Source Optimization")

    # Decision variables: whether to place a borehole or standpipe at a given location
    borehole_vars = model.addVars(potential_locations.shape[0], vtype=GRB.BINARY, name="Borehole")
    standpipe_vars = model.addVars(potential_locations.shape[0], vtype=GRB.BINARY, name="Standpipe")

    # Define costs
    costs = (
        cost_borehole * borehole_vars.sum() + 
        cost_standpipe * standpipe_vars.sum() + 
        cost_per_meter * sum(
            standpipe_vars[w_index] * min(
                calculate_distance(potential_locations.loc[w_index, 'Lat'], potential_locations.loc[w_index, 'Lon'], 
                                   existing_water_sources.loc[other_index, 'Lat'], existing_water_sources.loc[other_index, 'Lon'])
                for other_index in existing_water_sources.index
            )
            for w_index in potential_locations.index
        )
    )

    # Constraints: At least one water source must be selected
    model.addConstr(
        borehole_vars.sum() + standpipe_vars.sum() >= 1,
        name="At_Least_One_Source"
    )

    # Trouver distance a la pompe la plus proche
    distances = np.zeros((len(households),len(existing_water_sources)+len(potential_locations)))
    for h_index, household in households.iterrows():
        for w_index in range(potential_locations.shape[0]):
            if borehole_vars[w_index] > 0.5 or standpipe_vars[w_index] > 0.5:
                distances[h_index][w_index] = calculate_distance(household['Lat'], household['Lon'], potential_locations.loc[w_index, 'Lat'], potential_locations.loc[w_index, 'Lon'])
            else:
                distances[h_index][w_index] = np.inf
        for e_index in existing_water_sources.index:
            distances[h_index][e_index + len(potential_locations)] = calculate_distance(household['Lat'], household['Lon'], existing_water_sources.loc[e_index, 'Lat'], existing_water_sources.loc[e_index, 'Lon'])      

    distances = np.min(distances, axis=1)

    impact = sum(household['Nb capita'] * (distances[h_index] <= max_distance)
        for h_index, household in households.iterrows()
        )

    function = []
    optimised_impact = []
    optimised_costs = []

    # Optimize the model whilst increasing the minimum impact
    while True:
        model.optimize()
        
        # Retrieve and print solution
        if model.status == GRB.OPTIMAL:
            # Trouver distance a la pompe la plus proche
            distances = np.zeros((len(households),len(existing_water_sources)+len(potential_locations)))
            for h_index, household in households.iterrows():
                for w_index in range(potential_locations.shape[0]):
                    if borehole_vars[w_index].x > 0.5 or standpipe_vars[w_index].x > 0.5:
                        distances[h_index][w_index] = calculate_distance(household['Lat'], household['Lon'], potential_locations.loc[w_index, 'Lat'], potential_locations.loc[w_index, 'Lon'])
                    else:
                        distances[h_index][w_index] = np.inf
                for e_index in existing_water_sources.index:
                    distances[h_index][e_index + len(potential_locations)] = calculate_distance(household['Lat'], household['Lon'], existing_water_sources.loc[e_index, 'Lat'], existing_water_sources.loc[e_index, 'Lon'])      

            distances = np.min(distances, axis=1)

            impact_min = sum(household['Nb capita'] * (distances[h_index] <= max_distance)
                for h_index, household in households.iterrows()
                )

            optimal_sources = [(w_index, 'Borehole') for w_index in potential_locations.index if borehole_vars[w_index].X > 0.5] + \
                              [(w_index, 'Standpipe') for w_index in potential_locations.index if standpipe_vars[w_index].X > 0.5]
            optimised_impact.append(impact)
            optimised_costs.append(costs.getValue())
            function.append(optimal_sources)
            model.addConstr(impact >= impact_min, name = 'impact limits')
        else:
            print("No more optimal solutions found.")
            break
        
    return function, optimised_impact, optimised_costs