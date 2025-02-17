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

    # Combine existing water sources with potential new locations
    water_sources = pd.concat([existing_water_sources, potential_locations], ignore_index=True)

    # Create a model
    model = Model("Water Source Optimization")

    # Decision variables: whether to place a borehole or standpipe at a given location
    borehole_vars = model.addVars(potential_locations.index, vtype=GRB.BINARY, name="Borehole")
    standpipe_vars = model.addVars(potential_locations.index, vtype=GRB.BINARY, name="Standpipe")

    # Decision variables: number of people served by each water source
    served_vars = model.addVars(households.index, vtype=GRB.CONTINUOUS, name="Served")

    # Define impact as the number of people served within the max distance of a water source
    def impact():
        return sum(household['Nb capita'] 
                   for h_index, household in households.iterrows() 
                   if min(
                        calculate_distance(water_source['Lat'], water_source['Lon'], 
                                            household['Lat'], household['Lon'])
                            for w_index, water_source in water_sources.iterrows()
                        ) <= max_distance
                    )
    

    # Define a weighting factor to balance cost minimization and people served maximization
    weight_cost = 1.0
    weight_served = 1.0

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

    # Objective: Minimize costs while maximizing the number of people served
    model.setObjective(
        weight_cost * costs - weight_served * impact(),
        GRB.MINIMIZE
    )

    # Constraints: At least one water source must be selected
    model.addConstr(
        borehole_vars.sum() + standpipe_vars.sum() >= 1,
        name="At_Least_One_Source"
    )

    # # Constraints: Impact above a certain threshold
    # model.addConstr(
    #     impact() >= 1000,
    #     name="Impact_Threshold"
    # )

    model.update()

    # Solve the model
    model.optimize()

    # Check if the model was solved successfully
    if model.status == GRB.OPTIMAL:
        # Collect results
        optimal_boreholes = [w_index for w_index in potential_locations.index if borehole_vars[w_index].X > 0.5]
        optimal_standpipes = [w_index for w_index in potential_locations.index if standpipe_vars[w_index].X > 0.5]

        return optimal_boreholes, optimal_standpipes, impact(), costs.getValue()
    else:
        print("Optimization was not successful. Status:", model.status)
        return [], [], {}