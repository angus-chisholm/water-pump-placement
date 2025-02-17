from gurobipy import Model, GRB
import pandas as pd
import numpy as np
from utils import read_data, calculate_distance

def optimize_water_sources(data_file, max_distance, cost_borehole, cost_standpipe, cost_per_meter):
    # Load data
    data = read_data(data_file)
    
    # Filter households and water sources
    households = data[data['Type'] == 'Household']
    water_sources = data[data['Type'] == 'Hand Pump']

    # Create a model
    model = Model("Water Source Optimization")

    # Decision variables: whether to place a borehole or standpipe at a given location
    borehole_vars = model.addVars(water_sources.index, vtype=GRB.BINARY, name="Borehole")
    standpipe_vars = model.addVars(water_sources.index, vtype=GRB.BINARY, name="Standpipe")

    # Decision variables: number of people served by each water source
    served_vars = model.addVars(households.index, vtype=GRB.CONTINUOUS, name="Served")

    # Objective: Minimize costs while maximizing the number of people served
    model.setObjective(
        cost_borehole * borehole_vars.sum() + cost_standpipe * standpipe_vars.sum() + 
        cost_per_meter * sum(
            standpipe_vars[w_index] * min(
                calculate_distance(water_sources.loc[w_index, 'Lat'], water_sources.loc[w_index, 'Lon'], 
                                   water_sources.loc[other_index, 'Lat'], water_sources.loc[other_index, 'Lon'])
                for other_index in water_sources.index if other_index != w_index
            )
            for w_index in water_sources.index
        ) - served_vars.sum(),
        GRB.MAXIMIZE
    )

    # Constraints: Each household can only be served by one water source within the max distance
    for h_index, household in households.iterrows():
        model.addConstr(
            served_vars[h_index] <= sum(
                (borehole_vars[w_index] + standpipe_vars[w_index]) 
                for w_index, water_source in water_sources.iterrows()
                if calculate_distance(household['Lat'], household['Lon'], water_source['Lat'], water_source['Lon']) <= max_distance
            ),
            name=f"Serve_Household_{h_index}"
        )

    # Constraints: A standpipe must be connected to a borehole
    for w_index in water_sources.index:
        model.addConstr(
            standpipe_vars[w_index] <= sum(
                borehole_vars[other_index] 
                for other_index in water_sources.index if other_index != w_index
            ),
            name=f"Standpipe_Connection_{w_index}"
        )

    # Solve the model
    model.optimize()

    # Collect results
    optimal_boreholes = [w_index for w_index in water_sources.index if borehole_vars[w_index].X > 0.5]
    optimal_standpipes = [w_index for w_index in water_sources.index if standpipe_vars[w_index].X > 0.5]
    served_people = {h_index: served_vars[h_index].X for h_index in households.index}

    return optimal_boreholes, optimal_standpipes, served_people