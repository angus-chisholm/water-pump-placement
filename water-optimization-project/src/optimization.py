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


    # Define impact as the number of people served within the max distance of a water source
    impact = sum(household['Nb capita'] 
                   for h_index, household in households.iterrows() 
                   if min(
                        calculate_distance(water_source['Lat'], water_source['Lon'], 
                                            household['Lat'], household['Lon'])
                            for w_index, water_source in water_sources.iterrows()
                        ) <= max_distance
                    )

    
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


    minimised_function = []
    optimised_impact = []
    optimised_costs = []

    # Define a weighting factor to balance cost minimization and impact maximization
    impact_min = np.linspace(0,2000,21)

    for impact_level in impact_min:
        model.addConstr(
            impact >= impact_level,
            name="Impact_Threshold"
        )

        model.setObjective(
            costs,
            GRB.MINIMIZE
        )

        model.update()

        model.optimize()

        if model.status == GRB.OPTIMAL:
            optimal_boreholes = [w_index for w_index in potential_locations.index if borehole_vars[w_index].X > 0.5]
            optimal_standpipes = [w_index for w_index in potential_locations.index if standpipe_vars[w_index].X > 0.5]

            minimised_function.append((optimal_boreholes, optimal_standpipes))
            optimised_impact.append(impact())
            optimised_costs.append(costs.getValue())

        else:
            print("Optimization was not successful. Status:", model.status)
            return [], [], {}
        
    return minimised_function, optimised_impact, optimised_costs