from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np
from utils import read_data, calculate_distance_sq, calculate_distance


data_file = "C:/Users/angus/OneDrive/Documents/France/PVWPS Project/water-pump-placement/water-optimization-project/src/Map_village_20241227_data.csv"
max_distance = 600

data = read_data(data_file)

# Filter households and water sources
households = data[data['Type'] == 'Household']
existing_water_sources = data[data['Type'] == 'Hand Pump']

model = Model("Standpipe Optim")

# Add decision variables
X = model.addVars(2,vtype = GRB.CONTINUOUS, name = 'coords')

# Add auxiliary variables
V = model.addVars(households.shape[0],vtype = GRB.BINARY, name = 'within max distance')
W = model.addVars(households.shape[0],vtype = GRB.CONTINUOUS, name = 'min distance squared')

nbcapita = households['Nb capita'].to_numpy()


# Add constraints
for h_index, household in households.iterrows():
    for e_index, source in existing_water_sources.iterrows():
        model.addConstr(W[h_index] <= calculate_distance_sq(household['Lat'],household['Lon'],
                                                            source['Lat'],source['Lon']))
    model.addConstr(W[h_index] <= calculate_distance_sq(household['Lat'],household['Lon'],
                                                        X[0],X[1]))    

# Constants
# M is chosen to be as small as possible given the bounds on x and y
eps = 0.0001
M = 100 + eps

# If max_dist > dist to pump, then V = 1, otherwise V = 0
for j in range(households.shape[0]):
    model.addConstr(max_distance**2 >= W[j] + eps - M * (1 - V[j]), name="bigM_constr1")
    model.addConstr(max_distance**2 <= W[j] + M * V[j], name="bigM_constr2")
    
    
bounds = np.array([
    [data['Lat'].min(), data['Lat'].max()],  # Latitude bounds
    [data['Lon'].min(), data['Lon'].max()],  # Longitude bounds
])

model.addConstr(X[0] <= bounds[0,1], name='max_bound_lat')
model.addConstr(X[0] >= bounds[0,0], name='min_bound_lat')
model.addConstr(X[0] <= bounds[1,1], name='max_bound_lon')
model.addConstr(X[0] >= bounds[1,0], name='min_bound_lon')

model.setObjective(quicksum([nbcapita[j]*V[j] for j in range(len(V))]), GRB.MINIMIZE)

model.optimize()

if model.status == GRB.OPTIMAL:
    print('Optimum pos = {}').format(X[0].x,X[1].x)
else:
    print("No more optimal solutions found.")
    
