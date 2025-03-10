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

X = model.addVars(2,vtype = GRB.CONTINUOUS,name = 'position')
z = model.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'min dist')

bounds = np.array([
    [data['Lat'].min(), data['Lat'].max()],  # Latitude bounds
    [data['Lon'].min(), data['Lon'].max()],  # Longitude bounds
])

model.addConstr(X[1] <= bounds[0,1], name='max_bound_lat')
model.addConstr(X[1] >= bounds[0,0], name='min_bound_lat')
model.addConstr(X[0] <= bounds[1,1], name='max_bound_lon')
model.addConstr(X[0] >= bounds[1,0], name='min_bound_lon')

for i in range(existing_water_sources.shape[0]):
    model.addConstr(z <= calculate_distance_sq(X[0], X[1], existing_water_sources.iloc[i]['Lat'], existing_water_sources.iloc[i]['Lon']))
model.setObjective(z,GRB.MINIMIZE)

model.update()
model.optimize()

model.computeIIS()
model.write("infeasible_constraints.ilp")

if model.status == GRB.INFEASIBLE:
    print("Le modèle n'a pas de solution")
elif model.status == GRB.UNBOUNDED:
    print("Le modèle est non borné")
else:
    print("Min distance:", z.X)