import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum
from utils import read_data


#optimal_sources, impact, costs = optimize_water_sources(
#        data_file, potential_locations, max_distance=800, cost_borehole=5000, cost_standpipe=500, cost_per_meter=2
#    )

def optimise_water_sources(data_file, max_distance, cost_borehole, cost_standpipe, cost_pipe):
    '''Given a data file and costs, return collection of optimal sources, impact and costs for eeach of these sources'''

    #Read data file and return pandas dataframes of data, households and existing pumps
    data,households,pumps = read_data(data_file)

    pos_pumps = pumps[['Lon','Lat']].to_numpy()

    initial_pos = [data['Lon'].mean(), data['Lat'].min()]

    bounds = np.array([
        [data['Lon'].min(), data['Lon'].max()],  # Longitude bounds
        [data['Lat'].min(), data['Lat'].max()],  # Latitude bounds
    ])

    #Initialise model
    model = Model('Pump optimisation')

    #Decision Variables
    X = model.addMVar(shape=2,vtype = GRB.CONTINUOUS, name = 'Position') #X[0] = x (longitude), X[1] = y (latitude)

    #Auxiliary Variables
    D = model.addVar(name = 'Min distance to pump')
    xdiff = model.addVars(len(pos_pumps),lb=-GRB.INFINITY,name = 'xdiff')
    ydiff = model.addVars(len(pos_pumps),lb=-GRB.INFINITY,name = 'ydiff')

    #Constraints
    #model.addConstr(X[0] >= bounds[0,0],name = 'longitude minimum')
    #model.addConstr(X[0] <= bounds[0,1],name = 'longitude maximum')
    #model.addConstr(X[1] >= bounds[1,0],name = 'latitude minimum')
    #model.addConstr(X[1] <= bounds[1,1],name = 'latitude maximum')

    #Auxiliary Constraints
    for i,pos in enumerate(pos_pumps):
        model.addConstr(xdiff[i] == X[0]-pos[0],name=f"define_xdiff[{i}]")
        model.addConstr(ydiff[i] == X[1]-pos[1],name=f"define_ydiff[{i}]")
        model.addConstr(D ** 2 <= xdiff[i] ** 2 + ydiff[i] ** 2, name=f"D**2 <= dist{i}**2")
        

    #Objective
    #minimise cost (distance to nearest pump)
    #maximise impact (minimise negative impact) (distance to all households * nb capita)
    model.setObjective(-D,GRB.MAXIMIZE)

    for i,v in enumerate(model.getVars()):
        v.Start = initial_pos[i]

    model.update()
    model.optimize()
    model.params.outputflag = 0
    model.write('model.lp')

    if model.status == GRB.OPTIMAL:
        print(f'The optimial position is x = {X.x[0]}, y = {X.x[1]}; bounds are {bounds}; xdiff = {xdiff[0].x}, ydiff = {ydiff[0].x}')
        return [(X.x[0],X.x[1])], [0], [model.ObjVal]
    elif model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write('infeasible_model.ilp')
        




# Test function
file = r'water-optimization-project/data/Map_village_20241227_data.csv'
optimise_water_sources(file,max_distance=800, cost_borehole=5000, cost_standpipe=500, cost_pipe=2)
