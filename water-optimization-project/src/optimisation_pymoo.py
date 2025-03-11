import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
import geopy.distance as geo
from utils import read_data

data_file = r'water-optimization-project/data/Map_village_20241227_data.csv'

data,households,pumps = read_data(data_file)

pos_households = households[['Lon','Lat']].to_numpy() # For f1 (sum of weighted distances)
nb_capita = households['Nb capita'].to_numpy() # Define weight factors for each distance in pos_households

pos_pumps = pumps[['Lon','Lat']].to_numpy() # For f2 (min distance)
cost_borehole=5000
cost_standpipe=500
cost_per_meter=2

def impact(pump_positions, household_positions, household_capita, x = None):
    pump_positions_copy = pump_positions.copy()
    if isinstance(x, np.ndarray):
        pump_positions_copy = np.concatenate((pump_positions_copy,np.array([x])),axis = 0)
    min_pump_distance = np.zeros(len(household_positions))
    for index,pos_houseshold in enumerate(household_positions):
        dist = np.zeros(len(pump_positions_copy))
        for i,pump_pos in enumerate(pump_positions_copy):
            dist[i] = geo.great_circle(pump_pos, household_positions[index]).meters
        min_pump_distance[index] = np.min(dist)

    impact = np.sum(household_capita * (min_pump_distance))
    return impact

initial_impact = impact(pos_pumps,pos_households,nb_capita)


bounds = np.array([
        [data['Lon'].min(), data['Lat'].min()],  # Min bounds
        [data['Lon'].max(), data['Lat'].max()],  # Max bounds
    ])
    

class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=1, xl=bounds[0], xu=bounds[1])  # 2D search space

    def _evaluate(self, x, out, *args, **kwargs):
        # Compute weighted sum of distances to points in pos_households
        f1 = impact(pos_pumps,pos_households,nb_capita,x)-initial_impact  # Negative impact calculation with new x location
        
        # Compute minimum distance to any point in pos_pumps
        f2_distances = np.zeros(len(pos_pumps))
        for index,pump in enumerate(pos_pumps):
            f2_distances[index] = geo.great_circle(pump, x).meters
        f2 = cost_borehole+cost_standpipe+cost_per_meter*np.min(f2_distances)  # Minimum distance to any point in pos_pumps

        # Define constraint: f1 >= c ## could change to be above initial impact!!
        c = initial_impact - 0.5
        g = f1-c

        out["F"] = [f1, f2]  # Objective functions
        out["G"] = [g]  # Constraints (must be <= 0)



def optimise_nsgaII():
    # Define the NSGA-II optimization algorithm
    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    # Solve the problem
    problem = MyProblem()
    res = minimize(problem,
                algorithm,
                ('n_gen', 10),
                #termination=DefaultMultiObjectiveTermination(),
                seed=1,
                verbose=True)

    # Plot the Pareto front (objective values)
    Scatter().add(res.F).show()
    #best_index = np.argmin(res.F)
    return res.X,res.F

sol_pos,sol_val = optimise_nsgaII()


pos_pumps = np.concatenate((pos_pumps,sol_pos), axis = 0)

plt.scatter(households['Lon'].to_numpy(), households['Lat'].to_numpy(), c='blue', label='Household')
plt.scatter(pos_pumps[:,0],pos_pumps[:,1],c='red',label='pump')
plt.legend()
plt.show()