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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

data_file = r'data\Map_village_20241227_data.csv'

data,households,pumps = read_data(data_file)

pos_households = households[['Lon','Lat']].to_numpy() # For f1 (sum of weighted distances)
nb_capita = households['Nb capita'].to_numpy() # Define weight factors for each distance in pos_households

pos_pumps = pumps[['Lon','Lat']].to_numpy() # For f2 (min distance)
cost_conversion=5000
cost_standpipe=1000
cost_per_meter=6
consumption_person=5 #L/person/day

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
    def __init__(self, pump_specified=None):
        super().__init__(n_var=2, n_obj=2, n_constr=1, xl=bounds[0], xu=bounds[1])  # 2D search space
        self.pump_indices = []  # Store pump indices
        self.pump_specified = pump_specified  # Specify a pump index to be fixed

    def _evaluate(self, x, out, *args, **kwargs):

        # Compute weighted sum of distances to points in pos_households
        f1 = impact(pos_pumps,pos_households,nb_capita,x)-initial_impact  # Negative impact calculation with new x location
        
        # Compute minimum distance to any point in pos_pumps
        f2_distances = np.zeros(len(pos_pumps))
        for index,pump in enumerate(pos_pumps):
            f2_distances[index] = geo.great_circle(pump, x).meters
        
        # Add penalty term if the new pump is not closest to the specified pump
        if self.pump_specified is not None:
            f2 = cost_conversion+cost_standpipe+cost_per_meter*f2_distances[self.pump_specified]
        else:
            f2 = cost_conversion+cost_standpipe+cost_per_meter*np.min(f2_distances)  # Minimum distance to any point in pos_pumps



        # Define constraint: f1 >= c
        c = initial_impact
        g1 = f1-c

        out["F"] = [f1, f2]  # Objective functions
        out["G"] = [g1]  # Constraints (must be <= 0)

    def get_final_pump_indices(self, res):
        """Retrieve pump indices for the final generation."""
        self.pump_indices = []
        if self.pump_specified is not None:
            # If a pump is specified, use its index directly
            self.pump_indices = [self.pump_specified] * len(res.X)
        else:
            # If no pump is specified, find the closest pump for each solution
            for x in res.X:  # Iterate over the final generation's solutions
                f2_distances = np.zeros(len(pos_pumps))
                for index, pump in enumerate(pos_pumps):
                    f2_distances[index] = geo.great_circle(pump, x).meters
                pump_index = np.argmin(f2_distances)
                self.pump_indices.append(pump_index)


def optimise_nsgaII(pump_specified = None):
    # Define the NSGA-II optimization algorithm
    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    # Solve the problem
    problem = MyProblem(pump_specified)
    res = minimize(problem,
                algorithm,
                ('n_gen', 10),
                seed=1,
                verbose=True)
    
    problem.get_final_pump_indices(res)


    # Plot the Pareto front (objective values)
    # fig = px.scatter(x=res.F[:, 0], y=res.F[:,1], color=res.F[:, 0], labels={'x': 'Impact', 'y': 'Cost'},
    #                  title='Pareto Front', color_continuous_scale='Pinkyl')
    # fig.show()
    
    sol_pumps = np.array(problem.pump_indices)

    return res.X, res.F, sol_pumps

pump_specified = None  # Specify int pump index if constraint is needed
sol_pos,sol_val,sol_pumps = optimise_nsgaII(pump_specified) # Specify int pump index if constraint is needed

pos_pumps_new = sol_pos

plt.scatter(sol_val[:0],sol_val[:1],color = sol_val[:,0])
plt.xlabel('Impact (person-meters)')
plt.ylabel('Cost (Euros €)')
plt.title('Pareto Front')
plt.grid()
plt.show()

# Plot households and pumps with colorbar based on the first objective function
fig = px.scatter(x=households['Lon'].to_numpy(), y=households['Lat'].to_numpy(), color_discrete_sequence=['black'], labels={'x': 'Longitude', 'y': 'Latitude'}, title=f'Households and Pumps, existing pump {pump_specified} chosen')
fig.add_scatter(x=pos_pumps[:, 0], y=pos_pumps[:, 1], mode='markers', marker=dict(color='red'), name='Previous Pumps')
fig.add_scatter(x=pos_pumps_new[:, 0], y=pos_pumps_new[:, 1], mode='markers', marker=dict(color=sol_val[:, 0], colorscale='Pinkyl', colorbar=dict(title='Impact')), name='New Pumps')
fig.show()


# Find consumption at each standpipe for each possible solution

pos_pumps_with_new = np.array([np.concatenate((pos_pumps,[pump]), axis=0) for j, pump in enumerate(sol_pos)])


pump_index_min = np.zeros((len(pos_pumps_with_new),len(pos_households)))
for i,household in enumerate(pos_households):
    for x in range(len(pos_pumps_with_new)):
        pump_distances = np.zeros(len(pos_pumps_with_new[x]))
        for k, pump in enumerate(pos_pumps_with_new[x]):
            pump_distances[k] = geo.great_circle(household, pump).meters
        pump_index_min[x,i] = np.argmin(pump_distances)


# Initialize consumption_pump to store the total consumption for each pump
consumption_pump = np.zeros((len(pos_pumps_with_new), len(pos_pumps) + 1))  # +1 for the new pump in each solution

# Update consumption_pump based on the closest pump
for x in range(len(pos_pumps_with_new)):  # Iterate over solutions
    for i, household in households.iterrows():  # Iterate over households
        closest_pump_index = int(pump_index_min[x, i])  # Closest pump index
        consumption_pump[x, closest_pump_index] += household['Nb capita'] * consumption_person
        if closest_pump_index == 3:
            consumption_pump[x,sol_pumps[x]] += household['Nb capita'] * consumption_person  # Add consumption from the new pump to the closest existing pump

indices = np.arange(len(pos_pumps_with_new))
fig = make_subplots(rows=4, cols=1)

fig.add_trace(go.Scatter(
    x=indices,
    y=consumption_pump[:,0],
    name=f'Pump0,pos={pos_pumps[0]}',
    mode='markers',
    marker=dict(
        size=5,
        color=sol_val[:, 0], 
        colorscale='Pinkyl', 
        colorbar=dict(title='Impact'),
        showscale=True
    )
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=indices,
    y=consumption_pump[:,1],
    name=f'Pump1,pos={pos_pumps[1]}',
    mode='markers',
    marker=dict(
        size=5,
        color=sol_val[:, 0], 
        colorscale='Pinkyl', 
        colorbar=dict(title='Impact'),
        showscale=True
    )
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=indices,
    y=consumption_pump[:,2],
    name=f'Pump2,pos={pos_pumps[2]}',
    mode='markers',
    marker=dict(
        size=5,
        color=sol_val[:, 0], 
        colorscale='Pinkyl', 
        colorbar=dict(title='Impact'),
        showscale=True
    )
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=indices,
    y=consumption_pump[:,3],
    name=f'Pump3, new pump',
    mode='markers',
    marker=dict(
        size=5,
        color=sol_val[:, 0], 
        colorscale='Pinkyl', 
        colorbar=dict(title='Impact'),
        showscale=True
    )
), row=4, col=1)


fig.update_layout(height=800, width=1400, title_text=f"Consumption at each standpipe (last plot is new pump), , existing pump {pump_specified} chosen")
fig.show()