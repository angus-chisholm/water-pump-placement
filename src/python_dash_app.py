import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import geopy.distance as geo
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output, State

# Assuming utils.py is in the same directory
from utils import read_data

# --- Data Loading (Global, only load once) ---
data_file = r'data\Map_village_20241227_data.csv'
data, households, pumps = read_data(data_file)

pos_households = households[['Lon', 'Lat']].to_numpy()
nb_capita = households['Nb capita'].to_numpy()

pos_pumps = pumps[['Lon', 'Lat']].to_numpy()
cost_conversion = 5000
cost_standpipe = 500
cost_per_meter = 2
consumption_person = 5  # L/person/day



bounds = np.array([
    [data['Lon'].min(), data['Lat'].min()],  # Min bounds
    [data['Lon'].max(), data['Lat'].max()],  # Max bounds
])

# --- Helper Functions (from your original script) ---
def impact(pump_positions, household_positions, nb_capita, x=None):
    pump_positions_copy = pump_positions.copy()
    if isinstance(x, np.ndarray):
        pump_positions_copy = np.concatenate((pump_positions_copy, np.array([x])), axis=0)

    min_pump_distance = np.zeros(len(household_positions))
    for index, pos_household in enumerate(household_positions):
        dist = np.zeros(len(pump_positions_copy))
        for i, pump_pos in enumerate(pump_positions_copy):
            dist[i] = geo.great_circle(pump_pos, household_positions[index]).meters
        min_pump_distance[index] = np.min(dist)

    impact_val = np.sum(nb_capita * (min_pump_distance))
    return impact_val

initial_impact = impact(pos_pumps, pos_households, nb_capita=nb_capita)

class MyProblem(ElementwiseProblem):
    def __init__(self, pump_specified=None):
        super().__init__(n_var=2, n_obj=2, n_constr=1, xl=bounds[0], xu=bounds[1])
        self.pump_indices = []
        self.pump_specified = pump_specified

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = impact(pos_pumps, pos_households, nb_capita, x) - initial_impact

        f2_distances = np.zeros(len(pos_pumps))
        for index, pump in enumerate(pos_pumps):
            f2_distances[index] = geo.great_circle(pump, x).meters

        if self.pump_specified is not None:
            f2 = cost_conversion + cost_standpipe + cost_per_meter * f2_distances[self.pump_specified]
        else:
            f2 = cost_conversion + cost_standpipe + cost_per_meter * np.min(f2_distances)

        c = initial_impact
        g1 = f1 - c

        out["F"] = [f1, f2]
        out["G"] = [g1]

    def get_final_pump_indices(self, res):
        self.pump_indices = []
        for x_sol in res.X:
            f2_distances = np.zeros(len(pos_pumps))
            for index, pump in enumerate(pos_pumps):
                f2_distances[index] = geo.great_circle(pump, x_sol).meters
            pump_index = np.argmin(f2_distances)
            self.pump_indices.append(pump_index)


def optimise_nsgaII(pump_specified_idx=None, num_generations=10):
    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    problem = MyProblem(pump_specified=pump_specified_idx)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', num_generations),
                   seed=1,
                   verbose=True)

    problem.get_final_pump_indices(res)
    return res.X, res.F, np.array(problem.pump_indices)

# --- Dash App Layout ---
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Water Pump Optimization Dashboard"),

    html.Div([
        html.Label("Specify Pump Index (Optional):"),
        dcc.Input(
            id='pump-index-input',
            type='number',
            value=0,  # Default value
            min=0,
            max=len(pumps)-1,
            step=1,
            style={'marginRight': '10px'}
        ),
        html.Label("Number of Generations:"),
        dcc.Input(
            id='num-generations-input',
            type='number',
            value=10,  # Default value
            min=1,
            step=1,
            style={'marginRight': '10px'}
        ),
        html.Button('Run Optimization', id='run-button', n_clicks=0),
    ], style={'padding': '20px'}),

    html.Div(id='loading-output', children=[dcc.Loading(id="loading-1", type="default", children=html.Div(id="loading-output-1"))]),

    html.Div([
        dcc.Graph(id='pareto-front-graph'),
        dcc.Graph(id='locations-graph'),
        dcc.Graph(id='consumption-graph')
    ])
])

# --- Dash Callbacks ---
@app.callback(
    Output('pareto-front-graph', 'figure'),
    Output('locations-graph', 'figure'),
    Output('consumption-graph', 'figure'),
    Output('loading-output-1', 'children'), # To show loading state
    Input('run-button', 'n_clicks'),
    State('pump-index-input', 'value'),
    State('num-generations-input', 'value')
)
def update_graphs(n_clicks, pump_specified_input, num_generations_input):
    if n_clicks == 0:
        # Initial empty figures or placeholder
        return go.Figure(), go.Figure(), go.Figure(), ""

    # Clear previous loading message
    loading_message = "Optimizing... This might take a moment."

    # Run the optimization
    sol_pos, sol_val, sol_pumps = optimise_nsgaII(
        pump_specified_idx=pump_specified_input,
        num_generations=num_generations_input
    )

    # --- Pareto Front Graph ---
    fig_pareto = px.scatter(
        x=sol_val[:, 0], y=sol_val[:, 1],
        color=sol_val[:, 0],
        labels={'x': 'Impact', 'y': 'Cost'},
        title='Pareto Front',
        color_continuous_scale='Pinkyl'
    )

    # --- Locations Graph ---
    fig_locations = px.scatter(
        x=households['Lon'].to_numpy(), y=households['Lat'].to_numpy(),
        color_discrete_sequence=['black'],
        labels={'x': 'Longitude', 'y': 'Latitude'},
        title=f'Households and Pumps, existing pump {pump_specified_input} chosen'
    )
    fig_locations.add_trace(go.Scatter(
        x=pos_pumps[:, 0], y=pos_pumps[:, 1],
        mode='markers', marker=dict(color='red', size=8, symbol='triangle-up'),
        name='Previous Pumps'
    ))
    fig_locations.add_trace(go.Scatter(
        x=sol_pos[:, 0], y=sol_pos[:, 1],
        mode='markers',
        marker=dict(color=sol_val[:, 0], colorscale='Pinkyl', colorbar=dict(title='Impact'), size=10, symbol='star'),
        name='New Pumps'
    ))

    # --- Consumption Graph ---
    pos_pumps_with_new = np.array([np.concatenate((pos_pumps, [pump]), axis=0) for j, pump in enumerate(sol_pos)])
    consumption_pump = np.zeros((len(pos_pumps_with_new), len(pos_pumps) + 1))

    for x in range(len(pos_pumps_with_new)):
        for i, household_row in households.iterrows():
            closest_pump_index = -1
            min_dist = float('inf')
            for k, pump_loc in enumerate(pos_pumps_with_new[x]):
                dist = geo.great_circle(household_row[['Lon', 'Lat']].to_numpy(), pump_loc).meters
                if dist < min_dist:
                    min_dist = dist
                    closest_pump_index = k
            consumption_pump[x, closest_pump_index] += household_row['Nb capita'] * consumption_person

    indices = np.arange(len(pos_pumps_with_new))
    fig_consumption = make_subplots(rows=len(pos_pumps) + 1, cols=1,
                                     subplot_titles=[f'Pump {i}' for i in range(len(pos_pumps))] + ['New Pump'])

    for i in range(len(pos_pumps)):
        fig_consumption.add_trace(go.Scatter(
            x=indices,
            y=consumption_pump[:, i],
            name=f'Pump {i}, pos={pos_pumps[i].round(4)}',
            mode='markers',
            marker=dict(
                size=5,
                color=sol_val[:, 0],
                colorscale='Pinkyl',
                colorbar=dict(title='Impact')
            )
        ), row=i + 1, col=1)

    fig_consumption.add_trace(go.Scatter(
        x=indices,
        y=consumption_pump[:, len(pos_pumps)], # The last column is for the new pump
        name=f'Pump {len(pos_pumps)}, new pump',
        mode='markers',
        marker=dict(
            size=5,
            color=sol_val[:, 0],
            colorscale='Pinkyl',
            colorbar=dict(title='Impact') 
        )
    ), row=len(pos_pumps) + 1, col=1)

    fig_consumption.update_layout(height=400 * (len(pos_pumps) + 1), width=1400,
                                  title_text=f"Consumption at each standpipe (last plot is new pump), existing pump {pump_specified_input} chosen")

    return fig_pareto, fig_locations, fig_consumption, "" # Clear loading message on completion

if __name__ == '__main__':
    app.run_server(debug=True)