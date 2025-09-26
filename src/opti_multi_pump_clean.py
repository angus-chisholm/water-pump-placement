# Import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geopy.distance as geo
import pandas as pd
from scipy.interpolate import griddata, RBFInterpolator
from pymoo.core.variable import Real, Integer, Binary
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
import networkx as nx
import fluids
import tkinter as tk
from tkinter import messagebox, scrolledtext
import sys
import traceback


## Define functions

# Tools
def build_altitude_interpolator(df, kernel='linear'):
    """
    Build an interpolator that can predict altitude at any lat/lon
    using Radial Basis Function interpolation (works outside convex hull).
    
    Parameters:
        df (pd.DataFrame): must have 'lat', 'lon', 'alt' columns
        kernel (str): RBF kernel ('linear', 'cubic', 'thin_plate_spline', etc.)
        
    Returns:
        function: takes (lon, lat) and returns predicted altitude
    """
    # Prepare training points and values
    points = np.column_stack((df['Lat'], df['Lon']))
    values = df['Altitude'].values
    
    # Build RBF interpolator
    rbf = RBFInterpolator(points, values, kernel=kernel)
    
    def get_alt(point):
        (lon,lat) = point
        # Accepts scalars or arrays
        query_points = np.column_stack((np.atleast_1d(lat), np.atleast_1d(lon)))
        result = rbf(query_points)
        # Return scalar if scalar input
        return result if np.ndim(lat) > 0 else result[0]
    
    return get_alt
def pump_distance(
    pump_positions: np.ndarray,
    household_positions: np.ndarray,
    x: np.ndarray|None = None,
    ):
    """_summary_

    Args:
        pump_positions (np.ndarray): 
            Array of [lon, lat] positions of pumps
        household_positions (np.ndarray): 
            Array of [lon, lat] positions of households
        x (np.ndarray, optional): 
            Position of potential new pump [lon,lat] (only 1). Defaults to None.
        
    Returns:
        min_pump_distance (np.ndarray): 
            Array of distances (len(household_positions))
            to closest pump for each household
        min_pump_distance_index (np.ndarray): 
            Array (len(household_positions)) of 
            index of closest pump for each household
    """
    
    # Make copy and add x to pump array if it exists
    pump_positions_copy = pump_positions.copy()
    if isinstance(x, np.ndarray):
        pump_positions_copy = np.vstack((pump_positions_copy,np.array(x)))
        
    # Create arrays to assign values to
    min_pump_distance = np.zeros(len(household_positions))
    min_pump_distance_index = np.zeros(len(household_positions))
    
    for index,pos_household in enumerate(household_positions):
        dist = np.zeros(len(pump_positions_copy))
        for i,pump_pos in enumerate(pump_positions_copy):
            dist[i] = geo.great_circle(pump_pos, pos_household).meters
        min_pump_distance[index] = np.min(dist)
        min_pump_distance_index[index] = np.argmin(dist)
    
    return min_pump_distance, min_pump_distance_index
def get_consumption(min_indices, household_consumption):
    ### This is a DAILY value, and so if we want to calculate more realistic pressure losses/pump requirements, need to have a peak value i.e. daily-->peak Q function (graph of daily usage etc)
    """
    Takes indices of the closest pump from households as well as per household consumption
    Returns consumption at each pump
    """
    
    consumption_pump = np.zeros(len(household_consumption))
    for i,pump in min_indices:
        consumption_pump[pump] += household_consumption[i]
    
    return consumption_pump
def read_data_gogma(file_path):
    data_households, data_sources = pd.read_excel(
        file_path,
        sheet_name=['Position surveyed household (hh','Position water sources'],
        header=None).values()
    data_sources = data_sources.iloc[:,:5]
    data_sources.columns = ['ID','Name','Lon','Lat','Altitude']
    data_sources['Type'] = data_sources['Name'].str[1]
    data_sources = data_sources[(data_sources['Type']=='B') | (data_sources['Type']=='W')].drop('Name',axis=1)
    data_sources['Type'] = data_sources['Type'].replace('B','Borehole')
    data_sources['Type'] = data_sources['Type'].replace('W','Open Well')
    
    data_households.columns = ['ID','Lat','Lon','Altitude']
    data_household_capita = pd.read_excel(file_path,sheet_name='Choice source before install',usecols='A,F,G').iloc[1:]
    data_household_capita.columns = ['ID','Reading','Nb capita']
    data_household_capita = data_household_capita.sort_values(by='Reading',ascending=True)
    data_household_capita = data_household_capita.drop_duplicates(subset='ID').sort_values(by='ID').iloc[:-1].drop('Reading',axis=1)
    
    data_households['Type'] = 'Household'
    data_households = pd.merge(data_households, data_household_capita, on='ID', how='inner')    
    
    data = pd.concat([data_households,data_sources],ignore_index=True)
    households = data_households
    pumps = data_sources[data_sources['Type']=='Borehole']
    open_wells = data_sources[data_sources['Type']=='Open Well']
    
    
    return data,households,pumps,open_wells
def read_data(file_path):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    households = data[data['Type'] == 'Household']
    pumps = data[data['Type'] == 'Hand Pump']
    open_wells = data[data['Type'] == 'Open Well']

    return data, households, pumps, open_wells

# Pump + pipe flow
def pipe_and_pump_cost(
    alt1: float,
    alt2: float,
    length_pipe: float,
    flow_rate: float,
    pipe_costs: dict,
    pump_cost_per_watt: float,
    )-> tuple[float, float, float]:
    """_summary_

    Args:
        alt1 (float): 
            Altitude of point 1 (water start)
        alt2 (float): 
            Altitude of point 2 (water end)
        length_pipe (float): 
            Length of pipe (m)
        flow_rate (float): 
            Max flow rate required (m^3/s)
        pump_data (dict): 
            Details of pump(s) used with cost and head increase
        pipe_costs (dict): 
            Keys of pipe diameters, values of their cost/m (€/m)

    Returns:
        tuple[float, float, float]: 
            Minimum cost at this point, diameter of pipe chosen and nb pumps required
    """
    
    rho = 1000
    g=9.81
    mu = 1e-3
    epsilon = 0.005e-3
    K = 20 # To decide depending on how pipe flows

    min_pipe_cost = np.inf

    for d,cost_pipe in pipe_costs.items():
        V = flow_rate/(np.pi*d**2/4)
        ReD = rho*V*d/mu
        
        # Darcy friction factor using Clamond method
        f = fluids.friction_factor(Re = ReD,eD = epsilon/d)
        
        # Loss coefficients
        K_f = f*length_pipe/d
        K_output_flow = 1 # (releasing water to open atmosphere)
        K_total = K + K_f + K_output_flow
        
        # Pressure loss using gravity and losses
        p_loss = rho*g*(alt2-alt1) + 0.5*rho*V**2*K_total
        
        # Total cost using pumps and length of pipe
        total_cost = pump_cost_per_watt*p_loss*flow_rate + cost_pipe*length_pipe

        # Set the total cost, diameter and number of pumps required
        if total_cost < min_pipe_cost:
            min_pipe_cost = total_cost
            diameter_pipe = d
            pump_power = p_loss*flow_rate #get pump power required
        
    # Return minimum cost for given location
    if type(min_pipe_cost) != float:
        min_pipe_cost.astype(float)
        
    return min_pipe_cost, diameter_pipe, pump_power

# Impact functions
# minimise person-meters
def impact(original_pump_positions, household_positions, household_capita, x = None):
    min_pump_distance, min_pump_dist_indices = pump_distance(original_pump_positions, household_positions, x)

    impact = np.sum(household_capita * (min_pump_distance))/1000 # thousand-person-metres
    return impact
# % within 30 mins
def impact2(original_pump_positions, household_positions, household_capita, x = None):
    min_pump_distance, min_pump_dist_indices = pump_distance(original_pump_positions, household_positions, x)

    walking_speed = 15/1000 # 15min/km (in min/m)
    min_pump_time = min_pump_distance * walking_speed
    nb_within_margin = 0
    for i,t in enumerate(min_pump_time):
        if t*2 < 30:
            nb_within_margin += household_capita[i]
    
    percent_within_time = nb_within_margin/sum(household_capita)*100*-1 #negative percentage
    
    return percent_within_time

impact_dict = {
    "Person-Meters": impact,
    "Within 30 mins": impact2,
}

# Optimisation functions
class TopologyPositionProblem(ElementwiseProblem):
    def __init__(self, fixed_nodes, fixed_coords, fixed_heights,
                 house_coords, house_weights,
                 n_new_nodes,
                 bounds_xy,
                 impact_fn,
                 altitude_interpolator,
                 **kwargs):
        m = len(fixed_nodes)
        n = n_new_nodes
        
        x_bounds = bounds_xy[:,0]
        y_bounds = bounds_xy[:,1]
        vars = {}

        # Parents: n integers, each in [0, m + n - 1]
        for i in range(n):
            vars[f"parent_{i}"] = Integer(bounds=(0, m + n - 1))
        
        # Coordinates: n real-valued pairs (x, y)
        for i in range(n):
            vars[f"x_{i}"] = Real(bounds=x_bounds)
            vars[f"y_{i}"] = Real(bounds=y_bounds)
            
        # for i in range(n):
        #     vars[f"borehole_{i}"] = Binary()        
        
        super().__init__(vars= vars,  # parents + coords
                         n_obj=2,
                         n_constr=0,
                         **kwargs)
        self.fixed_nodes = fixed_nodes
        self.fixed_coords = fixed_coords
        self.fixed_heights = fixed_heights
        self.house_coords = house_coords
        self.house_weights = house_weights
        self.n_new_nodes = n_new_nodes
        self.bounds_xy = bounds_xy
        self.initial_impact = impact_fn(self.fixed_coords,
                                        self.house_coords,
                                        self.house_weights)
        self.impact_fn = impact_fn
        self.altitude_interpolator = altitude_interpolator
        
        #kwargs
        self.cost_standpipe = kwargs.get("cost_standpipe")
        self.cost_conversion = kwargs.get("cost_conversion")
        self.water_tower_height = kwargs.get("water_tower_height")
        self.pipe_costs = kwargs.get("pipe_costs")
        self.pump_cost_per_watt = kwargs.get("pump_cost_per_watt")
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        x format:
        [parent_0, parent_1, ..., parent_{n-1}, x_0, y_0, ..., x_{n-1}, y_{n-1}]
        """
        m = len(self.fixed_nodes)
        n = self.n_new_nodes
        parents = X[:n].astype(int)
        coords = X[n:3*n].reshape(n,2)
        converted_fixed_pumps = np.zeros(m)
        pipe_data = np.zeros(shape=(n,4))   # For each new node (and hence pipe),
                                            # data includes [node_x, node_y, pipe diameter, pump power]
        # borehole_binaries = np.zeros(n)
        
        for i,p in enumerate(parents):
            if p - m == i:
                # # If new boreholes are an option
                # borehole_binaries[i] = 1
                # Else: don't let a node have itself as a parent
                out["F"] = [1e6, 1e6]  # large penalty
                out["pipe_data"] = [pipe_data]
        
        # Build graph and check validity
        G = nx.DiGraph()
        all_nodes = list(self.fixed_nodes) + [f"N{i}" for i in range(n)]
        G.add_nodes_from(all_nodes)
        
        
        # Add edges child->parent for new nodes
        for i, p in enumerate(parents):
            # if borehole_binaries[i] == 0:
                #if a borehole does NOT exist here
                child = f"N{i}"
                if p < len(self.fixed_nodes):
                    parent = self.fixed_nodes[p]
                else:
                    parent = f"N{p - len(self.fixed_nodes)}"
                G.add_edge(child, parent)
            # elif borehole_binaries[i] == 1:
            #     # if a borehole DOES exist
            #     pass
            # else:
            #     print(borehole_binaries[i])
            #     raise ValueError("Borehole value not 0 or 1")
                
        
        # Check acyclic
        if not nx.is_directed_acyclic_graph(G):
            out["F"] = [1e6, 1e6]  # large penalty
            out["pipe_data"] = [pipe_data]
            return
        
        # Check connectivity: all new nodes connected to fixed nodes
        for i in range(n):
            # if borehole_binaries[i] == 0:
                child = f"N{i}"
                if not any(nx.has_path(G, child, fn) for fn in self.fixed_nodes):
                    out["F"] = [1e6, 1e6]
                    out["pipe_data"] = [pipe_data]
                    return
            # else:
            #     pass
            
        # Compute weighted sum of distances to points in pos_households with all nodes
        f1 = self.impact_fn(self.fixed_coords,self.house_coords,self.house_weights,coords)  # Negative impact calculation with new x location
        
        height_new_nodes = np.asarray([self.altitude_interpolator(i) for i in coords])
        
        # else:
        G_rev = G.reverse()
        f2 = 0
        for i,p in enumerate(parents):
            f2 += self.cost_standpipe
            # if borehole_binaries[i] == 1:
            #     f2 += self.cost_borehole
            #     pipe_data[i] = np.ones(shape=(1,4))
            #     continue
            coord_new_node = coords[i]
            alt2 = height_new_nodes[i]
            child = f"N{i}"
            
            if p < m:
                # Parent is a fixed node
                parent_coord = self.fixed_coords[p]
                alt1 = self.fixed_heights[p] + self.water_tower_height
                
                if converted_fixed_pumps[p] == 0:
                    f2 += self.cost_conversion
                    converted_fixed_pumps[p] = 1
            else:
                # Parent is a new node
                parent_index = p - m
                parent_coord = coords[parent_index]
                alt1 = height_new_nodes[parent_index]

            length_pipe = geo.great_circle(parent_coord, coord_new_node).meters
            descendants = nx.descendants(G_rev, child) # find how many nodes are connected
            num_descendants = len(descendants)                
            flow_rate = 1e-3 * (num_descendants+1) # Takes care of additional connections
            if flow_rate == 0:
                continue
            
            (cost_pipe_and_pumps, diameter_pipe, pump_power
                ) = pipe_and_pump_cost(alt1=alt1,alt2=alt2,
                                    length_pipe=length_pipe,
                                    flow_rate=flow_rate,
                                    pipe_costs=self.pipe_costs,
                                    pump_cost_per_watt=self.pump_cost_per_watt)
                
            pipe_data[i] = np.array([coord_new_node[0],coord_new_node[1],diameter_pipe, pump_power])
            
            f2 += cost_pipe_and_pumps
        
        out["F"] = [f1, f2]
        out["pipe_data"] = [pipe_data]
# -------------------------
# 1) Mixed variable sampler
# -------------------------
class MixedSampler(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        var_items = list(problem.vars.items())  # preserve variable order
        n_var = len(var_items)
        X = np.zeros((n_samples, n_var), dtype=float)

        for i, (name, var) in enumerate(var_items):
            if isinstance(var, Integer):
                low, high = var.bounds
                X[:, i] = np.random.randint(low, high + 1, size=n_samples)
            elif isinstance(var, Real):
                low, high = var.bounds
                X[:, i] = np.random.uniform(low, high, size=n_samples)
            elif isinstance(var, Binary):
                X[:, i] = np.random.randint(0, 2, size=n_samples)  # only 0 or 1
            else:
                raise NotImplementedError(f"Sampler not implemented for var type {type(var)}")

        return X
# -------------------------
# 2) Repair operator (integer/binary fixes)
# -------------------------
class RoundRepair(Repair):
    def _do(self, problem, X, **kwargs):
        X = X.copy()
        var_items = list(problem.vars.items())

        for i, (_, var) in enumerate(var_items):
            if isinstance(var, Integer):
                low, high = var.bounds
                X[:, i] = np.clip(np.round(X[:, i]), low, high)
            elif isinstance(var, Binary):
                # force strictly 0 or 1
                X[:, i] = np.where(X[:, i] >= 0.5, 1, 0)
            elif isinstance(var, Real):
                low, high = var.bounds
                X[:, i] = np.clip(X[:, i], low, high)
            # Choice and other types would be handled separately if needed

        return X
# -------------------------
# 3) Helper: numeric bounds attachment
# -------------------------
def attach_numeric_bounds(problem):
    var_items = list(problem.vars.items())
    xl = []
    xu = []
    for name, var in var_items:
        if isinstance(var, Integer) or isinstance(var, Real):
            low, high = var.bounds
        elif isinstance(var, Binary):
            low, high = 0, 1  # explicitly set binary bounds
        else:
            raise NotImplementedError(f"Bounds extraction not implemented for var type {type(var)}")

        xl.append(low)
        xu.append(high)

    problem.xl = np.array(xl, dtype=float)
    problem.xu = np.array(xu, dtype=float)

    return problem.xl, problem.xu, var_items

def calculate_optimal_placement(fixed_nodes, fixed_coords, fixed_heights,
                                house_coords, house_weights, n_new_nodes,
                                bounds_xy, impact_fn, altitude_interpolator,
                                **kwargs):    
    problem = TopologyPositionProblem(
        fixed_nodes=fixed_nodes,
        fixed_coords=fixed_coords,
        fixed_heights=fixed_heights,
        house_coords=house_coords,
        house_weights=house_weights,
        n_new_nodes=n_new_nodes,
        bounds_xy=bounds_xy,
        impact_fn=impact_fn,
        altitude_interpolator=altitude_interpolator,
        **kwargs
    )
    xl, xu, var_items = attach_numeric_bounds(problem)

    algorithm = NSGA2(
        pop_size=20,
        sampling=MixedSampler(),
        repair=RoundRepair()
    )

    res = minimize(problem,
                algorithm,
                ('n_gen', kwargs["simulation_generations"]),
                seed=4,
                verbose=True)

    return res

class InteractiveParetoPlot:
    def __init__(self, concatenated_result_vals, all_positions, all_result_vals, all_indices, 
                 households, pos_pumps, grid_x, grid_y, grid_z, initial_impact, pipe_data,impactfn,
                 max_nb_standpipes):
        # Store data
        self.concatenated_result_vals = concatenated_result_vals
        self.all_positions = all_positions
        self.all_result_vals = all_result_vals
        self.all_indices = all_indices
        self.households = households
        self.pos_pumps = pos_pumps
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.initial_impact = initial_impact
        self.pipe_data = pipe_data
        self.impactfn = impactfn
        self.max_nb_standpipes=max_nb_standpipes
        
        # Create figure and axes
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(11, 5))
        
        # Store references to plot elements for highlighting
        self.pareto_scatter = None
        self.pump_scatters = []
        self.pump_lines = []
        self.highlighted_pumps = []
        self.highlighted_lines = []
        self.data_box = None  # Single data box that gets updated
        self.n_tested = np.arange(1,self.max_nb_standpipes+1,dtype=int)
        
        # Current hover index
        self.current_hover_idx = None
        
        self.setup_plots()
        self.connect_events()
    
    def setup_plots(self):
        """Set up both plots"""
        # === PARETO PLOT (ax1) ===
        FILLED_MARKERS = list(Line2D.filled_markers)
        n_shape = {key: FILLED_MARKERS[i % len(FILLED_MARKERS)] for i, key in enumerate(self.n_tested)}
        
        # Store all scatter plots for hover detection
        self.pareto_scatters = []
        if self.impactfn == impact2:
            for n, k in enumerate(self.n_tested):
                scatter = self.ax1.scatter(
                    -1*self.all_result_vals[n][:, 0], 
                    self.all_result_vals[n][:, 1],
                    c=-1*self.all_result_vals[n][:, 0],
                    cmap='viridis',
                    vmin=-100,
                    vmax=0,
                    edgecolor='k',
                    s=20,
                    label=f'{k} new standpipes',
                    marker=n_shape.get(k,'s'),
                )
                self.pareto_scatters.append(scatter)
            
            self.ax1.set_xlabel('% of people within 30 mins of nearest improved water source', fontsize=12)
            self.ax1.set_xlim(50,105)
        else:
            for n, k in enumerate(self.n_tested):
                scatter = self.ax1.scatter(
                    self.all_result_vals[n][:, 0]-self.initial_impact, 
                    self.all_result_vals[n][:, 1],
                    c=self.all_result_vals[n][:, 0]-self.initial_impact,
                    cmap='viridis',
                    vmin=-100,
                    vmax=0,
                    edgecolor='k',
                    s=20,
                    label=f'{k} new standpipes',
                    marker=n_shape.get(k,'s'),
                )
                self.pareto_scatters.append(scatter)
        
            self.ax1.set_xlabel('Impact (thousand person-meters saved)', fontsize=12)
            
        self.ax1.set_ylabel('Cost (€)', fontsize=12)
        self.ax1.set_title('Pareto Front', fontsize=12)
        self.ax1.legend()
        self.ax1.grid(True)
        
        # === GEOGRAPHICAL PLOT (ax2) ===
        contour = self.ax2.contourf(self.grid_x, self.grid_y, self.grid_z, levels=20, cmap='terrain')
        
        # Plot households (black dots)
        self.ax2.scatter(self.households['Lon'], self.households['Lat'], color='black', label='Households')
        
        # Plot new standpipes and store references
        pump_idx = 0  # Index to track pumps across all configurations
        
        for n, k in enumerate(self.n_tested):
            if self.impactfn == impact2:
                for i in range(k):
                    pos_pumps_new_plot = self.all_positions[n][:,2*i:2*(i+1)]
                    sc = self.ax2.scatter(
                        pos_pumps_new_plot[:, 0], pos_pumps_new_plot[:, 1],
                        c=-1*self.all_result_vals[n][:, 0],
                        cmap='viridis', 
                        edgecolor='k',
                        vmin=50,
                        vmax=100,
                        s=60,
                        label='New Standpipes' if pump_idx == 0 else "",
                        marker=n_shape.get(k,'s'),
                        alpha=0.5
                    )
                    self.pump_scatters.append(sc)
                    
                    # Store lines for each solution in this pump configuration
                    lines_for_config = []
                    for j in range(len(pos_pumps_new_plot)):  # For each solution
                        solution_lines = []
                        if j < len(self.all_indices[n]) and i < self.all_indices[n].shape[1]:
                            if self.all_indices[n][j,i] < len(self.pos_pumps):
                                x = int(self.all_indices[n][j,i])
                                parent_coord = self.pos_pumps[x]
                            else:
                                x = int(self.all_indices[n][j,i] - len(self.pos_pumps))
                                if x < len(pos_pumps_new_plot):
                                    parent_coord = self.all_positions[n][j,2*x:2*(x+1)]
                                else:
                                    continue
                            
                            points_to_plot = np.array([pos_pumps_new_plot[j], parent_coord])
                            line, = self.ax2.plot(
                                points_to_plot[:, 0], points_to_plot[:, 1],
                                c='black',
                                alpha=0,
                                linewidth=1
                            )
                            solution_lines.append(line)
                        lines_for_config.append(solution_lines)
                    
                    self.pump_lines.append(lines_for_config)
                    pump_idx += 1
            else:
                for i in range(k):
                    pos_pumps_new_plot = self.all_positions[n][:,2*i:2*(i+1)]
                    sc = self.ax2.scatter(
                        pos_pumps_new_plot[:, 0], pos_pumps_new_plot[:, 1],
                        c=self.all_result_vals[n][:, 0]-self.initial_impact,
                        cmap='viridis', 
                        edgecolor='k',
                        vmin=-100,
                        vmax=0,
                        s=60,
                        label='New Standpipes' if pump_idx == 0 else "",
                        marker=n_shape.get(k,'s'),
                        alpha=0.5
                    )
                    self.pump_scatters.append(sc)
                    
                    # Store lines for each solution in this pump configuration
                    lines_for_config = []
                    for j in range(len(pos_pumps_new_plot)):  # For each solution
                        solution_lines = []
                        if j < len(self.all_indices[n]) and i < self.all_indices[n].shape[1]:
                            if self.all_indices[n][j,i] < len(self.pos_pumps):
                                x = int(self.all_indices[n][j,i])
                                parent_coord = self.pos_pumps[x]
                            else:
                                x = int(self.all_indices[n][j,i] - len(self.pos_pumps))
                                if x < len(pos_pumps_new_plot):
                                    parent_coord = self.all_positions[n][j,2*x:2*(x+1)]
                                else:
                                    continue
                            
                            points_to_plot = np.array([pos_pumps_new_plot[j], parent_coord])
                            line, = self.ax2.plot(
                                points_to_plot[:, 0], points_to_plot[:, 1],
                                c='black',
                                alpha=0,
                                linewidth=1
                            )
                            solution_lines.append(line)
                        lines_for_config.append(solution_lines)
                    
                    self.pump_lines.append(lines_for_config)
                    pump_idx += 1
        
        # Plot previous pumps (red)
        self.ax2.scatter(self.pos_pumps[:, 0], self.pos_pumps[:, 1], 
                        color='red', label='Previous Pumps', edgecolors='k', s=90)
        
        # Add colorbar
        if len(self.pump_scatters) > 0:
            cbar = self.fig.colorbar(self.pareto_scatters[-1], ax=self.ax2)
            if self.impactfn==impact2:
                cbar.set_label('% of people within 30 mins')
            else:
                cbar.set_label('Impact (thousand person-meters)')
        
        # Labels and title
        self.ax2.set_xlabel('Longitude')
        self.ax2.set_ylabel('Latitude')
        self.ax2.set_title('Households and Pumps')
        self.ax2.legend()
        self.ax2.grid(True)
        
        plt.tight_layout()
    
    def connect_events(self):
        """Connect mouse events"""
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
    
    def on_hover(self, event):
        """Handle hover events over the Pareto plot"""
        if event.inaxes == self.ax1:
            # Check each scatter plot for hover
            hover_idx = None
            cumulative_solutions = 0
            
            for n, scatter in enumerate(self.pareto_scatters):
                cont, ind = scatter.contains(event)
                if cont:
                    # Get the index within this scatter plot
                    local_idx = ind["ind"][0]
                    # Convert to global index
                    hover_idx = cumulative_solutions + local_idx
                    break
                cumulative_solutions += len(self.all_result_vals[n])
            
            if hover_idx is not None:
                if hover_idx != self.current_hover_idx:
                    self.highlight_solution(hover_idx)
                    self.current_hover_idx = hover_idx
            else:
                # Clear highlighting if not hovering over any point
                if self.current_hover_idx is not None:
                    self.clear_highlighting()
                    self.current_hover_idx = None
    
    def get_pipe_data_text(self, config_idx, solution_in_config):
        """Get formatted pipe data text for display"""
        try:
            # Access the pipe data for this specific solution
            data = self.pipe_data[config_idx][solution_in_config][0]
            return data
            
        except (IndexError, KeyError, TypeError):
            return "No data available"
    
    def highlight_solution(self, solution_idx):
        """Highlight the corresponding solution in the geographical plot"""
        self.clear_highlighting()
        
        # Find which configuration this solution belongs to
        cumulative_solutions = 0
        config_idx = None
        solution_in_config = None
        
        for n, k in enumerate(self.n_tested):
            num_solutions = len(self.all_result_vals[n])
            if cumulative_solutions <= solution_idx < cumulative_solutions + num_solutions:
                config_idx = n
                solution_in_config = solution_idx - cumulative_solutions
                break
            cumulative_solutions += num_solutions
        
        if config_idx is not None and solution_in_config is not None:
            # Highlight pumps for this specific solution
            pump_config_idx = 0
            for n in range(config_idx + 1):
                k = self.n_tested[n]
                if n == config_idx:
                    # This is our target configuration
                    for i in range(k):
                        if pump_config_idx < len(self.pump_scatters):
                            # Get the specific solution's pump positions
                            pos_pumps_new_plot = self.all_positions[n][:,2*i:2*(i+1)]
                            highlighted_scatter = self.ax2.scatter(
                                pos_pumps_new_plot[solution_in_config:solution_in_config+1, 0],
                                pos_pumps_new_plot[solution_in_config:solution_in_config+1, 1],
                                c='yellow',
                                s=150,
                                marker='*',
                                edgecolor='red',
                                linewidth=2,
                                zorder=10
                            )
                            self.highlighted_pumps.append(highlighted_scatter)
                            
                            # Highlight only the lines for this specific solution
                            if pump_config_idx < len(self.pump_lines):
                                if solution_in_config < len(self.pump_lines[pump_config_idx]):
                                    solution_lines = self.pump_lines[pump_config_idx][solution_in_config]
                                    for line in solution_lines:
                                        line.set_color('red')
                                        line.set_linewidth(3)
                                        line.set_alpha(1.0)
                                        self.highlighted_lines.append(line)
                        
                        pump_config_idx += 1
                else:
                    pump_config_idx += k
            
            # Show data box
            self.show_data_box(config_idx, solution_in_config)
            
            # Refresh the plot
            self.fig.canvas.draw_idle()
    
    def show_data_box(self, config_idx, solution_in_config):
        """Show data box with pipe information"""
        # Get the formatted data text
        data_text = self.get_pipe_data_text(config_idx, solution_in_config)
        
        data_string = ""
        for data_item in data_text:
            data_string += f"Lon: {data_item[0]:.3f}, Lat: {data_item[1]:.3f}, Diameter: {data_item[2]:.3f}m, Power: {data_item[3]:.3f}W; \n"
        
        # Create box properties
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black')
        
        # Add solution info to the text
        k = self.n_tested[config_idx]
        header = f"Solution {self.current_hover_idx}\n{k} standpipe{'s' if k > 1 else ''}\n---\n"
        full_text = header + data_string
        
        # Place text box in upper left of geographical plot
        self.data_box = self.ax2.text(
            0.02, 0.98, full_text, 
            transform=self.ax2.transAxes, 
            fontsize=8,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=props,
            zorder=15
        )
    
    def hide_data_box(self):
        """Hide the data box"""
        if self.data_box is not None:
            self.data_box.remove()
            self.data_box = None
    
    def clear_highlighting(self):
        """Clear all highlighting"""
        # Remove highlighted pumps
        for scatter in self.highlighted_pumps:
            scatter.remove()
        self.highlighted_pumps.clear()
        
        # Reset line properties
        for line in self.highlighted_lines:
            line.set_color('black')
            line.set_linewidth(1)
            line.set_alpha(0)
        self.highlighted_lines.clear()
        
        # Hide data box
        self.hide_data_box()
        
        # Refresh the plot
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the interactive plot"""
        plt.show()

## Main function

entry_widgets = {}

def run_optimisation_and_plot():
    
    """
    Setup, optimisation and plotting for the scenario with the given filename
    """
    try:       
        
        gui_kwargs = {
            "cost_conversion" : float(entry_widgets["Conversion Cost (€)"].get()),
            "cost_standpipe" : float(entry_widgets["Standpipe Cost (€)"].get()),
            "cost_borehole" : float(entry_widgets["Borehole Cost (€)"].get()),
            "consumption_person":float(entry_widgets["Per Capita Consumption (L/pers/day)"].get()), 
            "pump_cost_per_watt" : float(entry_widgets["Pump Cost (€/W)"].get()),
            "simulation_generations": int(entry_widgets["Number of Generations"].get()),
            "pipe_costs" : { # keys are pipe diameters, values are pipe costs €/m (source AO_GRP.1_GALLON_DUVAL_ROBILLOT_JIMENEZ)
                0.02: 0.65,
                0.025: 1.5,
                0.032: 2.69,
                0.04: 3.5,
                0.05: 5,
                0.06: 5,
            },
            "water_tower_height" : 4,
        }
        impactfn = impact_dict[entry_widgets["Impact Function"].get()]
        max_nb_standpipes = int(entry_widgets["Maximum Number of Standpipes"].get())
    
        # Setup data
        
        # Import data and define arrays (may need to rewrite read function depending on input)
        data_file = r'src\data\DO_NOT_DISTRIBUTE_DATA_GOGMA.xlsx'
        data,households,pumps,open_wells = read_data_gogma(data_file)
        pos_households = households[['Lon','Lat']].to_numpy() 
        nb_capita = households['Nb capita'].to_numpy()
        pos_pumps = pumps[['Lon','Lat']].to_numpy()
        bounds = np.array([
            [data['Lon'].min(), data['Lat'].min()],  # Min bounds
            [data['Lon'].max(), data['Lat'].max()],  # Max bounds
        ])
        
        # Create gridpoints for relief plot
        grid_x, grid_y = np.mgrid[data['Lon'].min():data['Lon'].max():200j, 
                                data['Lat'].min():data['Lat'].max():200j]
        grid_z = griddata((data['Lon'], data['Lat']), data['Altitude'], (grid_x, grid_y), method='cubic')
        
        # Create interpolator for altitude
        get_alt = build_altitude_interpolator(data)
        
        
        # Run optimiser
        all_result_vals = []
        all_indices = []
        all_positions = []
        all_boreholes = []
        pipe_data = []
        for i in range(1,max_nb_standpipes+1):
            res = calculate_optimal_placement(pumps.index.to_list(), pos_pumps,
                                            pumps['Altitude'].to_numpy(), pos_households,
                                            nb_capita, i, bounds, impactfn, get_alt,
                                            **gui_kwargs)
            all_result_vals.append(res.F)
            all_indices.append(res.X[:,:i])
            all_positions.append(res.X[:,i:3*i])
            #all_boreholes.append(res.X[:,3*i:4*i])
            pipe_data.append(res.pop.get("pipe_data"))
            
        concatenated_result_vals = np.concatenate(all_result_vals)
        
        
        # Set up plot
        interactive_plot = InteractiveParetoPlot(
            concatenated_result_vals, all_positions, all_result_vals, all_indices,
            households, pos_pumps, grid_x, grid_y, grid_z, 
            impactfn(pos_pumps, pos_households, nb_capita),
            pipe_data,impactfn, max_nb_standpipes
        )

        # Show the plot
        interactive_plot.show()    
    
    except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        # 1. Get the full traceback as a formatted string
        error_details = traceback.format_exc()
        
        # 2. Display the error message AND the traceback details
        messagebox.showerror(
            "Fatal Error in Optimization",
            f"An error occurred: {e}\n\n--- Traceback Details ---\n{error_details}"
        )
        
        # You can also print the traceback to the terminal for debugging
        print(error_details)


# GUI functions
def create_gui():
    root = tk.Tk()
    root.title("PVWPS Optimisation")
    
    # Create frames to organize the widgets
    main_frame = tk.Frame(root, padx=10, pady=10)
    main_frame.pack()
    
    return root, main_frame

def create_parameter_entry(parent_frame, param_name, default_value, row_num, entry_widgets, dropdown=False):
    tk.Label(parent_frame, text=param_name).grid(row=row_num, column=0, sticky="w")
    if dropdown==True:
        # entry_widgets[param_name].set(impact_dict.keys()[0])
        entry = tk.OptionMenu(parent_frame,default_value, list(impact_dict.keys()))
        entry.grid(row=row_num, column=1)
    else:
        entry = tk.Entry(parent_frame)
        entry.insert(0, str(default_value))
        entry.grid(row=row_num, column=1)
        entry_widgets[param_name] = entry


def main():
    """Run main function. 
    Things which can be modified:
    - filename in setup function
    - impact function defined in both optimisation and plotting functions
    """  
    # Define values
    default_kwargs = {
        "Conversion Cost (€)" : 5000,
        "Standpipe Cost (€)" : 1000,
        "Borehole Cost (€)" : 15000,
        "Per Capita Consumption (L/pers/day)":14, #L/person/day
        "Pump Cost (€/W)" : 1, # €/W Estimated! Needs verifying
        "Number of Generations": 100,
        "Maximum Number of Standpipes": 3,
    }
    root,main_frame = create_gui()
    row_num=0 
    for k,v in default_kwargs.items():
        create_parameter_entry(main_frame, k, v,row_num,entry_widgets)
        row_num+=1
    entry_widgets['Impact Function'] = tk.StringVar(root)
    entry_widgets['Impact Function'].set(list(impact_dict.keys())[0])
    tk.Label(main_frame, text='Impact Function').grid(row=row_num, column=0, sticky="w")
    dropdown = tk.OptionMenu(main_frame, entry_widgets['Impact Function'], *list(impact_dict.keys()))
    dropdown.grid(row=row_num, column=1)
    print(list(impact_dict.keys()))

        
        
    # Run button
    run_button = tk.Button(main_frame, text="Run Optimisation", command=run_optimisation_and_plot)
    run_button.grid(row=row_num+1, columnspan=2, pady=10)
    
    root.mainloop()
    
# Run code
if __name__ == "__main__":
    main()