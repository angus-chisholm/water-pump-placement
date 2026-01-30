# Standard library imports
import math
import os
import pickle
import traceback
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

# Third-party imports
import fluids
from geopy.distance import great_circle, lonlat
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, griddata
import requests

# Pymoo imports (Optimisation)
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.core.variable import Binary, Integer, Real
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.optimize import minimize


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
            Positions of potential new Fountains [lon,lat]. Defaults to None.
        
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
    min_pump_distance_index = np.zeros(len(household_positions), dtype=np.int8)
    
    for index,pos_household in enumerate(household_positions):
        dist = np.zeros(len(pump_positions_copy))
        for i,pump_pos in enumerate(pump_positions_copy):
            dist[i] = great_circle(lonlat(*pump_pos), lonlat(*pos_household)).meters
        min_pump_distance[index] = np.min(dist)
        min_pump_distance_index[index] = int(np.argmin(dist))
    
    return min_pump_distance, min_pump_distance_index



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

def pipe_calcs(alt1, alt2, length_pipe, flow_rate, pipe_costs, pump_cost_per_watt, head_1=0):
    
    # Pump + pipe flow + cost calculations
    rho = 1000
    g=9.81
    mu = 1e-3
    epsilon = 0.005e-3
    K = 17.95 ## 1 gate valve (for isolation), 1 check valve (to prevent backflow), 10 90o elbows, and 10 union connectors
    min_pipe_cost = np.inf
    
    for d,cost_pipe in pipe_costs.items():
        V = flow_rate/(np.pi*d**2/4)
        if V>=1.5:
            if d == list(pipe_costs.keys())[-1]:
                Exception(f"Err: No suitable pipe diameter found for flow rate {flow_rate}")
            continue
        ReD = rho*V*d/mu
        # Darcy friction factor
        f = fluids.friction_factor(Re = ReD,eD = epsilon/d)
        
        # Loss coefficients
        K_f = f*length_pipe/d
        K_total = K + K_f
        
        # Head change in pipe (negative if head loss)
        delta_H = (alt1-alt2) - 0.5*(V**2/g)*K_total
        head_2 = head_1 + delta_H # outlet head without pump
        
        # Total cost using pumps and length of pipe
        if head_2 < 3:
            outlet_head = 3 # minimum outlet head (3m)
            pump_head = outlet_head - delta_H - head_1              
            pump_power = rho*g*flow_rate*pump_head*1/0.7 # assuming 70% pump efficiency
            if pump_power<10:
                # no need for pump<10W
                pump_power = 0
                pump_head = 0
            total_cost = cost_pipe*length_pipe+pump_cost_per_watt*pump_power
        else:
            # no pump needed
            pump_head = 0
            pump_power = 0
            total_cost = cost_pipe*length_pipe

        # Set the total cost, diameter 
        if total_cost < min_pipe_cost:
            min_pipe_cost = total_cost
            diameter_pipe = d
            final_pump_power = pump_power
            final_pump_head = pump_head
            outlet_head = head_2
            
            
    # Return minimum cost for given location
    if not isinstance(min_pipe_cost,float):
        min_pipe_cost.astype(float)

    return min_pipe_cost, diameter_pipe, outlet_head, final_pump_power, final_pump_head


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

pumping_dict = {
    "Electric": "Electric",
    "Diesel": "Diesel",
}

class consumption_generator():
    def __init__(self, profile_file, nb_capita, G, household_positions, base_flow):
        self.usage_profile = pd.read_excel(profile_file,index_col="Heure")
        self.nb_capita = nb_capita
        self.graph = G
        self.graph_nodes = list(self.graph)
        self.node_dict = {self.graph_nodes[i]:i for i in range(len(self.graph_nodes))}
        self.base_flow = base_flow
        self.household_positions = household_positions
        self.pump_coords = np.array(list(nx.get_node_attributes(self.graph, 'coords').values()))
        self.min_indices = pump_distance(self.pump_coords, household_positions)[1]
        self.generate_pump_on_times()


    def get_consumption(self):
        # Calculate daily consumption
        """
        Takes indices of the closest pump from households as well as per household consumption
        Returns dict: {pump_index: total_consumption}
        """
        consumption_dict = {}
        for i, pump in enumerate(self.min_indices):
            pump_key = self.graph_nodes[pump]
            if pump_key not in consumption_dict:
                consumption_dict[pump_key] = 0
            consumption_dict[pump_key] += self.nb_capita[i]*10 # 10 L per capita per day
            
        for node in self.graph.nodes():
            if node not in consumption_dict:
                consumption_dict[node] = 0
            
        return consumption_dict

    def generate_value_gaussian(self, mean, std_dev, rng):
        return rng.normal(loc=mean, scale=std_dev)
    
    def generate_pump_on_times(self):
        # Generate pump on-times with proper handling of remainders and no overlap into next hour
        pump_on_times = {}

        consumption_dict = self.get_consumption()
        means = self.usage_profile["moyenne"].to_numpy()
        std_devs = self.usage_profile["std_dev"].to_numpy()
        for i, k in enumerate(consumption_dict.keys()):
            rng = np.random.default_rng(i)
            
            # Generate consumption distribution
            dist = np.array([self.generate_value_gaussian(m, s, rng) if s != 0 else 0 for m, s in zip(means, std_devs) ]).clip(min=0)
            dist = dist / dist.sum()

            # Calculate consumption and flowing times
            consumption = dist * consumption_dict[k]
            flowing_times = consumption / (self.base_flow * 60)  # minutes
            
            # For each hour, generate intervals with durations and random start times
            on_times_dict = {}

            for hour_idx, flowing_time in enumerate(flowing_times):
                remaining_time = flowing_time
                on_times_dict[hour_idx] = []
                while remaining_time > 0.01:
                    interval_duration = min(2, remaining_time)
                    remaining_time -= interval_duration
                    interval_duration_sec = math.floor(interval_duration * 60)
                    max_start_seconds = 3600 - interval_duration_sec

                    if max_start_seconds <= 0:
                        break
                    # Try until we find a non-overlapping slot
                    for _ in range(100):  # safety cap
                        start_second = math.floor(rng.uniform(0, max_start_seconds))
                        end_second = start_second + interval_duration_sec
                        overlaps = any(
                            not (end_second <= s or start_second >= s + d)
                            for s, d in on_times_dict[hour_idx]
                        )
                        if not overlaps:
                            on_times_dict[hour_idx].append((start_second, interval_duration_sec))
                            break
                    else:
                        # Could not place interval without overlap
                        break
            
            pump_on_times[k] = on_times_dict

        # Convert to DataFrame with hours as index and pump columns
        data = {f'pump_{k}': {h: pump_on_times[k].get(h,) for h in range(24)} 
                for k in pump_on_times.keys()}
        
        self.df_on_times = pd.DataFrame(data)
        
        # Put on times into graph attributes
        for pump_key, sched in pump_on_times.items():
            try:
                on_times = sched
            except KeyError:
                on_times = {}
            nx.set_node_attributes(self.graph, {pump_key: on_times}, 'on_times')
        
        return
    
    def calculate_flow_multiplier(self, intervals):
        """
        Input: List of (start, end) tuples.
        Output: List of (start, end, count) showing flow depth.
        """
        events = []
        for start, end in intervals:
            events.append((start, 1))  # Flow starts (+1)
            events.append((end, -1))   # Flow ends (-1)

        # Sort by time. If times are equal, process start (+1) before end (-1)
        events.sort()

        processed_profile = []
        current_depth = 0
        prev_time = events[0][0]

        for time, change in events:
            if time > prev_time:
                # Record the interval we just finished traversing
                if current_depth > 0:
                    processed_profile.append({
                        'start': prev_time,
                        'duration': time-prev_time,
                        'flow_multiplier': current_depth 
                    })
            
            current_depth += change
            prev_time = time
            
        return processed_profile


    def assign_flow_to_edges(self):
        
        # 1. Initialize a temporary list on every edge to hold raw time data
        for u, v in self.graph.edges():
            self.graph[u][v]['raw_intervals'] = []

        # 2. Iterate through every node to find demand
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            
            # print(node_data['on_times'] if 'on_times' in node_data else f"No on_times for node {node}...")
            if 'on_times' in node_data and node_data['on_times']:
                my_intervals = []
                
                # Iterate through the Hours (keys) and Times (values)
                for hour_key, time_list in node_data['on_times'].items():
                    for (start_sec, duration) in time_list:
                        
                        # CONVERSION: Turn relative hour times into absolute seconds of the day
                        # Formula: (Hour * 3600) + Seconds
                        absolute_start = (hour_key * 3600) + start_sec
                        absolute_end = absolute_start + duration
                        
                        my_intervals.append((absolute_start, absolute_end))
                
                # 3. Find the upstream edges for this specific node
                ancestors = nx.ancestors(self.graph, node) | {node}
                path_edges = self.graph.subgraph(ancestors).edges()
                
                # 4. Add this node's intervals to ALL those upstream edges
                for u, v in path_edges:
                    self.graph[u][v]['raw_intervals'].extend(my_intervals)
                    
        # 5. Final Pass: Calculate the overlaps for each edge
        for u, v in self.graph.edges():
            raw = self.graph[u][v].pop('raw_intervals') # Remove raw data to save memory
            if raw:
                # Run the helper function to get the "2x", "3x" markers
                self.graph[u][v]['flow_schedule'] = self.calculate_flow_multiplier(raw)
            else:
                self.graph[u][v]['flow_schedule'] = []
            
        return
    
    def assign_consumption_to_graph(self):
        self.generate_pump_on_times()
        self.assign_flow_to_edges()
        return self.graph


# Optimisation functions
class TopologyPositionProblem(ElementwiseProblem):
    def __init__(self, fixed_nodes, fixed_coords, fixed_heights,
                 house_coords, house_weights,
                 n_new_nodes,
                 bounds_xy,
                 impact_fn,
                 altitude_interpolator,
                 base_flow,
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
        self.base_flow = base_flow
        
        #kwargs
        self.cost_fountain = kwargs.get("cost_fountain")
        self.cost_conversion = kwargs.get("cost_conversion")
        self.water_tower_height = kwargs.get("water_tower_height")
        self.pipe_costs = kwargs.get("pipe_costs")
        self.pump_cost_per_watt = kwargs.get("pump_cost_per_watt")
        self.fountains_retrofitted = kwargs.get("fountains_retrofitted")
        self.pumping_method = kwargs.get("pumping_method")
        
        # Define pumping cost params
        # Diesel
        self.discount_rate = 0.05 # 5% discount rate
        self.specific_consumption = 0.35 #L/kWh for diesel pumps
        self.diesel_price = 1.22*0.84 # €/Litre (converted from https://www.globalpetrolprices.com/Burkina-Faso/diesel_prices/, 26 Jan 2026)
        self.oandm = 0.05 # €/kWh Operation and Maintenance (diesel)
        self.generator_cost_per_kw = 100*0.84 # €/kW (assuming demand is concentrated to 3hrs) for diesel generator operation
        
        #Electric
        self.peak_sun_hours = kwargs.get("peak_sun_hours")
        self.performance_ratio = 0.8
        self.panel_cost_per_kwp = 1000 #€/kWp
        self.efficiency_battery = 0.85
        self.depth_of_discharge = 0.7
        self.battery_cost_per_kwh = 236*0.84 #€/kWh (https://doi.org/10.1186/s13705-024-00480-1)
        
        
        
        self.fixed_heights = self.fixed_heights + self.water_tower_height
        self.head = np.zeros(self.n_new_nodes+len(self.fixed_nodes)) # Head at each node (initialise to zero)
        for i,h in enumerate(self.fixed_heights):
            self.head[i] = 0 # Assume fixed pumps have 0 head initially
        
        
        self.pipe_data_storage = {}  # Dict storage: key = tuple of X, value = graph
        self.chaining_penalty = {}
        
        
    def pipe_and_pump(self, G):
        
        for i in self.fixed_nodes:
            # perform breadth first search from each fixed pump
            for parent,child in nx.bfs_edges(G, i):
                max_q_pipe = (len(nx.descendants(G, child))+1)*self.base_flow/1000  #0.3 L/s per fountain
                G[parent][child]['flow_rate'] = max_q_pipe
                length_pipe = G[parent][child]['weight']
                
                alt1 = G.nodes[parent]['height']
                head_1 = G.nodes[parent]['head']
                    
                alt2 = G.nodes[child]['height']
                min_pipe_cost, diameter_pipe, outlet_head, pump_power, pump_head=pipe_calcs(
                    alt1,alt2,length_pipe,max_q_pipe,self.pipe_costs,self.pump_cost_per_watt, head_1)
                
                # Give node and edge attributes for the pressure, pipes etc.
                G.add_nodes_from([(child,{"head":outlet_head})])
                G.add_edges_from([(parent,child,{"diameter":diameter_pipe,
                                                    "pump_power":pump_power,"pump_head":pump_head,"pipe_cost":min_pipe_cost})])
        return G
    
    def calculate_running_costs(self, G):
        pumping_costs = 0
        edges = G.edges(data=True)     
        for edge in edges:
            pipe_vol = 0
            volume_pumped=0
            edge_costs = 0
            for flow in edge[2]['flow_schedule']:
                t = flow['duration']
                Q = flow['flow_multiplier']*self.base_flow/1000 # m3/s
                volume_pumped = Q*t # m3
                pipe_vol += volume_pumped
                
            pressure = edge[2]['pump_head']*9.81*1000 # Pa
            E = pressure*volume_pumped/0.7/3600000 # kWh, assuming 70% pump efficiency
            
            if self.pumping_method == "Electric":
                # Calculate electric pumping costs
                pv_power = E/(self.peak_sun_hours*self.performance_ratio) #kWp
                cost_pv = pv_power*self.panel_cost_per_kwp #€
                
                battery_capacity = 1.5*E/(self.efficiency_battery*self.depth_of_discharge) #kWp
                cost_battery = battery_capacity*self.battery_cost_per_kwh #€
                
                discounted_battery_costs = cost_battery+np.sum([cost_battery/((1+self.discount_rate)**year) for year in np.arange(5,21,4)]) # (replacement every 4 years + initial)
                
                total_cost = cost_pv + discounted_battery_costs
                edge_costs += total_cost
                pumping_costs += total_cost

            elif self.pumping_method == "Diesel":
                # Calculate diesel pumping costs
                annual_cost = (E*self.specific_consumption*self.diesel_price + E*self.oandm)*365 # €/year
                discounted_running_costs = np.sum([annual_cost/((1+self.discount_rate)**year) for year in range(1,21)]) # 20 year lifespan
                
                generator_cost = E/3*self.generator_cost_per_kw
                
                total_cost = discounted_running_costs + generator_cost
                edge_costs += total_cost
                pumping_costs += total_cost

            G.add_edge(*edge[0:2],pumping_cost = edge_costs)
            

        return pumping_costs, G
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        x format:
        [parent_0, parent_1, ..., parent_{n-1}, x_0, y_0, ..., x_{n-1}, y_{n-1}]
        """
        m = len(self.fixed_nodes)
        n = self.n_new_nodes
        parents = X[:n].astype(int)
        coords = X[n:3*n].reshape(n,2)
        converted_fixed_pumps = 0
        pipe_data = []   # Each graph for each solution
        self.fixed_costs = np.array([0 for _ in self.fixed_nodes]+[self.cost_fountain for _ in range(self.n_new_nodes)])

        
        # Build graph and check validity
        G = nx.DiGraph()
        all_nodes = list(self.fixed_nodes) + [f"N{i}" for i in range(n)]
        G.add_nodes_from(all_nodes)
        
        
        # Add edges parent->child for new nodes with length as weight
        for i, p in enumerate(parents):
            child = f"N{i}"
            if p < len(self.fixed_nodes):
                parent = self.fixed_nodes[p]
                parent_coord = self.fixed_coords[p]
                self.fixed_costs[p] = self.cost_conversion
                converted_fixed_pumps += 1
            else:
                parent = f"N{p - len(self.fixed_nodes)}"
                parent_coord = coords[p - len(self.fixed_nodes)]
            
            coord_new_node = coords[i]
            length = great_circle(lonlat(*parent_coord), lonlat(*coord_new_node)).meters
            G.add_edge(parent, child, weight=length)
            
        if n >= self.fountains_retrofitted:
            if converted_fixed_pumps != self.fountains_retrofitted:
                out["F"] = [1e6, 1e6]  # large penalty
                return
                
        # Check acyclic
        if not nx.is_directed_acyclic_graph(G):
            out["F"] = [1e6, 1e6]  # large penalty
            return
        
        # Check connectivity: all new nodes connected to fixed nodes
        for i in range(n):
            child = f"N{i}"
            if not any(nx.has_path(G, fn, child) for fn in self.fixed_nodes):
                out["F"] = [1e6, 1e6]
                return
            
        # Assign heights, fixed costs and known heads to nodes
        self.height_new_nodes = np.asarray([self.altitude_interpolator(i) for i in coords])
        self.all_heights = np.concatenate((self.fixed_heights,self.height_new_nodes))
        for i, h in enumerate(self.all_heights):
            G.nodes[all_nodes[i]]['height'] = h
            G.nodes[all_nodes[i]]['head'] = self.head[i]
            G.nodes[all_nodes[i]]['fixed_cost'] = self.fixed_costs[i]
            G.nodes[all_nodes[i]]['coords'] = self.fixed_coords[i] if i < m else coords[i - m]
        
        # Display initial graph with all data
        # self._display_initial_graph(G, all_nodes)

        # Compute weighted sum of distances to points in pos_households with all nodes
        f1 = self.impact_fn(self.fixed_coords,self.house_coords,self.house_weights,coords)  # Negative impact calculation with new x location


        cons_gen = consumption_generator(
            profile_file=r'src\data\usage_profile2.xlsx',
            nb_capita=self.house_weights,
            G=G,
            household_positions=self.house_coords,
            base_flow=self.base_flow)
        
        G = cons_gen.assign_consumption_to_graph()
        
        # Compute cost f2
        f2 = 0
        updated_graph = self.pipe_and_pump(G)
        
        pumping_costs, updated_graph = self.calculate_running_costs(updated_graph)
        f2 += pumping_costs
        
        f2 += np.sum(self.fixed_costs)
        f2 += sum(nx.get_edge_attributes(updated_graph,'pipe_cost').values())
        
        
        f1 = float(f1)
        f2 = float(f2)
        # print(X)
        # print(updated_graph.nodes['N0'])
        
                
        # Store graph in dictionary, keyed by solution X
        solution_key = tuple(X.flatten())
        self.pipe_data_storage[solution_key] = nx.freeze(updated_graph.copy())
        
        out["F"] = [f1, f2]

    def _display_initial_graph(self, G, all_nodes):
        """Display the initial graph with all node and edge data"""
        plt.close()
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Create layout using geographic coordinates scaled by edge weights
        pos = {}
        for node in G.nodes():
            coords = G.nodes[node].get('coords', (0, 0))
            pos[node] = np.array(coords)
        
        # Scale positions based on edge lengths to make arrow length proportional to weight
        # Find the maximum distance between any two connected nodes to normalize
        max_distance = 0
        distances = {}
        for u, v, data in G.edges(data=True):
            dist = np.linalg.norm(pos[u] - pos[v])
            distances[(u, v)] = dist
            max_distance = max(max_distance, dist)
        
        # Normalize positions so that edge lengths are proportional to their weight (length)
        if max_distance > 0:
            scale_factor = 1.0 / max_distance
            for node in pos:
                pos[node] = pos[node] * scale_factor
        
        # Draw nodes
        node_colors = []
        for node in G.nodes():
            node = str(node)
            if node.startswith('F'):
                node_colors.append('red')  # Fixed pumps in red
            else:
                node_colors.append('lightblue')  # New fountains in light blue
        
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=800,
            alpha=0.9,
            linewidths=2
        )
        
        # Draw edges with straight arrows
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            width=2,
            connectionstyle="arc3,rad=0"
        )
        
        # Create node labels with all node data
        node_labels = {}
        for node in G.nodes():
            node_data = G.nodes[node]
            height = node_data.get('height', 0)
            head = node_data.get('head', 0)
            fixed_cost = node_data.get('fixed_cost', 0)
            coords = node_data.get('coords', (0, 0))
            node_labels[node] = f"{node}\nH:{height:.1f}m\nHead:{head:.1f}m\nCost:€{fixed_cost:.0f}\nLon:{coords[0]:.4f}\nLat:{coords[1]:.4f}"
        
        nx.draw_networkx_labels(
            G, pos, node_labels, ax=ax,
            font_size=7,
        )
        
        # Create edge labels with all edge data
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            diameter = data.get('diameter', 'N/A')
            flow_rate = data.get('flow_rate', 'N/A')
            length = data.get('weight', 'N/A')
            pump_power = data.get('pump_power', 'N/A')
            pipe_cost = data.get('pipe_cost', 'N/A')
            
            if isinstance(flow_rate, (int, float)):
                flow_rate = f"{flow_rate*1000:.2f}L/s"
            if isinstance(diameter, (int, float)):
                diameter = f"{diameter:.4f}m"
            if isinstance(length, (int, float)):
                length = f"{length:.1f}m"
            if isinstance(pump_power, (int, float)):
                pump_power = f"{pump_power:.1f}W"
            if isinstance(pipe_cost, (int, float)):
                pipe_cost = f"€{pipe_cost:.2f}"
            
            edge_labels[(u, v)] = f"∅:{diameter}\nQ:{flow_rate}\nL:{length}\nPower:{pump_power}\nCost:{pipe_cost}"
        
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels, ax=ax,
            font_size=6
        )
        
        ax.set_title('Initial Graph - All Node and Edge Data', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        # Show and wait for window to close
        plt.show()
    
    def _enforce_minimum_distance(self, pos, min_distance=1.5):
        """Enforce minimum distance between nodes by repelling overlapping nodes"""
        import numpy as np
        
        nodes = list(pos.keys())
        positions = {node: np.array(pos[node]) for node in nodes}
        
        # Iteratively push nodes apart if they're too close
        for _ in range(10):  # Multiple iterations for convergence
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    p1 = positions[node1]
                    p2 = positions[node2]
                    dist = np.linalg.norm(p1 - p2)
                    
                    # If too close, push apart
                    if dist < min_distance and dist > 0:
                        direction = (p1 - p2) / dist
                        adjustment = (min_distance - dist) / 2
                        positions[node1] += direction * adjustment
                        positions[node2] -= direction * adjustment
        
        return {node: tuple(positions[node]) for node in nodes}

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
                                base_flow,
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
        base_flow=base_flow,
        **kwargs
    )
    xl, xu, var_items = attach_numeric_bounds(problem)
    
    # Callback to keep only pipe_data for non-dominated solutions (NDS)
    def keep_only_nds_pipe_data(algorithm):
        if algorithm.opt is not None and len(algorithm.opt) > 0 and len(problem.pipe_data_storage) > 0:
            # Get indices of non-dominated solutions from the full population
            nds_indices = set()
            for solution in algorithm.opt:
                # Find this solution in the population using closest match
                for i, pop_solution in enumerate(algorithm.pop):
                    if np.allclose(solution.X, pop_solution.X):
                        nds_indices.add(i)
                        break
            
            # Filter pipe_data_storage list to only keep NDS solutions
            filtered_storage = [problem.pipe_data_storage[i] for i in sorted(nds_indices) 
                              if i < len(problem.pipe_data_storage)]
            problem.pipe_data_storage = filtered_storage
    
    algorithm = NSGA2(
        pop_size=50,
        sampling=MixedSampler(),
        repair=RoundRepair(),
        mutation = PM(eta=5),
        crossover=SBX(eta=10,prob=0.9)
    )
    
    res = minimize(problem,
                algorithm,
                ('n_gen', kwargs["simulation_generations"]),
                seed=4,
                verbose=True,
    )
    
    # Retrieve pipe data ONLY for the final Pareto front solutions
    pipe_data_results = {}
    for X_solution in res.X:
        solution_key = tuple(X_solution.flatten())
        if solution_key in problem.pipe_data_storage:
            pipe_data_results[solution_key] = problem.pipe_data_storage[solution_key]
    
    return res, pipe_data_results

class InteractiveParetoPlot:
    def __init__(self, all_positions, all_result_vals, all_indices, 
                 households, pos_pumps, grid_x, grid_y, grid_z, initial_impact, pipe_data,impactfn,
                 max_nb_fountains, all_X=None):
        # Store data
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
        self.max_nb_fountains = max_nb_fountains
        self.all_X = all_X if all_X is not None else []
        
        # Create separate figures for each plot
        self.fig1, self.ax1 = plt.subplots(figsize=(10, 6))  # Pareto front
        self.fig1.suptitle('Pareto Front')
        
        self.fig2, self.ax2 = plt.subplots(figsize=(12, 10))  # Geographical map
        self.fig2.suptitle('Geographical Map')
        
        self.fig3, self.ax3 = plt.subplots(figsize=(10, 8))  # Network graph
        self.fig3.suptitle('Network Topology')
        
        # Store references to plot elements for highlighting
        self.pareto_scatter = None
        self.pump_scatters = []
        self.pump_lines = []
        self.highlighted_pumps = []
        self.highlighted_lines = []
        self.data_box = None  # Single data box that gets updated
        self.n_tested = np.arange(1,self.max_nb_fountains+1,dtype=int)
        
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
        if self.impactfn == "impact2":
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
                    label=f'{k} new fountains',
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
                    label=f'{k} new fountains',
                    marker=n_shape.get(k,'s'),
                )
                self.pareto_scatters.append(scatter)
        
            self.ax1.set_xlabel('Impact (thousand person-meters saved)', fontsize=12)
            
        self.ax1.set_ylabel('Cost (€)', fontsize=12)
        self.ax1.set_title('Pareto Front', fontsize=12)
        self.ax1.legend()
        self.ax1.grid(True)
        
        # === GEOGRAPHICAL PLOT (ax2) ===
        contour = self.ax2.contourf(self.grid_x, self.grid_y, self.grid_z, levels=5, cmap='terrain')
        
        # Plot households (black dots)
        self.ax2.scatter(self.households['Lon'], self.households['Lat'], color='black', label='Households')
        
        # Plot new fountains and store references
        pump_idx = 0  # Index to track pumps across all configurations
        
        for n, k in enumerate(self.n_tested):
            if self.impactfn == "impact2":
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
                        label='New fountains' if pump_idx == 0 else "",
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
                        label='New fountains' if pump_idx == 0 else "",
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
                                if x < len(self.all_positions[n]):
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
        
        # # Add colorbar
        # if len(self.pump_scatters) > 0:
        #     cbar = self.fig.colorbar(self.pareto_scatters[-1], ax=self.ax2)
        #     if self.impactfn=="impact2":
        #         cbar.set_label('% of people within 30 mins')
        #     else:
        #         cbar.set_label('Impact (thousand person-meters)')
        
        # Labels and title
        self.ax2.set_xlabel('Longitude')
        self.ax2.set_ylabel('Latitude')
        self.ax2.set_title('Households and Pumps')
        self.ax2.legend()
        self.ax2.grid(True)
        
        self.fig1.tight_layout()
        self.fig2.tight_layout()
        self.fig3.tight_layout()
    
    def connect_events(self):
        """Connect mouse events to the Pareto plot figure"""
        self.fig1.canvas.mpl_connect('motion_notify_event', self.on_hover)
    
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
    
    def get_pipe_data_text(self, config_idx, solution_in_config, solution_X=None):
        """Get formatted pipe data text for display"""
        try:
            # Access pipe_data by X key (solution_X required for dict-based storage)
            if config_idx < len(self.pipe_data):
                config_data = self.pipe_data[config_idx]
                
                # Dict-based storage: use X as key
                if isinstance(config_data, dict):
                    if solution_X is not None:
                        solution_key = tuple(solution_X.flatten())
                        for key in config_data.keys():
                            if np.allclose(solution_key, key):
                                solution_key = key
                                return config_data[solution_key]
                # Legacy list-based storage
                elif isinstance(config_data, list):
                    if solution_in_config < len(config_data):
                        return config_data[solution_in_config]
            
            return "No data available"
        except (IndexError, KeyError, TypeError) as e:
            print(f"Error retrieving pipe data: {e}")
            return "No data available"
    
    def highlight_solution(self, solution_idx):
        """Highlight the corresponding solution in the geographical plot"""
        self.clear_highlighting()
        
        # Find which configuration this solution belongs to
        cumulative_solutions = 0
        config_idx = None
        solution_in_config = None
        solution_X = None
        
        for n, k in enumerate(self.n_tested):
            num_solutions = len(self.all_result_vals[n])
            if cumulative_solutions <= solution_idx < cumulative_solutions + num_solutions:
                config_idx = n
                solution_in_config = solution_idx - cumulative_solutions
                solution_X = self.all_X[n][solution_in_config]
                break
            cumulative_solutions += num_solutions
        
        if config_idx is not None and solution_in_config is not None and solution_X is not None:
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
            self.show_data_box(config_idx, solution_in_config, solution_X)
            
            # Refresh all plots
            self.fig2.canvas.draw_idle()
            self.fig3.canvas.draw_idle()
            
    def get_node_families_dfs_by_level(self, G):
        """Get nodes grouped by family and level using DFS from root nodes."""
        # Find all root nodes
        root_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
        
        families = []
        
        for root in root_nodes:
            family_dict = {}
            visited = set()
            
            def dfs(node, level):
                if node not in visited:
                    visited.add(node)
                    # Add node to the level dictionary
                    if level not in family_dict:
                        family_dict[level] = []
                    family_dict[level].append(node)
                    
                    # Recursively visit successors at the next level
                    for successor in G.successors(node):
                        dfs(successor, level + 1)
            
            dfs(root, 0)
            if family_dict:
                families.append(family_dict)
        
        return families
    
    def show_data_box(self, config_idx, solution_in_config, solution_X=None):
        """Show data box with pipe information"""
        # Get the formatted data text
        data_graph = self.get_pipe_data_text(config_idx, solution_in_config, solution_X)
        families = self.get_node_families_dfs_by_level(data_graph)
        data_string = ""
        for i, family in enumerate(families):
            for level, nodes in family.items():
                for node in nodes:
                    node = str(node)
                    if node.startswith('N'):
                        node_data = data_graph.nodes[node]
                        parent = list(data_graph.predecessors(node))[0]
                        edge_data = data_graph.get_edge_data(parent, node)
                        
                        
                        lon, lat = node_data['coords']
                        fountain_head = node_data['head']
                        diameter = edge_data['diameter']
                        pump_power = edge_data['pump_power']
                        pump_head = edge_data['pump_head']
                        flow_rate = edge_data['flow_rate']*1000  # convert to L/s
                        
                        data_string += f"Lon: {lon:.3f}, Lat: {lat:.3f}, Fountain head: {fountain_head:.3f}m, \n Diameter: {diameter:.3f}m, Power: {pump_power:.3f}W, Head Change: {pump_head:.3f}m, Flow Rate: {flow_rate:.3f}L/s; \n"
        
        
        # Create box properties
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black')
        # print(solution_X)
        
        # Add solution info to the text
        k = self.n_tested[config_idx]
        header = f"Solution {self.current_hover_idx}\n{k} Fountain{'s' if k > 1 else ''}\n---\n"
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
        
        # Draw the network graph
        self.draw_network_graph(config_idx, solution_in_config, solution_X)
    
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
        
        # Clear graph plot
        self.ax3.clear()
        
        # Refresh all plots
        self.fig2.canvas.draw_idle()
        self.fig3.canvas.draw_idle()
    
    def draw_network_graph(self, config_idx, solution_in_config, solution_X=None):
        """Draw the network graph on ax3"""
        # Clear previous graph
        self.ax3.clear()
        
        # Get the graph
        graph = self.get_pipe_data_text(config_idx, solution_in_config, solution_X)
        
        # 1. Start with your geographic positions
        initial_pos = {node: np.array(graph.nodes[node].get('coords', (0, 0))) for node in graph.nodes()}

        # 2. Use spring_layout to 'nudge' nodes apart
        # k=0.1 to 0.5 controls the distance between nodes
        pos = nx.spring_layout(
            graph, 
            pos=initial_pos, 
            fixed=None,      # Allow all nodes to move slightly
            k=0.3,           # Increase this if labels still overlap
            iterations=50
            )
        
        if graph is None or len(graph.nodes()) == 0:
            self.ax3.text(0.5, 0.5, "No graph data", ha='center', va='center')
            self.ax3.set_xlim(0, 1)
            self.ax3.set_ylim(0, 1)
            return
        
        # Draw nodes
        node_colors = []
        for node in graph.nodes():
            node = str(node)
            if node.startswith('F'):
                node_colors.append('red')  # Fixed pumps in red
            else:
                node_colors.append('lightblue')  # New fountains in light blue
        
        nx.draw_networkx_nodes(
            graph, pos, ax=self.ax3,
            node_color=node_colors,
            node_size=500,
            alpha=0.9,
            linewidths=2
        )
        
        
        # Draw edges with straight arrows
        nx.draw_networkx_edges(
            graph, pos, ax=self.ax3,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle="arc3,rad=0"
        )
        # Create node labels with all node data
        node_labels = {}
        for node in graph.nodes():
            node_data = graph.nodes[node]
            head = node_data.get('head', 0)
            fixed_cost = node_data.get('fixed_cost', 0)
            coords = node_data.get('coords', (0, 0))
            node_labels[node] = f"{node}\nHead:{head:.1f}m\nCost:€{fixed_cost:.0f}\nLon:{coords[0]:.4f}\nLat:{coords[1]:.4f}"
        
        nx.draw_networkx_labels(
            graph, pos, node_labels, ax=self.ax3,
            font_size=7,
        )
        
        # Create edge labels with all edge data
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            diameter = data.get('diameter', 'N/A')
            flow_rate = data.get('flow_rate', 'N/A')
            length = data.get('weight', 'N/A')
            pump_power = data.get('pump_power', 'N/A')
            pipe_cost = data.get('pipe_cost', 'N/A')
            pumping_cost = data.get('pumping_cost', 'N/A')
            
            
            
            if isinstance(flow_rate, (int, float)):
                flow_rate = f"{flow_rate*1000:.2f}L/s"
            if isinstance(diameter, (int, float)):
                diameter = f"{diameter:.4f}m"
            if isinstance(length, (int, float)):
                length = f"{length:.1f}m"
            if isinstance(pump_power, (int, float)):
                pump_power = f"{pump_power:.1f}W"
            if isinstance(pipe_cost, (int, float)):
                pipe_cost = f"€{pipe_cost:.2f}"
            if isinstance(pumping_cost, (int, float)):
                pumping_cost = f"€{pumping_cost:.2f}"
            
            edge_labels[(u, v)] = f"∅:{diameter}\nQ:{flow_rate}\nL:{length}\nPower:{pump_power}\nPipe Cost:{pipe_cost}\n Pumping Cost:{pumping_cost}"
        
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels, ax=self.ax3,
            font_size=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="grey", alpha=0.6, ec="none"),
            label_pos=0.5,
            rotate=False,
        )
        
        
        self.ax3.set_title('Pipe Network Topology', fontsize=12, fontweight='bold')
        self.ax3.axis('off')
    
    def show(self):
        """Display the interactive plot"""
        plt.show()

## Main function

entry_widgets = {}

def save_output(all_positions, all_result_vals, all_indices, all_X, max_nb_fountains, pipe_data, initial_impact):
    # x format:
    #     [parent_0, parent_1, ..., parent_{n-1}, x_0, y_0, ..., x_{n-1}, y_{n-1}]
    # all positions is [[[x_0,y_0,...],[for each solution of n=1],...],[for each solution of n=max_nb_fountains etc]]
    
    # Create output folder if it doesn't exist
    output_dir = 'src/Output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp once for all files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create Excel writer object
    excel_filename = f'{output_dir}/{timestamp}_data_output.xlsx'
    pkl_filename = f'{output_dir}/{timestamp}_pipe_data_output.pkl'
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        sheet_name = f'data_summary'
        df_summary = pd.DataFrame({'initial_impact': [initial_impact]})
        df_summary.to_excel(writer, sheet_name=sheet_name, index=False)
        for i in range(max_nb_fountains):
            n_fountains = i + 1
            positions = all_positions[i]
            result_vals = all_result_vals[i]
            indices = all_indices[i]
            Xs = all_X[i]
            
            # Build header
            header = []
            # Parent indices
            header.extend([f'Fountain_{j+1}_Parent_index' for j in range(n_fountains)])
            # Longitudes and Latitudes
            header.extend([f'Fountain_{j+1}_{coord}' for j in range(n_fountains) for coord in ['Lon', 'Lat']])
            # Impact and Cost
            header.extend(['Impact', 'Cost (€)'])
            
            # Build data rows
            data_rows = []
            for k in range(len(Xs)):
                row = []
                # Parent indices for this solution
                for j in range(n_fountains):
                    row.append(indices[k,j])
                    # row.append(Xs[k][j])
                # Longitudes and Latitudes for this solution
                for j in range(n_fountains):
                    row.append(positions[k,2*j])
                    row.append(positions[k,2*j+1])
                    # row.append(Xs[k][n_fountains + 2*j])      # Lon
                    # row.append(Xs[k][n_fountains + 2*j + 1])  # Lat
                # Impact and Cost
                row.extend([result_vals[k][0], result_vals[k][1]])
                data_rows.append(row)
            
            # Create DataFrame and write to Excel
            df = pd.DataFrame(data_rows, columns=header)
            sheet_name = f'{n_fountains}_fountains'
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    with open(pkl_filename, 'wb') as pkl_file:
        pickle.dump(pipe_data, pkl_file)
    return

def setup_data(file_path):
    # Import data and define arrays (may need to rewrite read function depending on input)
    data,households,pumps,open_wells = read_data_gogma(file_path)
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
    
    return households, pumps, pos_households, nb_capita, pos_pumps, bounds, grid_x, grid_y, grid_z, get_alt

def get_psh(lat, lon, start_year=2018, end_year=2023):
    """
    Fetches monthly solar radiation data from the PVGIS API.
    """
    url = "https://re.jrc.ec.europa.eu/api/MRcalc"
    
    # Define parameters for the API call
    params = {
        'lat': lat,
        'lon': lon,
        'horirrad': 1,        # Global horizontal irradiation
        'optrad': 1,          # Irradiation at optimal tilt
        'startyear': start_year,
        'endyear': end_year,
        'outputformat': 'json'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()
        
        # Extract the monthly data series
        monthly_data = data['outputs']['monthly']
        df = pd.DataFrame(monthly_data)
        
        # Calculate Peak Sun Hours (PSH)
        # PVGIS returns kWh/m2/month
        # PSH = (Monthly Total) / (Days in Month)
        # Note: 1 kWh/m2 is equivalent to 1 PSH because PSH is defined at 1kW/m2.
        month_days = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
        worst_month = df.loc[df['H(i_opt)_m'].idxmin()]
        psh = worst_month['H(i_opt)_m']/month_days[int(worst_month['month'])]
        
        return psh

    except Exception as e:
        return f"Error: {e}"

def run_optimisation_and_plot():
    
    """
    Setup, optimisation and plotting for the scenario with the given filename
    """
    try:       
        
        labour_and_fixed_pipe = 3 #€/m
        base_flow = 0.3 #L/s
        
        gui_kwargs = {
            "cost_conversion" : float(entry_widgets["Conversion Cost (€)"].get()),
            "cost_fountain" : float(entry_widgets["Fountain Cost (€)"].get()),
            "pump_cost_per_watt" : float(entry_widgets["Pump Cost (€/W)"].get()),
            "simulation_generations": int(entry_widgets["Number of Generations"].get()),
            "pipe_costs" : { # keys are pipe diameters, values are pipe costs €/m (source https://www.siobati.com/boutique/tuyau-pression-pvc-pn10/)
                0.032: 1.27+labour_and_fixed_pipe,
                0.04: 1.78+labour_and_fixed_pipe,
                0.05: 3.30+labour_and_fixed_pipe,
                0.063: 2.90+labour_and_fixed_pipe,
                0.09: 5.59+labour_and_fixed_pipe,
            },
            "water_tower_height" : float(entry_widgets["Water Tower Height (m)"].get()),
            "fountains_retrofitted" : float(entry_widgets["Fountains Retrofitted"].get()),
            "pumping_method": pumping_dict[entry_widgets["Pumping Method"].get()]
        }
        
        impactfn = impact_dict[entry_widgets["Impact Function"].get()]
        max_nb_fountains = int(entry_widgets["Maximum Number of New Fountains"].get())
    
        # Setup data
        
        households, pumps, pos_households, nb_capita, pos_pumps, bounds, grid_x, grid_y, grid_z, get_alt = setup_data(r'..\DO_NOT_DISTRIBUTE\DO_NOT_DISTRIBUTE_DATA_GOGMA.xlsx')
        
        gui_kwargs["peak_sun_hours"] = get_psh(pos_pumps[0,1], pos_pumps[0,0]) # take the first pump position
        
        # Run optimiser
        all_result_vals = []
        all_indices = []
        all_positions = []
        all_X = []
        pipe_data = []
        for i in range(1,max_nb_fountains+1):
            res, res_pipe_data = calculate_optimal_placement(pumps.index.to_list(), pos_pumps,
                                            pumps['Altitude'].to_numpy(), pos_households,
                                            nb_capita, i, bounds, impactfn, get_alt, base_flow,
                                            **gui_kwargs)
            all_result_vals.append(res.F)
            all_indices.append(res.X[:,:i])
            all_positions.append(res.X[:,i:i+2*i])
            all_X.append(res.X)
            pipe_data.append(res_pipe_data)
            # print(res.X[0])
            # print(res_pipe_data[tuple(res.X[0].flatten())].nodes['N0'])
            
        initial_impact = impactfn(pos_pumps, pos_households, nb_capita)
            
        
        save_output(all_positions, all_result_vals, all_indices,all_X, max_nb_fountains, pipe_data, initial_impact)
        
        
        # Set up plot
        interactive_plot = InteractiveParetoPlot(
            all_positions, all_result_vals, all_indices,
            households, pos_pumps, grid_x, grid_y, grid_z, 
            initial_impact,
            pipe_data, impactfn, max_nb_fountains, all_X
        )

        # Show the plot
        interactive_plot.show()    
    
    except ValueError as e:
            messagebox.showerror("Invalid Input", "Please enter valid numbers.")
            # 1. Get the full traceback as a formatted string
            error_details = traceback.format_exc()
            
            # 2. Display the error message AND the traceback details
            messagebox.showerror(
                "Fatal Error in Optimization",
                f"An error occurred: {e}\n\n--- Traceback Details ---\n{error_details}"
            )
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        # 1. Get the full traceback as a formatted string
        error_details = traceback.format_exc()
        
        # 2. Display the error message AND the traceback details
        messagebox.showerror(
            "Fatal Error in Optimization",
            f"An error occurred: {e}\n\n--- Traceback Details ---\n{error_details}"
        )


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
        "Conversion Cost (€)" : 0,
        "Fountain Cost (€)" : 700,
        "Pump Cost (€/W)" : 1, # €/W Estimated! Needs verifying
        "Number of Generations": 5,
        "Maximum Number of New Fountains": 2,
        "Water Tower Height (m)" : 4,
        "Fountains Retrofitted": 1,
    }
    root,main_frame = create_gui()
    row_num=0 
    # entry_widgets = {}
    # row_num = 0 
    
    # 1. Generate numerical entries from kwargs
    for k, v in default_kwargs.items():
        create_parameter_entry(main_frame, k, v, row_num, entry_widgets)
        row_num += 1

    # 2. Add 'Impact Function' Dropdown
    entry_widgets['Impact Function'] = tk.StringVar(root)
    entry_widgets['Impact Function'].set(list(impact_dict.keys())[0])
    tk.Label(main_frame, text='Impact Function').grid(row=row_num, column=0, sticky="w")
    dropdown_impact = tk.OptionMenu(main_frame, entry_widgets['Impact Function'], *list(impact_dict.keys()))
    dropdown_impact.grid(row=row_num, column=1)
    row_num += 1 # Increment to the next row!

    # 3. Add 'Pumping Method' Dropdown (Using your new dictionary)
    entry_widgets['Pumping Method'] = tk.StringVar(root)
    entry_widgets['Pumping Method'].set(list(pumping_dict.keys())[0])
    tk.Label(main_frame, text='Pumping Method').grid(row=row_num, column=0, sticky="w")
    dropdown_pump = tk.OptionMenu(main_frame, entry_widgets['Pumping Method'], *list(pumping_dict.keys()))
    dropdown_pump.grid(row=row_num, column=1)
    row_num += 1

    # 4. Run button (placed at the bottom)
    run_button = tk.Button(main_frame, text="Run Optimisation", command=run_optimisation_and_plot)
    run_button.grid(row=row_num, columnspan=2, pady=10)
    
    root.mainloop()


# Run code
if __name__ == "__main__":
    main()