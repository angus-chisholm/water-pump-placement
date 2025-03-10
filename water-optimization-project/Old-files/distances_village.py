import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
import osmnx as ox
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Charger les données de la feuille excel
file_path = 'Map_village_20241227_t.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

#distinguish type of water source
conditions = [
    data['Type'].str.contains('Hand Pump'),
    data['Type'].str.contains('Open Well'),
    data['Type'].str.contains('Tank'),
]
values = [1, 2, 3]  # Corresponding values for each condition

data['Identifier'] = np.select(conditions, values, default=0)

# Extraire les coordonnées et types
types = data['Type']
latitudes = data['Lat']
longitudes = data['Lon']
altitudes = data['Altitude']

# Filtrer par type
household = data[data['Type'] == 'Household']
hand_pump = data[data['Type'] == 'Hand Pump']
open_well = data[data['Type'] == 'Open Well']
tank = data[data['Type'] == 'Tank']




# Trouver distance a la pompe la plus proche
distances = np.zeros((len(household),len(hand_pump)+len(open_well)+len(tank)))
for i in range(len(household)):
    household_longitude = household.loc[i]['Lon']
    household_latitude = household.loc[i]['Lat']
    for j in range(len(hand_pump)):
        pump_longitude = (hand_pump.iloc[j]).loc['Lon']
        pump_latitude = (hand_pump.iloc[j]).loc['Lat']
        distances[i,j] = ox.distance.great_circle(
            household_latitude,household_longitude,pump_latitude,pump_longitude,
            earth_radius=6371009)
    for k in range(len(open_well)):
        well_longitude = (open_well.iloc[k]).loc['Lon']
        well_latitude = (open_well.iloc[k]).loc['Lat']
        distances[i,k+len(hand_pump)] = ox.distance.great_circle(
            household_latitude,household_longitude,well_latitude,well_longitude,
            earth_radius=6371009)
    for l in range(len(tank)):
        tank_longitude = (tank.iloc[l]).loc['Lon']
        tank_latitude = (tank.iloc[l]).loc['Lat']
        distances[i,l+len(hand_pump)+len(open_well)] = ox.distance.great_circle(
            household_latitude,household_longitude,tank_latitude,tank_longitude,
            earth_radius=6371009)    
    
    
#separate water sources and household locations
water_sources = pd.concat([hand_pump.loc[:,['Identifier','Lon','Lat']],
                           open_well.loc[:,['Identifier','Lon','Lat']],
                           tank.loc[:,['Identifier','Lon','Lat']]],axis=0).to_numpy()
households = household.loc[:,['Type','Lon','Lat']].to_numpy()


def closest_water_source():
    #plot distance graph with households over the top
    vor = Voronoi(water_sources[:,1:])
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='black')
    
    ax.scatter(households[:,1],households[:,2],s = household.loc[:,'Nb capita']*3,c='blue',
               label = 'Households')
    ax.scatter(hand_pump.loc[:, ['Lon']].to_numpy(), 
               hand_pump.loc[:, ['Lat']].to_numpy(), 
               c='green', s=50, label='Hand Pump')
    ax.scatter(open_well.loc[:, ['Lon']].to_numpy(), 
               open_well.loc[:, ['Lat']].to_numpy(), 
               c='red', s=50, label='Open Well')
    ax.scatter(tank.loc[:, ['Lon']].to_numpy(), 
               tank.loc[:, ['Lat']].to_numpy(), 
               c='purple', s=50, label='Tank')
    
    
    # Ajouter des limites et graduations cohérentes
    rounded_lat_min = np.floor(data['Lat'].min() * 1000) / 1000  # Arrondi à 3 décimales
    rounded_lat_max = np.ceil(data['Lat'].max() * 1000) / 1000
    rounded_lon_min = np.floor(data['Lon'].min() * 1000) / 1000
    rounded_lon_max = np.ceil(data['Lon'].max() * 1000) / 1000
    plt.ylim(rounded_lat_min, rounded_lat_max)
    plt.xlim(rounded_lon_min, rounded_lon_max)
    
    # Ajouter des détails
    plt.title('Households and their closest water source', fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.grid(True)
    plt.legend()
    
    # Afficher le graphique
    plt.show()


def closest_pump():
    #plot distance graph with households over the top
    vor = Voronoi(hand_pump.loc[:,['Lon','Lat']].to_numpy())
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='black')
    
    ax.scatter(households[:,1],households[:,2],s = household.loc[:,'Nb capita']*3,c='blue',
               label = 'Households')
    ax.scatter(hand_pump.loc[:, ['Lon']].to_numpy(), 
               hand_pump.loc[:, ['Lat']].to_numpy(), 
               c='green', s=50, label='Hand Pump')
    
    
    # Ajouter des limites et graduations cohérentes
    rounded_lat_min = np.floor(data['Lat'].min() * 1000) / 1000  # Arrondi à 3 décimales
    rounded_lat_max = np.ceil(data['Lat'].max() * 1000) / 1000
    rounded_lon_min = np.floor(data['Lon'].min() * 1000) / 1000
    rounded_lon_max = np.ceil(data['Lon'].max() * 1000) / 1000
    plt.ylim(rounded_lat_min, rounded_lat_max)
    plt.xlim(rounded_lon_min, rounded_lon_max)
    
    # Ajouter des détails
    plt.title('Households and their closest hand pump', fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.grid(True)
    plt.legend()
    
    # Afficher le graphique
    plt.show()

#find distances to nearest tank or handpump
distances_to_pick = []
for i in range(np.shape(distances)[1]):
    if 0<=i and i<len(hand_pump):
        distances_to_pick.append(i)
    elif i> len(hand_pump)+len(open_well)-1:
        distances_to_pick.append(i)
  

    
def time_to_pump_plot():
    
                    
    # Créer une grille
    grid_x, grid_y = np.mgrid[household['Lon'].min():household['Lon'].max():200j, 
                              household['Lat'].min():household['Lat'].max():200j]
    grid_z = griddata((household['Lon'], household['Lat']), 
                      np.min(distances[:,distances_to_pick]*0.0125*2,
                      axis=1), (grid_x, grid_y), method='cubic')
    

    # Créer la figure
    plt.figure(figsize=(12, 8))

    # Distances comme contours
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='terrain')
    plt.colorbar(label="Time (mins)")

    # Tracer chaque type par-dessus
    plt.scatter(household['Lon'], household['Lat'], c='blue', alpha=0.7,s = household.loc[:,'Nb capita']*5, label='Household')
    plt.scatter(hand_pump['Lon'], hand_pump['Lat'], c='green', alpha=0.7, label='Hand Pump')
    plt.scatter(open_well['Lon'], open_well['Lat'], c='red', alpha=0.7, label='Open Well')
    plt.scatter(tank['Lon'], tank['Lat'], c='purple', alpha=0.7, label='Tank')

    # Ajouter des limites et graduations cohérentes
    rounded_lat_min = np.floor(data['Lat'].min() * 1000) / 1000  # Arrondi à 3 décimales
    rounded_lat_max = np.ceil(data['Lat'].max() * 1000) / 1000
    rounded_lon_min = np.floor(data['Lon'].min() * 1000) / 1000
    rounded_lon_max = np.ceil(data['Lon'].max() * 1000) / 1000
    plt.ylim(rounded_lat_min, rounded_lat_max)
    plt.xlim(rounded_lon_min, rounded_lon_max)

    # Ajouter des détails
    plt.title('Round trip time to walk to closest water pump (12.5min/km)', fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.grid(True)
    plt.legend()

    # Afficher le graphique
    plt.show()


def calcul_households_within_time(time):
    distance = time/(2*0.0125)
    nb_people_within_time = 0
    for i in range(len(distances)):
        if np.min(distances[:,distances_to_pick],axis=1)[i] < distance:
            nb_people_within_time += (household.iloc[i]).loc['Nb capita']
            
    return nb_people_within_time

print(calcul_households_within_time(15))
time_to_pump_plot()

    

