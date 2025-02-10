import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np

# Charger les données de la feuille excel
file_path = 'Map_village_20241227_t.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

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

# Créer une grille pour le relief
grid_x, grid_y = np.mgrid[data['Lon'].min():data['Lon'].max():200j, 
                          data['Lat'].min():data['Lat'].max():200j]
grid_z = griddata((data['Lon'], data['Lat']), data['Altitude'], (grid_x, grid_y), method='cubic')

# Créer la figure
plt.figure(figsize=(12, 8))

# Tracer le relief en 2D
contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='terrain')
plt.colorbar(label="Altitude (m)")

# Tracer chaque type par-dessus
plt.scatter(household['Lon'], household['Lat'], c='blue', alpha=0.7, label='Household')
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
plt.title('Carte des installations avec relief', fontsize=16)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.grid(True)
plt.legend()

# Afficher le graphique
plt.show()
