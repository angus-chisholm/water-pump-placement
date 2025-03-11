import pandas as pd

def read_data(file_path):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    households = data[data['Type'] == 'Household']
    pumps = data[data['Type'] == 'Hand Pump']
    return data, households, pumps

def calculate_distance_sq(lat1, lon1, lat2, lon2):
    '''Calculate sqaured distance between 2 points - euclidian 
    (using lat,lon = x, y as distances are small)'''
    
    return (lat1-lat2)**2+(lon1-lon2)**2

def calculate_distance(lat1, lon1, lat2, lon2):
    from geopy.distance import great_circle
    
    # Calculate the distance between two geographical points
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)
    return great_circle(point1, point2).meters