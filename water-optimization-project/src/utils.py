def read_data(file_path):
    import pandas as pd
    
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    return data

def calculate_distance(lat1, lon1, lat2, lon2):
    from geopy.distance import great_circle
    
    # Calculate the distance between two geographical points
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)
    return great_circle(point1, point2).meters

def prepare_data(data):
    # Prepare the data for optimization
    households = data[data['Type'] == 'Household']
    water_sources = data[data['Type'] == 'Hand Pump']
    
    return households, water_sources

def filter_within_distance(households, water_sources, max_distance):
    # Filter households that are within a specified distance from water sources
    filtered_households = []
    
    for _, household in households.iterrows():
        for _, source in water_sources.iterrows():
            distance = calculate_distance(household['Lat'], household['Lon'], source['Lat'], source['Lon'])
            if distance <= max_distance:
                filtered_households.append(household)
                break
                
    return filtered_households