import pandas as pd

def read_data(file_path):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    households = data[data['Type'] == 'Household']
    pumps = data[data['Type'] == 'Hand Pump']
    open_wells = data[data['Type'] == 'Open Well']

    return data, households, pumps, open_wells
