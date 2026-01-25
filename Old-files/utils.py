import pandas as pd
import requests

def read_data(file_path):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    households = data[data['Type'] == 'Household']
    pumps = data[data['Type'] == 'Hand Pump']
    open_wells = data[data['Type'] == 'Open Well']

    return data, households, pumps, open_wells

def get_elevation(lat, long):
    api_url = ('https://api.open-meteo.com/v1/elevation'
            f'?latitude={lat}&longitude={long}')
    try:
        # Make the API request
        response = requests.get(api_url)

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Parse the JSON response into a Python dictionary
        data = response.json()

        # Access the 'results' list
        results = data.get("elevation")
        print(results)

        # Check if the list is not empty
        if results:
            # # Get the first item in the results list (if you expect only one)
            # first_result = results[0]

            # # Extract the 'elevation' value
            elevation = results[0]#first_result.get("elevation")
            print(elevation)
        else:
            print("No results found in the API response.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    return elevation

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

def add_elevations_to_gogma(file_path):
    data_households, data_sources = pd.read_excel(
        file_path,
        sheet_name=['Position surveyed household (hh','Position water sources'],
        header=None).values()
    
    data_households.columns = ['ID','Lat','Lon','Altitude']
    data_sources['Altitude'] = 0
    alts = []
    for i,row in data_households.iterrows():
        alts.append(get_elevation(row['Lat'],row['Lon']))
        print(alts)
    
    data_households['Altitude'] = alts
        
    data_households.to_excel(r'src\data\altitudes_households.xlsx',sheet_name='Position surveyed household (hh',
                             header=False)
    
    # data_sources = data_sources.iloc[:,:4]
    # data_sources.columns = ['ID','Name','Lon','Lat']
    # data_sources['Altitude'] = 0
    # alts = []
    # for i,row in data_sources.iterrows():
    #     alts.append(get_elevation(row['Lat'],row['Lon']))
    #     print(alts)
        
    # data_sources['Altitude'] = alts
    # data_sources.to_excel(r'src\data\altitudes_sources.xlsx',sheet_name='Position water sources',
    #                          header=False)
    
    
# file = r'src\data\DO_NOT_DISTRIBUTE_DATA_GOGMA.xlsx'
# d,h,p,w = read_data_gogma(file)
# print(d)
