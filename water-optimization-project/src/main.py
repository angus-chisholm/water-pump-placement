import pandas as pd

def main():
    # Load data from CSV
    data_file = 'water-optimization-project/data/Map_village_20241227_data.csv'

    # Define potential new locations for water sources
    potential_locations = pd.DataFrame({
        'Type': ['Potential'] * 5,
        'Lon': [-11.432, -11.433, -11.434, -11.435, -11.436],
        'Lat': [10.980, 10.981, 10.982, 10.983, 10.984],
        'Altitude': [370, 371, 372, 373, 374],
        'Nb capita': [0] * 5,
        'Drink': [0] * 5,
        'Cook': [0] * 5,
        'Hygiene': [0] * 5,
        'Laundry': [0] * 5,
        'Usage': [0] * 5
    })

    # Initialize optimization process
    optimal_boreholes, optimal_standpipes, served_people = optimize_water_sources(
        data_file, potential_locations, max_distance=800, cost_borehole=5000, cost_standpipe=500, cost_per_meter=2
    )

    # Output results
    print("Optimal Placement and Type of Water Sources:")
    print("Boreholes:", optimal_boreholes)
    print("Standpipes:", optimal_standpipes)
    print("Served People:", served_people)

if __name__ == "__main__":
    main()


## max distance - 20 minute round trip at 12.5 min/km
## costs - in USD and given by Gemini - so likely inaccurate