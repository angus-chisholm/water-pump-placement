import pandas as pd
from optimization import optimize_water_sources

def main():
    # Load data from CSV
    data_file = 'data/Map_village_20241227_data.csv'
    data = pd.read_csv(data_file)

    # Initialize optimization process
    optimal_sources = optimize_water_sources(data, max_distance=800, cost_borehole=5000, cost_standpipe=500, cost_per_meter=2)

    # Output results
    print("Optimal Placement and Type of Water Sources:")
    for source in optimal_sources:
        print(source)

if __name__ == "__main__":
    main()


## max distance - 20 minute round trip at 12.5 min/km
## costs - in USD and given by Gemini - so likely inaccurate 