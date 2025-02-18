import pandas as pd
import plotly.express as px
from optimization import optimize_water_sources

def main():
    # Load data from CSV
    data_file = 'water-optimization-project/data/Map_village_20241227_data.csv'

    # Define potential new locations for water sources
    potential_locations = pd.DataFrame({'Type': ['Potential'] * 5,
        'Lon': [-11.423, -11.428, -11.433, -11.438, -11.443],
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
    optimal_sources, impact, costs = optimize_water_sources(
        data_file, potential_locations, max_distance=800, cost_borehole=5000, cost_standpipe=500, cost_per_meter=2
    )

    # Create a DataFrame for the plot
    plot_data = pd.DataFrame({
        'Cost': costs,
        'Impact': [-i for i in impact],  # Negative impact
        'Optimal Sources': [str(sources) for sources in optimal_sources]
    })

    # Create an interactive plot
    fig = px.scatter(plot_data, x='Cost', y='Impact', hover_data=['Optimal Sources'])
    fig.update_layout(title='Cost vs. Impact with Optimal Sources', xaxis_title='Cost (USD)', yaxis_title='Negative Impact')
    fig.show()

    # Output results    
    print("Optimal Placement and Type of Water Sources:")
    print("Optimal Sources:", optimal_sources)
    print("People within max distance:", impact)
    print("Total Costs (USD):", round(sum(costs), 2))

if __name__ == "__main__":
    main()


## max distance - 20 minute round trip at 12.5 min/km
## costs - in USD and given by Gemini - so likely inaccurate