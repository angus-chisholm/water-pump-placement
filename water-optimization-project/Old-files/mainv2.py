import pandas as pd
import plotly.express as px
from optimizationv2 import optimize_water_sources

def main():
    # Load data from CSV
    data_file = 'C:/Users/angus/OneDrive/Documents/France/PVWPS Project/water-pump-placement/water-optimization-project/src/Map_village_20241227_data.csv'

    # Initialize optimization process
    optimal_pos, impact, costs = optimize_water_sources(
        data_file, max_distance=600, cost_borehole=5000, cost_standpipe=500, cost_per_meter=2
    )

    # Create a DataFrame for the plot
    plot_data = pd.DataFrame({
        'Cost': costs,
        'Impact': [-i for i in impact],  # Negative impact
        'Optimal Sources': optimal_pos
    })

    # Create an interactive plot
    fig = px.scatter(plot_data, x='Cost', y='Impact', hover_data=['Optimal Sources'])
    fig.update_layout(title='Cost vs. Impact with Optimal Sources', xaxis_title='Cost (USD)', yaxis_title='Negative Impact')
    fig.show()

    # Output results    
    print("Optimal Placement and Type of Water Sources:")
    print("Optimal Sources:", optimal_pos)
    print("People within max distance:", impact)
    print("Total Costs (USD):", round(sum(costs), 2))

if __name__ == "__main__":
    main()


## max distance - 20 minute round trip at 12.5 min/km
## costs - in USD and given by Gemini - so likely inaccurate
