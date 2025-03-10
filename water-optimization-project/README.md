# Water Optimization Project

This project aims to optimize the placement and type of water sources in a village to minimize costs while maximizing the number of people served within a specified distance from a clean water source. The optimization is performed using Gurobi, a powerful optimization solver.

## Project Structure

```
water-optimization-project
├── data
│   └── Map_village_20241227_data.csv
├── src
│   ├── main.py
│   ├── optimization.py
│   └── utils.py
├── requirements.txt
└── README.md
```

### Data

- **data/Map_village_20241227_data.csv**: Contains data points for households and water sources, including coordinates, altitude, and water usage statistics.

### Source Code

- **src/main.py**: Entry point of the application. Handles user input for the data file, initializes the optimization process, and outputs the results, including the optimal placement and type of water sources.

- **src/optimization.py**: Contains the optimization logic using Gurobi. Defines the optimization model, including objective functions for minimizing costs and maximizing the impact (number of people served within a specified distance from a water source). Includes functions to set up decision variables and constraints based on the input data.

- **src/utils.py**: Includes utility functions for data processing, such as reading the CSV file, calculating distances between points, and preparing data for the optimization model.

### Requirements

To run this project, you need to install the following dependencies:

- Gurobi
- pandas
- numpy
- other necessary libraries

You can install the required packages by running:

```
pip install -r requirements.txt
```

### Usage

1. Place the `Map_village_20241227_data.csv` file in the `data` directory.
2. Run the main application using:

```
python src/main.py
```

3. Follow the prompts to input the necessary parameters for the optimization process.

### Optimization Problem Overview

The goal of this optimization project is to determine the optimal placement and type of new water sources in the village. The optimization model will minimize the costs associated with establishing these water sources while maximizing the number of households that can access clean water within a specified distance. The model will take into account various factors such as household water usage, distances to potential water sources, and the overall budget for the project.