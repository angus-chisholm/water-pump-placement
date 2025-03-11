import pandas as pd
from gurobipy import *

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

print(potential_locations.shape[0])

for w_index in range(potential_locations.shape[0]):
    print(w_index)

model = Model("test")

# Decision variables: whether to place a borehole or standpipe at a given location
borehole_vars = model.addVars(potential_locations.shape[0], vtype=GRB.BINARY, name="Borehole")

for i in range(potential_locations.shape[0]):
    print(borehole_vars[i].x)