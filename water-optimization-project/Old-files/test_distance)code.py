import gurobipy as gp
from gurobipy import GRB

m = gp.Model()

loc = [(358, 309), (275, 126), (424, 154)]
loads = [8, 5, 8]
cost = [4, 6, 5]
n = len(loc)

x, y = m.addVar(name="x"), m.addVar(name="y")
u = m.addVars(n, name="u")
xdiff = m.addVars(n, lb=-GRB.INFINITY, name="xdiff")
ydiff = m.addVars(n, lb=-GRB.INFINITY, name="ydiff")

m.setObjective(gp.quicksum(loads[i] * cost[i] * u[i] for i in range(n)), GRB.MINIMIZE)

for i in range(n):
    m.addConstr(xdiff[i] == loc[i][0] - x, name=f"define_xdiff[{i}]")
    m.addConstr(ydiff[i] == loc[i][1] - y, name=f"define_ydiff[{i}]")
    m.addConstr(u[i] ** 2 >= xdiff[i] ** 2 + ydiff[i] ** 2, name=f"define_u[{i}]")

m.optimize()
m.write('model-test.lp')

print("Cost:", m.ObjVal)
print(f"Optimal solution: x={x.X}, y={y.X}")