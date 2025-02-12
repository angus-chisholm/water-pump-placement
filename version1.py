# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:33:25 2025

@author: angus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from gurobipy import *

data = np.array([[1,3],[4,1]])

plt.scatter(data[:,0],data[:,1])
plt.xlim(0,5)
plt.ylim(0,5)
plt.show()

def distance(p1,p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


costs = np.array([100,50,10])

m = Model("Point optimisation")

#create variables
X = m.addMVar(shape = (2), vtype = GRB.CONTINUOUS,name="Coords")
P = m.addVar(vtype=GRB.BINARY, name = "type")

#add constraints
xlims = [0,5]
ylims = [0,5]

c1 = m.addConstr(X >= np.array([xlims[0],ylims[0]]), name = "lim lower")
c2 = m.addConstr(X <= np.array([xlims[1],ylims[1]]), name = "lim upper")

#add objective
distances = []
for i in range(np.shape(data)[0]):
    distances.append(distance(X,data[i,:]))
    
distances = np.array(distances)

point = []
length = 

Obj_fn = costs[0]*P +costs[1]*(1-P)+costs[2]*(1-P)*length

m.setObjective(,GRB.MINIMIZE)

# -- Choix d'un paramétrage --
m.params.outputflag = 0 # mode muet


# -- Mise à jour du modèle  --
m.update()


# -- Affichage en mode texte du PL --
print(m.display())

# Résolution
m.optimize()

# Affichage de la solution optimale 
print("coords = {}".format(X.x,m.objVal))