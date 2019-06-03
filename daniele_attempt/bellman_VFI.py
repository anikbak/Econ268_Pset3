# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 12:03:32 2019

@author: danic
"""
import numpy as np
from numba import njit
import scipy.optimize as opt
import scipy.linalg as linalg
import matplotlib.pyplot as plt

beta = 0.9
w = 1
r = 0.01
amax=20
amin=-2
N = 200
M = 2
# set up grid
a_grid = np.linspace(amin, amax, num=N)

V_old = np.zeros([N,M])     #initializing value function

#state transition matrix
pUE=0.5
pEU=0.038
b=0.15
L = pUE / (pUE + pEU)  # labor endowment normalized to 1
e_grid = np.array([b, 1 - (1 - L) / L * b])
Pi = np.array([[1 - pUE, pUE], [pEU, 1 - pEU]])

#computing consumption in each scenario
a_tom = np.tile(a_grid,(N,1))
a_tod = np.transpose(a_tom)
Cs = w*e_grid[0] + (1+r)*a_tod - a_tom
neg_c = (Cs <= 0)
Cs[neg_c] = 10
Us = np.log(Cs)
Us[neg_c] = -np.Inf
max_locs = np.argmax(Us +  np.tile(V_old @ (beta * Pi[0,]),(N,1)),axis=1)
apol = a_grid[max_locs]
cpol = w*e_grid[0] + (1+r)*a_grid - apol
V_new[0,] = np.log(cpol) + V_old @ (beta * Pi[0,])