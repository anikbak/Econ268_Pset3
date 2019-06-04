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
#import quantecon

alpha = 0.64        #labor share
beta = 0.98267      #discount factor 
gamma = 0.4         #Frisch elasticity
w = 1               #wage
r = 0.01            #interest rate
amax=20             #maximal assets
amin=-2             #borrowing limit
N = 200             #number of grid points for assets
M = 2               #number of grid points for productivity
h = 1/3             #hours worked
B = 166.3           #disutility of labor


# set up grid
a_grid = np.linspace(amin, amax, num=N)


#state transition matrix
pUE=0.5
pEU=0.038
b=0.15
L = pUE / (pUE + pEU)  # labor endowment normalized to 1
e_grid = np.array([b, 1 - (1 - L) / L * b])
Pi = np.array([[1 - pUE, pUE], [pEU, 1 - pEU]])


def backward_VFI(VE_p, VU_p, Pi_p, a_grid, e_grid, r, w, beta, B, h, gamma):
    """Computes one backward iteration in the VFI algorithm
    Parameters
    ----------
    VE_p     : array (S*A), next period value function for employed
    VU_p     : array (S*A), next period value function for unemployed
    Pi_p     : array (S*S), Markov matrix for skills tomorrow
    a_grid   : array (A), asset grid
    e_grid   : array (A), skill grid
    r        : scalar, ex-post interest rate
    w        : scalar, wage
    beta     : scalar, discount rate today
    B        : disutility of labor
    h        : hours worked
    gamma    : Frisch elasticity

    Returns
    ----------
    V  : array (S*A), marginal value of assets today
    a  : array (S*A), asset policy today
    c  : array (S*A), consumption policy today
    """
    N = len(a_grid)
    M = len(e_grid)

    #initializing next period V
    V = np.zeros([N,M])
    apol = np.zeros([N,M])
    cpol = np.zeros([N,M])
    #computing consumption in each scenario
    a_tom = np.tile(a_grid,(N,1))
    a_tod = np.transpose(a_tom)
    atmp = (1+r)*a_tod - a_tom
    Cs = np.tile(w*e_grid*h,(N,N,1)) +  np.repeat(atmp[:, :, np.newaxis], M, axis=2)
    
    #imposing -Inf utility whenever there is negative consumption
    neg_c = (Cs <= 0)
    Cs[neg_c] = 10
    Us = np.log(Cs) - B*(h**(1 + 1/gamma))/(1+ 1/gamma)
    Us[neg_c] = -np.Inf

    ExpV = np.maximum(VE_p,VU_p)

    for i in range(M):        
        tomax = Us[:,:,i] +  np.tile(ExpV @ (beta * Pi_p[i,]),(N,1))
        max_locs = np.argmax(tomax,axis=1)
        apol[:,i] = a_grid[max_locs]
        cpol[:,i] = w*e_grid[i]*h + (1+r)*a_grid - apol[:,i]
        V[:,i] = np.max(tomax,axis=1)
        
    return V, apol, cpol



#optimization parameters
tol = 1e-6
maxiter = 1000

#initializing VFI 
dist = np.inf   #improvement step
it = 0          #iteration number

VE_p = np.zeros([N,M])     #initializing value function for employed
VU_p = np.zeros([N,M])     #initializing value function for employed

while dist>tol and it<maxiter:
    VE, apolE, cpolE = backward_VFI(VE_p, VU_p, Pi, a_grid, e_grid, r, w, beta, B, 1/3, gamma)
    VU, apolU, cpolU = backward_VFI(VE_p, VU_p, Pi, a_grid, e_grid, r, w, beta, B, 0, gamma)

    V_p = np.maximum(VE_p,VU_p)    
    dist = np.amax(abs(V_p - np.maximum(VE,VU)))
    VE_p = np.copy(VE)
    VU_p = np.copy(VU)
    if it%100==0:
        print("Iteration # " + str(it) + "- step = " + str(dist))
    it = it+1 
    
    
plt.subplot(111)
plt.plot(a_grid,VE_p[:,1],label="Employed")
plt.plot(a_grid,VU_p[:,1],label="Unemployed")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)    