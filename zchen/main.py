# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:07:44 2019

@author: chen-
"""
import ks2
import numpy as np
import matplotlib.pyplot as plt

delta=0.025
alpha=0.36
hbar=1/3

ss=ks2.ks_ss(delta=delta,alpha=alpha,hbar=hbar,amax=20,a_lower=0)

T=100

J=ks2.get_J(ss, T)
G=ks2.get_G(J, T)

#####Aggregate shock 
rho_z=0.95
sigma_z=0.007

dZ = sigma_z*rho_z**(np.arange(T))

irfs=ks2.td_linear(G, dZ, outputs=('r', 'w', 'K', 'L' , 'Y'))


plt.figure()
plt.plot(ss['a_grid'][0:85],ss['a'][2,0:85])
plt.title('Policy Function: Next period assets')
plt.savefig('a_pol.png')

plt.figure()
plt.plot((irfs['r']/ss['r'])[1:50])
plt.title(r'$r$ response to one standard deviation $Z$ shocks with $\rho=0.95$')
plt.savefig('irf_r.png')

plt.figure()
plt.plot((irfs['w']/ss['w'])[1:50])
plt.title(r'$w$ response to one standard deviation $Z$ shocks with $\rho=0.95$')
plt.savefig('irf_w.png')

plt.figure()
plt.plot((irfs['K']/ss['K'])[1:50])
plt.title(r'$K$ response to one standard deviation $Z$ shocks with $\rho=0.95$')
plt.savefig('irf_K.png')

plt.figure()
plt.plot((irfs['L']/ss['L'])[1:50])
plt.title(r'Labor Participation response to one standard deviation $Z$ shocks with $\rho=0.95$')
plt.savefig('irf_L.png')

plt.figure()
plt.plot((irfs['Y']/ss['Y'])[1:50])
plt.title(r'Output response to one standard deviation $Z$ shocks with $\rho=0.95$')
plt.savefig('irf_Y.png')