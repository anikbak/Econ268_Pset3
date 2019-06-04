# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:07:44 2019

@author: chen-
"""
import ks2
import numpy as np
import matplotlib.pyplot as plt

delta=0.025
alpha=0.64
hbar=1/3

ss=ks2.ks_ss(delta=delta,alpha=alpha,hbar=hbar)

T=100

J=ks2.get_J(ss, T)
G=ks2.get_G(J, T)

#####Aggregate shock 
rho_z=0.95
sigma_z=0.007

dZ = sigma_z*rho_z**(np.arange(T))

irfs=ks2.td_linear(G, dZ, outputs=('r', 'w', 'K'))

irfs['L']=(((irfs['r']+ss['r']+delta)/alpha/np.exp(dZ))**(1/(1-alpha))*(irfs['K']+ss['K'])-ss['L'])/hbar

plt.figure()
plt.plot(irfs['r']/ss['r'])
plt.title(r'$r$ response to one standard deviation $Z$ shocks with $\rho=0.95$')
plt.savefig('ifs_r.png')

plt.figure()
plt.plot(irfs['w']/ss['w'])
plt.title(r'$w$ response to one standard deviation $Z$ shocks with $\rho=0.95$')
plt.savefig('ifs_w.png')

plt.figure()
plt.plot(irfs['K']/ss['K'])
plt.title(r'$K$ response to one standard deviation $Z$ shocks with $\rho=0.95$')
plt.savefig('ifs_K.png')

plt.figure()
plt.plot(irfs['L']/ss['L'])
plt.title(r'Labor Participation response to one standard deviation $Z$ shocks with $\rho=0.95$')
plt.savefig('ifs_L.png')