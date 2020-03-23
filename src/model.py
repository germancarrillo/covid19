import os
import math
import itertools
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.optimize import curve_fit

#
def state_plotter(times, states, fig_num,titles):
    num_states = np.shape(states)[0]
    num_cols = int(np.ceil(np.sqrt(num_states)))
    num_rows = int(np.ceil(num_states / num_cols))
    plt.figure(fig_num)
    plt.clf()
    fig, ax = plt.subplots(num_rows, num_cols, num=fig_num, clear=True, squeeze=False)
    for n in range(num_states):
        row = n // num_cols
        col = n % num_cols
        ax[row][col].plot(times, states[n], 'k.:')
        ax[row][col].set(xlabel='Time',ylabel='$y_{:0.0f}(t)$ vs. Time'.format(n),title =titles[n])        
    for n in range(num_states, num_rows * num_cols): fig.delaxes(ax[n // num_cols][n % num_cols])
    print(row,col)
    ax[row][col].plot(sol.y[1:].sum(axis=0));
    fig.tight_layout()



# Define derivative function
def f(t, y, alpha_p, lambda_p ,mu_p):
    # parameters
    alpha_p=alpha_p[0]; lambda_p=lambda_p[0]; mu_p=mu_p[0]

    # populations
    Lambda_f,S,I,R,D = y[0],y[1],y[2],y[3],y[4]

    N = S+I+R+D
    Lambda_f = beta_p*I/N
    
    dSdt = -Lambda_f*S             - mu_p*S     
    dIdt = +Lambda_f*S - alpha_p*I - mu_p*I
    dRdt =             + alpha_p*I - mu_p*R 
    dDdt =                                  + mu_p*I + mu_p*S + mu_p*R
    
    dXdt = [Lambda_f,dSdt, dIdt, dRdt, dDdt]
    return dXdt




# Define time spans, initial values, and constants
tspan    = np.linspace(0, 1, 50)

Lambda_init = 0.5
S_init      = 89
I_init      = 10
R_init      = 1
D_init      = 0

yinit    = [Lambda_init,S_init,I_init,R_init,D_init]

alpha_p  = [30/52.]
lambda_p = [ 0.5  ]
mu_p     = [ 1/40.]

# Solve differential equation
sol = solve_ivp(lambda t, y: f(t, y, alpha_p, lambda_p, mu_p), [tspan[0], tspan[-1]], yinit, t_eval=tspan)

# Plot states
state_plotter(sol.t, sol.y, 1,['Lambda','Susceptibles','Infected','Recovered','Deaths'])

plt.show()
