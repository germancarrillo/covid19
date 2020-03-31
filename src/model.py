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
from scipy.optimize  import curve_fit
from scipy.optimize  import minimize, rosen, rosen_der


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
        if n==0: ax[row][col].plot(times, states.sum(axis=0),label='Tot');
        ax[row][col].set(xlabel='Time [days]',ylabel='$y_{:0.0f}(t)$ vs. Time'.format(n),title =titles[n])        
    for n in range(num_states, num_rows * num_cols): fig.delaxes(ax[n // num_cols][n % num_cols])
    fig.tight_layout()



# Define derivative function
def differential_evolution(t, y, parameter):

    parameter = parameter[0]
    
    # populations
    S,IU,ID,R,D = y[0],y[1],y[2],y[3],y[4]

    N = S+IU+ID+R+D
    parameter['Lambda_f'  ] = parameter['beta_p']*IU/N + parameter['lambda_p']
    parameter['alpha_DR_p'] = 1/(1/parameter['alpha_p'] - 1/parameter['delta_p'])
        
    dSdt  = -parameter['Lambda_f']*S                 
    dIUdt = +parameter['Lambda_f']*S - parameter['delta_p']*IU - parameter['alpha_p']*IU
    dIDdt =                          + parameter['delta_p']*IU                           - parameter['alpha_DR_p']*ID - parameter['mu_p']*ID        
    dRdt =                                                     + parameter['alpha_p']*IU + parameter['alpha_DR_p']*ID  
    dDdt =                                                                                                            + parameter['mu_p']*ID
    
    dXdt = [dSdt, dIUdt, dIDdt, dRdt, dDdt]
    
    return dXdt

##
def create_model_1A(timespan,*parameter):
    
    # define time spans, initial values
    tspan    = np.linspace(0, timespan, timespan+1)

    c = ['Susceptibles','Infected-Undiagnosed','Infected-Diagnosed','Recovered','Deaths']
    compartments = dict()
    for ix,i in enumerate(c): compartments[i]=ix
    
    # initial population, normalized to 100% of country population,
    S_init      = 100-0.00001
    IU_init     = 0.00001
    ID_init     = 0.00001
    R_init      = 0
    D_init      = 0

    # intial conditions
    yinit    = [S_init,IU_init,ID_init,R_init,D_init]
    
    # Solve differential equation
    sol = solve_ivp(lambda t, y: differential_evolution(t, y, parameter), [tspan[0], tspan[-1]], yinit, t_eval=tspan)
    
    return sol,compartments

def fit_model(df,country):

    # retrieve confirmed and deaths  
    df_c = df.loc[df.index.get_level_values('country_region')==country].pivot_table(index='date',columns='case_type',values='density')
    df_c = df_c.loc[df_c.Confirmed > 0.001]/1000

    # number of days for time span fit
    days = ( df_c.index.get_level_values(0).unique()[-1] - df_c.index.get_level_values(0).unique()[0] ).days

    # fixed parameters
    fixed_parameters  = dict(alpha_p  = 0.0001,    # 
                             lambda_p = 0.0001)
                                 
    # parameters to feet
    fitted_parameters = dict(beta_p   = 1E-06,     #
                             mu_p     = 1E-06,     # 
                             delta_p  = 1E-06)                        

    locals().update(fitted_parameters) 
    
    # function definition
    def func(x,beta_p,mu_p,delta_p):
        for i in fitted_parameters.keys(): fitted_parameters[i] = locals()[i]
        parameters = dict(fixed_parameters,**fitted_parameters)
        sol,compartments = create_model_1A(days,parameters)        
        return np.hstack([sol.y[compartments['Infected-Diagnosed']],sol.y[compartments['Deaths']]])

    normalizations =  df_c.values.max(axis=0) 
    input_vector = np.hstack(np.divide( df_c.values , normalizations).T)
    
    popt, pcov = curve_fit(func,np.arange(days),input_vector,bounds=(0, np.ones(len(fitted_parameters))))

    for ix,i in enumerate(fitted_parameters.keys()): fitted_parameters[i]=popt[ix]
    parameters = dict(fixed_parameters,**fitted_parameters)

    sol,compartments = create_model_1A(days*10,parameters)    
    state_plotter(sol.t, sol.y, 1,list(compartments.keys()))
    
    fig = plt.figure()
    plt.suptitle(country+' - Infected-Diagnosed')
    plt.plot(df_c.Confirmed.values,'o',markersize=2, label='data')
    plt.plot(sol.y[compartments['Infected-Diagnosed']]*normalizations[0],label='model')
    plt.legend(); plt.grid()

    fig = plt.figure()
    plt.suptitle(country+' - Deaths')
    plt.plot(df_c.Deaths.values,'o',markersize=2,label='data')
    plt.plot(sol.y[compartments['Deaths']]*normalizations[1],label='model')
    plt.legend(); plt.grid()
 
 
