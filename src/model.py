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

##
def plot_fit_results(df_c,sol,compartments,normalizations,country):

    os.makedirs('plots/model/'+country+'/',exist_ok=True)    
    state_plotter(sol.t, sol.y, 1,list(compartments.keys()),country)
    data_plotter(df_c,sol,compartments,normalizations,country)
        
##    
def data_plotter(df_c,sol,compartments,normalizations,country):

    index = pd.date_range(start=df_c.index[0],periods=sol.t.size,freq='1D')
    
    fig = plt.figure(figsize=(12,8))
    
    for ix,i in enumerate(['Infected-Diagnosed','Deaths']):
        ax = fig.add_subplot(221+ix)
        ax.set_title(country+' - '+i)    
        ax.plot(index[:df_c.index.size],(sol.y[compartments[i]]*normalizations[ix])[:df_c.index.size],'r',label='fitted-model')
        ax.plot(index,sol.y[compartments[i]]*normalizations[ix],'r--',label='model-projection')
        ax.plot(df_c['Confirmed' if ix==0 else i],'o',markersize=2, label='data')
        ax.legend(); ax.grid(); ax.set_xlabel('Datetime'); ax.set_ylabel('Cases [%]');

    for ix,i in enumerate(['Infected-Diagnosed','Deaths']):
        ax = fig.add_subplot(223+ix)
        ax.set_title(country+' - '+i+', Daily Cases')        
        ax.plot(index[1:df_c.index.size+1],(sol.y[compartments[i]]*normalizations[ix])[1:df_c.index.size+1] - (sol.y[compartments[i]]*normalizations[ix])[:df_c.index.size],'r',label='fitted-model')
        ax.plot(index[1:],(sol.y[compartments[i]]*normalizations[ix])[1:] - (sol.y[compartments[i]]*normalizations[ix])[:-1] ,'r--',label='model-projection')
        ax.plot(df_c['Confirmed' if ix==0 else i] - df_c['Confirmed' if ix==0 else i].shift(1),'o',markersize=2, label='data')
        ax.legend(); ax.grid(); ax.set_xlabel('Datetime'); ax.set_ylabel('Cases [%]');

    fig.autofmt_xdate()
    fig.savefig('plots/model/'+country+'/infected_diagnosed.png')


#
def state_plotter(times, states, fig_num,titles,country):
    
    num_states = np.shape(states)[0]
    num_cols = int(np.ceil(np.sqrt(num_states)))
    num_rows = int(np.ceil(num_states / num_cols))
    plt.figure(fig_num)
    plt.suptitle(country+' - Comparments Evolution')
    plt.clf()
    fig, ax = plt.subplots(num_rows, num_cols, num=fig_num, clear=True, squeeze=False)
    for n in range(num_states):
        row = n // num_cols
        col = n % num_cols
        ax[row][col].plot(times, states[n], 'k',linewidth=2)
        if n==0: ax[row][col].plot(times, states.sum(axis=0),label='Tot');
        ax[row][col].set(xlabel='Time [days]',ylabel='$y_{:0.0f}(t)$ vs. Time'.format(n),title =titles[n])        
    for n in range(num_states, num_rows * num_cols): fig.delaxes(ax[n // num_cols][n % num_cols])    
    fig.tight_layout()
    plt.savefig('plots/model/'+country+'/compartments.png')

# Define derivative function
def differential_evolution(t, y, parameter):

    parameter = parameter[0]
    
    # populations
    S,IU,ID,R,D = y[0],y[1],y[2],y[3],y[4]
    
    N = np.exp(S) + np.exp(IU) + np.exp(ID) + np.exp(R) + np.exp(D)

    N = 100
    
    # local infection rate shall be time dependent as it is influenced by isolation -> proxy from containment measures:
    parameter['local_infection_rate'] = parameter['beta1_p'] + (parameter['containment'][max(0,int(t-1))])*parameter['beta2_p']
    
    # external force of infection can be modelled based on flight flux ( as a proxy to mobilty restrictions / travel ban )
    parameter['external_force'] = parameter['flights_infected'][max(0,int(t-1))]*parameter['lambda_p'] 

    # infection rate:
    parameter['Lambda_f'  ] = parameter['local_infection_rate']*np.exp(IU)/N + parameter['external_force']      

    # diagnosed to recovered
    parameter['alpha_DR_p'] = 1/(1/parameter['alpha_p'] - 1/parameter['delta_p'])   

    dSdt  = (-parameter['Lambda_f'])
    dIUdt = (+parameter['Lambda_f']*np.exp(S) - parameter['delta_p']*np.exp(IU) - parameter['alpha_p']*np.exp(IU)                                                                     )*np.exp(-IU)     
    dIDdt = (                                 + parameter['delta_p']*np.exp(IU)                                   - parameter['alpha_DR_p']*np.exp(ID) - parameter['mu_p']*np.exp(ID) )*np.exp(-ID)      
    dRdt  = (                                                                   + parameter['alpha_p']*np.exp(IU) + parameter['alpha_DR_p']*np.exp(ID)                                )*np.exp(-R )      
    dDdt  = (                                                                                                                                          + parameter['mu_p']*np.exp(ID) )*np.exp(-D )      
    
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
    S_init      = np.log(100)
    IU_init     = -25#-np.inf
    ID_init     = -25#-np.inf
    R_init      = -25#-np.inf
    D_init      = -25#-np.inf

    # intial conditions
    yinit    = [S_init,IU_init,ID_init,R_init,D_init]

    # Solve differential equation
    sol = solve_ivp(lambda t, y: differential_evolution(t, y, parameter), [tspan[0], tspan[-1]], yinit, t_eval=tspan)
    
    return sol,compartments


def fit_model(df,country):

    # retrieve confirmed and deaths  
    df_c = df.pivot_table(index='date',columns='case_type',values='density')
    df_c = df_c.loc[df_c.Confirmed > 0.001]/1000

    # number of days for time span fit
    days = ( df_c.index.get_level_values(0).unique()[-1] - df_c.index.get_level_values(0).unique()[0] ).days

    # scenarios

    period_factor = 3 
    
    containment      = df.loc[df.case_type=='Deaths','containment'].loc[df_c.index]
    containment      = containment.values/containment.max()
    flights_infected = (df.loc[df.case_type=='Deaths','flights_infected']*df.loc[df.case_type=='Deaths','flights']).loc[df_c.index]
    flights_infected = flights_infected.values/flights_infected.max()
    
    containment      = np.append( containment     , np.ones(containment     .size*(period_factor-1))*containment     [-1])   
    flights_infected = np.append( flights_infected, np.ones(flights_infected.size*(period_factor-1))*flights_infected[-1])   
    
    # fixed parameters
    fixed_parameters  = dict(alpha_p          = 1/10.,
                             containment      = containment, 
                             flights_infected = flights_infected )
                             
    # parameters to feet
    fitted_parameters = dict(beta1_p  = 1E-06,
                             beta2_p  = 1E-06, 
                             delta_p  = 1E-06, 
                             lambda_p = 1E-06,
                             mu_p     = 1E-06)

    locals().update(fitted_parameters) 

    sol,compartments = create_model_1A(days*period_factor,parameters)    
    state_plotter(sol.t, np.exp(sol.y), 1,list(compartments.keys()),country)
    
    # function definition
    def func(x,beta1_p,beta2_p,delta_p,lambda_p,mu_p):
        for i in fitted_parameters.keys(): fitted_parameters[i] = locals()[i]
        parameters = dict(fixed_parameters,**fitted_parameters)
        sol,compartments = create_model_1A(days,parameters)        
        return np.exp( np.hstack([sol.y[compartments['Infected-Diagnosed']],sol.y[compartments['Deaths']]]) )

    normalizations =  df_c.values.max(axis=0) 
    input_vector = np.hstack(np.divide( df_c.values , normalizations).T)
    
    popt, pcov = curve_fit(func,np.arange(days),input_vector,bounds=(0, np.ones(len(fitted_parameters))))

    for ix,i in enumerate(fitted_parameters.keys()): fitted_parameters[i]=popt[ix]
    parameters = dict(fixed_parameters,**fitted_parameters)

    sol,compartments = create_model_1A(days*period_factor,parameters)    

    sol.y = np.exp(sol.y)
    
    plot_fit_results(df_c,sol,compartments,normalizations,country)
    

