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
def smooth(x,window_len=11,window='hanning'): 
    if window_len<3: 
        return x 
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]] 
    if window == 'flat': #moving average 
        w=np.ones(window_len,'d') 
    else:   
        w=eval('np.'+window+'(window_len)') 
    y=np.convolve(w/w.sum(),s,mode='same') 
    return y[window_len:-window_len+1] 

##
def create_scenario(scenario,parameters_i,period):
       
    parameters_i['period'  ] = period
    parameters_i['scenario'] = scenario

    def persist(p,period):
        p = np.append( p, np.ones(period - p.size + 1)*p[-1])
        return p
    def steps_days(p,period):
        p = np.append( p, np.ones(2)*p[-1]  )        
        p = np.append( p, np.ones(7)*2*1./3 )        
        p = np.append( p, np.ones(7)*2*1./3 )
        p = np.append( p, np.ones(max(0,period-p.size)+1)*0.)
        return p
    def linear(p,period):
        p = np.append( p, np.ones(2)*p[-1]  )                     
        epsilon = (1-p[-1])/(period-p.size)
        for i in range(period - p.size + 1):
            p = np.append(p, p[-1]+epsilon)
        return p
    
    if scenario == 'persistence':     
        parameters_i['containment'     ] = persist(parameters_i['containment'     ],parameters_i['period'])
        parameters_i['flights_infected'] = persist(parameters_i['flights_infected'],parameters_i['period'])
        return parameters_i

    if scenario == 'ease-containment':
        parameters_i['containment'     ] = steps_days(parameters_i['containment'     ],parameters_i['period'])
        parameters_i['flights_infected'] = persist   (parameters_i['flights_infected'],parameters_i['period'])
        return parameters_i

    if scenario == 'ease-travelban':
        parameters_i['containment'     ] = persist(parameters_i['containment'     ],parameters_i['period'])
        parameters_i['flights_infected'] = linear (parameters_i['flights_infected'],parameters_i['period'])
        return parameters_i

    if scenario == 'ease-containment-and-travelban': 
        parameters_i['containment'     ] = steps_days(parameters_i['containment'     ],parameters_i['period'])
        parameters_i['flights_infected'] = linear    (parameters_i['flights_infected'],parameters_i['period'])
        return parameters_i
    
    return parameters_i

##
def plot_fit_results(df,sol,country,parameters):
    df_c,containment,flights_infected,input_vector,weigths,days =  prepare_data(df)
    os.makedirs('plots/model/'+country+'/',exist_ok=True)    
    data_plotter(df_c,sol,country,parameters)

##    
def data_plotter(df_c,sol,country,parameters):

    index = pd.date_range(start=df_c.index[0],periods=sol.t.size,freq='1D')
    
    fig = plt.figure(figsize=(15,12))

    fig.suptitle(parameters['scenario'])

    normalize_input_vector = sol.y[sol.compartments['Deaths']].max()/df_c.Deaths.max()
    df_n = df_c*normalize_input_vector
    
    for ix,i in enumerate(['Infected-Diagnosed','Deaths']):
        ax0 = fig.add_subplot(331+ix)
        ax0.set_title(country+' - '+i)    
        ax0.plot(index[:index.size],(sol.y[sol.compartments[i]])[:index.size],'r',label='fitted-model')
        ax0.plot(index,sol.y[sol.compartments[i]],'r--',label='model-projection')
        ax0.plot(df_n['Confirmed' if ix==0 else i],'o',markersize=2, label='data')
        ax0.set_ylim([ax0.get_ylim()[0], 0.5 if ix==0 else 0.05])
        ax0.vlines(datetime.today(),ymin=0,ymax=ax0.get_ylim()[1],color='k',linestyle='--',label='today')
        ax0.legend(); ax0.grid(); ax0.set_xlabel('Datetime'); ax0.set_ylabel('Cases [%]');

    for ix,i in enumerate(['Infected-Diagnosed','Deaths']):
        ax = fig.add_subplot(334+ix,sharex=ax0)
        ax.set_title(country+' - '+i+', Daily Cases')        
        ax.plot(index[1:],(sol.y[sol.compartments[i]])[1:index.size] - (sol.y[sol.compartments[i]])[0:index.size-1],'r',label='fitted-model')
        ax.plot(index[1:],(sol.y[sol.compartments[i]])[1:] - (sol.y[sol.compartments[i]])[:-1] ,'r--',label='model-projection')
        ax.plot(df_n['Confirmed' if ix==0 else i] - df_n['Confirmed' if ix==0 else i].shift(1),'o',markersize=2, label='data')
        ax.vlines(datetime.today(),ymin=0,ymax=ax.get_ylim()[1],color='k',linestyle='--',label='today')
        ax.legend(); ax.grid(); ax.set_xlabel('Datetime'); ax.set_ylabel('Cases [%]');

    for ix,i in enumerate(['containment','flights_infected']):
        ax = fig.add_subplot(333+ix*3,sharex=ax0)
        ax.set_title(country+' - Infection Factor: '+i.replace('_',' ').title())
        ax.plot(index,parameters[i],'g',label=i.replace('_',' ').title())
        ax.vlines(datetime.today(),ymin=0,ymax=ax.get_ylim()[1],color='k',linestyle='--',label='today')
        ax.legend(); ax.grid(); ax.set_xlabel('Datetime'); ax.set_ylabel('Normalized');

    for i,ix in sol.compartments.items():
        ax = fig.add_subplot(6,3,13+ix,sharex=ax0)
        ax.plot(index,sol.y[ix],'k',label=i.replace('_',' ').title())
        ax.vlines(datetime.today(),ymin=0,ymax=ax.get_ylim()[1],color='k',linestyle='--')                
        ax.legend(); ax.set_xlabel('Datetime'); ax.set_ylabel('Population [%]'); ax.set_ylim([ax.set_ylim()[0],ax.set_ylim()[1]*1.3])
        
    ax = fig.add_subplot(6,3,13+ix+1,sharex=ax0)
    ax.plot(index,sol.y.sum(axis=0),'k',label='Total'.title())
    ax.vlines(datetime.today(),ymin=0,ymax=ax.get_ylim()[1],color='k',linestyle='--')                    
    ax.legend(); ax.set_xlabel('Datetime'); ax.set_ylabel('Population [%]');  ax.set_ylim([-1,110])
    
    fig.autofmt_xdate()
    fig.savefig('plots/model/'+country+'/infected_diagnosed.png')


# Define derivative function
def differential_evolution(t, y, parameter):

    parameter = parameter[0]
    
    # populations
    S,IU,ID,R,D = y[0],y[1],y[2],y[3],y[4]

    # total population
    N = np.exp(S) + np.exp(IU) + np.exp(ID) + np.exp(R) + np.exp(D)
   
    # local infection rate shall be time dependent as it is influenced by isolation -> proxy from containment measures:
    parameter['local_infection_rate'] = parameter['beta1_p'] + (1 - parameter['containment'][int(t)])*parameter['beta2_p']
    
    # external force of infection can be modelled based on flight flux ( as a proxy to mobilty restrictions / travel ban )
    parameter['external_force'] = parameter['lambda1_p'] + parameter['flights_infected'][int(t)]*parameter['lambda2_p']

    # infection rate:
    parameter['Lambda_f'] = parameter['local_infection_rate']*np.exp(IU)/N + parameter['external_force']      

    '''
    Lambda Infection Rate:
    beta1  & beta2   -> local infection rates ( beta1:base-intrinsic beta2:containment-influenced)
    lamda1 & lambda2 -> external force of infection (lambda1:base-itrinsic lambda2:flight-prevalence-weighted-proxy)

    Diagnosis Rate:
    delta            -> diagnosis rate 
    
    Recovery Rate:
    alpha1           -> recovery rate of undiagnosed
    alpha2           -> recovery rate of diagnosed 
    '''
    
    dSdt  = ( - parameter['Lambda_f']*np.exp(S) +                                                                                                       + parameter['gamma_p']*np.exp(R )                                  )*np.exp(-S )  
    dIUdt = ( + parameter['Lambda_f']*np.exp(S) - parameter['delta_p']*np.exp(IU) - parameter['alpha1_p']*np.exp(IU)                                                                                                       )*np.exp(-IU)     
    dIDdt = (                                   + parameter['delta_p']*np.exp(IU)                                    - parameter['alpha2_p']*np.exp(ID)                                     - parameter['mu_p']*np.exp(ID) )*np.exp(-ID)      
    dRdt  = (                                                                     + parameter['alpha1_p']*np.exp(IU) + parameter['alpha2_p']*np.exp(ID) - parameter['gamma_p']*np.exp(R )                                  )*np.exp(-R )      
    dDdt  = (                                                                                                                                                                               + parameter['mu_p']*np.exp(ID) )*np.exp(-D )      
    
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
    sol.compartments = compartments

    return sol

##
def prepare_data(df):

    # retrieve confirmed and deaths  
    df_c = df.pivot_table(index='date',columns='case_type',values='density')
    df_c = df_c.loc[df_c.Confirmed > 0.001]/1000.
    df_c = df_c.fillna(0)
    
    # number of days for time span fit
    days = ( df_c.index.get_level_values(0).unique()[-1] - df_c.index.get_level_values(0).unique()[0] ).days

    containment      = df.loc[df.case_type=='Deaths','containment'].loc[df_c.index].values
    #containment     = smooth(containment,11)
    containment      = containment/np.max(containment)
    
    flights_infected = (df.loc[df.case_type=='Deaths','flights_infected']*df.loc[df.case_type=='Deaths','flights']).loc[df_c.index].values
    flights_infected = smooth(flights_infected,19)
    flights_infected = flights_infected/np.max(flights_infected)
    
    normalizations = df_c.values.max(axis=0) 
    input_vector   = np.hstack(np.divide( df_c.values , normalizations).T)
    
    linear_weights = np.arange(0,input_vector.size)/(input_vector.size-1)
    input_vector   = input_vector * linear_weights 

    weights = (linear_weights,normalizations)
    
    return df_c,containment,flights_infected,input_vector,weights,days

##
def fit_model(df):

    df_c,containment,flights_infected,input_vector,weights,days =  prepare_data(df)
        
    # fixed parameters
    fixed_parameters  = dict(containment      = containment, 
                             flights_infected = flights_infected)
                             
    # parameters to feet
    fitted_parameters = dict(alpha1_p  = 0.100,
                             alpha2_p  = 0.100,
                             beta1_p   = 0.050,
                             beta2_p   = 0.050,
                             gamma_p   = 0.100,
                             delta_p   = 0.050, 
                             lambda1_p = 0.001,
                             lambda2_p = 0.001,
                             mu_p      = 0.050)
    locals().update(fitted_parameters) 

    # function definition
    def func(x,alpha1_p,alpha2_p,beta1_p,beta2_p,gamma_p,delta_p,lambda1_p,lambda2_p,mu_p):        
        for i in fitted_parameters.keys(): fitted_parameters[i] = locals()[i]
        parameters = dict(fixed_parameters,**fitted_parameters)        
        sol = create_model_1A(days,parameters)        

        ID = np.exp( sol.y[sol.compartments['Infected-Diagnosed']] )
        D  = np.exp( sol.y[sol.compartments['Deaths']] )
        
        ID = ID/ID.max()
        D  = D /D.max()
        
        model_vector = np.hstack([ID,D])
        model_vector = model_vector*weights[0]
        return model_vector

    # calibrate model
    popt, pcov = curve_fit(func,np.arange(days),input_vector,bounds=(0, np.ones(len(fitted_parameters))))

    # update parameters with fitted values
    for ix,i in enumerate(fitted_parameters.keys()): fitted_parameters[i]=popt[ix]
    parameters = dict(fixed_parameters,**fitted_parameters)

    parameters['weights'] = weights
    
    parameters['days'  ] = days
    parameters['period'] = days
    
    parameters['scenario'] = 'calibration'

    return parameters

##
def evaluate_model(parameters):
    
    # evaluate model
    sol = create_model_1A(parameters['period'],parameters)    

    # back to non log equations
    sol.y = np.exp(sol.y)
    
    return sol

##
def run_model(df,country):

    # fit model
    parameters = fit_model(df)

    # evaluate calibration
    sol = evaluate_model(parameters)    
    plot_fit_results(df,sol,country,parameters)
    plt.show()
    
    # evaluate scenarios
    for scenario in ['persistence','ease-containment','ease-travelban','ease-containment-and-travelban']:
        period = parameters['days']*4
        parameters_i = create_scenario(scenario,parameters.copy(),period)
        sol = evaluate_model(parameters_i)    
        plot_fit_results(df,sol,country,parameters_i)
        
    plt.show()    
    
