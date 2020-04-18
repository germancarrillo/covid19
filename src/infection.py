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
def estimate_force(df_covid,df_airp,country):

    df_covid = df_covid.reset_index().set_index(['date','country_region'])

    def density(x,df_covid):
        try:
            return df_covid.loc[x]
        except:
            return np.nan

    df_airp = df_airp.loc[df_covid.index.get_level_values('date')[0]:df_covid.index.get_level_values('date')[-1]]
    df_airp['date'] = df_airp.index.floor('1D')             
    df_airp = df_airp.reset_index().set_index(['date','departureAirportCountry'])
           
    df_covid_deaths    = df_covid.loc[df_covid.case_type=='Deaths'   ].density
    df_covid_confirmed = df_covid.loc[df_covid.case_type=='Confirmed'].density
                
    df_airp['fatality'] = df_airp.index.map( lambda x:density(x,df_covid_deaths   ) )
    df_airp['infected'] = df_airp.index.map( lambda x:density(x,df_covid_confirmed) )

    df_covid_country = df_covid.loc[df_covid.index.get_level_values('country_region') == country].droplevel(1)  
    df_covid_country['flights'         ] = df_airp.fatality.groupby('date').size()
    df_covid_country['flights_infected'] = None
    df_covid_country.loc[df_covid_country.case_type=='Deaths'   ,'flights_infected'] = df_airp.fatality.groupby('date').sum()
    df_covid_country.loc[df_covid_country.case_type=='Confirmed','flights_infected'] = df_airp.infected.groupby('date').sum()

    return df_covid_country 
