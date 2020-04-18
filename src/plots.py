import os
import math
import itertools
import numpy as np
import pandas as pd

from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from matplotlib import cm

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

from scipy import interpolate

countries = dict(Switzerland= 'r',
                 Spain      = 'y',
                 France     = 'b',
                 Italy      = 'g',
                 US         = 'k')
##
def make_plots(df):

    os.makedirs('plots/cases/',exist_ok=True)
    plot_totals(df)
    print('total cases plotted')    
    for case_type in df.index.get_level_values('case_type').unique():
        plot_cases(df,case_type,'cases')
        plot_cases(df,case_type,'cases,daily')
    print('country cases plotted')
    for case_type in df.index.get_level_values('case_type').unique():
        plot_cases(df,case_type,'density')
        plot_cases(df,case_type,'density,daily')
    print('country densities plotted')
    
    os.makedirs('plots/maps/',exist_ok=True) 
    for date in df.index.get_level_values('date').unique():
        for case_type in df.index.get_level_values('case_type').unique():       
            df_date=df.loc[(df.index.get_level_values('date'     )==date     )&
                           (df.index.get_level_values('case_type')==case_type)].droplevel([0,2])                                 
            plot_map(df_date,date,case_type)
    print('map plotted')

##
def plot_totals(df):

    df_tot = df.groupby(['date','case_type']).sum().reset_index().pivot(index='date',columns='case_type',values='cases')
    fig = plt.figure(figsize=(10,8))
    plt.plot(df_tot.Confirmed,'k',linewidth=2,label='Confirmed')
    plt.plot(df_tot.Deaths   ,'r',linewidth=2,label='Deaths'   )
    #plt.plot(df_tot.Active   ,'g',linewidth=2,label='Active'   )
    #plt.plot(df_tot.Recovered,'b',linewidth=2,label='Recovered')
    plt.ylabel('Total Cases'); plt.legend(); plt.grid()
    plt.savefig('plots/cases/Total_cases.png')
    plt.close()     
    
##
def plot_cases(df,case_type,options):

    daily  = True if 'daily' in options else False
    target = 'density'if 'density' in options else 'cases'
    
    fig = plt.figure(figsize=(10,8))
    for country,country_color in countries.items():
        df_i = df.loc[(df.index.get_level_values('country_region')==country)&(df.index.get_level_values('case_type')==case_type) ][target].droplevel([1,2])
        if daily: df_i = df_i - df_i.shift(1)
        plt.plot(df_i,color=country_color,linewidth=2,label=country)        
        if 'log' in options: plt.semilogy()
    plt.suptitle('Daily New Cases' if daily else 'Aggregated Cases')
    plt.legend(); plt.grid(); plt.ylabel(case_type+('/100k inhabitants' if target=='density' else '')); plt.xlabel('Datetime'); plt.tight_layout()
    plt.savefig('plots/cases/'+case_type+'_countries'+('_per_100k' if target=='density' else '')+('_dailychange' if daily else '' )+'.png')
    #plt.close()

##
def plot_map(df,date,case_type):

    file_name = 'plots/maps/map_'+case_type+'_'+str(date).split(' ')[0]+'.png'
    if os.path.exists(file_name): return 
    
    shapename = 'admin_0_countries'
    countries_shp = shpreader.natural_earth(resolution='110m',category='cultural', name=shapename)    
    fig = plt.figure(figsize=(20,10))
    fig.suptitle('Case Type: '+case_type)
    ax = plt.axes(projection=ccrs.PlateCarree())
    for country in shpreader.Reader(countries_shp).records():
        density = df.loc[df.NAME == country.attributes['NAME']].density
        density = 0 if density.size==0 else density[0]
        z = density/df.density.max()    
        color = (0.95-(0.95*z),0.95-(0.95*z),0.95-(0.95*z))
        ax.add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=color,label=country.attributes['NAME_LONG'])
    plt.savefig(file_name)
    plt.close()
