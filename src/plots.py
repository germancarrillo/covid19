import os
import math
import numpy as np
import pandas as pd
import datadotworld as dw
import itertools

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
                 Germany    = 'g',
                 Italy      = 'b',
                 China      = 'k')
##
def make_plots(df):

    os.makedirs('plots',exist_ok=True) 
    for case_type in df.index.get_level_values('case_type').unique(): plot_cases(df,case_type,'cases')
    print('cases plotted')
    
    for case_type in df.index.get_level_values('case_type').unique(): plot_cases(df,case_type,'density')
    print('densities plotted')
    
    for date in df.index.get_level_values('date'):
        for case_type in df.index.get_level_values('case_type').unique():       
            df_date=df.loc[(df.index.get_level_values('date'     )==date     )&
                           (df.index.get_level_values('case_type')==case_type)].droplevel([0,2])                                 
            plot_map(df_date,date,case_type)
    print('map plotted')
##
def plot_cases(df,case_type,target):
                    
    fig = plt.figure(figsize=(10,8))
    for country,country_color in countries.items():
        plt.plot( df.loc[(df.index.get_level_values('country_region')==country  )&
                         (df.index.get_level_values('case_type'     )==case_type) ][target].droplevel([1,2])
                  ,color=country_color,linewidth=2,label=country)
    plt.semilogy()
    plt.legend(); plt.grid(); plt.ylabel(case_type+'/100k inhabitants'); plt.xlabel('Datetime')
    plt.savefig('plots/'+case_type+'_per_100k.png')
    plt.close()
        
##
def plot_map(df,date,case_type):
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
    plt.savefig('plots/map_'+case_type+'_'+str(date).split(' ')[0]+'.png')
    plt.close()
