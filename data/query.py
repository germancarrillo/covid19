import os
import math
import numpy as np
import pandas as pd
import datadotworld as dw
import matplotlib as mpl
import matplotlib.pyplot as plt

from datetime import datetime
from mpl_toolkits.basemap import Basemap
from matplotlib import cm

from scipy import interpolate

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import itertools
import numpy as np

##
def query_dataworld():
    results = dw.query('covid-19-data-resource-hub/covid-19-case-counts', 'SELECT * FROM  covid_19_cases')
    return results.dataframe

def plot_map(df,d):
    shapename = 'admin_0_countries'
    countries_shp = shpreader.natural_earth(resolution='110m',category='cultural', name=shapename)    
    fig = plt.figure(figsize=(20,10)) 
    ax = plt.axes(projection=ccrs.PlateCarree())
    for country in shpreader.Reader(countries_shp).records():
        density = df.loc[df.NAME == country.attributes['NAME']].density
        density = 0 if density.size==0 else density[0]
        z = density/df.density.max()    
        color = (0.95-(0.95*z),0.95-(0.95*z),0.95-(0.95*z))
        print(d,country.attributes['NAME_LONG'],density,color)
        ax.add_geometries([country.geometry], ccrs.PlateCarree(),facecolor=color,label=country.attributes['NAME_LONG'])
    plt.savefig('plots/'+str(d)+'.png')
    plt.close()
            
##
if __name__== "__main__":
    df_raw = query_dataworld()
    df = df_raw.set_index(['date','country_region']).sort_index() 

    df = df.groupby(['date','country_region','case_type']).sum()
        
    shapename = 'admin_0_countries'
    countries_shp = shpreader.natural_earth(resolution='110m',category='cultural', name=shapename)
    df['POP_EST'] = None
    df['NAME'   ] = None
  
    for country in shpreader.Reader(countries_shp).records():
        country_index = ''        
        if country.attributes['POSTAL'   ] in df.droplevel(0).index: country_index=country.attributes['POSTAL'   ]
        if country.attributes['NAME'     ] in df.droplevel(0).index: country_index=country.attributes['NAME'     ]
        if country.attributes['NAME_ALT' ] in df.droplevel(0).index: country_index=country.attributes['NAME_ALT' ]
        if country.attributes['NAME_LONG'] in df.droplevel(0).index: country_index=country.attributes['NAME_LONG']
        if country_index != '':
            df.loc[df.index.get_level_values(1)==country_index,'POP_EST'] = country.attributes['POP_EST']
            df.loc[df.index.get_level_values(1)==country_index,'NAME'   ] = country_index
    df['density'] = (df.cases/df.POP_EST*100000).fillna(0)        

    fig = plt.figure(figsize=(10,8))
    for c in ['Italy','Switzerland','Spain','Germany','France','China']:
        plt.plot( df.loc[(df.index.get_level_values('country_region')==c)&(df.index.get_level_values('case_type')=='Confirmed')].density.droplevel([1,2]),linewidth=2,label=c)
    plt.semilogy()
    plt.legend(); plt.grid(); plt.ylabel('ConfirmedCases/100k inhabitants'); plt.xlabel('Datetime')
    plt.show()

    
    fig = plt.figure(figsize=(10,8))
    for c in ['Italy','Switzerland','Spain','Germany','France','China']:
        plt.plot( df.loc[(df.index.get_level_values('country_region')==c)&(df.index.get_level_values('case_type')=='Deaths')].density.droplevel([1,2]),linewidth=2,label=c)
    plt.semilogy()
    plt.legend(); plt.grid(); plt.ylabel('Fatalities/100k inhabitants'); plt.xlabel('Datetime')
    plt.show()

  
