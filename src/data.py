import os
import math
import numpy as np
import pandas as pd
import datadotworld as dw
import itertools

from datetime import datetime

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader


##
def query_dataworld():
    results = dw.query('covid-19-data-resource-hub/covid-19-case-counts', 'SELECT * FROM  covid_19_cases')
    return results.dataframe
            
##
def enhance(df):
    
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
    return df

##
def save(df):
    os.makedirs('data/',exist_ok=True) 
    df.to_pickle('data/data.pkl.tgz')
