import os
import math
import json
import requests
import itertools

from datetime import datetime

import numpy as np
import pandas as pd
import datadotworld as dw

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader


##
def query_dataworld():
    results = dw.query('covid-19-data-resource-hub/covid-19-case-counts', 'SELECT * FROM  covid_19_cases')
    return results.dataframe
##
def query_flightstats():

    flightstats_restapi_hist_host = 'https://api.flightstats.com/flex/flightstatus/historical/rest/v3'
    output_type = 'json'
    object_type = 'airport/status'
    dep_arr     = 'arr'
    appId       = os.getenv('FLIGHTSTATS_APPID')
    appKey      = os.getenv('FLIGHTSTATS_APPKEY')
    postfix     = '&utc=false&numHours=6&maxFlights=100000'

    df_flights = []
    
    for airport in ['GVA','ZRH','BSL']:
        for date in pd.date_range(start='2020-01-01 00:00:00',end='2020-03-27 00:00:00',freq='6H'):
            year  = str(date.year )
            month = str(date.month).zfill(2)    
            day   = str(date.day  ).zfill(2)    
            hour  = str(date.hour ).zfill(2)
            
            query = flightstats_restapi_hist_host+'/'+output_type+'/'+object_type+'/'+airport+'/'+dep_arr+'/'+year+'/'+month+'/'+day+'/'+hour+'?appId='+appId+'&appKey='+appKey+postfix
            
            r = requests.get(query)
            df_flights.append( pd.read_json( json.dumps( r.json() ['flightStatuses']) ))
            
    df_flight = pd.concat(df_flights).set_index('flightId')

    return df_flight


##
def consolidate_country_names(df):
    df['country_region'] = df['country_region'].apply(lambda x:x.replace('Korea, South','Republic of Korea')) 
    return df

##
def enhance(df):

    df = consolidate_country_names(df)        
    df = df.set_index(['date','country_region']).sort_index() 
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
        if country.attributes['NAME_LONG'] in df.droplevel(0).index: country_index=country.attributes['NAME_LONG']

        
        if country_index != '':
            df.loc[df.index.get_level_values(1)==country_index,'POP_EST'] = country.attributes['POP_EST']
            df.loc[df.index.get_level_values(1)==country_index,'NAME'   ] = country_index
    df['density'   ] = (df.cases/df.POP_EST*100000).fillna(0)        
    return df

##
def save(df):
    os.makedirs('data/',exist_ok=True) 
    df.to_pickle('data/data.pkl.tgz')
