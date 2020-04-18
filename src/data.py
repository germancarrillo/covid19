import os
import math
import json
import string
import requests
import itertools

from datetime import datetime

import numpy as np
import pandas as pd
import datadotworld as dw

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

##
def create_airport_info():
    from pyairports.airports import Airports 
    airports = Airports() 

    a=[] 
    for i in string.ascii_uppercase: 
        for j in string.ascii_uppercase:   
            for k in  string.ascii_uppercase:  
                try: 
                    x = airports.lookup(i+j+k) 
                    a.append({'iata':i+j+k,'country':x.country,'city':x.city,'lat':x.lat,'lon':x.lon}) 
                except: 
                    continue 
    df = pd.DataFrame(a).set_index('iata')
    save(df,name='raw/airports/airport_info')
    return df

##
def query_dataworld():
    save_name = 'raw/dataworld/all_'+str(datetime.now().date())
    if os.path.exists('data/'+save_name+'.pkl.tgz'):
        print('data found in',save_name)
        return pd.read_pickle('data/'+save_name+'.pkl.tgz')
    else:
        results = dw.query('covid-19-data-resource-hub/covid-19-case-counts', 'SELECT * FROM  covid_19_cases')
        df = results.dataframe
        save(df,name='raw/dataworld/all_'+str(datetime.now().date()))
        return df
##
def enhance_dataworld(df):

    def consolidate_country_names(df):
        df['country_region'] = df['country_region'].apply(lambda x:x.replace('Korea, South','Republic of Korea'))    
        return df
    
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

    save(df,name='dataworld')    
    return df


#
def query_airpassangers(country):
    
    # retrived airport data-base to list all airports in the coutry
    df_airports = create_airport_info()
    df_airports = df_airports.loc[ np.isin ( df_airports.index, pd.read_pickle('data/raw/airports/large_airports.pkl.tgz').sort_index().index.dropna() ) ]   

    airports = df_airports.loc[df_airports.country==country].index
    
    df_flights = query_flightstats(airports)
    
    save(df_flights,name='flights_country')    
    
    return df_flights
    
##
def query_flightstats(airports):

    print('querying airports:',*airports)
    
    # templates for historical airport flights
    flightstats_restapi_hist_host = 'https://api.flightstats.com/flex/flightstatus/historical/rest/v3'
    output_type = 'json'
    object_type = 'airport/status'
    dep_arr     = 'arr'
    appId       = os.getenv('FLIGHTSTATS_APPID')
    appKey      = os.getenv('FLIGHTSTATS_APPKEY')
    postfix     = '&utc=false&numHours=6&maxFlights=100000&codeType=IATA&extendedOptions=excludeAppendix'
    
    # loop over airports and slpit in request of 6H (max allowed by flightstats API)
    dfs = []        
    for airport in airports:
        for date in pd.date_range(start='2020-01-01 00:00:00',end=datetime.now()-pd.to_timedelta('6H'),freq='6H'):

            year  = str(date.year )
            month = str(date.month).zfill(2)    
            day   = str(date.day  ).zfill(2)    
            hour  = str(date.hour ).zfill(2)
            
            df = None
            
            save_name = 'raw/flights/airport_'+airport+'_'+year+'_'+month+'_'+day+'_'+hour
            
            if os.path.exists('data/'+save_name+'.pkl.tgz'):                
                df = pd.read_pickle('data/'+save_name+'.pkl.tgz')
            else:                 
                query = flightstats_restapi_hist_host+'/'+output_type+'/'+object_type+'/'+airport+'/'+dep_arr+'/'+year+'/'+month+'/'+day+'/'+hour+'?appId='+appId+'&appKey='+appKey+postfix
                try:
                    r = requests.get(query)                    
                    df = pd.read_json( json.dumps( r.json() ['flightStatuses']) )
                    save(df,name=save_name)
                except:
                    print('query failed:',query)
                else:
                    print('query retreived:',query)
                
            dfs.append(df)

    # concat info for all dates and airports 
    df = pd.concat(dfs).set_index('flightId')

    return df

##
def enhance_flightstats(df,country): 

    from pyairports.airports import Airports 
    airports = Airports()     
    def airport_coutry_lookup(x):
        try:
            return airports.lookup(x).country
        except:
            return np.nan
        
    # Simplify dataframe with only relevant info
    dfs = pd.DataFrame()
    dfs['arrivalDate'            ] = df.arrivalDate.apply(lambda x:x.get("dateLocal"))
    dfs['flightEquipment'        ] = df.flightEquipment.apply(lambda x:x.get("actualEquipmentIataCode")) 
    dfs['arrivalAirportFsCode'   ] = df.arrivalAirportFsCode
    dfs['departureAirportFsCode' ] = df.departureAirportFsCode
    dfs['departureAirportCountry'] = dfs.departureAirportFsCode.apply(airport_coutry_lookup)
    dfs['status'                 ] = df.status

    # Filter out flights non Landed fligths (status=='L') and local-national flights
    dfs = dfs.loc[(dfs.status=='L')&(dfs.departureAirportCountry!=country)]
        
    dfs = dfs.set_index(['arrivalDate']).sort_index()
    dfs.index = pd.to_datetime(dfs.index)

    save(dfs,name='flights_'+country)
    return dfs

##
def query_containmentdata():

    save_name = 'raw/containment/cont_all_'+str(datetime.now().date())
    if os.path.exists('data/'+save_name+'.pkl.tgz'):
        print('data found in',save_name)
        return pd.read_pickle('data/'+save_name+'.pkl.tgz')
    else:
        os.system('wget https://storage.googleapis.com/static-covid/Containment%20measures/countermeasures_db_johnshopkins.csv data/raw/containment/countermeasures_db_johnshopkins.csv')
        df = pd.read_csv('data/raw/containment/countermeasures_db_johnshopkins.csv',index_col='Date',parse_dates=True)
        save(df,name=save_name)
        return df

##
def enhance_containmentdata(df):
    
    cols = ['Domestic travel restriction',
            'Nonessential business suspension',
            'Assisting people to stay home',
            'Gatherings banned',
            'Activity cancellation',        
            'School closure',
            'Contact tracing',
            'Public cleaning',
            'Mask wearing',
            'Miscellaneous hygiene measures',
            'Public interaction and hygiene']

    df_ = df[cols].select_dtypes(['number']).fillna(0).astype('bool')
    df_['Country'] = df['Country']
    df = df_.reset_index().set_index(['Date','Country'])    
    df = df.sum(axis=1).reset_index().set_index('Date').pivot_table(index='Date',columns='Country',values=0)

    return df

##
def combined_datasets(df_covid,df_airp,df_cont,country):
    import src.infection as infection

    df = infection.estimate_force(df_covid,df_airp,country)
    
    df['containment'] = df_cont[country] 

    df = df.dropna()
    
    return df

##
def save(df,name=''):
    
    for i in ['raw/dataworld','raw/flights','raw/airports']: os.makedirs('data/'+i,exist_ok=True) 
    df.to_pickle('data/'+name+'.pkl.tgz')
    
    
