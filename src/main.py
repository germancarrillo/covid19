import os
import sys
import math
import getopt

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import src.model as model
import src.plots as plots
import src.data as data
import src.infection as infection

##
if __name__== "__main__":

    # default
    country = 'Switzerland'
    
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv,'i:c:')
    if len(args)>0: country = args[0]

    print('processing country',country)

    df_airp = data.query_airpassangers(country)
    df_airp = data.enhance_flightstats(df_airp,country)

    print('flightstats passenger data retrived')
    
    df_covid = data.query_dataworld()
    df_covid = data.enhance_dataworld(df_covid)

    print('dataworld covid set retrived')

    df_cont = data.query_containmentdata()
    df_cont = data.enhance_containmentdata(df_cont)

    print('containemdataworld covid set retrived')

    df = data.combined_datasets(df_covid,df_airp,df_cont,country)

    model.run_model(df,country)
    
    #plots.make_plots(df_covid)    
    #plt.close()


    
