import os
import math
import numpy as np
import pandas as pd
import datadotworld as dw
import itertools

from datetime import datetime

import src.plots as plots
import src.data as data


##
if __name__== "__main__":
    
    df_raw = data.query_dataworld()
    df = df_raw.set_index(['date','country_region']).sort_index() 
    df = df.groupby(['date','country_region','case_type']).sum()
    df = data.enhance(df)


    plots.make_plots(df)
    data.save(df)
