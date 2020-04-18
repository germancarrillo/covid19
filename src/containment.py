import os
import math
import itertools
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt


def 

os.system('wget https://storage.googleapis.com/static-covid/Containment%20measures/countermeasures_db_johnshopkins.csv data/raw/containment/')
df = pd.read_csv('countermeasures_db_johnshopkins.csv',index_col='Date',parse_dates=True) 
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

for country in ['Switzerland','Germany','Italy','Spain','New Zealand','South Korea','China']: 
    df_i = df.loc[df.Country==country] 
    plt.plot( df_i[cols].select_dtypes(['number']).fillna(0).astype('bool').sum(axis=1),label=country) 
plt.legend(); plt.show()                                                                                                                                                                                                                                                                                              
