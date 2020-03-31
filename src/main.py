import os
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import src.model as model
import src.plots as plots
import src.data as data


##
if __name__== "__main__":
    
    df_raw = data.query_dataworld()

    df = data.enhance(df_raw)

    model.fit_model(df,'Switzerland')
    plt.show()

    model.fit_model(df,'Republic of Korea')
    plt.show() 

    
    #plots.make_plots(df)
    #data.save(df)
