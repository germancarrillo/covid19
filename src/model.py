import os
import math
import itertools
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.integrate import solve_ivp


#
def susceptible(t, y):
    return -0.5 * y


sol = solve_ivp(susceptible, [0, 10], [2, 4, 8])
