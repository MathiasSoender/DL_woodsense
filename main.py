import io
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

# Importing data
df = pd.read_csv(os.getcwd()+'/Data/sensor1.csv')
