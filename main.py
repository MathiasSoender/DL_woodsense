import matplotlib.pyplot as plt
import pandas as pd
import os

cwd = os.getcwd()
df = pd.read_csv(cwd+'/data/sensor1.csv')
print(df.head())