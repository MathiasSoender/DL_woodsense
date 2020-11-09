#File for data preproccessing
import io
import json
import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from plotting import plot_time_series_class

RANDOM_SEED = 42

#Change directory
cwd = os.getcwd()

#Get the data out in the right format
df = pd.read_csv(cwd+'/data/sensor1.csv',infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])

#Find the null values for each attribute
print(df.isnull().sum())
# This shows that we have a lot of null values. ^

#Filling out null values with values from the closest previous time
df = df.astype('float32')

#Forward will all NaN values with previous existing values
df.fillna(method='ffill', inplace=True)

print(df.isnull().sum())
#^This shows us that we have one attribute where all values are Nan. Apart from this, is looks like it worked.

print(df.head())

#################################################

#shuffle the elements
df = df.sample(frac=1.0)

#Splitting the data in to training, validation and test
train_df, val_df = train_test_split(
  df,
  test_size=0.15,
  random_state=RANDOM_SEED
)
val_df, test_df = train_test_split(
  val_df,
  test_size=0.33,
  random_state=RANDOM_SEED
)
