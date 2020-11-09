import io
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

cwd = os.getcwd()
df = pd.read_csv(cwd+'/data/sensor1.csv')
print(df.head())

df_sensor_1 = df[df['sensor_id'] == 1]
_ = df_sensor_1.set_index('timestamp')[['temperature', 'humidity', 'ohms', 'moisture']].plot(subplots=True, figsize=(23, 14))
plt.show()