import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

# Importing data
df = pd.read_csv(os.getcwd()+'/Data/sensor1.csv')

# Understanding Data
# print(df.tail())

# Humidity at sensor
# ax = df.humidity.plot.line()
# ax = df.humidity.plot.hist(bins=100)


# Weather Humidity
# ax = df.weather_humidity.plot.hist(bins=100)

# Wind
# ax = df.weather_wind_dir.plot.density()
# ax = df.weather_wind_dir.plot.hist(bins=100)

# Season
# print(df.timestamp.min())
# print(df.timestamp.max())

# Correlation Matrix
# corrMatrix = df[df['sensor_id'] == 4].corr()
# sn.heatmap(corrMatrix)
# plt.show()

# Plotting
# ax.plot()
# ax.set_xlabel("Degrees")
# ax.set_ylabel("Frequency")
# ax.set_title("Wind Direction")
# plt.show()

# Plotting BY SENSOR
# df = pd.read_csv(os.getcwd()+'/Data/sensor1.csv')
# fig, axs = plt.subplots(6,7)
# fig.set_size_inches(15, 15)
# for (i, plots) in enumerate(axs.flatten()):
#     df_s1_m = df[df["sensor_id"] == i+1]["humidity"]
#     plots.hist(df_s1_m, bins = 30, range=(0,100))


# fig.tight_layout()
# fig.show()