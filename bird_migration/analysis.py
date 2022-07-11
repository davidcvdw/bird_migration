import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

dfs = pd.read_pickle(r"new_results_constant\processed\europe_birds_wind.p")

# Total simulated birds
num_sims = len(glob("new_results_constant" + '/*.p'))
num_birds = 3329
total_birds = num_sims * num_birds

# Arrived birds per country
arrived_country = dfs[dfs.country.notna()].groupby('country')['bird'].count()
# Total arrived birds
total_arrived = sum(arrived_country)

# Number of hours with succesive arrivals
len(dfs.departure.unique())

# We want to look at which days/hours/time of the day is most succesful
## UTC-4!!
from astral import Observer, sun
import datetime as dt
from timezonefinder import TimezoneFinder
tf = TimezoneFinder()

def set_tz(x):
    tz = tf.timezone_at(lng=x.lon, lat=x.lat)
    return x.departure.tz_convert(tz)

def sunset(x):
    tz = tf.timezone_at(lng=x.lon, lat=x.lat)
    observer = Observer(x.lat, x.lon)
    
    date = x.departure
    date_1 = x.departure - pd.DateOffset(days=1) 
    
    sunset = sun.sunset(observer, date, tzinfo=tz)
    sunset_1 = sun.sunset(observer, date_1, tzinfo=tz)
    
    t_sunset = (date - sunset) / pd.Timedelta(hours=1)
    t_sunset_1 = (date - sunset_1) / pd.Timedelta(hours=1)
    
    return min(t_sunset, t_sunset_1, key=abs)
    
def sunrise(x):
    
    tz = tf.timezone_at(lng=x.lon, lat=x.lat)
    observer = Observer(x.lat, x.lon)
    date = x.departure
    sunrise = sun.sunrise(observer, date, tzinfo=tz)
    t_sunrise = (sunrise - date) / pd.Timedelta(hours=1)    
    
    return t_sunrise

def delta_night(x):
    return min(x.sunset, x.sunrise, key=abs)

dfs0 = dfs[dfs.t == 0].copy()

index = pd.DatetimeIndex(dfs0['departure'])

dfs0.reset_index(drop=True, inplace=True)

dfs0['departure'] = dfs0['departure'].dt.tz_localize('UTC')
dfs0['departure'] = dfs0.apply(set_tz, axis=1)

dfs0['sunset'] = dfs0.apply(sunset, axis=1)
# dfs0['sunrise'] = dfs0.apply(sunrise, axis=1)

# dfs0['delta_night'] = dfs0.apply(delta_night, axis=1)

# dfs0['dep_from_sunset'] = dfs0.apply(time_from_sunset, axis = 1)

hours = [dt.time(i).strftime('%H:%M') for i in range(24)]
hours.append(hours[0])
for i in range(24):
    arrival_h = len(dfs0['departure'].iloc[index.indexer_between_time(hours[i], hours[i+1], include_end=False)])
    print("Departing between {} and {} UTC: {}".format(hours[i], hours[i+1], arrival_h))

print("\nTotal arrived birds: {}".format(total_arrived))
print("Total simulated birds: {}".format(total_birds))
print("% Birds arrived: {:.3%}\n".format(total_arrived/total_birds))
print("Arrived birds per country \n{}\n".format(arrived_country))

# Take-off information
take_off = dfs[dfs.t == 0][['p', 'lat', 'lon', 'bearing', 'airspeed']].describe()[1:]
print("Take off information\n{}\n".format(take_off))

# Arrival information
arrival = dfs[dfs.country.notna()][['t', 'p', 'lat', 'lon']].describe()[1:]
print("Arrival information\n{}\n".format(arrival))

# Information per time step
time_step = dfs.groupby('t')[['p', 'lat', 'lon']].describe().iloc[:,[0,1,2,9,10,17,18]]
print("Information per time step\n{}\n".format(time_step))

# Sun infromation
bins = list(range(-12,13))
ax = dfs0.sunset.hist(bins=bins, grid=False, density=True)
ax.set_xticks(list(range(-12,13,2)))
ax.set_xlabel('Hours from sunset')
ax.set_ylabel('Number of birds')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

# Wind information
wind = dfs[['u', 'v']].describe()[1:]
# Compare against others???

# Bearing information
import numpy as np
ax = np.degrees(dfs0.bearing).hist(bins = int(180/5), grid=False, density=True)
ax.set_xticks(list(range(90, 280, 20)))
ax.set_xlabel('Bearing in degrees')
ax.set_ylabel('Number of birds')
plt.savefig('hist', dpi=1000)

# Explore high bearing
mask = dfs0[dfs0.bearing > np.pi].bird
high_bearing = dfs[dfs.bird.isin(mask)].copy()
high_bearing[['t', 'p', 'airspeed', 'lat', 'lon', 'u', 'v']].describe()[1:]
high_bearing[high_bearing.t==0].departure # departure dates
high_bearing[high_bearing.country.notna()].country # countries of arrival

    