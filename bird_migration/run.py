from BirdMigration import BirdMigration
from tqdm import tqdm
import numpy as np
import xarray as xr

START_BIRDS = np.insert(
    np.flip(
        np.genfromtxt(
            'startBirdPosition.csv', 
            delimiter=',',
            skip_header=1)
        ), 
    0, 1000, axis=1
)

da = xr.open_dataset(
    'autumn_data.grib', 
    engine='cfgrib'
)
TIMES = da.time.values
PRESSURE_LVLS = da.isobaricInhPa.values
BOUNDS = np.array(
    [[min(da.latitude.values), max(da.latitude.values)],
     [min(da.longitude.values), max(da.longitude.values)]]
)

START_TIME = 0
UPDATE_TIME = 1
RUNTIME = 40

START_BIRDS = START_BIRDS[
    (START_BIRDS[:, 1] >= BOUNDS[0,0]) & 
    (START_BIRDS[:, 1] <= BOUNDS[0,1]) & 
    (START_BIRDS[:, 2] >= BOUNDS[1,0]) & 
    (START_BIRDS[:, 2] <= BOUNDS[1,1])
]
NUM_BIRDS = START_BIRDS.shape[0]
SELF_SPEED = (7*np.random.rand(NUM_BIRDS)) + 8

for t in tqdm(range(len(TIMES)-RUNTIME+1)):
    
    BEARING = np.random.uniform(
        np.pi/2, 
        3*np.pi/2, 
        NUM_BIRDS
    )
    SELF_SPEED = (7*np.random.rand(NUM_BIRDS*N_SIMS)) + 8
    
    WIND = np.stack((
        da['u'].sel(time = xr.DataArray(TIMES[t:t+RUNTIME])).values, 
        da['v'].sel(time = xr.DataArray(TIMES[t:t+RUNTIME])).values
    ))
       
    birds = BirdMigration(
        START_BIRDS, 
        WIND, 
        TIMES, 
        PRESSURE_LVLS,
        BOUNDS
    )
    model = birds.model(
        START_TIME, 
        UPDATE_TIME, 
        RUNTIME, 
        BEARING, 
        self_speed=SELF_SPEED
    )

    file_name = f"df_{str(TIMES[t])[:13]}_{str(TIMES[t+RUNTIME])[:13]}.p"
    model['airspeed'] = np.nan
    model.loc[:SELF_SPEED.shape[0] - 1, 'airspeed'] = SELF_SPEED
    model.to_pickle(fr"results\{file_name}")
    
    )
    
    
