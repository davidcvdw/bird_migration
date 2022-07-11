from bird_migration import BirdMigration
from tqdm import tqdm
import numpy as np
from xarray import open_dataset, DataArray
from glob import glob
from warnings import filterwarnings
from pandas import DataFrame, read_pickle, Timedelta
from geopandas import read_file, GeoDataFrame, points_from_xy
from geopandas.datasets import get_path
from geopandas.tools import sjoin
from scipy.interpolate import RegularGridInterpolator

filterwarnings("ignore") 

# where to save files
dst_path = "new_results_constant"

## RUN SIMULATIONS

START_BIRDS = np.insert(
    np.flip(
        np.genfromtxt(
            r"datasets\startBirdPosition.csv", 
            delimiter=',',
            skip_header=1)
        ), 
    0, 1000, axis=1
)

da = open_dataset(
    r"datasets\autumn_data.grib", 
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

for t in tqdm(range(716)):
    
    np.random.seed(t)
    
    BEARING = np.random.uniform(
        np.pi/2, 
        3*np.pi/2, 
        NUM_BIRDS
    )

    SELF_SPEED = (7*np.random.rand(NUM_BIRDS)) + 8
    
    WIND = np.stack((
        da['u'].sel(time = DataArray(TIMES[t:t+RUNTIME])).values, 
        da['v'].sel(time = DataArray(TIMES[t:t+RUNTIME])).values
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
        bearing=BEARING, 
        self_speed=SELF_SPEED
    )

    file_name = f"df_{str(TIMES[t])[:13]}_{str(TIMES[t+RUNTIME])[:13]}.p"
    model['airspeed'] = np.nan
    model.loc[:SELF_SPEED.shape[0] - 1, 'airspeed'] = SELF_SPEED
    model.to_pickle(fr"{dst_path}\{file_name}")  
    
## PROCESSES: WHICH BIRDS IN EUROPE?

bird_count = 1
dfs = []

files = glob(f"{dst_path}" + '/*.p')
for file in tqdm(files):
    file_name = file.replace(f"{dst_path}\\", '')
    
    model = read_pickle(fr"{file}")
    
    START_TIME = model.loc[0, 't']
    RUNTIME = model.loc[model.index[-1], 't']
    BOUNDS = np.array([[30,70],
                       [-90,10]])
    
    world = read_file(
        get_path('naturalearth_lowres')
    )
    world.drop(['pop_est', 'iso_a3', 'gdp_md_est'], 
               axis=1, 
               inplace = True
    )
    
    gdf_model = GeoDataFrame(
        model, 
        crs="EPSG:4326", 
        geometry=points_from_xy(
            model.lon, 
            model.lat
        )
    )
    gdf_model['departure'] = np.datetime64(file_name[3:16])
    # join current gdf with world metadata
    joined = sjoin(
        gdf_model, 
        world, 
        op="within", 
        how='left'
    )
    # remove countries and continent not (in) europe
    joined['name'].where(
        joined.continent == 'Europe', 
        np.nan, 
        inplace=True
    )
    joined['continent'].where(
        joined.continent == 'Europe', 
        np.nan, 
        inplace=True
    )
    # filter birds only in europe
    europe = joined[joined.continent == 'Europe'].drop_duplicates('bird')
        
    birds = []
    grouped = joined.groupby('bird')
    for bird in europe.bird.unique():
        group = grouped.get_group(bird)
        group['bird'] = int(bird_count)
        bird_count += 1
        t = europe[europe.bird == bird].t.values[0]
        birds.append(group[group.t <= t].values.tolist())
    birds = [row for rows in birds for row in rows]
    dfs.append(birds)
    
dfs = DataFrame([row for rows in dfs for row in rows], 
    columns = list(joined.columns)
)
dfs.drop(
    world.columns.drop('name'),
    axis=1, 
    inplace=True
)
dfs.drop('index_right', axis=1, inplace=True)
dfs = dfs.astype({'bird': 'int32', 't':'int32'})
dfs.rename({'name': 'country'}, axis=1, inplace=True)
dfs.airspeed = dfs.airspeed.ffill()
dfs.to_pickle(fr"{dst_path}\processed\europe_birds.p")

## Evaluating wind speeds

wind_comps = []
for date in tqdm(dfs.departure.unique()):
    
    df2 = dfs[dfs.departure == date].reset_index(drop=True)
    
    # We need to remove the final timestep, because we do not fly then
    df = df2[~df2.country.notna()]
    
    list_dates = [df.loc[0, 'departure'] + 
                  Timedelta(hours=i) for i in range(40)]
    
    WIND = np.stack((
        da['u'].sel(time = DataArray(list_dates)).values, 
        da['v'].sel(time = DataArray(list_dates)).values
    ))
    
    f = RegularGridInterpolator((
        np.arange(WIND.shape[0]), 
        np.arange(WIND.shape[1]), 
        np.flip(PRESSURE_LVLS), 
        np.arange(
            BOUNDS[0,0], 
            BOUNDS[0,1] + 0.25, 
            0.25).astype(np.float32
        ),  
        np.arange(
            BOUNDS[1,0], 
            BOUNDS[1,1] + 0.25, 
            0.25).astype(np.float32)
        ),
        WIND
    )
    
    u = f(np.column_stack((np.zeros(len(df)), df.t.values, df.p.values, df.lat.values, df.lon.values)))
    v = f(np.column_stack((np.ones(len(df)), df.t.values, df.p.values, df.lat.values, df.lon.values)))
    
    wind_comps.append(np.column_stack((u,v)))

dfs.loc[~dfs.country.notna(), ['u', 'v']] = np.concatenate(wind_comps)

dfs.to_pickle(fr"{dst_path}\processed\europe_birds_wind.p")
