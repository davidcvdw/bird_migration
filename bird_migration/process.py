from glob import glob
from warnings import filterwarnings
from tqdm import tqdm
from pandas import DataFrame, read_pickle
import numpy as np
import geopandas as gpd
# import movingpandas as mpd
# import matplotlib.pyplot as plt

filterwarnings("ignore") 

bird_count = 1
dfs = []

files = glob("results" + '/*.p')
for file in tqdm(files):
    file_name = file.replace('results\\', '')
    
    model = read_pickle(fr"{file}")
    
    START_TIME = model.loc[0, 't']
    RUNTIME = model.loc[model.index[-1], 't']
    BOUNDS = np.array([[30,70],
                       [-90,10]])
    
    world = gpd.read_file(
        gpd.datasets.get_path('naturalearth_lowres')
    )
    world.drop(['pop_est', 'iso_a3', 'gdp_md_est'], 
               axis=1, 
               inplace = True
    )
    
    gdf_model = gpd.GeoDataFrame(
        model, 
        crs="EPSG:4326", 
        geometry=gpd.points_from_xy(
            model.lon, 
            model.lat
        )
    )
    gdf_model['departure'] = np.datetime64(file_name[3:16])
    # join current gdf with world metadata
    joined = gpd.tools.sjoin(
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
dfs = dfs.astype({'bird': int, 't': int})
dfs.rename({'name': 'country'}, axis=1, inplace=True)

dfs.to_pickle(r"results\processed\europe_birds.p")
    
    # df_traj = model[model['bird'].isin(europe.bird)]
    
    # if df_traj.empty:
    #     continue
    # df_traj.to_pickle(fr"results\processed\{file_name}")
    
    # gdf_traj = gpd.GeoDataFrame(
    #     df_traj, 
    #     crs="EPSG:4326", 
    #     geometry=gpd.points_from_xy(
    #         df_traj.lon, 
    #         df_traj.lat
    #     )
    # )
    # 
    # ax = world.plot(color='white', edgecolor='black')
    # ax.set_xlim(BOUNDS[1,0], BOUNDS[1,1])
    # ax.set_ylim(BOUNDS[0,0], BOUNDS[0,1])
    
    # gdf_traj[gdf_traj.t == START_TIME].plot(
    #     ax=ax, 
    #     color='red', 
    #     markersize=20
    # )
    # gdf_traj[gdf_traj.t == RUNTIME].plot(
    #     ax=ax, 
    #     color='green', 
    #     markersize=20
    # )
    
    # traj_collection = mpd.TrajectoryCollection(
    #     gdf_traj, 
    #     'bird', 
    #     t='t', 
    #     crs="EPSG:4326"
    # )
    # traj_collection.plot(
    #     ax=ax, 
    #     column='bird', 
    # )
    
    # plt.savefig(fr"results\processed\images\{file_name[3:] + 'ng'}")