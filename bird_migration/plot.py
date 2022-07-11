import numpy as np
import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import matplotlib.pyplot as plt
from tqdm import tqdm
from warnings import filterwarnings

filterwarnings("ignore") 

path = "new_results_constant"

dfs = pd.read_pickle(fr"{path}\processed\europe_birds.p")

world = gpd.read_file(
    gpd.datasets.get_path('naturalearth_lowres')
)
BOUNDS = np.array([[30,70],
                   [-90,10]])

grouped = dfs.groupby('departure')
for date in tqdm(dfs.departure.unique()):
    group = grouped.get_group(date)
    
    gdf_traj = gpd.GeoDataFrame(
        group, 
        crs="EPSG:4326", 
        geometry=gpd.points_from_xy(
            group.lon, 
            group.lat
        )
    )
    
    ax = world.plot(color='white', edgecolor='black')
    ax.set_xlim(BOUNDS[1,0], BOUNDS[1,1])
    ax.set_ylim(BOUNDS[0,0], BOUNDS[0,1])
    
    gdf_traj[gdf_traj.t == 0].plot(ax=ax, color='green', markersize=20)
    gdf_traj[gdf_traj.country.notna()].plot(ax=ax, color='red', markersize=20)
    
    traj_collection = mpd.TrajectoryCollection(
        gdf_traj, 
        'bird', 
        t='t', 
        crs="EPSG:4326"
    )
    traj_collection.plot(
        ax=ax, 
        column='bird', 
    )
    
    plt.savefig(fr"{path}\processed\images\{str(date)[:13] + '.png'}", dpi=1000)
    plt.clf()