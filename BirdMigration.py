import numpy as np
from pandas import DataFrame
from scipy.interpolate import RegularGridInterpolator
# import geopandas as gpd
# import movingpandas as mpd
# import matplotlib.pyplot as plt

class BirdMigration:

    R_EARTH = 6378137.0

    def __init__(
            self, 
            start_birds, 
            wind, 
            dates, 
            pressure_lvls, 
            bounds
    ):
        '''
        Create agent based model for migration simulation

        Parameters
        ----------
        start_birds : numpy.ndarray
            nx3 array containing pressure level, lat, long of each bird
        wind : numpy.ndarray
            5D array containing wind values.
        dates : numpy.ndarray
            Dates and times of the wind array.
        pressure_lvls : numpy.ndarray
            Pressure levels of the wind array.
        bounds : numpy.ndarray
            Bounds of the wind array.
        '''
        
        self.start_birds = start_birds
        self.wind = wind
        self.dates = dates
        self.pressure_lvls = pressure_lvls
        self.bounds = bounds
    
        self.f = RegularGridInterpolator((
            np.arange(wind.shape[0]), 
            np.arange(wind.shape[1]), 
            np.flip(pressure_lvls), 
            np.arange(
                bounds[0,0], 
                bounds[0,1] + 0.25, 
                0.25).astype(np.float32
            ),  
            np.arange(
                bounds[1,0], 
                bounds[1,1] + 0.25, 
                0.25).astype(np.float32)
            ),
            wind
        )

        
    def bearing_from_loc(
            self, 
            loc, 
            goal
    ):
        '''
        Get the bearing from the position of a bird to its goal

        Parameters
        ----------
        loc : numpy.ndarray
            Array containing current bird position.
        goal : numpy.ndarray
            2D-Array containing goal of each bird.

        Returns
        -------
        float
            Bearing to the goal.

        '''
        
        if loc.shape[1] == 2:
            
            phi1 = np.radians(loc[:,0])
            labda1 = np.radians(loc[:,1])
            phi2 = np.radians(goal[:,0])
            labda2 = np.radians(goal[:,1])
            y = np.sin(labda2-labda1) * np.cos(phi2)
            x = np.cos(phi1) * np.sin(phi2) - \
                np.sin(phi1) * np.cos(phi2) * np.cos(labda2 - labda1)
            
            return np.arctan2(y, x)
        
        if loc.shape[1] == 3:
            
            phi1 = np.radians(loc[:,1])
            labda1 = np.radians(loc[:,2])
            phi2 = np.radians(goal[:,1])
            labda2 = np.radians(goal[:,2])
            y = np.sin(labda2-labda1) * np.cos(phi2)
            x = np.cos(phi1)*np.sin(phi2) - \
                np.sin(phi1)*np.cos(phi2)*np.cos(labda2-labda1)
            
            return np.arctan2(y, x)
        
    def tailwind(
            self, 
            u, 
            v, 
            bearing, 
            self_speed
    ):
        '''
        Calculate tailwind of the bird

        Parameters
        ----------
        u : numpy.ndarray
            Array containing u component of wind speed for each bird.
        v : numpy.ndarray
            Array containing v component of wind speed for each bird.
        bearing : numpy.ndarray
            Array containing bearing of each bird.
        self_speed : int
            Air speed of the bird.

        Returns
        -------
        tailwind : numpy.ndarray
            Tailwind of each bird.

        '''
        
        # calculate the angle between wind and bird
        alpha = np.where(
            (u >= 0) == (v >= 0), 
            bearing, 
            np.where(
                (u > 0) == (v < 0), 
                np.pi - bearing,
                np.where(
                    (u < 0) == (v > 0), 
                    2*np.pi - bearing,
                    bearing - np.pi
                )
            )
        )

        u_bird = np.sin(alpha) * self_speed # u component of bird speed
        v_bird = np.cos(alpha) * self_speed # v component of bird speed
        
        w = np.sqrt(u ** 2 + v ** 2)
        cos_beta = (u*u_bird + v*v_bird) / (w*self_speed)
        tailwind = w * cos_beta
        
        return tailwind
    
    def optimal_height(
            self, 
            locs, 
            bearing, 
            t, 
            self_speed, 
            f, 
            pressure_lvls, 
            threshold, 
            step, 
            num_neighbors
    ):
        '''
        Calculate pressure level with best tailwind for each bird

        Parameters
        ----------
        locs : numpy.ndarray
            Current location of birds
        bearing : numpy.ndarray
            Current heading of birds
        t : TYPE
            DESCRIPTION.
        self_speed : int
            Airspeed of birds
        f : RegularGridInterpolator
            Interpolating function
        pressure_lvls : numpy.ndarray
            Array containing pressure levels
        threshold : float
            Threshold that determines how much better new height must be
        step : int
            Stepsize for pressure levels to calculate new tailwind
        num_neighbors : int
            Number of neighboring pressure levels to evaluate

        Returns
        -------
        p : numpy.ndarray
            Optimal pressure level/height.

        '''
        
        # we start with the current height
        p = locs[:, 1].copy()
        
        u = f(np.insert(locs, 0, 0, axis=1))
        v = f(np.insert(locs, 0, 1, axis=1))
        
        tail_wind = self.tailwind(
            u, 
            v, 
            bearing, 
            self_speed
        )
        
        # at t = 1 we go up
        # we keep climbing until wind support does not increase significantly   
        
        if t == 1:
            
            ps = np.array([
                np.arange(
                    max(pressure_lvls),
                    min(pressure_lvls) - step, 
                    -step
                ) 
                for x in p
            ])
            
            for column in ps.T:
                test_locs = np.column_stack((
                    locs[:,0],
                    column, locs[:,2:]
                ))
                u = f(np.insert(test_locs, 0, 0, axis=1))
                v = f(np.insert(test_locs, 0, 1, axis=1))
                
                new_tail_wind = self.tailwind(u, v, bearing, self_speed)
                
                mask = new_tail_wind > tail_wind + threshold
                p[np.where(mask)] = column[np.where(mask)]
                
        
        # we want to check above and below after departing    
        else:
            
            ps_up = np.array([
                np.arange(
                    x-step, 
                    x - step*num_neighbors, 
                    -step) 
                for x in p
            ])
            ps_up[ps_up <= min(pressure_lvls)] = min(pressure_lvls)
            ps_down = np.array([
                np.arange(x+step, 
                          x + step*num_neighbors, 
                          step)
                for x in p
            ])
            # we cannot fly above these values
            ps_down[ps_down >= max(pressure_lvls)] = max(pressure_lvls) - step        
            # afterwards we cannot go to 1000 hPa, only to 990.
            ps = np.column_stack((ps_down, ps_up))
            
            for column in ps.T:
                test_locs = np.column_stack((
                    locs[:, 0], 
                    column, 
                    locs[:,2:]
                ))
                u = f(np.insert(test_locs, 0, 0, axis=1))
                v = f(np.insert(test_locs, 0, 1, axis=1))
                
                new_tail_wind = self.tailwind(
                    u,
                    v,
                    bearing,
                    self_speed
                )
                
                mask = new_tail_wind > tail_wind + threshold
                p[np.where(mask)] = column[np.where(mask)]
                
        return p
    
    def model(self, 
              start_time: int, 
              update_time:int, 
              runtime: int, 
              bearing=None, 
              goal=None, 
              comp=0.1, 
              self_speed:int=12):
        '''
        
        Parameters
        ----------
        start_time : int
            Time of dates interval to start simulation.
        update_time : int
            Update time of simulation in hours.
        runtime : int
            Time of dates interval to end simulation.
        bearing : numpy.ndarray, optional
            Heading of the bird. The default is None.
        goal : numpy.ndarray, optional
            Goal of the bird. The default is None.
        comp : float, optional
            Stoachatic error for flying towards the goal. The default is 0.1.
        self_speed : int, optional
            Airspeed of the bird. The default is 12.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing information of birds at each timestep.

        '''
        
        self.start_time = start_time
        self.runtime = runtime
        num_birds = self.start_birds.shape[0]
        
        # array with the current timepoint (0), start height in P 
        locs = np.hstack((
            np.zeros((num_birds, 1), dtype=int) + start_time,
            self.start_birds
        ))

        #for the model where you only look at the initial bearing
        if isinstance(bearing, np.ndarray):
            
            for t in range(1, runtime+1, update_time):
                
                
                # select current information
                current_locs = locs[(t-1)*num_birds:t*num_birds, :]
                current_bearing = bearing[(t-1)*num_birds:t*num_birds]
                
                p = self.optimal_height(
                    current_locs, 
                    current_bearing, 
                    t, 
                    self_speed, 
                    self.f, 
                    self.pressure_lvls, 
                    threshold=1, 
                    step=10, 
                    num_neighbors=5
                )
                    
                current_locs[:,1] = p
                
                # wind speed components
                u_w = self.f(np.insert(current_locs, 0, 0, axis=1))
                v_w = self.f(np.insert(current_locs, 0, 1, axis=1))
                
                # bird airspeed components
                u_b = self_speed * np.sin(current_bearing)
                v_b = self_speed * np.cos(current_bearing) 

                # calculating new position
                lat_1 = locs[(t-1)*num_birds:t*num_birds:,2]
                phi_1 = np.radians(lat_1)
                lon_1 = locs[(t-1)*num_birds:t*num_birds:,3]
                labda_1 = np.radians(lon_1)
                
                d = update_time * 3600 * np.sqrt(
                    (u_w + u_b) ** 2 + (v_w + v_b) ** 2
                ) 
                brng = np.arctan((u_w + u_b) / (v_w + v_b))
                
                phi_2 = np.arcsin(
                    np.sin(phi_1)*np.cos(d/self.R_EARTH) + \
                    np.cos(phi_1)*np.sin(d/self.R_EARTH)*np.cos(brng)
                )
                labda_2 = labda_1 + \
                    np.arctan2(
                        np.sin(brng)*np.sin(d/self.R_EARTH)*np.cos(phi_1),
                        np.cos(d/self.R_EARTH)-np.sin(phi_1)*np.sin(phi_2)
                )
                lat_2 = np.degrees(phi_2)
                lon_2 = np.degrees(labda_2)
                
                # making sure we stay on our map
                lat_2[lat_2 < self.bounds[0,0]] = self.bounds[0,0]
                lat_2[lat_2 > self.bounds[0,1]] = self.bounds[0,1]
                lon_2[lon_2 < self.bounds[1,0]] = self.bounds[1,0]
                lon_2[lon_2 > self.bounds[1,1]] = self.bounds[1,1]
                
                # calculate new bearing
                new_bearing = (
                    self.bearing_from_loc(
                        np.column_stack((lat_2, lon_2)),
                        np.column_stack((lat_1, lon_1))
                    ) + np.pi) % (2*np.pi)
                bearing = np.concatenate((
                    bearing, 
                    new_bearing
                ))
                
                locs = np.vstack((
                    locs, 
                    np.column_stack((
                        np.zeros(num_birds,dtype=int)+start_time+t, 
                        p, 
                        lat_2, 
                        lon_2
                    ))
                ))
     
        if isinstance(goal, np.ndarray):
            
            #calculate shortest angle to the goal and add a stochastic error
            bearing = self.bearing_from_loc(
                locs[(t-1)*num_birds:t*num_birds,1:], 
                goal
            ) + (2*np.random.rand(7)-1) * comp
            
            for t in range(1, runtime+1, update_time):
               
                # select current information
                current_locs = locs[(t-1)*num_birds:t*num_birds, :]
                current_bearing = bearing[(t-1)*num_birds:t*num_birds]
                
                current_locs[:,1] = p
                
                # wind speed components
                u_w = self.f(np.insert(current_locs, 0, 0, axis=1))
                v_w = self.f(np.insert(current_locs, 0, 1, axis=1))
                
                # bird airspeed components
                u_b = self_speed * np.sin(current_bearing)
                v_b = self_speed * np.cos(current_bearing)

                # calculating new position
                lat_1 = locs[(t-1)*num_birds:t*num_birds:,2]
                phi_1 = np.radians(lat_1)
                lon_1 = locs[(t-1)*num_birds:t*num_birds:,3]
                labda_1 = np.radians(lon_1)
                
                d = update_time*3600 * np.sqrt(
                    (u_w + u_b) ** 2 + (v_w + v_b) ** 2
                )
                brng = np.arctan((u_w + u_b) / (v_w + v_b))
                
                phi_2 = np.arcsin(
                    np.sin(phi_1)*np.cos(d/self.R_EARTH) + \
                        np.cos(phi_1)*np.sin(d/self.R_EARTH)*np.cos(brng)
                )
                labda_2 = labda_1 + \
                    np.arctan2(
                        np.sin(brng)*np.sin(d/self.R_EARTH)*np.cos(phi_1), 
                        np.cos(d/self.R_EARTH)-np.sin(phi_1)*np.sin(phi_2)
                ) 
                lat_2 = np.degrees(phi_2)
                lon_2 = np.degrees(labda_2)
                
                # making sure we stay on our map
                lat_2[lat_2 < self.bounds[0,0]] = self.bounds[0,0]
                lat_2[lat_2 > self.bounds[0,1]] = self.bounds[0,1]
                lon_2[lon_2 < self.bounds[1,0]] = self.bounds[1,0]
                lon_2[lon_2 > self.bounds[1,1]] = self.bounds[1,1]
                
                locs = np.vstack((
                    locs,
                    np.column_stack((
                        np.zeros(num_birds,dtype=int)+start_time+t,
                        p, 
                        lat_2, 
                        lon_2
                    ))
                ))
                
                # calculate new bearing
                new_bearing = self.bearing_from_loc(
                    locs[(t-1)*num_birds:t*num_birds,1:], 
                    goal
                ) + (2*np.random.rand(7)-1) * comp
                
                bearing = np.concatenate((
                    bearing, 
                    new_bearing
                ))
            
        df = DataFrame(
            np.column_stack((
                np.tile(
                    np.arange(1, num_birds+1), 
                    self.runtime+1
                ), 
                locs, 
                bearing
            )), 
            columns=['bird', 
                     't', 
                     'p', 
                     'lat', 
                     'lon', 
                     'bearing']
        )
        
        # getting datetimes
        df['t'] = df['t'].apply(lambda row: self.dates[int(row)])
            
        return df
    
    # def plot(self, model):
    #     '''
    #     DOES NOT WORK PROPERLY YET

    #     Parameters
    #     ----------
    #     model : TYPE
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     '''
        
        
    #     num_birds = self.start_birds.shape[0]
        
    #     gdf_start = gpd.GeoDataFrame(DataFrame(data={'bird': list(range(1, num_birds+1))}), 
    #                                   geometry=gpd.points_from_xy(model.loc[model.t == self.dates[self.start_time],'lon'], 
    #                                                               model.loc[model.t == self.dates[self.start_time],'lat']), crs="EPSG:4326")
    #     gdf_stop = gpd.GeoDataFrame(DataFrame(data={'bird': list(range(1, num_birds+1))}), 
    #                                 geometry=gpd.points_from_xy(model.loc[model.t == self.dates[self.runtime+self.start_time],'lon'], 
    #                                                             model.loc[model.t == self.dates[self.runtime+self.start_time],'lat']), crs="EPSG:4326")
        
                  
    #     world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    #     ax = world.plot(color='white', edgecolor='black')
    #     ax.set_xlim(self.bounds[1,0], self.bounds[1,1])
    #     ax.set_ylim(self.bounds[0,0], self.bounds[0,1])

    #     gdf_start.plot(ax=ax, color='red', markersize=20)
    #     gdf_stop.plot(ax=ax, color='green', markersize=20)

    #     model.rename(columns={'t': 'bird'}, inplace=True)
    #     model['bird'] = np.tile(np.arange(1, num_birds+1), self.runtime+1)        
    #     model['t'] = np.repeat([self.dates[self.start_time] + pd.Timedelta(hours, 'h') for hours in range(self.runtime+1)], num_birds)

    #     df_traj = DataFrame(model)

    #     geo_df = gpd.GeoDataFrame(df_traj, 
    #                               geometry=gpd.points_from_xy(df_traj.lon, df_traj.lat), 
    #                               crs="EPSG:4326")

    #     traj_collection = mpd.TrajectoryCollection(geo_df, 'bird', t='t', crs="EPSG:4326")
    #     traj_collection.plot(ax=ax, column='bird')

    #     plt.show()