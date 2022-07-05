import numpy as np
from pandas import DataFrame
from scipy.interpolate import RegularGridInterpolator

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
        
        self.num_birds = self.start_birds.shape[0]
    
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
            
            return (np.arctan2(y, x) + 2*np.pi) % (2*np.pi)
        
        if loc.shape[1] == 3:
            
            phi1 = np.radians(loc[:,1])
            labda1 = np.radians(loc[:,2])
            phi2 = np.radians(goal[:,0])
            labda2 = np.radians(goal[:,1])
            y = np.sin(labda2-labda1) * np.cos(phi2)
            x = np.cos(phi1)*np.sin(phi2) - \
                np.sin(phi1)*np.cos(phi2)*np.cos(labda2-labda1)
            
            return (np.arctan2(y, x) + 2*np.pi) % (2*np.pi)
    
    def get_angle_with_y(
            self,
            u,
            v,
            alpha
        ):
        '''
        alpha: arctan(u/v)
        '''
        
        # calculate the angle between wind and bird
        brng  = np.where(
            (u >= 0) & (v >= 0), 
            alpha, 
            np.where(
                (u > 0) & (v < 0), 
                np.pi - alpha,
                np.where(
                    (u < 0) & (v > 0), 
                    2*np.pi - alpha,
                    alpha + np.pi
                )
            )
        )
        
        return brng
    
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
    
    def get_bird_speed(self, self_speed, bearing):
        u_b = self_speed * np.sin(bearing)
        v_b = self_speed * np.cos(bearing)
        return u_b, v_b
    
    def get_current_degrees(self, locs):
        lat_1 = locs[:,2]
        lon_1 = locs[:,3]
        return lat_1, lon_1
    
    def get_current_radians(self, lat, lon):
        phi_1 = np.radians(lat)
        lambda_1 = np.radians(lon)
        return phi_1, lambda_1
    
    def get_ground_distance(self, u, v):
        d = self.update_time*3600 * np.sqrt(
            (u) ** 2 + (v) ** 2
        )
        alpha = abs(np.arctan(u / v))
        return d, alpha
      
    def get_new_radians(self, phi_1, lambda_1, d, brng):
        phi_2 = np.arcsin(
            np.sin(phi_1)*np.cos(d/self.R_EARTH) + \
                np.cos(phi_1)*np.sin(d/self.R_EARTH)*np.cos(brng)
        )
        labda_2 = lambda_1 + \
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
        
        return lat_2, lon_2
    
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
        self.update_time = update_time
        self.runtime = runtime
        
        # array with the current timepoint (0), start height in P 
        locs = []
        
        loc = np.hstack((
            np.zeros((self.num_birds, 1), dtype=int) + start_time,
            self.start_birds
        ))
        
        locs.append(loc)
        
        bearings = []

        #for the model where you only look at the initial bearing
        if isinstance(bearing, np.ndarray):
            bearings.append(bearing)
            for t in range(1, runtime+1, update_time):
            
                # select current information
                current_locs = locs[t-1]
                current_bearing = bearings[t-1]
                
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
                u_b, v_b = self.get_bird_speed(self_speed, bearing)

                # calculating new position
                lat_1, lon_1 = self.get_current_degrees(current_locs)
                phi_1, lambda_1 = self.get_current_radians(lat_1, lon_1)
                
                u = u_w + u_b
                v = v_w + v_b
                
                d, alpha = self.get_ground_distance(u, v)
                brng = self.get_angle_with_y(u, v, alpha)

                lat_2, lon_2 = self.get_new_radians(phi_1, lambda_1, d, brng)
                
                bearings.append(current_bearing)
                
                # calculate new location
                new_loc = np.column_stack((
                    np.zeros(
                        self.num_birds,
                        dtype=int)+start_time+t, 
                    p, 
                    lat_2, 
                    lon_2))
     
                locs.append(new_loc)
                
        if isinstance(goal, np.ndarray):
            #calculate shortest angle to the goal
            bearing = self.bearing_from_loc(
                loc[:,1:], 
                goal
            )
            bearings.append(bearing)
            for t in range(1, runtime+1, update_time):
               
                # select current information
                current_locs = locs[t-1]                
                current_bearing = bearings[t-1]
                
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
                u_b, v_b = self.get_bird_speed(self_speed, bearing)

                # calculating new position
                lat_1, lon_1 = self.get_current_degrees(current_locs)
                phi_1, lambda_1 = self.get_current_radians(lat_1, lon_1)
                
                u = u_w + u_b
                v = v_w + v_b
                
                d, alpha = self.get_ground_distance(u, v)
                brng = self.get_angle_with_y(u, v, alpha)

                lat_2, lon_2 = self.get_new_radians(phi_1, lambda_1, d, brng)
                
                # calculate new location
                new_loc = np.column_stack((
                    np.zeros(
                        self.num_birds,
                        dtype=int)+start_time+t, 
                    p, 
                    lat_2, 
                    lon_2))
                
                locs.append(new_loc)
                
                #calculate shortest angle to the goal
                bearing = self.bearing_from_loc(
                    new_loc[:,1:], 
                    goal
                )
                bearings.append(bearing)
     
        df = DataFrame(
            np.asarray(locs).reshape((t+1)*self.num_birds, 4),
        columns=['t', 'p', 'lat', 'lon']
        )
        
        # getting datetimes
        df.insert(0, 
                  'bird',
                  np.tile(np.arange(1, self.num_birds+1), self.runtime+1)
        )
        df['bearing'] = np.asarray(bearings).flatten()
            
        return df    
