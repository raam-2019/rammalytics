import pdb
from datetime import datetime, timedelta
import statistics
import math
import logging

import numpy as np
import pandas as pd

import weather_requests
import data_wrangler

__author__ = "Steven Wangen"
__version__ = "0.1"
__email__ = "srwangen@wisc.edu"
__status__ = "Development"


logger = logging.getLogger(__name__)
log_level = logging.INFO
logging.basicConfig(level=log_level,
                        format='%(asctime)s %(levelname)s %(message)s')


class Prediction:

    def __init__(self, course, analysis_window_size, current_segment_index):
        
        self.ftp = 335
        
        self.prediction_df = course.segment_df.iloc[current_segment_index:current_segment_index + analysis_window_size]

        # extend dataframe for predictive variables
        self.prediction_df['wind_speed(m/s)'] = None
        self.prediction_df['wind_speed_confidence_level'] = None
        self.prediction_df['wind_direction'] = None
        self.prediction_df['wind_direction_confidence_level'] = None
        self.prediction_df['predicted_power(watts)'] = None
        self.prediction_df['headwind(m/s)'] = None
        self.prediction_df['segment_speed(km/h)'] = None
        self.prediction_df['segment_duration(s)'] = None
        self.prediction_df['segment_tss'] = 0
        self.prediction_df['predicted_arrival_time'] = None
        self.prediction_df['predicted_finishing_time'] = None
        self.prediction_df['wind_speed+2hr'] = None
        self.prediction_df['wind_speed+2hr_confidence_level'] = None
        self.prediction_df['wind_direction+2hr'] = None
        self.prediction_df['wind_direction+2hr_confidence_level'] = None
        self.prediction_df['headwind+2hr(m/s)'] = None
        self.prediction_df['segment_calories'] = None


        # approximate time to finish each segment in df
        first_segment = True
        for index, row in self.prediction_df.iterrows():
            if first_segment:
                row['length(m)'] = row['length(m)'] - course.distance_along_segment
                first_segment = False
                hours_from_now = 0
                segment_start_time = datetime.now()
            else:
                segment_start_time = previous_row['predicted_finishing_time']
                hours_from_now = round((segment_start_time - datetime.now()).seconds / 3600)
                

            # TODO!!!
            # get power approximation - to be based on slope
            # predicted_power = predict_power_from_slope(slope)
            # predicted_power = predict_power_from_slope_tss(slope, tss)
            row['predicted_power(watts)'] = self.predict_segment_power(row['slope'])
                
            # query wind_speed from wc api
            wind_speed_data = weather_requests.best_estimate_wind_speed(row['from_lat'], row['from_lon'], row['from_elevation'], hours_from_now)
            row['wind_speed(m/s)'] = statistics.mean(wind_speed_data[0]['windspeed_range(m/s)'])
            row['wind_speed_confidence_level'] = wind_speed_data[0]['windspeed_probability'] / 100

            # get wind_speed 2 hours later
            future_wind_speed_data = weather_requests.best_estimate_wind_speed(row['from_lat'], row['from_lon'], row['from_elevation'], (hours_from_now+2))
            row['wind_speed+2hr'] = statistics.mean(future_wind_speed_data[0]['windspeed_range(m/s)'])
            row['wind_speed+2hr_confidence_level'] = future_wind_speed_data[0]['windspeed_probability'] / 100

            # query wind direction from wc api
            wind_direction_data = weather_requests.best_estimate_wind_direction(row['from_lat'], row['from_lon'], row['from_elevation'], hours_from_now)
            row['wind_direction'] = statistics.mean(wind_direction_data[0]['wind_direction_range'])
            row['wind_direction_confidence_level'] = wind_direction_data[0]['wind_direction_probability'] / 100

            # get wind direction 2 hours later
            future_wind_direction_data = weather_requests.best_estimate_wind_direction(row['from_lat'], row['from_lon'], row['from_elevation'], (hours_from_now+2))
            row['wind_direction+2hr'] = statistics.mean(future_wind_direction_data[0]['wind_direction_range'])
            row['wind_direction+2hr_confidence_level'] = future_wind_direction_data[0]['wind_direction_probability'] / 100

            row['headwind(m/s)'] = self.calculate_headwind(row['bearing'], row['wind_speed(m/s)'], row['wind_direction'])
            row['headwind+2hr(m/s)'] = self.calculate_headwind(row['bearing'], row['wind_speed+2hr'], row['wind_direction+2hr'])

            row['segment_speed(km/h)'] = self.calculate_speed(row['predicted_power(watts)'], row['slope'], row['headwind(m/s)'], row['from_elevation'])
            row['segment_duration(s)'] = ((row['length(m)'] / 1000) / row['segment_speed(km/h)']) * 3600
            
            row['predicted_arrival_time'] = segment_start_time
            row['predicted_finishing_time'] = segment_start_time + timedelta(seconds=row['segment_duration(s)'])

            row['segment_tss'] = self.get_tss([row['predicted_power(watts)']], self.ftp, row['segment_duration(s)'])
            row['segment_calories'] = ((row['predicted_power(watts)'] * row['segment_duration(s)']) / 4.18) / 0.24

            # actually update the df with the updated row
            self.prediction_df.at[index] = row

            previous_row = row

        # WRITE TO DYNAMO
        data_wrangler.write_prediction_to_database(self.prediction_df)



    # def predict_segment_power(self, accumulated_tss, slope):
    def predict_segment_power(self, slope):
        # linear regression
        return 141.0758 + 589.2302 * slope




    def calculate_headwind(self, rider_bearing, wind_speed, wind_direction):
        # return in m/s - same as wind_speed input
        relative_wind_angle = min((2 * math.pi) - abs(rider_bearing - wind_direction), abs(rider_bearing - wind_direction))
        headwind = math.cos(relative_wind_angle) * wind_speed
        return headwind




    def calculate_speed(self, power, slope, headwind, elevation):
        # heavily based off of JFP's impementation
        # power in watts
        # Cx = The air penetration coefficient. We will use Cx = 0.25 by default.
        # f = friction (assumed to be 0.01)
        # W = rider weight (72.5 kg)
        # slope in percent
        # headwind in m/s at 10 m high
        # elevation in meters

        Cx = 0.25
        f = 0.01
        W = 72.5
        G = 9.81
        air_pressure = 1 - 0.000104 * elevation # calc from elevation
        Cx = Cx * air_pressure # adjustment for air pressure
        headwind = (0.1 ** 0.143) * headwind

        roots = np.roots([Cx, 2 * Cx * headwind, Cx * headwind ** 2 + W * G * (slope / 100.0 + f), -power])
        roots = np.real(roots[np.imag(roots) == 0])
        roots = roots[roots > 0]

        speed = np.min(roots)

        if speed + headwind < 0:
            roots = np.roots([-Cx, -2 * Cx * headwind, -Cx * headwind ** 2 + W * G * (slope/100.0 + f), -power])
            roots = np.real(roots[np.imag(roots) == 0])
            roots = roots[roots>0]
            if len(roots) > 0:
                speed = np.min(roots)  

        # convert from m/s to km/h
        kph = speed * 3.6
        return kph




    def calculate_apparent_wind_angle_and_speed(self, rider_speed, rider_direction, wind_speed, wind_direction):

        # from sheldonbrown.com (heart) https://www.sheldonbrown.com/brandt/wind.html
         
        relative_wind_angle = min((2 * math.pi) - abs(wind_speed - wind_direction), abs(wind_speed - wind_direction))
        relative_wind_speed = math.sqrt((rider_speed + wind_speed * math.cos(relative_wind_angle))**2 + (wind_speed * math.sin(relative_wind_angle))**2)
        return (relative_wind_angle, relative_wind_speed)




    def get_tss(self, power_vector, functional_training_threshold, duration):
        normalized_power = self.calculate_np(power_vector)
        intensity_factor = self.calculate_if(normalized_power, functional_training_threshold)
        tss = self.calculate_tss(duration, normalized_power, intensity_factor, functional_training_threshold)
        return tss




    def calculate_np(self, power_vector):
        
        """ Calculate the normalized power from a vector of power observations
            
            Keyword arguments:
            power_vector -- the vector of power obserations
            
            Result returned in Watts (or same units the power is reported in)
        """
        
        quartic_values = [i ** 4 for i in power_vector]
        quartic_np = sum(quartic_values)
        np = quartic_np ** 0.25
        return np
        



    def calculate_if(self, np, ftp):
        
        """Calculate the intensity factor
        
            Keyword arguments:
            np -- normalized power (in watts)
            ftp - functional threshold power (in watts)
        """
        intensity_factor = np / ftp
        return intensity_factor




    def calculate_tss(self, t, np, intensity_factor, ftp):
        
        """ Calculate the training stress score (TSS)
        
            Keyword arguments:
            t -- duration of workout in seconds
            np -- normalized power
            intensity_factor -- intensity factor
            ftp - functional threshold power
            
            Result in dimensionless (has no units)
        """
        
        tss = (t * np * intensity_factor) / (ftp * 36)
        return tss




    def calculate_training_load(self, tss, current_training_load, window_size):
        """ Calculate the chronic training load (CTL)
        
            tss -- vector of tss data
            training_load -- training load (CTL OR ATL) from yesterday
            window_size -- duration during which to calculate training load;
                           traditionally, chronic load is the average over the 
                           last 42 days, acute is over last 7 days
        """
        
        training_load = (current_training_load * exp(-1/window_size)) + \
                        (tss * exp(1 - exp(-1/window_size)))
        
        return training_load

