import pdb
from datetime import datetime, timedelta
import statistics
import math
import logging
import time
import threading
import multiprocessing as mp

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

    def __init__(self, course, analysis_window_size, current_segment_index, wind_observations):
        logging.info("analysis window size = {}".format(analysis_window_size))
        prediction_window = 24 # hours

        self.prediction_df = course.segment_df.iloc[current_segment_index:current_segment_index + analysis_window_size]

        # detailed analysis of the evolution of course
        analysis_results = self.model_course_evolution(analysis_window_size, self.prediction_df, wind_observations, course)
        data_wrangler.write_prediction_to_database2(analysis_results)

        # calculate the cost of rest
        prediction_windows = [4, 8, 12] #, 24, 48
        best_guess_speed = 27 #kph
        for hours in prediction_windows:
            analysis_window_size = course.find_segment_after_x_hours(hours, best_guess_speed)
            logging.info('calculating cost_of_rest for {} hour window over {} segments'.format(hours, analysis_window_size))
            cost_of_rest = self.calculate_cost_of_rest(analysis_window_size, self.prediction_df, wind_observations, course)
            data_wrangler.write_cost_of_rest_to_database(hours, cost_of_rest)
        



    
    def model_course_evolution(self, analysis_window_size, prediction_df, wind_observations, course):
        logging.info("going to evolve course for {} segments".format(analysis_window_size))
        
        try:
            # perform predictions
            logging.info("modeling segment")
            prediction_start = time.time()

            self.ftp = 335
            
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
            self.prediction_df['segment_calories'] = None

            self.prediction_df['plus_2_wind_speed(m/s)'] = None
            self.prediction_df['plus_2_wind_speed_confidence_level'] = None
            self.prediction_df['plus_2_wind_direction'] = None
            self.prediction_df['plus_2_wind_direction_confidence_level'] = None
            self.prediction_df['plus_2_headwind(m/s)'] = None
            self.prediction_df['plus_2_segment_speed(km/h)'] = None
            self.prediction_df['plus_2_segment_duration(s)'] = None
            self.prediction_df['plus_2_predicted_arrival_time'] = None
            self.prediction_df['plus_2_predicted_finishing_time'] = None
            self.prediction_df['plus_2_segment_tss'] = None
            self.prediction_df['plus_2_segment_calories'] = None

            # approximate time to finish each segment in df
            first_segment = True
            rows = []

            for index, row in self.prediction_df.iterrows():
                result = {}
                if first_segment:
                    result['length(m)'] = row['length(m)'] - course.distance_along_segment
                    first_segment = False
                    hours_from_now = 0
                    segment_start_time = datetime.now()
                    plus_2_segment_start_time = datetime.now() + timedelta(hours=2)
                    hours_from_now_plus_2 = 2
                else:
                    segment_start_time = previous_row['predicted_finishing_time']
                    plus_2_segment_start_time = previous_row['plus_2_predicted_finishing_time']
                    hours_from_now = round((segment_start_time - datetime.now()).seconds / 3600)
                    hours_from_now_plus_2 = round((plus_2_segment_start_time - datetime.now()).seconds / 3600)       

                # write all pertinent data to a dictionary
                result['slope'] = row['slope']
                result['segment_id'] = row['segment_id']
                result['bearing'] = row['bearing']
                result['to_elevation'] = row['to_elevation']
                result['from_elevation'] = row['from_elevation']
                result['length(m)'] = row['length(m)']
                result['segment_id'] = row['segment_id']
                result['cumulative_distance_to_segment'] = row['cumulative_distance_to_segment']

                # 
                # TODO: predicted_power = predict_power_from_slope_tss(slope, tss)
                result['predicted_power(watts)'] = self.predict_segment_power(result['slope'])
                
                # current course evolution    
                result['wind_speed(m/s)'] = wind_observations[result['segment_id']]['wind_speed_data'][hours_from_now]['windspeed_range(m/s)']
                # result['wind_speed_confidence_level'] = wind_observations[result['segment_id']]['wind_speed_data'][hours_from_now]['windspeed_probability'] / 100
                result['wind_direction'] = wind_observations[result['segment_id']]['wind_direction_data'][hours_from_now]['wind_direction_range']
                # result['wind_direction_confidence_level'] = wind_observations[result['segment_id']]['wind_direction_data'][hours_from_now]['wind_direction_probability'] / 100
                result['rh'] = wind_observations[result['segment_id']]['rh'][hours_from_now]
                # result['heat_index'] = wind_observations[result['segment_id']]['heat_index'][hours_from_now]
                result['temp'] = wind_observations[result['segment_id']]['temp'][hours_from_now]

                result['headwind(m/s)'] = self.calculate_headwind(result['bearing'], result['wind_speed(m/s)'], result['wind_direction']) 
                result['segment_speed(km/h)'] = self.calculate_speed(result['predicted_power(watts)'], result['slope'], result['headwind(m/s)'], result['from_elevation'])
                result['segment_duration(s)'] = ((result['length(m)'] / 1000) / result['segment_speed(km/h)']) * 3600
                result['predicted_arrival_time'] = segment_start_time
                result['predicted_finishing_time'] = segment_start_time + timedelta(seconds=result['segment_duration(s)'])
                result['segment_tss'] = self.get_tss([result['predicted_power(watts)']], self.ftp, result['segment_duration(s)'])
                result['segment_calories'] = ((result['predicted_power(watts)'] * result['segment_duration(s)']) / 4.18) / 0.24

                # 2+ hour sim
                result['plus_2_wind_speed(m/s)'] = wind_observations[result['segment_id']]['wind_speed_data'][hours_from_now_plus_2]['windspeed_range(m/s)']
                # result['plus_2_wind_speed_confidence_level'] = wind_observations[result['segment_id']]['wind_speed_data'][hours_from_now_plus_2]['windspeed_probability'] / 100
                result['plus_2_wind_direction'] = wind_observations[result['segment_id']]['wind_direction_data'][hours_from_now_plus_2]['wind_direction_range']
                # result['plus_2_wind_direction_confidence_level'] = wind_observations[result['segment_id']]['wind_direction_data'][hours_from_now_plus_2]['wind_direction_probability'] / 100

                result['plus_2_headwind(m/s)'] = self.calculate_headwind(result['bearing'], result['plus_2_wind_speed(m/s)'], result['plus_2_wind_direction']) 
                result['plus_2_segment_speed(km/h)'] = self.calculate_speed(result['predicted_power(watts)'], result['slope'], result['plus_2_headwind(m/s)'], result['from_elevation'])
                result['plus_2_segment_duration(s)'] = ((result['length(m)'] / 1000) / result['plus_2_segment_speed(km/h)']) * 3600
                result['plus_2_predicted_arrival_time'] = plus_2_segment_start_time
                result['plus_2_predicted_finishing_time'] = plus_2_segment_start_time + timedelta(seconds=result['plus_2_segment_duration(s)'])
                result['plus_2_segment_tss'] = self.get_tss([result['predicted_power(watts)']], self.ftp, result['plus_2_segment_duration(s)'])
                result['plus_2_segment_calories'] = ((result['predicted_power(watts)'] * result['plus_2_segment_duration(s)']) / 4.18) / 0.24

                # actually update the df with the updated row
                rows.append(result)
                # self.prediction_df.at[index] = row

                previous_row = result

            prediction_end = time.time()
            logging.info("course evolution analysis took: {} seconds".format(prediction_end - prediction_start))

        except Exception as e:
            logging.error(e)

        return rows





    def calculate_cost_of_rest(self, analysis_window_size, prediction_df, wind_observations, course):

        logging.info("going to evaluate {} of {} segments".format(analysis_window_size, prediction_df.size))
        # perform one evolution of course accounting for 2hr rest at each segment
        logging.info("Beginning cost of rest calculation - prediction_df.size = {}".format(prediction_df.size))
        cor_start = time.time()
        costs_of_rest = {}
        total_times = []
        self.ftp = 335
        
        for i in range(0, (analysis_window_size-1)):
            # approximate time to finish each segment in df
            first_segment = True

            for index, row in self.prediction_df.iterrows():

                if first_segment:
                    first_segment = False
                    segment_number = 0
                    hours_from_now = 0
                    segment_start_time = datetime.now()
                    section_start_time = segment_start_time
                    segment_length = row['length(m)'] - course.distance_along_segment
                    
                else:
                    segment_number += 1
                    segment_start_time = predicted_finishing_time
                    hours_from_now = round((segment_start_time - datetime.now()).seconds / 3600)
                   
                if segment_number == i:
                    # add 2hr rest
                    segment_start_time += timedelta(hours=2)

                # TODO!!!
                # get power approximation - to be based on slope
                # predicted_power = predict_power_from_slope(slope)
                # predicted_power = predict_power_from_slope_tss(slope, tss)
                power = self.predict_segment_power(row['slope'])
                    
                # wind data
                wind_speed = wind_observations[row['segment_id']]['wind_speed_data'][hours_from_now]['windspeed_range(m/s)']
                wind_direction = wind_observations[row['segment_id']]['wind_direction_data'][hours_from_now]['wind_direction_range']

                if wind_speed is None:
                    raise Exception("calculate_cost_of_rest(): exception trying to retrieve wind data...")

                headwind = self.calculate_headwind(row['bearing'], wind_speed, wind_direction) 
                segment_speed = self.calculate_speed(power, row['slope'], headwind, row['from_elevation'])
                segment_duration = ((segment_length / 1000) / segment_speed) * 3600
                predicted_arrival_time = segment_start_time
                predicted_finishing_time = segment_start_time + timedelta(seconds=segment_duration)
                
                previous_row = row

            total_times.append(predicted_finishing_time - section_start_time)

            min_total_time = min(total_times)

        time_of_rest = []

        logging.info("total_times length = {}".format(len(total_times)))
        for row_count in range(0, len(total_times)-1):
            segment_data = {}
            segment_id = prediction_df.iloc[row_count]['segment_id']
            segment_data['cost_of_rest'] = (total_times[row_count] - min_total_time).seconds
            segment_data['elevation'] = prediction_df.iloc[row_count]['from_elevation']
            segment_data['cumulative_distance_to_segment'] = prediction_df.iloc[row_count]['cumulative_distance_to_segment']
            segment_data['segment_id'] = segment_id
            time_of_rest.append(segment_data)

        cor_end = time.time()

        logging.info("Cost of rest evaluation took: {} seconds".format(cor_end - cor_start))

        return time_of_rest





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


