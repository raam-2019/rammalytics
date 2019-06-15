import pdb
from datetime import datetime, timedelta
import statistics
import math
import logging
import time
import threading
import multiprocessing as mp
import course

import numpy as np
import pandas as pd
import pickle

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

    def __init__(self, rider_number, course, analysis_window_size, current_segment_index):
        logging.info("analysis window size = {}".format(analysis_window_size))

        self.rider_number = rider_number

        # subset the course data to only reflect the window size (from current segment: current segment + window)
        self.prediction_df = course.segment_df.iloc[current_segment_index:current_segment_index + analysis_window_size]

        
            


    
    def model_course_evolution(self, analysis_window_size, wind_data, course, TEST):
        logging.info("going to evolve course for {} segments".format(analysis_window_size))
        
        try:
            # perform predictions
            logging.info("modeling segment")
            prediction_start = time.time()
            pdb.set_trace()
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
            # self.prediction_df['rh'] = None
            # self.prediction_df['temp'] = None

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

            # model time to finish each segment
            first_segment = True
            rows = []
            
            logging.info('iterating through self.prediction_df to create predictions...')
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
                if result['segment_id'] not in wind_data.keys():
                    pass
                result['wind_speed(m/s)'] = wind_data[result['segment_id']]['wind_speed_data'][hours_from_now]['windspeed_range(m/s)']
                # result['wind_speed_confidence_level'] = wind_data[result['segment_id']]['wind_speed_data'][hours_from_now]['windspeed_probability'] / 100
                result['wind_direction'] = wind_data[result['segment_id']]['wind_direction_data'][hours_from_now]['wind_direction_range']
                # result['wind_direction_confidence_level'] = wind_data[result['segment_id']]['wind_direction_data'][hours_from_now]['wind_direction_probability'] / 100
                # result['rh'] = wind_data[result['segment_id']]['rh'][hours_from_now]
                # result['heat_index'] = wind_data[result['segment_id']]['heat_index'][hours_from_now]
                # result['temp'] = wind_data[result['segment_id']]['temp'][hours_from_now]

                result['headwind(m/s)'] = self.calculate_headwind(result['bearing'], result['wind_speed(m/s)'], result['wind_direction']) 
                result['segment_speed(km/h)'] = self.calculate_speed(result['predicted_power(watts)'], result['slope'], result['headwind(m/s)'], result['from_elevation'])
                result['segment_duration(s)'] = ((result['length(m)'] / 1000) / result['segment_speed(km/h)']) * 3600
                result['predicted_arrival_time'] = segment_start_time
                result['predicted_finishing_time'] = segment_start_time + timedelta(seconds=result['segment_duration(s)'])
                result['segment_tss'] = self.get_tss([result['predicted_power(watts)']], self.ftp, result['segment_duration(s)'])
                result['segment_calories'] = ((result['predicted_power(watts)'] * result['segment_duration(s)']) / 4.18) / 0.24

                # 2+ hour sim
                result['plus_2_wind_speed(m/s)'] = wind_data[result['segment_id']]['wind_speed_data'][hours_from_now_plus_2]['windspeed_range(m/s)']
                # result['plus_2_wind_speed_confidence_level'] = wind_data[result['segment_id']]['wind_speed_data'][hours_from_now_plus_2]['windspeed_probability'] / 100
                result['plus_2_wind_direction'] = wind_data[result['segment_id']]['wind_direction_data'][hours_from_now_plus_2]['wind_direction_range']
                # result['plus_2_wind_direction_confidence_level'] = wind_data[result['segment_id']]['wind_direction_data'][hours_from_now_plus_2]['wind_direction_probability'] / 100

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
            pdb.set_trace()
            logging.error(e)

        if TEST:
            pickle.dump(rows, open( "analysis_results_rider_{}.p".format(self.rider_number), "wb" ) )
        else:
            data_wrangler.write_prediction_to_database2(rows, self.rider_number)





    def calculate_cost_of_rest(self, wind_data, distance_to_start_of_simulation, hours, TEST):

        """
        Calculates the 'cost of rest' metric by evolving the model of a set of segments
        with a two-hour delay (rest) at each segment - returns the difference in completion time
        between the evaluated scenario (rest at segment) and the rest scenario providing the shortest
        completion time


        Parameters:

            analysis_window_size (int): 
            prediction_df (dataframe): subset of course data from most current location to x number of
                                       segments, where x = analysis_window_size
            wind_data (dataframe): a dataframe containing wind data for a 360 hour period for each 
                                   start location of a segment within the prediction_df
            course (Course object): an object with an in-memory representation of the entire course

        Returns:
            list: list of the relative increase in completion time of the scenario if rest is implemented
                  at a specific segment (indicated by index / list position)

        """

        best_guess_speed = 27 #kph

        # figure out how many segments to evaluate
        (analysis_window_size, elapsed_time, elapsed_distance) = self.find_segment_after_x_hours(hours, best_guess_speed, self.prediction_df)
        logging.info('calculating cost_of_rest - approximating {} hour window will cover {} segments, or {} kms over {} hours'.format(hours, analysis_window_size, elapsed_distance/1000, elapsed_time/3600))
            

        logging.info("going to evaluate {} of {} possible segments".format(analysis_window_size, self.prediction_df.size))
        # perform one evolution of course accounting for 2hr rest at each segment
        logging.info("Beginning cost of rest calculation - prediction_df.size = {}".format(self.prediction_df.size))
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
                    segment_length = row['length(m)'] - distance_to_start_of_simulation
                    
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
                wind_speed = wind_data[row['segment_id']]['wind_speed_data'][hours_from_now]['windspeed_range(m/s)']
                wind_direction = wind_data[row['segment_id']]['wind_direction_data'][hours_from_now]['wind_direction_range']

                if wind_speed is None:
                    raise Exception("calculate_cost_of_rest(): exception trying to retrieve wind data...")

                headwind = self.calculate_headwind(row['bearing'], wind_speed, wind_direction) 
                segment_speed = self.calculate_speed(power, row['slope'], headwind, row['from_elevation'])
                segment_duration = ((segment_length / 1000) / segment_speed) * 3600
                predicted_arrival_time = segment_start_time
                predicted_finishing_time = segment_start_time + timedelta(seconds=segment_duration)
                
                previous_row = row

            total_times.append(predicted_finishing_time - section_start_time)

        # calcuate the cost of rest at each segment
        time_of_rest = []
        min_total_time = min(total_times)

        logging.info("total_times: length = {}".format(len(total_times)))
        for row_count in range(0, len(total_times)-1):
            segment_data = {}
            segment_id = self.prediction_df.iloc[row_count]['segment_id']
            segment_data['cost_of_rest'] = (total_times[row_count] - min_total_time).seconds
            segment_data['elevation'] = self.prediction_df.iloc[row_count]['from_elevation']
            segment_data['cumulative_distance_to_segment'] = self.prediction_df.iloc[row_count]['cumulative_distance_to_segment']
            segment_data['segment_id'] = segment_id
            time_of_rest.append(segment_data)

        cor_end = time.time()

        logging.info("Cost of rest evaluation took: {} seconds".format(cor_end - cor_start))

        # write the results to the database
        if TEST:
            pickle.dump(cost_of_rest, open( "cost_of_rest_{}_rider_{}.p".format(hours, self.rider_number), "wb" ) )
        else:
            data_wrangler.write_cost_of_rest_to_database(hours, cost_of_rest, self.rider_number)





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

        roots = np.roots([Cx, 2 * Cx * headwind, Cx * headwind ** 2 + W * G * (slope + f), -power])
        roots = np.real(roots[np.imag(roots) == 0])
        roots = roots[roots > 0]

        if len(roots) == 0:
            pdb.set_trace()
            roots
        speed = np.min(roots)

        if speed + headwind < 0:
            roots = np.roots([-Cx, -2 * Cx * headwind, -Cx * headwind ** 2 + W * G * (slope+ f), -power])
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





    def find_segment_after_x_hours(self, hours, speed, prediction_df):

        target_time = 3600 * hours
        logging.info('target_time = {}'.format(target_time))
        elapsed_distance = 0
        elapsed_time = 0
        mps = (speed * 1000) / (60 * 60)
        i = 0
        for index, row in prediction_df.iterrows():
            i += 1
            elapsed_distance += row['length(m)']
            elapsed_time += row['length(m)'] / mps
            if elapsed_time > target_time:
                logging.info("finished: elapsed_time = {}".format(elapsed_time))
                return (i, elapsed_time, elapsed_distance)






if __name__ == '__main__':

    #for testing: kansas city:
    lat = 33.86697222222222
    lon = -113.39761111111112

    course_object = course.Course()
    
    # get next n segments in a dataframe for prediction
    analysis_window_size = 1000
    
    # make sure weather runs
    last_weather_et = 0

    try:
        read_lat = lat
        read_lon = lon    

        # determine course segment
        current_segment_index = course_object.find_current_course_segment(read_lat, read_lon)

        # get weather (if necessary)
        # if it's been 15 min
        wind_df = course_object.segment_df.iloc[current_segment_index:current_segment_index + analysis_window_size]

        if ((last_weather_et + 1800) < time.time()):
            last_weather_et = time.time()
            logging.info("getting fresh weather data (this could take a few minutes...)")
            wind_data = weather_requests.query_wind_data(analysis_window_size, wind_df)

        # make predictions
        if len(wind_data.keys()) != 0:
            p = Prediction(course_object, analysis_window_size, current_segment_index, wind_data)

    except Exception as e:     
        logging.error('Exception caught in main.run(): {}'.format(e))


