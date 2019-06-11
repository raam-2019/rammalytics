import requests
import pdb
import csv
import time
import boto3
import pandas as pd
import pickle
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint
from decimal import Decimal
import uuid
import logging
import threading
import os

__author__ = "Steven Wangen"
__version__ = "0.1"
__email__ = "srwangen@wisc.edu"
__status__ = "Development"


logger = logging.getLogger(__name__)
log_level = logging.INFO
logging.basicConfig(level=log_level,
                        format='%(asctime)s %(levelname)s %(message)s')


API_KEY = os.environ['TWC_API_KEY']



def query_wind_data(prediction_window, wind_df):
    logging.debug("Beginning query of wind data...")
    weather_read_start = time.time()

    wind_observations = {}

    """
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply(self.get_weather_for_row, args=(row, wind_observations, prediction_window,)) for row in self.prediction_df.iterrows()]
    pool.close()
    """

    for index, row in wind_df.iterrows():
        
        # iterate over all of the points in the analysis window, and fetch their wind predictions for that time range
        twc_thread = threading.Thread(target=get_weather_for_row, args=(row, wind_observations, prediction_window,))
        twc_thread.start()
        twc_thread.join()
    

    weather_read_end = time.time()
    logging.info("Query of wind data took: {} seconds".format(weather_read_end - weather_read_start))

    return wind_observations






def best_estimate_wind_speed(latitude, longitude, elevation, forecast_range):

    try:
        # get data from the api
        forecast = get_wind_speed_probability_forecast_for_point(latitude, longitude, elevation)

        predicted_windspeed = []

        for i in range(0, (forecast_range - 1)):
        
        # pull out the data relevant to the predicted arrival time (hours from now)
            bin_edges = forecast['forecasts1Hour']['discretePdfs'][0]['binEdges'][i]
            bin_values = forecast['forecasts1Hour']['discretePdfs'][0]['binValues'][i]

            # get the bin edges of the max value
            max_p_indexes = get_highest_probability_bin_indexes(bin_values)
            
            for i in max_p_indexes:
                observation = {}
                observation['windspeed_range(m/s)'] = (bin_edges[i-1], bin_edges[i])
                observation['windspeed_probability'] = bin_values[i]
                predicted_windspeed.append(observation)

        return predicted_windspeed

    except Exception as e:
        logging.error("ERROR in weather_requests.best_estimate_wind_speed(): {}".format(e))
        return None



def best_estimate_wind_direction(latitude, longitude, elevation, forecast_range):

    try:
        # get data from the api
        forecast = get_wind_direction_probability_forecast_for_point(latitude, longitude, elevation)
        
        predicted_wind_direction = []

        for i in range(0, (forecast_range - 1)):
            # pull out the data relevant to the predicted arrival time (hours from now)
            bin_edges = forecast['forecasts1Hour']['discretePdfs'][0]['binEdges'][i]
            bin_values = forecast['forecasts1Hour']['discretePdfs'][0]['binValues'][i]

            # get the bin edges of the max value
            max_p_indexes = get_highest_probability_bin_indexes(bin_values)
            
            for i in max_p_indexes:
                observation = {}
                observation['wind_direction_range'] = (bin_edges[i-1], bin_edges[i])
                observation['wind_direction_probability'] = bin_values[i]
                predicted_wind_direction.append(observation)

        return predicted_wind_direction  
        
    except Exception as e:
        logging.error("ERROR in weather_requests.best_estimate_wind_speed(): {}".format(e))
        return None


#####################
# query weather API's


def get_current_conditions_for_point(latitude, longitude, elevation):
    
    """
    Function to retrieve current weather conditions at a lat/long/elevation.

    Parameters:

        latitude (float): the latitude of the coordinate
        longitude (float): the longitude of the coordinate
        elevation (float): height AMSL

    Returns:
        dict: the observed weather conditions

    """

    # https://api.weather.com/v1/geocode/34.063/-84.217/observations/timeseries.json?hours=5&language=en-US&units=m&
    base_url = 'https://api.weather.com/v1/'

    # defining a params dict for the parameters to be sent to the API 
    request_params = {
        "geocode": f"{latitude},{longitude}",
        "hours": 1,
        "language": 'en=US',
        "format": "json",
        "apiKey": API_KEY
        }
          
    # sending get request and saving the response as response object 
    r = requests.get(url = base_url, params = request_params) 
      
    # extracting data in json format 
    data = r.json()

    observation = {}
    observation['wind_speed'] = data['observations']['wspd']
    observation['wind_direction'] = data['observations']['wdir']
    observation['temperature'] = data['observations']['temp']

    return observation





#########################
# Historical observations


def get_historical_weather_for_point(latitude, longitude, timestamp):
    
    """
    Function to retrieve historical weather conditions ata lat/long at a given time.
    Weather company API endpoint for 'historical' data (but doesn't specify time):
    https://docs.google.com/document/d/1TlF0nNIWN1fP760hsyOZjr8JjARKzyE7hYaMkaWdAT0/edit
    Doesn't appear to work w/ our current api key.

    Parameters:

        latitude (float): the latitude of the coordinate
        longitude (float): the longitude of the coordinate
        timestamp (str): time of the observed conditions

    Returns:
        dict: the observed weather conditions

    """

    # api endpoint
    # historical weather api w/ example placeid (not working w/ our key): 
    # API documentation at https://docs.google.com/document/d/1TlF0nNIWN1fP760hsyOZjr8JjARKzyE7hYaMkaWdAT0/edit 
    base_url = 'https://api.weather.com/v3/wx/conditions/historical/hourly/1day'

    # defining a params dict for the parameters to be sent to the API 
    request_params = {
        "geocode": f"{latitude},{longitude}",
        "elevation": elevation,
        "landuse": 1,
        "format": "json",
        "units": "m",
        "apiKey": API_KEY
        }
          
    # sending get request and saving the response as response object 
    r = requests.get(url = base_url, params = request_params) 
      
    # extracting data in json format 
    data = r.json()

    observation = {}
    observation['wind_speed'] = None
    observation['wind_direction'] = None
    observation['temperature'] = None
    observation['windspeed'] = None

    return observation




#########################
# Probabalistic forecasts


def get_wind_speed_probability_forecast_for_point(latitude, longitude, elevation):
    # API documented at: https://docs.google.com/document/d/1dGcs1dcPSqpRoniCJ1vYyLPR7gf9BavxxrQUtPDWxiU/edit
    
    base_url = "https://api.weather.com/v3/wx/forecast/probabilistic"
    
    request_params = {
        "geocode": str(latitude) + ',' + str(longitude),
        "elevation": elevation,
        "landuse": 1,
        "format": "json",
        "units": "m",
        "discretePdfs": "windSpeed:fine",
        "hours": 240,
        "apiKey": API_KEY
    }
    
    req = requests.get(base_url, request_params)
    
    if req.status_code != 200:
        
        raise Exception('TWC API responded w/ status code = {} from url {}'.format(req.status_code, req.url))
    else:    
        return req.json()






def get_wind_direction_probability_forecast_for_point(latitude, longitude, elevation):
    # model: # https://api.weather.com/v3/wx/forecast/probabilistic?geocode=48,-32&elevation=2000&landuse=1&format=json&units=e&discretePdfs=windSpeed:coarse&hours=360&apiKey=c124315d967a40b8a4315d967a60b820
    
    base_url = "https://api.weather.com/v3/wx/forecast/probabilistic"
    
    request_params = {
        "geocode": str(latitude) + ',' + str(longitude),
        "elevation": elevation,
        "landuse": 1,
        "format": "json",
        "units": "m",
        "discretePdfs": "windDirection:fine",
        "hours": 240,
        "apiKey": API_KEY
    }

    r = requests.get(base_url, request_params)
    return r.json()



def get_highest_probability_bin_indexes(array):
    m = max(array)
    indexes = [i for i, j in enumerate(array) if j == m]
    return indexes



def get_weather_for_row(row, wind_observations, prediction_window):
    weather_observation = {}
    logging.info("pulling wind speed data...")
    weather_observation['wind_speed_data'] = best_estimate_wind_speed(row['from_lat'], row['from_lon'], row['from_elevation'], 120)
    logging.info("pulling wind direction data...")
    weather_observation['wind_direction_data'] = best_estimate_wind_direction(row['from_lat'], row['from_lon'], row['from_elevation'], 120)
    wind_observations[row['segment_id']] = weather_observation
    

if __name__ == "__main__":

    latitude="33.201550" 
    longitude="-117.369750"
    elevation="120"

    get_wind_speed_probability_forecast_for_point(latitude, longitude, elevation)
    speed_estimate = best_estimate_wind_speed(latitude, longitude, elevation, 2)
    direction_estimate = best_estimate_wind_direction(latitude, longitude, elevation, 2)
    
    print(get_probabalistic_conditions(latitude, longitude, elevation))

    e['wind_speed_data']['forecasts1Hour']['discretePdfs']['binValues']


