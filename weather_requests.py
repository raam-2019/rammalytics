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


__author__ = "Steven Wangen"
__version__ = "0.1"
__email__ = "srwangen@wisc.edu"
__status__ = "Development"



API_KEY = 'c124315d967a40b8a4315d967a60b820'



def best_estimate_wind_speed(latitude, longitude, elevation, hours_from_now):

    try:
        # get data from the api
        forecast = get_wind_speed_probability_forecast_for_point(latitude, longitude, elevation)

        # pull out the data relevant to the predicted arrival time (hours from now)
        bin_edges = forecast['forecasts1Hour']['discretePdfs'][0]['binEdges'][hours_from_now]
        bin_values = forecast['forecasts1Hour']['discretePdfs'][0]['binValues'][hours_from_now]

        # get the bin edges of the max value
        max_p_indexes = get_highest_probability_bin_indexes(bin_values)
        predicted_windspeed = []
        for i in max_p_indexes:
            observation = {}
            observation['windspeed_range(m/s)'] = (bin_edges[i-1], bin_edges[i])
            observation['windspeed_probability'] = bin_values[i]
            predicted_windspeed.append(observation)

        return predicted_windspeed

    except Exception as e:
        logging.error("ERROR in weather_requests.best_estimate_wind_speed(): {}".format(e))
        return None



def best_estimate_wind_direction(latitude, longitude, elevation, hours_from_now):

    try:
        # get data from the api
        forecast = get_wind_direction_probability_forecast_for_point(latitude, longitude, elevation)
        
        # pull out the data relevant to the predicted arrival time (hours from now)
        bin_edges = forecast['forecasts1Hour']['discretePdfs'][0]['binEdges'][hours_from_now]
        bin_values = forecast['forecasts1Hour']['discretePdfs'][0]['binValues'][hours_from_now]

        # get the bin edges of the max value
        max_p_indexes = get_highest_probability_bin_indexes(bin_values)
        predicted_wind_direction = []
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
    
    r = requests.get(base_url, request_params)
    return r.json()






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


    

if __name__ == "__main__":

    latitude="33.201550" 
    longitude="-117.369750"
    elevation="120"

    get_wind_speed_probability_forecast_for_point(latitude, longitude, elevation)
    speed_estimate = best_estimate_wind_speed(latitude, longitude, elevation, 2)
    direction_estimate = best_estimate_wind_direction(latitude, longitude, elevation, 2)
    pdb.set_trace()
    
    print(get_probabalistic_conditions(latitude, longitude, elevation))


    e['wind_speed_data']['forecasts1Hour']['discretePdfs']['binValues']


