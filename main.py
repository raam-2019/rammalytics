import math
import pdb
import pickle
import logging
import time
from xml.etree import ElementTree
import requests

import shapely
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint
import pandas as pd


import course
import data_wrangler
import prediction
import weather_requests


__author__ = "Steven Wangen"
__version__ = "0.1"
__email__ = "srwangen@wisc.edu"
__status__ = "Development"


logger = logging.getLogger(__name__)
log_level = logging.INFO
logging.basicConfig(level=log_level,
                        format='%(asctime)s %(levelname)s %(message)s')


def run():

    TEST = False

    # load the course data
    course_object = course.Course()
    
    # get next n segments in a dataframe for prediction
    analysis_window_size = 1200
    
    # make sure weather runs
    last_weather_et = 0

    # do stuff    
    while True:

        # heartbeat
        data_wrangler.heartbeat()
        
        try:
            # get current location
            # from s3
            coords = get_rider_lat_long()

            if coords is None:
                # from trackleaders
                coords = ping_track_leaders()

            # parse results
            read_lat = coords[0]
            read_lon = coords[1]

            if read_lat is not None: 

                # determine course segment
                current_segment_index = course_object.find_current_course_segment(read_lat, read_lon)

                # get weather for next n segments if it's been 30 min
                if TEST:
                    wind_data = pickle.load( open( "wind.p", "rb" ) )
                else:
                    if ((last_weather_et + 1800) < time.time()):
                        wind_df = course_object.segment_df.iloc[current_segment_index:current_segment_index + (analysis_window_size + 200)]
                        last_weather_et = time.time()
                        logging.info("getting fresh weather data (this could take a few minutes...)")
                        wind_data = weather_requests.query_wind_data(analysis_window_size, wind_df)                

                # make predictions
                if len(wind_data.keys()) != 0:
                    p = prediction.Prediction(course_object, analysis_window_size, current_segment_index, wind_data)

        except Exception as e:     
            logging.error('Exception caught in main.run(): {}'.format(e))

        else:
            logging.error("dataframe populated by IoT datastore is empty!!!")




def get_rider_lat_long():
    try:
        current_df = data_wrangler.bucket_csv_to_df()

        # check if there is any data returned
        if current_df.size > 0:

            # get lat/long from current_Df
            most_recent_row = current_df.ix[current_df['timestamp'].idxmax()]

            # check for nan cat
            if math.isnan(most_recent_row['coordinates']):
                return None

            logging.info("most recent row = {}".format(most_recent_row))
            read_lat = eval(most_recent_row['coordinates'])[1]
            read_lon = eval(most_recent_row['coordinates'])[0]

            if read_lat is None: 

                most_recent_row = current_df.ix[current_df['coordinates'].notnull()]
                read_lat = eval(most_recent_row['coordinates'])[1]
                read_lon = eval(most_recent_row['coordinates'])[0]

                return (read_lat, read_lon)

        else:
            return None

    except Exception as e:
        logging.error('Exception caught trying to parse lat/lon from s3 csv: {}'.format(e))





# determine which segment
def ping_track_leaders():
    trackleaders_url = "http://trackleaders.com/spot/raam19/fullfeed.xml"
    dave_race_id = 52
    trackleaders_data = []
    r = requests.get(url = trackleaders_url)
    tree = ElementTree.fromstring(r.content)
    children = tree.getchildren()
    # print(ElementTree.tostring(child, encoding='utf8').decode('utf8'))k
    for child in children:
        racer_data = {}
        racer_observations = []
        # each child = feed from one racer (trackleader feed)
        for sub_child in child:
            # parse the racer data
            # get the racer id
            if sub_child.tag == 'trackleaders_racer_ID':
                racer_data['racer_id'] = int(sub_child.text)
            elif sub_child.tag == 'message':
                observation = {}
                for c in sub_child:
                    if c.tag == 'latitude':
                        observation['lat'] = c.text
                    elif c.tag == 'longitude':
                        observation['lon'] = c.text
                    elif c.tag == 'elevation':
                        observation['elevation'] = c.text
                    elif c.tag == 'timeInGMTSecond':
                        observation['timestamp'] = c.text
                racer_observations.append(observation)
        racer_data['observations'] = racer_observations
        trackleaders_data.append(racer_data)
    for racer in trackleaders_data:
        if racer['racer_id'] == dave_race_id:
            # find newest entry
            timelist = []
            for observation in racer['observations']:
                timelist.append(observation['timestamp'])
            newest = max(timelist)
            # get the position for that time
            for observation in racer['observations']:
                if observation['timestamp'] == newest:
                    return (float(observation['lat']), float(observation['lon']))





# pickle weather data
def pickle_weather(wind_data):
    pickle.dump( wind_data, open( "wind.p", "wb" ) )



# make predictions

def predict(current_lat, current_lon):

    #for testing: kansas city = 39.0997, -94.5786
    current_read_position = Point(current_lat, current_lon)

    # snap to a position on the course line
    distance_along_segment = course_line.project(current_read_position)
    course_position = course_line.interpolate(distance_along_segment)

    # find the nearest vertex on the course
    nearest_vertex = nearest_points(course_mp_line, course_position)
    


if __name__ == '__main__':

    run()










