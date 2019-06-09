import math
import pdb
import pickle
import logging


import shapely
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint
import pandas as pd


import course
import data_wrangler
import prediction


__author__ = "Steven Wangen"
__version__ = "0.1"
__email__ = "srwangen@wisc.edu"
__status__ = "Development"


logger = logging.getLogger(__name__)
log_level = logging.INFO
logging.basicConfig(level=log_level,
                        format='%(asctime)s %(levelname)s %(message)s')


def run():

    # load the course data
    course_object = course.Course()

    while True:

        # heartbeat
        data_wrangler.heartbeat()

        # get current location
        current_df = data_wrangler.bucket_csv_to_df()

        # check if there is any data returned
        if current_df.size > 0:
            # get lat/long from current_Df
            most_recent_row = current_df.ix[current_df['timestamp'].idxmax()]
            
            read_lat = eval(most_recent_row['coordinates'])[0]
            read_lon = eval(most_recent_row['coordinates'])[1]

            # determine course segment
            current_segment_index = course_object.find_current_course_segment(read_lat, read_lon)

            # get next n segments in a dataframe for prediction
            prediction_window_size = 100
            p = prediction.Prediction(course_object, prediction_window_size, current_segment_index)

        else:
            logging.error("dataframe populated by IoT datastore is empty!!!")



# determine which segment

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
    """
    course = course.Course()
    #for testing: kansas city: = 
    current_lat = 39.0997
    current_lon = -94.5786
    current_segment_index = course.find_current_course_segment(current_lat, current_lon)
    pdb.set_trace()
    """
    run()











