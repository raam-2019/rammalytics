import logging
from datetime import datetime
import uuid
import pickle
import math
import pdb

import shapely
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint
import pandas as pd

import data_wrangler


__author__ = "Steven Wangen"
__version__ = "0.1"
__email__ = "srwangen@wisc.edu"
__status__ = "Development"


logger = logging.getLogger(__name__)
log_level = logging.INFO
logging.basicConfig(level=log_level,
                        format='%(asctime)s %(levelname)s %(message)s')


class Course:

    def __init__(self):

        self.segments = pickle.load( open( "segments.pkl", "rb" ) )
        self.segment_df = self.create_segment_df(self.segments)
        self.points = self.get_point_list(self.segments)
        self.course_line = LineString(self.points)
        self.course_mp_line = MultiPoint(self.points)
        self.current_course_position = None
        self.distance_along_segment = None


    def find_current_course_segment(self, read_lat, read_lon):
        
        # snap to a position on the course line
        current_read_position = Point(read_lon, read_lat)
        
        # should always be distance from beginning of segment
        distance_along_segment = self.course_line.project(current_read_position)

        self.current_course_position = self.course_line.interpolate(distance_along_segment)

        # find the nearest vertex on the course
        nearest_vertex = nearest_points(self.course_mp_line, self.current_course_position)

        # parse out coordinates
        # course location
        lat_a = self.current_course_position.xy[1][0]
        lon_a = self.current_course_position.xy[0][0]

        # vertex location 
        lat_b = nearest_vertex[0].xy[1][0]
        lon_b = nearest_vertex[0].xy[0][0]

        # identify current segment
        heading = self.calculate_heading(lat_a, lon_a, lat_b, lon_b)

        # find segments that contain the nearest vertex, see which has the matching bearing
        
        f_lat_match = self.segment_df.loc[self.segment_df['from_lat'] == lat_b]
        t_lat_match = self.segment_df.loc[self.segment_df['to_lat'] == lat_b]

        # grab the difference between the observed heading and the course headings, both ways (forwards and backwards)
        bearing_dict = {}
        if not f_lat_match.empty:
            bearing_dict['flm_diff_1'] = abs(f_lat_match['bearing'].values[0]) - abs(heading)
            bearing_dict['flm_diff_2'] = abs(f_lat_match['bearing'].values[0]) - abs(heading - math.pi)
        if not t_lat_match.empty:
            bearing_dict['tlm_diff_1'] = abs(t_lat_match['bearing'].values[0]) - abs(heading)
            bearing_dict['tlm_diff_2'] = abs(t_lat_match['bearing'].values[0]) - abs(heading - math.pi)
        
        bd_value_list = list(bearing_dict.values())
        bd_key_list = list(bearing_dict.keys())

        position = bd_value_list.index(min(bd_value_list))
        answer = bd_key_list[position]

        if answer == 'flm_diff_1':
            current_segment_index = self.segment_df.index[self.segment_df['from_lat'] == lat_b][0]
        elif answer == 'flm_diff_2':
            current_segment_index = self.segment_df.index[self.segment_df['from_lat'] == lat_b][0]
        elif answer == 'tlm_diff_1':
            current_segment_index = self.segment_df.index[self.segment_df['to_lat'] == lat_b][0]
        elif answer == 'tlm_diff_2':
            current_segment_index = self.segment_df.index[self.segment_df['to_lat'] == lat_b][0]

        self.current_segment_index = current_segment_index
        self.distance_along_segment = self.distance_in_m(lat_a, lon_a, lat_b, lon_b)

        return current_segment_index




    def find_segment_after_x_hours(self, hours, speed):

        target_time = 3600 * hours * speed
        elapsed_time = 0
        i = 0
        for index, row in self.segment_df.iterrows():
            i += 1
            elapsed_time += row['length(m)'] * speed
            if elapsed_time > target_time:
                return i




    def get_point_list(self, segments):
        
        points = []
        
        for segment in segments:
            points.append(Point(segment['begin']['longitude'], segment['begin']['latitude']))
        
        return points



    def create_segment_df(self, segments):

        rows = []

        for segment in segments:
            rows.append({
                          'from_lat': segment['begin']['latitude'], 
                          'from_lon': segment['begin']['longitude'], 
                          'from_elevation': segment['begin']['elevation'] if 'elevation' in segment['begin'].keys() else "",
                          'to_lat': segment['end']['latitude'], 
                          'to_long': segment['end']['longitude'], 
                          'to_elevation': segment['end']['elevation'] if 'elevation' in segment['end'].keys() else "",
                          'length(m)': segment['length(m)'], 
                          'bearing': segment['bearing'], 
                          'slope': segment['slope'] if 'slope' in segment.keys() else None,
                          'segment_id': segment['segment_id'],
                          'cumulative_distance_to_segment': segment['cumulative_distance_to_segment']
                      }) 

        df = pd.DataFrame(rows)

        return df



    def calculate_heading(self, lat_a, lon_a, lat_b, lon_b):
        
        x = math.cos(lat_b) * math.sin(lon_a - lon_b)
        y = math.cos(lat_a) * math.sin(lat_b) - math.sin(lat_a) * math.cos(lat_b) * math.cos(lon_a - lon_b)
        bearing = math.atan2(x,y)
        return bearing



    def distance_in_m(self, lat1, lon1, lat2, lon2):
        # haversine function
        lat1_rad=math.radians(lat1)
        lat2_rad=math.radians(lat2)
        lon1_rad=math.radians(lon1)
        lon2_rad=math.radians(lon2)
        delta_lat=lat2_rad-lat1_rad
        delta_lon=lon2_rad-lon1_rad
        a=math.sqrt((math.sin(delta_lat/2))**2+math.cos(lat1_rad)*math.cos(lat2_rad)*(math.sin(delta_lon/2))**2)
        d=2*6371000*math.asin(a)
        return d


if __name__ == '__main__':
    segments = pickle.load( open( "segments.pkl", "rb" ) )
    try:
        upload_tstamp = datetime.now()
        course_model_id = str(uuid.uuid4())    
        for segment in segments:
            data_wrangler.write_segments_to_dynamo(segment, upload_tstamp, course_model_id)
    except Exception as e:
        logging.error('error writing course to dyanmodb: ' + e)

