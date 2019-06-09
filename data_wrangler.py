import requests
import pdb
import csv
import time
import boto3
import pandas as pd
import pickle
from decimal import Decimal
import uuid
import io
import logging
from datetime import datetime


__author__ = "Steven Wangen"
__version__ = "0.1"
__email__ = "srwangen@wisc.edu"
__status__ = "Development"



BUCKET_NAME = "raam-test-run-datastore-dump"


session = boto3.Session()



def load_course_data():
    segments = pickle.load( open( "segments.pkl", "rb" ) )
    points = []
    for segment in segments:
        points.append(Point(segment['begin']['latitude'], segment['begin']['longitude']))

    course_line = LineString(points)
    course_mp_line = MultiPoint(points)



###########################
# Write out to the dynamoDB

def write_segments_to_dynamo(segment, upload_tstamp, course_model_id):

    dynamodb = boto3.resource('dynamodb')
    # course_prescription_table = dynamodb.Table('course_prescription')
    table = dynamodb.Table('course_data')
    entry = {
        'course_model_id': course_model_id,
        'timestamp': str(upload_tstamp),
        'key': str(uuid.uuid4()),
        'begin': {
            'latitude': Decimal(str(segment['begin']['latitude'])),
            'longitude': Decimal(str(segment['begin']['longitude'])),
            'id': str(segment['begin']['id']),
            'elevation': Decimal(str(segment['begin']['elevation']))
            },
        'end': {
            'latitude': Decimal(str(segment['end']['latitude'])),
            'longitude': Decimal(str(segment['end']['longitude'])),
            'id': str(segment['begin']['id']),
            'elevation': Decimal(str(segment['end']['elevation']))
            },
        'length(m)': Decimal(str(segment['length(m)'])),
        'bearing': Decimal(str(segment['bearing'])),
        'slope': Decimal(str(segment['slope'])),
        'cumulative_distance_to_segment': Decimal(str(segment['cumulative_distance_to_segment'])),
        'segment_id': segment['segment_id']
        }

    table.put_item(Item = entry)




def write_prediction_to_database(prediction_df):
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('testing')
    model_run_id = str(uuid.uuid4())
    model_tstamp = datetime.now()
    for index, row in prediction_df.iterrows():

        entry = {
            'key': str(uuid.uuid4()),
            'model_run': model_run_id,
            'model_run_tstamp': str(model_tstamp),
            'segment_id': row['segment_id'],
            'wind_speed(m/s)': Decimal(str(row['wind_speed(m/s)'])),
            'wind_speed_confidence_level': Decimal(str(row['wind_speed_confidence_level'])),
            'wind_direction': Decimal(str(row['wind_direction'])),
            'wind_direction_confidence_level': Decimal(str(row['wind_direction_confidence_level'])),
            'predicted_power(watts)': Decimal(str(row['predicted_power(watts)'])),
            'headwind(m/s)': Decimal(str(row['headwind(m/s)'])),
            'segment_speed(km/h)': Decimal(str(row['segment_speed(km/h)'])), 
            'segment_duration(s)': Decimal(str(row['segment_duration(s)'])),
            'segment_tss': Decimal(str(row['segment_tss'])),
            'predicted_arrival_time': str(row['predicted_arrival_time']),
            'predicted_finishing_time': str(row['predicted_finishing_time']),
            'cumulative_distance_to_segment': Decimal(str(row['cumulative_distance_to_segment'])),
            'course_bearing': Decimal(str(row['bearing'])),
            'wind_speed+2hr': str(Decimal(row['wind_speed+2hr'])),
            'wind_speed+2hr_confidence_level': str(Decimal(row['wind_speed+2hr_confidence_level'])),
            'wind_direction+2hr': str(Decimal(row['wind_direction+2hr'])),
            'wind_direction+2hr_confidence_level': str(Decimal(row['wind_direction+2hr_confidence_level'])),
            'headwind+2hr': str(Decimal(row['headwind+2hr(m/s)'])),
            'segment_calories': str(Decimal(row['segment_calories']))
        }
        try:
            table.put_item(Item = entry)
        except Exception as e:
            logging.error('Item = ' + entry)
            logging.error(e)
            pass





def heartbeat():
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('rammalytics_heartbeat')
    key = str(uuid.uuid4())
    tstamp = datetime.now()
    
        entry = {
            'key': key,
            'timestamp': tstamp
        }
        try:
            table.put_item(Item = entry)
        except Exception as e:
            logging.error('Item = ' + entry)
            logging.error(e)
            pass



###############################
# get the data into this module



def bucket_csv_to_df():

    # get a handle on s3
    s3 = boto3.client('s3')
    s3r = boto3.resource('s3')

    # get the most recently updated object in the bucket
    get_last_modified = lambda obj: int(obj['LastModified'].strftime('%s'))

    # grab the name of the most recent file
    objs = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix='testing_datastore_1min')['Contents']
    last_added = [obj['Key'] for obj in sorted(objs, key=get_last_modified)][0]
 
    """
    # get a handle on the object you want (i.e. your file)
    response = s3r.Object(BUCKET_NAME, last_added).get()["Body"].read() 
    csv_content = response.decode('utf-8').split()
    df = pd.read_csv(csv_content)
    """

    # other way: (https://stackoverflow.com/questions/37703634/how-to-import-a-text-file-on-aws-s3-into-pandas-without-writing-to-disk)
    logging.info('data_wrangler.bucket_csv_to_df(): reading csv for processing - {}'.format(last_added))
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=last_added)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))

    return df



