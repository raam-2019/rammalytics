import requests
import pdb
import csv
import time
import pickle
from decimal import Decimal
import uuid
import io
import logging
from datetime import datetime

import boto3
import pandas as pd


__author__ = "Steven Wangen"
__version__ = "0.1"
__email__ = "srwangen@wisc.edu"
__status__ = "Development"


logger = logging.getLogger(__name__)
log_level = logging.INFO
logging.basicConfig(level=log_level,
                        format='%(asctime)s %(levelname)s %(message)s')


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
    table = dynamodb.Table('raamalytics')
    model_run_id = str(uuid.uuid4())
    model_tstamp = str(datetime.now())
    
    logging.info('data_wrangler.write_prediction_to_database(): writing out {} records w/ timestamp: {} and id: {}'.format(prediction_df.size, model_tstamp, model_run_id))
    
    with table.batch_writer() as batch:
    
        write_count = 0

        for index, row in prediction_df.iterrows():

            entry = {
                'key': str(uuid.uuid4()),
                'model_run': model_run_id,
                'model_run_tstamp': model_tstamp,
                'segment_id': row['segment_id'],
                'wind_speed_m_per_s': Decimal(str(row['wind_speed(m/s)'])),
                'wind_speed_confidence_level': Decimal(str(row['wind_speed_confidence_level'])),
                'wind_direction': Decimal(str(row['wind_direction'])),
                'wind_direction_confidence_level': Decimal(str(row['wind_direction_confidence_level'])),
                'predicted_power_watts': Decimal(str(row['predicted_power(watts)'])),
                'headwind_m_per_s': Decimal(str(row['headwind(m/s)'])),
                'segment_speed_km_per_h': Decimal(str(row['segment_speed(km/h)'])), 
                'segment_duration_s': Decimal(str(row['segment_duration(s)'])),
                'segment_tss': Decimal(str(row['segment_tss'])),
                'predicted_arrival_time': str(row['predicted_arrival_time']),
                'predicted_finishing_time': str(row['predicted_finishing_time']),
                'cumulative_distance_to_segment': Decimal(str(row['cumulative_distance_to_segment'])),
                'course_bearing': Decimal(str(row['bearing'])),
                'segment_calories': Decimal(str(row['segment_calories'])),

                'wind_speed_plus_2hr': Decimal(str(row['plus_2_wind_speed(m/s)'])),
                'wind_speed_plus_2hr_confidence_level': Decimal(str(row['plus_2_wind_speed_confidence_level'])),
                'wind_direction_plus_2hr': Decimal(str(row['plus_2_wind_direction'])),
                'wind_direction_plus_2hr_confidence_level': Decimal(str(row['plus_2_wind_direction_confidence_level'])),
                'headwind_plus_2hr': Decimal(str(row['plus_2_headwind(m/s)'])),

                'segment_speed_plus_2hr': Decimal(str(row['plus_2_segment_speed(km/h)'])),
                'segment_duration_plus_2hr': Decimal(str(row['plus_2_segment_duration(s)'])),
                'predicted_arrival_time_plus_2hr': Decimal(str(row['plus_2_predicted_arrival_time'])),
                'predicted_finishing_time_plus_2hr': Decimal(str(row['plus_2_predicted_finishing_time'])),
                'tss_plu_2_hr': Decimal(str(row['plus_2_segment_tss'])),
                'calories_plus_2hr': Decimal(str(row['plus_2_segment_calories']))
            }

            try:
                batch.put_item(Item = entry)
                # table.put_item(Item = entry)
                # print(entry)
                write_count += 1

            except Exception as e:
                logging.error('Item = ' + str(entry))
                logging.error(e)
                pass

    logging.info("wrote {} of {} records to dynamodb".format(write_count, prediction_df.size))





def write_prediction_to_database2(rows):
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('raamalytics')
    model_run_id = str(uuid.uuid4())
    model_tstamp = str(datetime.now())
    
    logging.info('data_wrangler.write_prediction_to_database(): writing out {} records w/ timestamp: {} and id: {}'.format(len(rows), model_tstamp, model_run_id))
    
    with table.batch_writer() as batch:
    
        write_count = 0

        for row in rows:
            
            entry = {
                'key': str(uuid.uuid4()),
                'model_run': model_run_id,
                'model_run_tstamp': model_tstamp,
                'segment_id': row['segment_id'],
                'wind_speed_m_per_s': Decimal(str(row['wind_speed(m/s)'])),
                # 'wind_speed_confidence_level': Decimal(str(row['wind_speed_confidence_level'])),
                'wind_direction': Decimal(str(row['wind_direction'])),
                # 'wind_direction_confidence_level': Decimal(str(row['wind_direction_confidence_level'])),
                'predicted_power_watts': Decimal(str(row['predicted_power(watts)'])),
                'headwind_m_per_s': Decimal(str(row['headwind(m/s)'])),
                'segment_speed_km_per_h': Decimal(str(row['segment_speed(km/h)'])), 
                'segment_duration_s': Decimal(str(row['segment_duration(s)'])),
                'segment_tss': Decimal(str(row['segment_tss'])),
                'predicted_arrival_time': str(row['predicted_arrival_time']),
                'predicted_finishing_time': str(row['predicted_finishing_time']),
                'cumulative_distance_to_segment': Decimal(str(row['cumulative_distance_to_segment'])),
                'course_bearing': Decimal(str(row['bearing'])),
                'segment_calories': Decimal(str(row['segment_calories'])),
                
                'wind_speed_plus_2hr': Decimal(str(row['plus_2_wind_speed(m/s)'])),
                # 'wind_speed_plus_2hr_confidence_level': Decimal(str(row['plus_2_wind_speed_confidence_level'])),
                'wind_direction_plus_2hr': Decimal(str(row['plus_2_wind_direction'])),
                # 'wind_direction_plus_2hr_confidence_level': Decimal(str(row['plus_2_wind_direction_confidence_level'])),
                'headwind_plus_2hr': Decimal(str(row['plus_2_headwind(m/s)'])),

                'segment_speed_plus_2hr': Decimal(str(row['plus_2_segment_speed(km/h)'])),
                'segment_duration_plus_2hr': Decimal(str(row['plus_2_segment_duration(s)'])),
                'predicted_arrival_time_plus_2hr': str(row['plus_2_predicted_arrival_time']),
                'predicted_finishing_time_plus_2hr': str(row['plus_2_predicted_finishing_time']),
                'tss_plu_2_hr': Decimal(str(row['plus_2_segment_tss'])),
                'calories_plus_2hr': Decimal(str(row['plus_2_segment_calories']))
            }
        
            try:
                batch.put_item(Item = entry)
                # table.put_item(Item = entry)
                # print(entry)
                write_count += 1

            except Exception as e:
                logging.error('Item = ' + str(entry))
                logging.error(e)
                pass

    logging.info("wrote {} of {} records to dynamodb".format(write_count, len(rows)))





def write_cost_of_rest_to_database(hours, rows):
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('raamalytics')
    model_run_id = str(uuid.uuid4())
    model_tstamp = str(datetime.now())
    
    logging.info('data_wrangler.write_cost_of_rest_to_database(): writing out {} records w/ timestamp: {} and id: {}'.format(len(rows), model_tstamp, model_run_id))
    
    with table.batch_writer() as batch:
    
        write_count = 0

        for row in rows:

            entry = {
                "key": str(uuid.uuid4()),
                "prediction_tstamp": str(model_tstamp),
                "model_run": model_run_id,  
                "window_size_hours": str(Decimal(str(hours))),
                "segment_id": row["segment_id"],
                "elevation": str(Decimal(str(row["elevation"]))),
                "cumulative_distance_to_segment": str(Decimal(str(row["cumulative_distance_to_segment"]))),
                "cost_of_rest_s": str(Decimal(str(row["cost_of_rest"])))
            }
            
            try:
                batch.put_item(Item = entry)
                # table.put_item(Item = entry)
                # print(entry)
                write_count += 1

            except Exception as e:
                logging.error('Item = ' + str(entry))
                logging.error(e)
                pass

    logging.info("wrote {} records to dynamodb".format(write_count))







def heartbeat():
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('rammalytics_heartbeat')
    key = str(uuid.uuid4())
    tstamp = str(datetime.now())
    
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



