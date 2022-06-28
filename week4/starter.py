#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd

def read_data(taxi_type,year,month):
    print(f'Reading: {taxi_type} data for {year} {month}')
    df = pd.read_parquet(f'https://nyc-tlc.s3.amazonaws.com/trip+data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet')
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy() 

    return df

def make_prediction():
    taxi_type = sys.argv[1]
    year = int(sys.argv[2])
    month = int(sys.argv[3])

    print(f'Processing: {taxi_type} data for {year} {month}')

    categorical = ['PUlocationID', 'DOlocationID']

    df = read_data(taxi_type,year,month)

    print(f'{df.columns}')
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    df['prediction'] = y_pred
    print(df['prediction'].mean())

if __name__=="__main__": 
    make_prediction()
