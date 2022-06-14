import pandas as pd
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger


@task
def read_data(path):
    logger = get_run_logger()
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date: str):
    logger = get_run_logger()
    if date:
        run_date = datetime.strptime(date, '%Y-%m-%d')
    else:
        run_date = datetime.today()

    logger.debug(f'Run date: {run_date}')

    train_date = run_date - relativedelta(months=2)
    val_date = run_date - relativedelta(months=1)

    train_path = f'./data/fhv_tripdata_{train_date.year}-{train_date.month:02d}.parquet'
    val_path = f'./data/fhv_tripdata_{val_date.year}-{val_date.month:02d}.parquet'

    return train_path,val_path,run_date

@flow
def main(date=None):
    logger = get_run_logger()
    train_path, val_path, run_date = get_paths(date).result()
    logger.info(f'Train file: {train_path} &&  Val file: {val_path}')

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    #Save the assets
    dv_file = f'models/dv-{run_date.strftime("%Y-%m-%d")}.b'
    model_file = f'models/model-{run_date.strftime("%Y-%m-%d")}.bin'
    with open(dv_file, "wb") as f_out:
        pickle.dump(dv, f_out)
    with open(model_file, "wb") as f_out:
        pickle.dump(dv, f_out)

    run_model(df_val_processed, categorical, dv, lr)

#main(date="2021-08-15")
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
        flow=main,
        name="hw_model_training",
        schedule=CronSchedule(cron="0 9 15 * *"),
        flow_runner=SubprocessFlowRunner(),
        tags=["ml"]
)

