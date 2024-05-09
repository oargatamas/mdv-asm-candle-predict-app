import dask.dataframe as dd
import pandas as pd
import re
from datetime import datetime


def setup():
    pd.set_option('future.no_silent_downcasting', True)
    pd.set_option('display.max_columns', None)
    return


def prepare_candle_data():
    setup()

    raw_data_location = '../../data/raw/eurusd_m1_candles_2024-01-31_2024-05-08.csv'  # Todo move to environment and/or CLI variable
    ddf = dd.read_csv(
        raw_data_location,
        sep='\t',
        blocksize=25e6)

    ddf = ddf.reset_index()

    # Convert date and time into a unix timestamp
    ddf = ddf.assign(timestamp=lambda df: pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M:%S'))
    ddf = ddf.assign(timestamp=lambda df: df['timestamp'].apply(lambda x: x.timestamp(), meta=('timestamp', 'float64')))

    # Calculate candle direction
    ddf = ddf.assign(candle_type=lambda x: x['open'] > x['close'])
    ddf = ddf.replace({False: 0, True: 1})

    # Drop not needed columns
    ddf = ddf.drop(columns=['date', 'time', 'spread', 'vol'])
    ddf = ddf.compute()
    print(ddf.dtypes)
    print(ddf.head())
    ddf.to_csv("training_eurusd_m1.csv")

    return


def timestamp_map(df):
    # return pd.to_datetime(df['date'] + df['time'], format='%Y.%m.%d%H:%M:%S')
    return


prepare_candle_data()
