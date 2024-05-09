import dask.dataframe as dd
import pandas as pd
from datetime import datetime


def prepare_candle_data():
    raw_data_location = '../../data/raw/eurusd_m1_candles_2024-01-31_2024-05-08.csv'  # Todo move to environment and/or CLI variable
    ddf = dd.read_csv(
        raw_data_location,
        dtype={
            "date": "str",
            "time": "str",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "tickvol": "float64",
            "vol": "float64",
            "spread": "float64"
        },
        sep='\t',
        blocksize=25e6)

    ddf = ddf.assign(timestamp=lambda x: pd.to_datetime(x['date'] + x['time'], format='%Y.%m.%d%H:%M:%S', utc=True))
    ddf = ddf.drop(columns=['date', 'time'])

    ddf = ddf.assign(candle_type=lambda x: pd.to_numeric(x['open'] > x['close'], downcast='float'))

    print(ddf.dtypes)
    pd.options.display.max_columns = None
    print(ddf.head(3))

    # ddf.to_csv("final_data")

    return


prepare_candle_data()
