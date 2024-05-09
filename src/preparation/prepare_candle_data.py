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
    # Nos a feladat a fenti csv-ből összehozni a következőket: ( ez amúgy egy metatrader candle export )
    # a date és time -ből össze kéne hozni egy UTC timestamp-et
    # a candle type-ot ki kéne találni ( piros vagy zöld gyertya vagy ha jobban tetszik felfele mutat vagy lefele)


    # Todo ez még működik viszont a ddf.head()-et elcseszi: ValueError: Length of values (99957) does not match length of index (3)
    ddf = ddf.assign(timestamp=lambda x: pd.to_datetime(x['date'] + x['time'], format='%Y.%m.%d%H:%M:%S', utc=True))
    # Todo szerintem még ez is jó
    ddf = ddf.drop(columns=['date', 'time'])

    # Todo ezt az Istennek se tudom rávenni, hogy 0 és 1 legyen. Bool-t nem tudok beadni a hálózatnak...
    ddf = ddf.assign(candle_type=lambda x: x['open'] > x['close'])
    ddf = ddf.assign(candle_type=lambda x: pd.to_numeric(x['candle_type']))

    print(ddf.dtypes)
    pd.options.display.max_columns = None
    print(ddf.head(3))

    # ddf.to_csv("final_data")

    return


prepare_candle_data()
