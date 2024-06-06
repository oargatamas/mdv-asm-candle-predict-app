import dask.dataframe as dd
import pandas as pd
import re
import numpy as np


def setup():
    pd.set_option('future.no_silent_downcasting', True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    return


def prepare_candle_data():
    setup()
    sliding_window = 3  # 60
    prediction_window = 5
    train_window = sliding_window - prediction_window

    raw_data_location = '../../data/raw/eurusd_m1_candles_2024-01-31_2024-05-08.csv'  # Todo move to environment and/or CLI variable
    ddf = dd.read_csv(
        raw_data_location,
        sep='\t',
        blocksize=25e6)

    # ddf = ddf.reset_index()
    print(ddf.dtypes)

    # Converting date and time as a number -> this will be suitable for the ML model comparing to date time object
    ddf = ddf.assign(timestamp=lambda df: df['date'] + ' ' + df['time'])
    ddf = ddf.assign(timestamp=lambda df: df['timestamp'].apply(
        lambda x: int(re.sub('[^0-9]', '', x)[:-2]),
        meta=('timestamp', 'int64')))

    # Calculate candle direction 0 -> bullish candle , 1 -> bearish candle
    ddf = ddf.assign(candle_type=lambda x: x['open'] > x['close'])
    ddf = ddf.replace({False: 0, True: 1})

    # To keep time series direction
    ddf = ddf.sort_values(by=['timestamp'], ascending=True)

    # Drop not needed columns
    ddf = ddf.drop(columns=['date', 'time', 'spread', 'vol'])

    ddf = ddf.set_index('timestamp')

    base_columns = ddf.columns
    # columns_to_add = [col + str(j + 2) for j in range(0, sliding_window) for col in base_columns]
    # ddf = ddf.assign(**{col: lambda df: v_lookup(col, df) for col in columns_to_add})

    # -------------------------------

    ddf_copy = ddf.copy()  # Putting original data into a copy dataframe
    # Index is the first timestamp of the moving window

    for i in range(1, sliding_window):
        suffix = str(i)
        print('Exporting forward data. Iteration: ' + suffix)
        ddf_copy = ddf_copy.join(ddf.shift(-int(suffix)), rsuffix=suffix)  # Join shifted versions

    # Add 0 to the first columns
    temp_columns = list(ddf_copy.columns)
    for i in range(len(ddf_copy.columns)//sliding_window):
        temp_columns[i] = temp_columns[i] + "0"
    ddf_copy.columns = temp_columns

    print(ddf_copy.tail())  # The last N rows contains NaN because of the window making procedure

    # -------------------------------

    ddf = ddf.compute()

    print('Final training set created.')
    print(ddf.head(10))
    print(len(ddf))
    print(ddf.dtypes)

    ddf.to_csv("training_eurusd_m1.csv")

    return


def assign(df):
    return df.query('timestamp == 202401312158')['open']


if __name__ == '__main__':
    prepare_candle_data()
