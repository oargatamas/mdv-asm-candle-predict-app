import dask.dataframe as dd
import pandas as pd
import re

def setup():
    pd.set_option('future.no_silent_downcasting', True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    return


def prepare_candle_data():

    sliding_window = 60  # 60
    prediction_window = 5
    input_window = sliding_window - prediction_window

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
    ddf = ddf.assign(direction=lambda x: x['open'] > x['close'])
    ddf = ddf.replace({False: 0, True: 1})

    # To keep time series direction
    ddf = ddf.sort_values(by=['timestamp'], ascending=True)

    # Drop not needed columns
    ddf = ddf.drop(columns=['date', 'time', 'spread', 'vol'])

    ddf = ddf.set_index('timestamp')

    # Index is the first timestamp of the moving window
    print('Shifting candle data to fill the sliding window...')
    ddf_copy = ddf.copy()
    for i in range(1, sliding_window):
        suffix = str(i)
        ddf_copy = ddf_copy.join(ddf.shift(int(suffix)), rsuffix='_'+suffix)  # Join shifted versions

    ddf = ddf_copy.dropna()

    # Labeling series with
    ddf = ddf.rename(columns={
        'open': 'open_0',
        'close': 'close_0',
        'high': 'high_0',
        'low': 'low_0',
        'tickvol': 'tickvol_0',
        'direction': 'direction_0'
    })

    labels = {}
    for i in range(0, sliding_window):
        is_input = i < input_window
        prefix = 'in' if is_input else 'out'
        suffix = (input_window - i - 1) if is_input else i - input_window
        labels['open_' + str(i)] = '{prefix}_open_{suffix}'.format(prefix=prefix, suffix=suffix)
        labels['close_' + str(i)] = '{prefix}_close_{suffix}'.format(prefix=prefix, suffix=suffix)
        labels['high_' + str(i)] = '{prefix}_high_{suffix}'.format(prefix=prefix, suffix=suffix)
        labels['low_' + str(i)] = '{prefix}_low_{suffix}'.format(prefix=prefix, suffix=suffix)
        labels['tickvol_' + str(i)] = '{prefix}_tickvol_{suffix}'.format(prefix=prefix, suffix=suffix)
        labels['direction_' + str(i)] = '{prefix}_direction_{suffix}'.format(prefix=prefix, suffix=suffix)

    ddf = ddf.rename(columns=labels)


    ddf = ddf.compute()

    print('Final training set created.')
    print('Head elements:')
    print(ddf.head(10))
    print('Tail elements:')
    print(ddf.tail(10))
    print('{rows} rows exported'.format(rows= len(ddf)))

    ddf.to_csv("training_eurusd_m1.csv")

    return




if __name__ == '__main__':
    setup()
    prepare_candle_data()
