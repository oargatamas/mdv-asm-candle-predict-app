from src.preparation.prepare_candle_data import prepare_candle_data
from src.training.train_time_series import train_time_series


def start_pipeline():
    prepare_candle_data()
    train_time_series()
    return

if __name__ == '__main__':
    start_pipeline()
