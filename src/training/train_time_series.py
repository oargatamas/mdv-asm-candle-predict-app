from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras


def setup():
    pd.set_option('future.no_silent_downcasting', True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    return


def train_time_series():
    setup()

    sliding_window = 60  # 60
    prediction_window = 5
    input_window = sliding_window - prediction_window
    bar_columns = 5
    center_index = input_window * bar_columns

    # Load the data and split it between train and test sets
    raw_data_location = '../../data/prepared/training_eurusd_m1.csv'  # Todo move to environment and/or CLI variable
    df = pd.read_csv(raw_data_location, sep=',')

    train, test = train_test_split(df, test_size=0.2)

    x_train = train.iloc[:, 1:center_index]
    y_train = train.iloc[:, (center_index + 1):]

    x_test = test.iloc[:, 1:center_index]
    y_test = test.iloc[:, (center_index + 1):]

    # x_train = x_train.values.reshape((1, x_train.shape[0], x_train.shape[1]))
    # x_test = x_test.values.reshape((1, x_test.shape[0], y_train.shape[1]))

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_train.head())

    print("y_train shape:", y_train.shape)
    print(y_train.shape[0], "test samples")
    print(y_train.sample())

    model = get_model(x_train.shape, y_train.shape[1])

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
        ],
    )

    batch_size = 128
    epochs = 20

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath="../model/model_at_epoch_{epoch}.keras"),
        #keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
    ]

    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.15,
        callbacks=callbacks,
    )
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Model score:")
    print(score)

    model.save("../model/final_model.keras")
    model.summary()

    input = x_test.iloc[0].values.reshape(1, -1)
    print(input)

    result = model.predict(input)
    print("Prediction: ")
    print(result)

    print("Reality: ")
    print(y_test.iloc[0].values.reshape(1, -1))


def get_model(input_shape, output_shape):
    return keras.Sequential(
        [
            keras.layers.Input((input_shape[1],)),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(output_shape, activation="sigmoid"),
            keras.layers.Reshape([-1, 1]),
        ]
    )

if __name__ == '__main__':
    train_time_series()
