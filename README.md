# mdv-asm-candle-predict-app
Stock Exchange ML application aiming to predict the next N chart candles based on the previous M number of candles.


# Explain Input Data 
Input data is a regular stock price candle history exported from MetaTrader5 client. The exact method of export described [here](https://strategyquant.com/doc/quantdatamanager/how-to-export-data-from-metatrader-5/)
After the export the CSV will have the following columns/fields: 

- `date`: Date of the period aggregated in the candle
- `time`: Time of the period aggregated in the candle
- `open`: Actual price of the instrument the beginning of the period
- `high`: Maximum price of the instrument the in the period
- `low`: Minimum price of the instrument the in the period
- `close`: Actual price of the instrument the end of the period
- `tickvol`: Number of price changing event in the period
- `vol`: Number of trades made in the period
- `spread`: Have no clue...

# Data Preparation Steps 

### Basic field cleaning
1. Convert date and time fields into a timestamp
2. Determine the candle direction ( bullish / bearish ) based on the open and close prices
3. Drop the spread and volume columns. At the moment I cannot see the potential in this data

### Prepare Sliding Window
Basic idea is that we give the current chart window to the model, and we expect from the model to give back next possible candles in the chart. For that we need to train the model with the following statements: 

![sliding_window_sample.JPG](doc/Fsliding_window_sample.JPG)

-  `red` -> The complete window (training sample = row in training data)
-  `yellow` -> The simulated history (model input data) 
-  `purple` -> The next candles (expected prediction)

Each row in the source csv describing one candle in the chart. To be able to produce that data we have to search and copy the upcoming `N` rows (N = size of data window) for each row.
In case of `N=3` the following should be the expectation:

![sliding_window_csv_expectation.JPG](doc%2Fsliding_window_csv_expectation.JPG)

# Describing Model