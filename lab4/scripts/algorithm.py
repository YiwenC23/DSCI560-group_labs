from yfinance_retrieve import connect_db, SessionLocal, StockData, TickerIndex, get_DBdata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf


def load_data():
    engine = connect_db()
    stock_data = pd.read_sql_table('stock_data', engine)
    return stock_data

def moving_average(data, window):
    return pd.Series(data).rolling(window=window).mean()

def moving_average_prediction(df, column, window):
    df[f"MA_{window}"] = moving_average(df[column], window)
    return df

def exponential_smoothing(data, trend, seasonal, seasonal_periods):
    """
    Builds an Exponential Smoothing model.
    
    :param trend: str, 'add' or 'mul', specifies the type of trend component.
    :param seasonal: str, 'add' or 'mul', specifies the type of seasonal component.
    """
    model = ExponentialSmoothing(data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fitted_model = model.fit()
    return fitted_model

# Function to generate predictions using the fitted Exponential Smoothing model
def exponential_smoothing_prediction(model, steps):
    """
    Generate future predictions using the fitted Exponential Smoothing model.
    
    :param steps: int, number of steps to predict into the future.
    """
    predictions = model.forecast(steps)
    return predictions

def seasonal_autoarima(train_data, seasonal_periods):
    model = pm.auto_arima(train_data, seasonal=True, m=seasonal_periods, stepwise=True, trace=True)
    return model

def seasonal_autoarima_prediction(model, steps):
    predictions = model.predict(n_periods=steps)
    return predictions

def model_evaluation(actual, predicted, model_name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    results = {f'MAE_{model_name}': mae, f'RMSE_{model_name}': rmse}

    return results

# Plot results

if __name__ == "__main__":
    stock_data =load_data()

    plot_acf(stock_data, lags=30)  # Adjust lags as needed
    plt.title('ACF of Stock Time Series')
    plt.show()

    # slice stock_data to closing price on weekly basis / daily basis

    proportion = int(len(stock) * 0.8) # subject to change ...
    training_data, testing_data = stock[:proportion], stock[proportion:]
    metrics = {}

    # moving average
    stock = moving_average_prediction(stock, 3)
    metrics = model_evaluation(stock['MA_3'], stock['close'], 'MA') # subject to change

    # exponential smoothing
    stock["ES"] = np.nan
    for i in range(10, len(stock)):
        last_10_data = stock[column].iloc[i-10:i]
        model = exponential_smoothing(last_10_data, trend='add', seasonal='add', seasonal_periods=12)
        predictions = exponential_smoothing_prediction(model, steps=1)
        stock.loc[stock.index[i], "ES"] = predictions[0]

    metrics = model_evaluation(stock['ES'], stock['close'], 'ES')

    # auto-arima (train vs test)
    stock["ARIMA1"] = np.nan
    ARIMA = seasonal_autoarima(training_data, 12)
    predictions = seasonal_autoarima_prediction(ARIMA, steps=len(testing_data))
    stock.loc[stock.index[len(training_data)], "ARIMA1"] = predictions[0]

    metrics = model_evaluation(stock['ARIMA1'], stock['close'], 'ARIMA1')

    # auto-arima
    stock["ARIMA2"] = np.nan
    for i in range(10, len(stock)):
        last_10_data = stock[column].iloc[i-10:i]
        model = seasonal_autoarima(training_data, 12)
        predictions = seasonal_autoarima_prediction(ARIMA, steps=1)
        stock.loc[stock.index[i], "ARIMA2"] = predictions[0]

    metrics = model_evaluation(stock['ARIMA2'], stock['close'], 'ARIMA2')


# Define the function to retrieve data from the database
# def get_DBdata():
#     with SessionLocal() as session:
#         stock_data = session.query(StockData).all()
#         df = pd.DataFrame([s.__dict__ for s in stock_data])
#         return(df)

if __name__ == "__main__":
    get_DBdata()
