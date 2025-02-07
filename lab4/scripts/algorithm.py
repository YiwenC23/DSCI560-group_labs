from yfinance_retrieve import connect_db
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def load_data():
    engine = connect_db()
    stock_data = pd.read_sql_table('stock_data', engine)
    return stock_data

def moving_average(data, window):
    return pd.Series(data).rolling(window=window).mean()

def moving_average_prediction(df, column, window):
    df[f"MA_{window}"] = moving_average(df[column], window)
    return df

def exponential_smoothing(data, trend='add', seasonal='add', seasonal_periods=12):
    """
    Builds an Exponential Smoothing model.
    
    :param trend: str, 'add' or 'mul', specifies the type of trend component.
    :param seasonal: str, 'add' or 'mul', specifies the type of seasonal component.
    """
    model = ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fitted_model = model.fit()
    return fitted_model

# Function to generate predictions using the fitted Exponential Smoothing model
def exponential_smoothing_prediction(model, steps=12):
    """
    Generate future predictions using the fitted Exponential Smoothing model.
    
    :param steps: int, number of steps to predict into the future.
    """
    predictions = model.forecast(steps)
    return predictions

def seasonal_autoarima(train_data, seasonal_periods=12):
    model = pm.auto_arima(train_data, seasonal=True, m=seasonal_periods, stepwise=True, trace=True)
    return model

def seasonal_autoarima_prediction(model, steps=12):
    predictions = model.predict(n_periods=steps)
    return predictions

def model_evaluation(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    return {'MAE': mae, 'RMSE': rmse}


def generate_future_predictions(data, window, forecast_steps):
    """
    Generate future predictions using the moving average.
    
    :param data: Full dataset (Pandas Series).
    :param window: The window size for the moving average.
    :param forecast_steps: Number of steps to forecast into the future.
    :return: Pandas Series containing the future predictions.
    """
    future_predictions = []
    last_window = data.iloc[-window:].values  # Use the last 'window' values
    
    for _ in range(forecast_steps):
        future_pred = last_window.mean()  # Calculate the moving average
        future_predictions.append(future_pred)
        last_window = np.append(last_window[1:], future_pred)  # Update the window
    
    # Create a date range for the future predictions
    future_dates = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq='B')[1:]
    return pd.Series(future_predictions, index=future_dates)


# Plot results

if __name__ == "__main__":
    stock_data =load_data()
    proportion = int(len(stock_data) * 0.8)
    training_data, testing_data = data[:proportion], data[proportion:]

    # moving average
    