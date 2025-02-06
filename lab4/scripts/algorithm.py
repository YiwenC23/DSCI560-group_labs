from yfinance_retrieve import connect_db
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm


def load_data():
    engine = connect_db()
    stock_data = pd.read_sql_table('stock_data', engine)
    return stock_data

# training data and testing data

def moving_average(data, window, forecast_steps):
    """
    Predict future stock prices using Moving Average.
    """
    predictions = []
    for _ in range(forecast_steps):
        moving_avg = data[-window:].mean()  # Use the last 'window' observations
        predictions.append(moving_avg)
        data = pd.concat([data, pd.Series([moving_avg])], ignore_index=True)
    return pd.Series(predictions)

# ma_predictions = moving_average(stock_data['close'], window=10, forecast_steps=10)
# print("Moving Average Predictions:", ma_predictions)

def exponential_smoothing(data, forecast_steps):
    """
    Predict future stock prices using Simple Exponential Smoothing.
    """
    model = SimpleExpSmoothing(data)
    model_fit = model.fit()  # Automatically optimize alpha
    predictions = model_fit.forecast(steps=forecast_steps)
    return predictions

# es_predictions = exponential_smoothing(stock_data['Close'], forecast_steps=10)
# print("Exponential Smoothing Predictions:", es_predictions)


def autoarima_model(train_data, test_data, seasonal=False, m=12):
    """
    Builds and predicts using AutoARIMA model.
    """
    
    # Fit the AutoARIMA model
    model = pm.auto_arima(train_data, seasonal=seasonal, m=m, stepwise=True, trace=True)

    # Make predictions on the test data
    forecast = model.predict(n_periods=len(test_data))
    
    return model, forecast

# train_data = stock_data['close'][:'2024-01-01']  # Adjust as needed
# test_data = stock_data['close']['2024-01-02':]
# model, forecast = autoarima_model(train_data, test_data, seasonal=True, m=12)

def evaluate_models(stock_data, forecast_steps=10):
    # Split data into training and testing sets (for example)
    train_data = stock_data['close'][:'2024-01-01']
    test_data = stock_data['close']['2024-01-02':]

    # Moving Average Predictions
    ma_predictions = moving_average(train_data, window=10, forecast_steps=forecast_steps)
    ma_rmse, ma_mae = calculate_metrics(test_data[:forecast_steps], ma_predictions)
    print(f"Moving Average - RMSE: {ma_rmse}, MAE: {ma_mae}")

    # Exponential Smoothing Predictions
    es_predictions = exponential_smoothing(train_data, forecast_steps=forecast_steps)
    es_rmse, es_mae = calculate_metrics(test_data[:forecast_steps], es_predictions)
    print(f"Exponential Smoothing - RMSE: {es_rmse}, MAE: {es_mae}")

    # AutoARIMA Model Predictions
    model, forecast = autoarima_model(train_data, test_data, seasonal=True, m=12)
    arima_rmse, arima_mae = calculate_metrics(test_data[:forecast_steps], forecast)
    print(f"AutoARIMA - RMSE: {arima_rmse}, MAE: {arima_mae}")

    # LSTM Model Predictions
    model, forecast = lstm_model(train_data, test_data, time_steps=60)
    lstm_rmse, lstm_mae = calculate_metrics(test_data[:forecast_steps], forecast)
    print(f"LSTM - RMSE: {lstm_rmse}, MAE: {lstm_mae}")


# Plot results