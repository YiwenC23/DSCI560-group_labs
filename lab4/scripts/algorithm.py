import sys
import pandas as pd
import yfinance as yf
import sqlalchemy as sql
import matplotlib.pyplot as plt
from yfinance_retrieve import StockData
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import warnings

from database import SessionLocal

warnings.filterwarnings('ignore')


def missing_data(df):
    for column in df.columns:
            if df[column].isnull().sum() > 0:
                df[column] = df[column].interpolate(method='linear', inplace=True, limit_direction='both')
    
    return df

def moving_average(data, window):
    return pd.Series(data).rolling(window=window).mean()

def moving_average_prediction(df, column, window):
    df["MA"] = moving_average(df[column], window)
    return df

def exponential_smoothing(data, trend):
    """
    Builds an Exponential Smoothing model.
    
    :param trend: str, 'add' or 'mul', specifies the type of trend component.
    :param seasonal: str, 'add' or 'mul', specifies the type of seasonal component.
    """
    model = ExponentialSmoothing(data, trend=trend)
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

def autoarima(train_dataing):
    model = pm.auto_arima(training_data, seasonal=False, stepwise=True, trace=True)
    return model

def autoarima_prediction(model, steps):
    predictions = model.predict(n_periods=steps)
    return predictions

def model_evaluation(actual, predicted, model_name, dict):
    actual = actual.dropna()
    predicted = predicted.dropna()

    lens = len(actual) - len(predicted)
    actual = actual.iloc[lens:]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    dict[f'MAE_{model_name}'] = mae
    dict[f'RMSE_{model_name}'] = rmse

    return dict

def decision_signal(metrics, stock_data, ARIMA_model):
    mae_dict = {key: value for key, value in metrics.items() if 'MAE' in key}
    rmse_dict = {key: value for key, value in metrics.items() if 'RMSE' in key}

    mae_dict = sorted(mae_dict.items(), key=lambda item: item[1])
    rmse_dict = sorted(rmse_dict.items(), key=lambda item: item[1])

    model_names = ['MA', 'ARIMA', 'ES']
    rank = 1
    rank_dict = {'MA':0, 'ARIMA':0, 'ES':0}

    for model in model_names:
        for key in mae_dict.keys():
            if model in key:
                rank_dict[key] += rank
                rank += 1
        for key in rmse_dict.keys():
            if model in key:
                rank_dict[key] += rank
                rank += 1
    
    best_model = min(rank_dict, key=rank_dict.get)

    if best_model == 'MA':
        prediction = stock_data['close'].iloc[-3:].mean()
    elif best_model == 'ES':
        last_10_data = stock_data['close'].iloc[-10:]
        ES = exponential_smoothing(last_10_data, trend='add')
        prediction = exponential_smoothing_prediction(ES, steps=1)
    elif best_model == 'ARIMA':
        prediction = autoarima_prediction(ARIMA_model, steps = 1)


    past_20_day_avg = stock_data['close'].iloc[-20:].mean()

    if prediction > past_20_day_avg:
        return 'sell'
    elif prediction < past_20_day_avg:
        return 'buy'
    else:
        return 'no move'


if __name__ == "__main__":

    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = missing_data(stock_data)
    stock_data = stock_data[['date', 'close']]
    stock_data.set_index('date', inplace=True)
    # stock_data = stock_data.asfreq('D')
    # stock_data.index = pd.to_datetime(stock_data.index)
    # stock_data = stock_data.asfreq('B')

    stock_data_weekly = stock_data['close'].resample('W').mean()
    stock_data_weekly = stock_data_weekly.reset_index()
    stock_data_weekly.rename(columns={'close': 'weekly_avg_close'}, inplace=True)

    # print(stock_data_weekly)

    # plot_acf(stock_data['close'], lags=52)  # Adjust lags as needed
    # plt.title('ACF of Stock Time Series for Daily Data')
    # plt.show()

    # plot_acf(stock_data_weekly['weekly_avg_close'], lags=52)  # Adjust lags as needed
    # plt.title('ACF of Stock Time Series for Weekly Data')
    # plt.show()

    # daily version
    proportion = int(len(stock_data) * 0.8)
    training_data, testing_data = stock_data[:proportion], stock_data[proportion:]
    metrics = {}

    # moving average
    MA_prediction = moving_average_prediction(stock_data, 'close', 3)
    metrics = model_evaluation(stock_data['close'], stock_data['MA'], 'MA', metrics)

    # exponential smoothing
    stock_data["ES"] = np.nan
    for i in range(10, len(stock_data)):
        last_10_data = stock_data['close'].iloc[i-10:i]
        ES = exponential_smoothing(last_10_data, trend='add')
        ES_predictions = exponential_smoothing_prediction(ES, steps=1)
        stock_data.iloc[i,stock_data.columns.get_loc("ES")] = ES_predictions

    metrics = model_evaluation(stock_data['close'], stock_data['ES'], 'ES', metrics)

    # auto-arima
    stock_data["ARIMA"] = np.nan
    ARIMA = autoarima(training_data)
    ARIMA_predictions = autoarima_prediction(ARIMA, steps = len(testing_data))
    for i, pred in enumerate(ARIMA_predictions):
        stock_data.iloc[len(training_data)+i, stock_data.columns.get_loc("ARIMA")] = pred

    metrics = model_evaluation(stock_data['close'], stock_data['ARIMA'], 'ARIMA', metrics)

    