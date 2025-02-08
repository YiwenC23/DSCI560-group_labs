import sys
import pandas as pd
import yfinance as yf
import sqlalchemy as sql
from sqlalchemy.orm import declarative_base, sessionmaker
import matplotlib.pyplot as plt
from yfinance_retrieve import StockData
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import warnings

warnings.filterwarnings('ignore')

Base = declarative_base()

#* Define the function to connect to the database
def connect_db():
    global engine, SessionLocal
    try:
        #? Get the database credentials from the user
        db_username = input("Please enter the username for the database: ")
        db_password = input("Please enter the password for the database: ")
        db_name = input("Please enter the database name: ")
        
        print(f"Establishing connection to {db_name} database as {db_username}...")
        #? Create database engine with connection pool configuration
        engine = sql.create_engine(
            f"mysql+pymysql://{db_username}:{db_password}@localhost/{db_name}",
            pool_size=20,     # Number of maintained idle connections in the pool
            pool_recycle=3600,    # Recycle connections hourly to prevent connection timeout
            max_overflow=10,    # Allow up to 10 additional connections to the pool
            pool_pre_ping=True,     # Validate connection viability before use
            echo=False    # Disable engine logging
            )
        
        #? Create the Session Library
        SessionLocal = sessionmaker(
            bind=engine,
            autocommit=False,    # Require explicit commit() for transaction control
            autoflush=False,    # Delay SQL emission until flush()/commit() called, enables batch operations
            expire_on_commit=False,    # Keep object attributes accessible after commit
            class_=sql.orm.Session    # Use the SQLAlchemy Session class
        )
        
        #? Initialize database schema if not exists
        Base.metadata.create_all(bind=engine)
        
        print("Successfully connected to the database!")
    
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)


def get_DBdata():
    with SessionLocal() as session:
        stock_data = session.query(StockData).all()
        df = pd.DataFrame([s.__dict__ for s in stock_data])
        df= df.drop(columns=["_sa_instance_state"], errors="ignore")
        return df

def missing_data(df):
    for column in df.columns:
            if df[column].isnull().sum() > 0:
                df[column] = df[column].interpolate(method='linear', inplace=True, limit_direction='both')
    
    return df

def moving_average(data, window):
    return pd.Series(data).rolling(window=window).mean()

def moving_average_prediction(df, column, window):
    df[f"MA_{window}"] = moving_average(df[column], window)
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

if __name__ == "__main__":
    connect_db()
    stock_data = get_DBdata()

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
    metrics = model_evaluation(stock_data['close'], stock_data['MA_3'], 'MA_3', metrics)

    # exponential smoothing
    stock_data["ES"] = np.nan
    for i in range(10, len(stock_data)):
        last_10_data = stock_data['close'].iloc[i-10:i]
        ES = exponential_smoothing(last_10_data, trend='add')
        ES_predictions = exponential_smoothing_prediction(ES, steps=1)
        stock_data.iloc[i,stock_data.columns.get_loc("ES")] = ES_predictions

    metrics = model_evaluation(stock_data['close'], stock_data['ES'], 'ES', metrics)

    # auto-arima
    stock_data["ARIMA1"] = np.nan
    ARIMA = autoarima(training_data)
    ARIMA_predictions = autoarima_prediction(ARIMA, steps = len(testing_data))
    for i, pred in enumerate(ARIMA_predictions):
        stock_data.iloc[len(training_data)+i, stock_data.columns.get_loc("ARIMA1")] = pred

    metrics = model_evaluation(stock_data['close'], stock_data['ARIMA1'], 'ARIMA1', metrics)

    print(metrics)
