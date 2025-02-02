import sys
import datetime
import sqlalchemy
import pymysql
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
from yfinance_retrieve import connect_db

def missing_data(df):
    for column in df.columns:
            if df[column].isnull().sum() > 0:
                df[column] = df[column].interpolate(method='linear', inplace=True, limit_direction='both')
    
    return df

def daily_return(df):
    df['Daily Return'] = df.groupby('ticker')['close'].pct_change() * 100  
    return df

def volatility(df):
    df['Volatility'] = df.groupby('ticker')['Daily Return'].rolling(window=20).std().reset_index(level=0, drop=True) * (252 ** 0.5)
    return df

def moving_average(df, window):
    df['Moving Average'] = df.groupby('ticker')['close'].rolling(window=window).mean().reset_index(level=0, drop=True)
    return df

def VWAP(df):
    df['VWAP'] = (df['close']*df['volume']).groupby(df['ticker']).cumsum() / df['volume'].groupby(df['ticker']).cumsum()
    return df


if __name__ == "__main__":
    engine = connect_db()

    stock_data = pd.read_sql_table('stock_data', engine)
    stock_data['date'] = pd.to_datetime(stock_data['date'])

    stock_data = missing_data(stock_data)

    stock_data = daily_return(stock_data)

    stock_data = volatility(stock_data)

    stock_data = moving_average(stock_data, 20)

    stock_data = VWAP(stock_data)

    print(stock_data)
