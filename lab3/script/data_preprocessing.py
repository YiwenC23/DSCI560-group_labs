import sys
import datetime
import sqlalchemy
import pymysql
import pandas as pd
import yfinance as yf

# def create_connection():
#     connection = mysql.connector.connect(
#         host='localhost',
#         user='root',
#         password='!Mhl020725'
#     )
#     return connection

# def fetch_stock_data(connection, ticker):
#     query = f"SELECT * FROM stocks WHERE ticker = '{ticker}' ORDER BY date;"
    
#     df = pd.read_sql(query, con=connection) 

#     return df

# connection = create_connection()

# if connection:
#     ticker = 'AAPL'
#     df = fetch_stock_data(connection, ticker)
    
#     print(df.head())

#     connection.close() 

def query():
	data = pd.read_sql("SELECT * FROM stock_data", my_db)
	print(data)

def missing_data(df):
    for column in df.columns:
            if df[column].isnull().sum() > 0:
                df[column] = df[column].interpolate(method='linear', inplace=True, limit_direction='both')
    
    return df

def daily_return(df):
    df['Daily Return'] = df.groupby('Ticker')['Close'].pct_change() * 100  
    return df

def volatility(df):
    df['Volatility'] = df.groupby('Ticker')['Daily Return'].rolling(window=20).std() * (252 ** 0.5)
    return df

def moving_average(df, window):
    df['Moving Average'] = df.groupby('Ticker')['Close'].rolling(window=window).mean().reset_index(level=0, drop=True)
    return df

def VWAP(df):
    df['VWAP'] = df.groupby('Ticker').apply(lambda x: (x['Close'] * x['Volume']).cumsum() / x['Volume'].cumsum()).reset_index(level=0, drop=True)
    return df


if __name__ == "__main__":
    db_user = input("Please enter the username for the database: ")
    db_password = input("Please enter the password for the database: ")
    db_name = input("Please enter the database name: ")
    my_db = sqlalchemy.create_engine(f"mysql+pymysql://{db_user}:{db_password}@localhost/{db_name}")

    connection = engine.connect()

    query()

    stock_data = pd.read_sql_table('stock_data', connection)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    stock_data = missing_data(stock_data)

    stock_data = daily_return(stock_data)

    stock_data = volatility(stock_data)

    stock_data = moving_average(stock_data, 20)

    stock_data = VWAP(stock_data)