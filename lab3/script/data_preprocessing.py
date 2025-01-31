import sys
import datetime
import sqlalchemy
import pymysql
import pandas as pd
import yfinance as yf



def query():
	data = pd.read_sql("SELECT * FROM stock_data", my_db)
	print(data)

def missing_data(df):
    for column in df.columns:
            if df[column].isnull().sum() > 0:
                df[column].interpolate(method='linear', inplace=True, limit_direction='both')
    
    return df

def format_data(df):

    return df

if __name__ == "__main__":
    db_user = input("Please enter the username for the database: ")
    db_password = input("Please enter the password for the database: ")
    db_name = input("Please enter the database name: ")
    my_db = sqlalchemy.create_engine(f"mysql+pymysql://{db_user}:{db_password}@localhost/{db_name}")

    connection = engine.connect()

    stock_data = pd.read_sql_table('stock_data', connection)
    stock_data = missing_data(stock_data)



    query()