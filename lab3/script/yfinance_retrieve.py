import sys
import datetime
import sqlalchemy
import pymysql
import pandas as pd
import yfinance as yf

my_db = sqlalchemy.create_engine("mysql+pymysql://root:yiwen960131@localhost/dsci560")

def query():
	data = pd.read_sql("SELECT * FROM Course", my_db)
	print(data)

if __name__ == "__main__":
	query()