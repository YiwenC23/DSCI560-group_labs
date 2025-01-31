import sys
import sqlalchemy
import pymysql
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def connect_db():
	try:
		db_username = input("Please enter the username for the database: ")
		db_password = input("Please enter the password for the database: ")
		db_name = input("Please enter the database name: ")
		my_db = sqlalchemy.create_engine(f"mysql+pymysql://{db_username}:{db_password}@localhost/{db_name}")
		return my_db

	except Exception as e:
		print(e)
		sys.exit()


def stock_retrieve():
	try:
		ticker = ["AAPL", "MSFT", "AMZN", "NVDA"]
		data = yf.download(ticker, start="2000-01-01", end="2025-01-30", interval="1d")
		#####
		data = data.drop(1).reset_index(drop=True)
		data.rename(columns={'Price':"Date"}, inplace=True)
		#####
		return data
	except Exception as e:
		print(e)
		return None


def data_reformat(data):

	data = pd.DataFrame(data)


if __name__ == "__main__":
	
	connect_db()
	stock_retrieve()