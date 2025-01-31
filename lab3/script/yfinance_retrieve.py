import sys
import datetime
import sqlalchemy
import pymysql
import pandas as pd
import yfinance as yf


def query():
	data = pd.read_sql("SELECT * FROM Course", my_db)
	print(data)


if __name__ == "__main__":
	db_password = input("Please enter the password for the database: ")
	db_name = input("Please enter the database name: ")
	my_db = sqlalchemy.create_engine(f"mysql+pymysql://root:{db_password}@localhost/{db_name}")

	query()