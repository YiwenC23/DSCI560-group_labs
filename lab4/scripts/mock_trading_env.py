import os
import sys
import sqlalchemy
import pandas as pd
import yfinance as yf
from sqlalchemy import text
from datetime import datetime


def connect_db(db_username=None, db_password=None, db_name=None):
    try:
        # If credentials are not provided, prompt for them
        if not all([db_username, db_password, db_name]):
            db_username = input("Please enter the username for the database: ")
            db_password = input("Please enter the password for the database: ")
            db_name = input("Please enter the database name: ")
        engine = sqlalchemy.create_engine(f"mysql+pymysql://{db_username}:{db_password}@localhost/{db_name}")
        
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stock_data(
                    date DATE NOT NULL,
                    ticker VARCHAR(10) NOT NULL,
                    close DECIMAL(12, 4),
                    high DECIMAL(12, 4),
                    low DECIMAL(12, 4),
                    open DECIMAL(12, 4),
                    volume BIGINT,
                    PRIMARY KEY (date, ticker)
                    );
            """))
            
            # Check if the index already exists
            check_ticker_index = conn.execute(text("""
                            SELECT COUNT(1) indexExists
                                FROM INFORMATION_SCHEMA.STATISTICS
                                WHERE table_schema = 'dsci560'
                                    AND table_name = 'stock_data'
                                    AND index_name = 'idx_ticker';"""))
            if check_ticker_index.fetchone()[0] == 0:
                conn.execute(text("""CREATE INDEX idx_ticker ON stock_data (ticker);"""))
            
            # Check if the date index already exists
            check_date_index = conn.execute(text("""
                            SELECT COUNT(1) indexExists
                                FROM INFORMATION_SCHEMA.STATISTICS
                                WHERE table_schema = 'dsci560'
                                    AND table_name='stock_data'
                                    AND index_name='idx_date';"""))
            if check_date_index.fetchone()[0] == 0:
                conn.execute(text("""CREATE INDEX idx_date ON stock_data (date);"""))
        
        print("Database connected successfully")
        return engine
    
    except Exception as e:
        print(e)
        sys.exit()


def stock_retrieve(tickers):
    try:
        # Get AAPL ticker object
        hist_data = pd.DataFrame()
        for i in tickers:
            ticker = yf.Ticker(i)
            
            # Get historical price data
            tck_data = ticker.history(start="2025-01-15", end="2025-01-31", interval="1d")
            # Drop the last two columns
            tck_data = tck_data.iloc[:, :-2]
            # format the date to YYYY-MM-DD
            tck_data.index = tck_data.index.strftime("%Y-%m-%d")
            # Insert the ticker symbol as the first column
            tck_data.insert(0, "Ticker", i)
            hist_data = pd.concat([hist_data, tck_data])
            # Sort the data by Date and Ticker
            hist_data = hist_data.sort_values(["Date", "Ticker"])
        
        # Get all available information
#        info = aapl.info
        
        # Get additional data
#        dividends = aapl.dividends
        
#        splits = aapl.splits
#        actions = aapl.actions
        
        print("Stock data retrieved successfully.")
        return hist_data
    
    except Exception as e:
        print(e)
        sys.exit()


def insert_db(engine, data):
    try:
        with engine.connect() as conn:
            # Convert DataFrame to records for row-by-row processing
            records = data.reset_index().to_dict("records")
            
            for record in records:
                # Check if this specific record exists
                check_data = conn.execute(text("""
                    SELECT COUNT(1) dataExists 
                    FROM stock_data 
                    WHERE date = :date AND ticker = :ticker
                    """),
                    {"date": record["Date"], "ticker": record["Ticker"]}
                ).fetchone()
                
                if check_data[0] == 0:
                    # Insert only if record doesn't exist
                    conn.execute(text("""
                        INSERT INTO stock_data (date, ticker, open, high, low, close, volume)
                        VALUES (:date, :ticker, :open, :high, :low, :close, :volume)
                        """),
                        {
                            "date": record["Date"],
                            "ticker": record["Ticker"],
                            "open": record["Open"],
                            "high": record["High"],
                            "low": record["Low"],
                            "close": record["Close"],
                            "volume": record["Volume"]
                        }
                    )
            
            conn.commit()
            print("Stock data has successfully been inserted into the database.")
    
    
    except Exception as e:
        print(f"Filed to insert data: {e}")
        sys.exit()


if __name__ == "__main__":
    default_tickers = ["AAPL", "NVDA"]
    db_username = input("Please enter the username for the database: ")
    db_password = input("Please enter the password for the database: ")
    db_name = input("Please enter the database name: ")
    db_engine = connect_db(db_username, db_password, db_name)
    
    stock_data = stock_retrieve(default_tickers)
    
    insert_db(db_engine, stock_data)