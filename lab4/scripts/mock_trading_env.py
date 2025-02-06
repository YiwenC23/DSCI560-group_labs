import sys
import requests
import pandas as pd
import yfinance as yf
import sqlalchemy as sql
from datetime import timedelta
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base() # Declare the base class for the database


#* Define the StockData Class
class StockData(Base):
    #? Define the table name and columns
    __tablename__ = "stock_data"
    date = sql.Column(sql.Date, primary_key=True)
    ticker = sql.Column(sql.String, primary_key=True)
    open = sql.Column(sql.Numeric)
    high = sql.Column(sql.Numeric)
    low = sql.Column(sql.Numeric)
    close = sql.Column(sql.Numeric)
    volume = sql.Column(sql.BigInteger)
    
    #? Table Constraints Configuration
    __table_args__ = (
        sql.PrimaryKeyConstraint("date", "ticker", name="pk_stock_data"),
        {"extend_existing": True}    # Allow the table to be extended if it already exists
    )


#* Define Ticker Index Class
class TickerIndex(Base):
    __tablename__ = "ticker_index"
    ticker = sql.Column(sql.String(10), primary_key=True)
    start_date = sql.Column(sql.Date, index=True)
    end_date = sql.Column(sql.Date, index=True)
    
    #? Table Constraints Configuration
    __table_args__ = (
        sql.PrimaryKeyConstraint("ticker", name="pk_ticker_index"),
        sql.Index("idx_ticker_dates", "ticker", "start_date", "end_date", unique=True),
        {"extend_existing": True}
    )


#* Define the function to connect to the database
def connect_db():
    try:
        #? Get the database credentials from the user
        db_username = input("Please enter the username for the database: ")
        db_password = input("Please enter the password for the database: ")
        db_name = input("Please enter the database name: ")
        
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
        
        print("Database connected successfully")
        return engine, SessionLocal
    
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)


#TODO: Need to update the algorithm for last date of the stock data
#* Define the function for timestamp comparison algorithm
def compare_timestamp(session, tickers_ranges):
    if not isinstance(tickers_ranges, list):
        tickers_ranges = [tickers_ranges]
    
    tickers = [tr[0] for tr in tickers_ranges]
    indexes = session.query(TickerIndex).filter(TickerIndex.ticker.in_(tickers)).all()
    index_dict = {idx.ticker: idx for idx in indexes}
    
    results = {}
    for ticker, new_start, new_end in tickers_ranges:
        idx = index_dict.get(ticker)
        
        if not idx:
            results[ticker] = [(new_start, new_end)]
            continue
        
        if new_start <= idx.end_date and new_end >= idx.start_date:
            results[ticker] = []
            continue
        
        left = (new_start, idx.start_date - timedelta(days=1)) if new_start < idx.start_date else None
        right = (idx.end_date + timedelta(days=1), new_end) if new_end > idx.end_date else None
        
        valid = []
        if left and left[0] <= left[1]:
            valid.append(left)
        if right and right[0] <= right[1]:
            valid.append(right)
        
        results[ticker] = valid
    
    return results


#* Define the function to insert data into the database
def insert_db(data):
    try:
        with SessionLocal() as session:
            records = data.reset_index()[["date", "ticker", "open", "high", "low", "close", "volume"]]
            
            session.bulk_insert_mappings(StockData, records.to_dict(orient="records"))
            session.commit()
            print("Stock data has successfully been inserted into the database.")
    
    except Exception as e:
        print(f"Filed to insert data: {e}")
        sys.exit(1)


#* Define the function to retrieve historical stock data
def stock_retrieve(ticker_list, start_date, end_date):
    try:
        raw_data = yf.download(
            ticker_list,
            group_by="Ticker",
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False
        )
        
        stock_data = (
            raw_data.stack(level=0, future_stack=True)
            .rename_axis(["Date", "Ticker"])
            .reset_index(level=1)
            .reset_index(drop=False)
            .rename(columns={
                "Date": "date",
                "Ticker": "ticker",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            .rename_axis(columns=None)
        )
        
        print("Stock data retrieved successfully.")
        return stock_data.set_index("date")
    
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    engine, SessionLocal = connect_db()