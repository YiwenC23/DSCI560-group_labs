import sys
import requests
import pandas as pd
import yfinance as yf
import sqlalchemy as sql
from datetime import datetime, timedelta
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base() # Declare the base class for the database


#* Define the StockData Class
class StockData(Base):
    #? Define the table name and columns
    __tablename__ = "stock_data"
    date = sql.Column(sql.Date, primary_key=True)
    ticker = sql.Column(sql.String(10), primary_key=True)
    open = sql.Column(sql.Float)
    high = sql.Column(sql.Float)
    low = sql.Column(sql.Float)
    close = sql.Column(sql.Float)
    volume = sql.Column(sql.Integer)
    
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


# TODO: Define the user asset table class.


#* Define the function to connect to the database
def connect_db():
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
        return engine, SessionLocal
    
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)


#* Define the function for timestamp processing algorithm
def compare_timestamp(session, tickers_ranges):
    
    results = {}
    
    #? Ensure tickers_ranges is a list
    if not isinstance(tickers_ranges, list):
        tickers_ranges = [tickers_ranges]
    
    #? Retrieve existing ticker index records for all tickers at once.
    tickers = [tr[0] for tr in tickers_ranges]
    indexes = session.query(TickerIndex).filter(TickerIndex.ticker.in_(tickers)).all()
    index_dict = {idx.ticker: idx for idx in indexes}
    
    for ticker, new_start, new_end in tickers_ranges:
        idx = index_dict.get(ticker)
        
        #! Case 1: If no stored data exists, the complete timeline is the new data range
        if not idx:
            results[ticker] = {
                "complete_range": (new_start, new_end),
                "missing_segments": [(new_start, new_end)]
            }
        else:
            #? Get the stored data interval and the new data interval
            results[ticker] = {}
            idx_interval = (idx.start_date, idx.end_date)
            new_interval = (new_start, new_end)
            
            #? The overall data range is the minimum start date and the maximum end date
            overall_start = min(new_interval[0], idx_interval[0])
            overall_end = max(new_interval[1], idx_interval[1])
            results[ticker]["complete_range"] = (overall_start, overall_end)
            
            #? Initialize the missing segments
            missing_segments = []
            
            #? Handle the gap relationships
            #! Case 2: New data interval left-apart from the stored data interval
            if new_interval[1] < idx_interval[0]:
                gap_start = new_interval[0]
                gap_end = idx_interval[0] - timedelta(days=1)
                if gap_start <= gap_end:
                    missing_segments.append((gap_start, gap_end))
            
            #! Case 3: New data interval right-apart from the stored data interval
            elif new_interval[0] > idx_interval[1]:
                gap_start = idx_interval[1] + timedelta(days=1)
                gap_end = new_interval[1]
                if gap_start <= gap_end:
                    missing_segments.append((gap_start, gap_end))
            
            #? Handle the containment relationships
            else:
                #! Case 4: New data interval left-overlaps with the stored data interval
                if new_interval[0] < idx_interval[0]:
                    missing_segments.append((new_interval[0], idx_interval[0] - timedelta(days=1)))
                #! Case 5: New data interval right-overlaps with the stored data interval
                if new_interval[1] > idx_interval[1]:
                    missing_segments.append((idx_interval[1] + timedelta(days=1), new_interval[1]))
            
            results[ticker]["missing_segments"] = missing_segments
    
    return results


#* Defined function for data insertion
def insert_db(data):
    try:
        with SessionLocal() as session:
            records = data.reset_index()[["date", "ticker", "open", "high", "low", "close", "volume"]]
            
            session.bulk_insert_mappings(StockData, records.to_dict(orient="records"))
            session.commit()
            print("\nSuccessfully inserted stock data into the database!")
    
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
        
        return stock_data.set_index("date")
    
    except Exception as e:
        print(e)
        sys.exit(1)


#* Define Workflow Function
def workflow(ticker_list, start_date, end_date):
    try:
        start_dateObj = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dateObj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        with SessionLocal() as session:
            ticker_ranges = [(tk, start_dateObj, end_dateObj) for tk in ticker_list]
            needed_ranges = compare_timestamp(session, ticker_ranges)
            
        all_data = []
        for tk in ticker_list:
            needed_tr = needed_ranges.get(tk)
            print(f"\nRanges for {tk}: {needed_tr}\n")
            
            overall_range = needed_tr["complete_range"]
            missing_segments = needed_tr["missing_segments"]
            
            #! If there are no missing segments (and an index exists), the complete range is already stored.
            if needed_tr and not missing_segments:
                print(f"{tk}: Data from {overall_range[0]} to {overall_range[1]} already exists in the database.")
                continue
            
            print(f"\n{tk}: Fetching data for complete range {overall_range[0]} to {overall_range[1]}...")
            #? Fetch the missing segments data and insert into the database
            for ms in missing_segments:
                all_data.append(stock_retrieve([tk], ms[0], ms[1]))
            
            print("Successfully retrieved stock data!")
            #? Update/insert the ticker index with the new complete range
            new_index = TickerIndex(
                ticker=tk,
                start_date=overall_range[0],
                end_date=overall_range[1]
            )
            session.merge(new_index)
        
        #? If any new data was fetched, combine and insert it into the database
        if all_data:
            combined_data = pd.concat(all_data)
            insert_db(combined_data)
        
        session.commit()
    
    except Exception as e:
        print(f"Workflow failed: {e}")
        sys.exit(1)


# TODO: Define the function for retrieving real-time price data of stocks that are in the database.

# TODO: Define the function for trading transactions.

# TODO: Define the function for calculating the portfolio's total value and total gain/loss. (With the portfolio's composition)

# TODO: Define the function to display the portfolio's total value and total gain/loss in real-time.

# TODO: Define the function for interactive trading interface.


if __name__ == "__main__":
    engine, SessionLocal = connect_db()
    while True:
        print("\nWelcome to the Mock Trading Environment!")
        print("\nYour Options Are:")
        print("1. Add stock to portfolio")
        print("2. Remove stock from portfolio")
        print("3. Display all portfolios")
        print("4. Exit")
        userschoice = input("Select one of the above: ")
        
        if userschoice == "1":
            symbol = [input("Enter stock symbol: ").upper()]
            start_date = input("Enter start date (YYYY-MM-DD): ")
            end_date = input("Enter end date (YYYY-MM-DD): ")
            workflow(symbol, start_date, end_date)
        elif userschoice == "2":
            symbol = input("Enter the stock symbol to remove: ").upper()
            continue
        elif userschoice == "3":
#            displayingpf()
            continue
        elif userschoice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")
    sys.exit(0)