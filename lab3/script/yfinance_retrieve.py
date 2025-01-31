import sys
import sqlalchemy
import pandas as pd
import yfinance as yf


def connect_db():
    try:
        db_username = input("Please enter the username for the database: ")
        db_password = input("Please enter the password for the database: ")
        db_name = input("Please enter the database name: ")
        engine = sqlalchemy.create_engine(f"mysql+pymysql://{db_username}:{db_password}@localhost/{db_name}")
        '''
        with engine.connect() as conn:
            conn.execute(" \
                CREATE TABLE IF NOT EXISTS stock_data( \
                    date DATE NOT NULL, \
                    ticker VARCHAR(10) NOT NULL, \
                    close DECIMAL(12, 4), \
                    high DECIMAL(12, 4), \
                    low DECIMAL(12, 4), \
                    open DECIMAL(12, 4), \
                    volume BIGINT, \
                    PRIMARY KEY (date, ticker) \
                    ); \
            ")
            
            conn.execute(" \
                CREATE INDEX IF NOT EXISTS idx_ticker \
                ON stock_data (ticker); \
            ")
            
            conn.execute(" \
                CREATE INDEX IF NOT EXISTS idx_date \
                ON stock_data (date); \
            ")
        '''
        print("Database connected successfully")
        return engine
    
    except Exception as e:
        print(e)
        sys.exit()


def stock_retrieve():
    try:
        tickers = ["AAPL", "NVDA"]
        # Get AAPL ticker object
        hist_data = pd.DataFrame()
        for i in tickers:
            ticker = yf.Ticker(i)
        
            # Get all available information
#            info = aapl.info
            
            # Get historical price data
            tck_data = ticker.history(start="2025-01-20", end="2025-01-30", interval="1d")
            # Drop the last two columns
            tck_data = tck_data.iloc[:, :-2]
            # format the date to YYYY-MM-DD
            tck_data.index = tck_data.index.strftime('%Y-%m-%d')
            # Insert the ticker symbol as the first column
            tck_data.insert(0, 'Ticker', i)
            hist_data = pd.concat([hist_data, tck_data])
            # Sort the data by Date and Ticker
            hist_data = hist_data.sort_values(['Date', 'Ticker'])
        
        # Get additional data
#        dividends = aapl.dividends
#        splits = aapl.splits
#        actions = aapl.actions
        
        return hist_data
    except Exception as e:
        print(e)
        return None


def insert_db(engine, data):
    try:
        
        # Write to database
        data.to_sql(
            name="stock_data",
            con=engine,
            if_exists="append",
            chunksize=1000
        )
    
    except Exception as e:
        print(f"Filed to insert data: {e}")
        sys.exit()


if __name__ == "__main__":
#    ticker = ["AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA"]
    db_engine = connect_db()
    
    raw_data = stock_retrieve()
    
    insert_db(db_engine, raw_data)