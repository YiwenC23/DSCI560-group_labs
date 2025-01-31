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
        
        with engine.connect() as conn:
            conn.execute(" \
                CREATE TABLE IF NOT EXISTS stock_data( \
                    date DATE NOT NULL, \
                    ticker VARCHAR(10) NOT NULL \
                    close DECIMAL(12, 4)\
                    high DECIMAL(12, 4)\
                    low DECIMAL(12, 4) \
                    open DECIMAL(12, 4) \
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
        
        print("Database connected successfully")
        return engine
    
    except Exception as e:
        print(e)
        sys.exit()


def insert_db(engine, data):
    try:
        # Reformat dataframe into desired format
        data = pd.DataFrame(data, columns=data.columns)
        data.columns = [f"{col[0]}_{col[1]}" for col in data.columns]
        
        data.columns = data.columns.str.split('_', n=1, expand=True)
        data.columns.names = ["Metric", "Ticker"]
        
        long_df = data.stack("Ticker", future_stack=True).reset_index()
        
        long_df.columns = ["Date", "Ticker", "Close", "High", "Low", "Open", "Volume"]
        result = long_df[["Date", "Ticker", "Close", "High", "Low", "Open", "Volume"]]
        result = result.set_index("Date")
        
        # Write to database
        result.to_sql(
            name="stock_data",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=1000
        )
        
    except Exception as e:
        print(f"Filed to insert data: {e}")
        sys.exit()


if __name__ == "__main__":
    ticker = ["AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA"]
    
    db_engine = connect_db()
    
    raw_data = yf.download(
        ticker,
        start="2000-01-01",
        end="2025-01-30",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False
    )
    
    insert_db(db_engine, raw_data)