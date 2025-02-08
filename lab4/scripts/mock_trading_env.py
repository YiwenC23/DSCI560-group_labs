import sys
import pandas as pd
import yfinance as yf
import sqlalchemy as sql
from sqlalchemy.orm import declarative_base, sessionmaker

from yfinance_retrieve import StockData

Base = declarative_base()

#* Define the function to connect to the database
def connect_db():
    global engine, SessionLocal
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
    
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)


def get_DBdata():
    with SessionLocal() as session:
        stock_data = session.query(StockData).all()
        df = pd.DataFrame([s.__dict__ for s in stock_data])
        df= df.drop(columns=["_sa_instance_state"], errors="ignore")
        return df


if __name__ == "__main__":
    connect_db()
    print(get_DBdata())


# TODO: Define the function for retrieving real-time price data of stocks that are in the database.

# TODO: Define the function for trading transactions.

# TODO: Define the function for calculating the portfolio's total value and total gain/loss. (With the portfolio's composition)

# TODO: Define the function to display the portfolio's total value and total gain/loss in real-time.

# TODO: Define the function for interactive trading interface.