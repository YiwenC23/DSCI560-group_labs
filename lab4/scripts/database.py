import sys
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

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


#* Define User Asset Table Class
class UserAsset(Base):
    __tablename__ = "user_asset"
    ticker = sql.Column(sql.String(10), primary_key=True) # Ticker symbol
    datetime = sql.Column(sql.DateTime, primary_key=True) # Purchase date
    quantity = sql.Column(sql.Integer, nullable=False) # Quantity purchased at a time
    price = sql.Column(sql.Float, nullable=False) # Purchase price
    
    #? Table Constraints Configuration
    __table_args__ = (
        sql.PrimaryKeyConstraint("ticker", "datetime", name="pk_userasset"),  # Explicit composite PK
        sql.ForeignKeyConstraint(["ticker"], ["ticker_index.ticker"], name="fk_userasset_ticker"),
        sql.Index("idx_userasset_ticker", "ticker"),  # Faster lookups by ticker
        {"extend_existing": True}
    )
    
    @property
    def transaction_cost(self): # Total purchase price when a transaction made on a ticker
        return self.quantity * self.price
    
    @classmethod
    def ticker_total_cost(cls, session, ticker):
        total = session.query(
            sql.func.sum(cls.quantity * cls.price)
        ).filter(
            cls.ticker == ticker
        ).scalar()
        return total
    
    @classmethod
    def total_portfolio_value(cls, session):
        tickers = session.query(cls.ticker).distinct().all()
        
        total_value = 0
        for (ticker, ) in tickers:
            total_value += cls.ticker_total_cost(session, ticker)
        return total_value


#* Define the function to connect to the database
def connect_db():
    
    #? Get the database credentials from the user
    db_username = input("Please enter the username for the database: ")
    db_password = input("Please enter the password for the database: ")
    db_name = input("Please enter the database name: ")
    
    conn_url = f"mysql+pymysql://{db_username}:{db_password}@localhost/{db_name}"
    
    try:
        
        print(f"Establishing connection to {db_name} database as {db_username}...")
        #? Create database engine with connection pool configuration
        engine = sql.create_engine(
            conn_url,
            pool_size=20,     # Number of maintained idle connections in the pool
            pool_recycle=3600,    # Recycle connections hourly to prevent connection timeout
            max_overflow=10,    # Allow up to 10 additional connections to the pool
            pool_pre_ping=True,     # Validate connection viability before use
            echo=False    # Disable engine logging
            )
        

        
        print("Successfully connected to the database!")
    
        return engine
    except Exception as e:
        print(f"Connection failed: {e}")
        raise

#? Create the engine
engine = connect_db()

#? Create the tables
Base.metadata.create_all(bind=engine)

#? Create the Session Library
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,    # Require explicit commit() for transaction control
    autoflush=False,    # Delay SQL emission until flush()/commit() called, enables batch operations
    expire_on_commit=False,    # Keep object attributes accessible after commit
    class_=sql.orm.Session    # Use the SQLAlchemy Session class
)
