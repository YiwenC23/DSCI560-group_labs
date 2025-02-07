from sqlalchemy import create_engine, Column, String, Float, Integer, Date
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import yfinance as yf

# Define Base for declarative models
Base = declarative_base()
# Live Stock Price Model
class StockPrice(Base):
    __tablename__ = 'stock_prices'    
    date = Column(Date, primary_key=True)
    ticker = Column(String(10), primary_key=True)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    open = Column(Float)
    volume = Column(Integer)

#PLACEHOLDER for Mock Trading Environment Model
class MockTradingEnvironment(Base):
    __tablename__ = 'mock_trading'
    ticker = Column(String(10), primary_key=True)
    shares_held = Column(Integer)
    cash_balance = Column(Float)
    avg_purchase_price = Column(Float)

def connect_to_databases():
#Establish connections to both databases
    try:
        # Replace with your actual database connection strings
        live_db_url = 'PLACEHOLDERLINK://username:password@localhost/live_stock_db'
        mte_db_url = 'PLACEHOLDERLINK://username:password@localhost/mock_trading_db'
        # Create engines
        live_engine = create_engine(live_db_url)
        mte_engine = create_engine(mte_db_url)
        # Create sessions
        LiveSession = sessionmaker(bind=live_engine)
        MTESession = sessionmaker(bind=mte_engine)
        live_session = LiveSession()
        mte_session = MTESession()
        return live_session, mte_session    
    except SQLAlchemyError as e:
        print(f"Database connection error: {e}")
        raise

def calculate_daily_profit(live_session, mte_session, ticker):
# Calculate daily profit for a specific stock
    try:
        # Fetch most recent stock price data
        latest_price = live_session.query(StockPrice).filter_by(ticker=ticker).order_by(StockPrice.date.desc()).first()        
        # Fetch trading environment data
        trading_data = mte_session.query(MockTradingEnvironment).filter_by(ticker=ticker).first()        
        if not latest_price or not trading_data:
            print(f"Insufficient data for {ticker}")
            return None        
        # Calculate daily profit
        opening_price = latest_price.open
        closing_price = latest_price.close
        shares_held = trading_data.shares_held        
        daily_profit = (closing_price - opening_price) * shares_held        
        # Profit status
        if daily_profit > 0:
            print(f"{ticker}: You are in profit. Daily Profit: ${daily_profit:.2f}")
        elif daily_profit < 0:
            print(f"{ticker}: You are in loss. Daily Loss: ${abs(daily_profit):.2f}")
        else:
            print(f"{ticker}: No profit or loss")        
        return daily_profit    
    except Exception as e:
        print(f"Error calculating daily profit: {e}")
        return None

def main():
    try:
        # Connect to databases PLACEHOLDER
        live_session, mte_session = connect_to_databases()        
        # Example tickers to track
        tickers = ['AAPL', 'GOOGL', 'MSFT']        
        # Calculate profit for each ticker
        for ticker in tickers:
            calculate_daily_profit(live_session, mte_session, ticker)        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close sessions
        if 'live_session' in locals():
            live_session.close()
        if 'mte_session' in locals():
            mte_session.close()

if __name__ == "__main__":
    main()
