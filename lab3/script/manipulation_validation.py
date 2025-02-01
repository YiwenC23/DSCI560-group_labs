from sqlalchemy import text
from datetime import datetime
import yfinance as yf
from sqlalchemy import create_engine, Column, String, Float, Integer, Date, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

def get_db_url():
    print("\nDatabase Connection Details")
    print("-" * 50)
    print("NOTE: Enter credentials without any quotes")
    db_username = input("Username: ").strip()
    db_password = input("Password: ").strip().replace('"', '').replace("'", '')  # Remove any quotes
    db_name = input("Database name: ").strip()
    return f"mysql+pymysql://{db_username}:{db_password}@localhost/{db_name}"

try:
    db_url = get_db_url()
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
except Exception as e:
    print(f"\nError connecting to database: {e}")
    print("Please make sure your credentials are correct and try again.")
    exit(1)

Base = declarative_base()

class StockData(Base):
    __tablename__ = 'stock_data'
    
    date = Column(Date, primary_key=True)
    ticker = Column(String(10), primary_key=True)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    open = Column(Float)
    volume = Column(Integer)

def validation(symbol):
    try:
        stock = yf.Ticker(symbol)
        if stock.history(period="1d").empty:
            return False
        return True
    except Exception as e:
        return False

def addingstock(symbol, start_date, end_date):
    if not validation(symbol):
        print(f"Invalid stock symbol: {symbol}")
        return
    try:
        # Download stock data
        stockdata = yf.download(symbol, start=start_date, end=end_date)
        if stockdata.empty:
            print(f"No data available for {symbol} in the specified date range")
            return
            
        # Process each row of data
        for index, row in stockdata.iterrows():
            # Extract numeric values directly
            stock_entry = StockData(
                date=index.date(),
                ticker=symbol,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume'])
            )
            try:
                session.merge(stock_entry)
                session.commit()
            except Exception as e:
                session.rollback()
                print(f"Error adding entry for date {index.date()}: {e}")
                continue
                
        print(f"Stock data for {symbol} has been added successfully")
    except Exception as e:
        print(f"Error adding stock data: {e}")
        session.rollback()

def removingstock(symbol):
    try:
        result = session.execute(
            text("DELETE FROM stock_data WHERE ticker = :symbol"),
            {"symbol": symbol}
        )
        session.commit()
        if result.rowcount > 0:
            print(f"Removed {symbol} from database")
        else:
            print(f"{symbol} not found in database")
    except Exception as e:
        print(f"Error removing stock: {e}")
        session.rollback()

def displayingpf():
    try:
        stocks = session.execute(text("SELECT * FROM stock_data ORDER BY date, ticker"))
        rows = list(stocks)
        if not rows:
            print("\nNo stock data found in the database.")
            return
            
        print("\nCurrent Stock Data:")
        print("-" * 80)
        for stock in rows:
            print(f"Date: {stock.date}, Symbol: {stock.ticker}, Open: {stock.open:.2f}, "
                  f"High: {stock.high:.2f}, Low: {stock.low:.2f}, Close: {stock.close:.2f}, "
                  f"Volume: {stock.volume}")
        print("-" * 80)
    except Exception as e:
        print(f"Error displaying stock data: {e}")

def main():
    while True:
        print("\nYour Options Are:")
        print("1. Add stock to portfolio")
        print("2. Remove stock from portfolio")
        print("3. Display all portfolios")
        print("4. Exit")
        userschoice = input("Select one of the above: ")

        if userschoice == "1":
            symbol = input("Enter stock symbol: ").upper()
            start_date = input("Enter start date (YYYY-MM-DD): ")
            end_date = input("Enter end date (YYYY-MM-DD): ")
            addingstock(symbol, start_date, end_date)
        elif userschoice == "2":
            symbol = input("Enter the stock symbol to remove: ").upper()
            removingstock(symbol)
        elif userschoice == "3":
            displayingpf()
        elif userschoice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()

