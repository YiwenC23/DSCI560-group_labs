from sqlalchemy import text
from datetime import datetime
import yfinance as yf
from sqlalchemy import create_engine, Column, String, Float, Integer, Date, ForeignKey, DateTime
# 1. add a stock to the portfolio
# 2. remove a stock from the portfolio
# 3. display all portfolios

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
def get_db_url():
    db_username = input("Please enter the username for the database: ")
    db_password = input("Please enter the password for the database: ")
    db_name = input("Please enter the database name: ")
    return f"mysql+pymysql://{db_username}:{db_password}@localhost/{db_name}"
db_url = get_db_url()
engine = create_engine(db_url)
Session = sessionmaker(bind=engine)
session = Session()

base = declarative_base()
class portfolio(base):
    __tablename__ = 'portfolio'
    id = Column(Integer, primary_key=True)
    created_date = Column(DateTime, default=datetime.now)
    symbol =Column(String(10))
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    #adj_close = Column(Float)
    volume = Column(Integer)
    date = Column(Float)

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
        print(f"invalid stock")
        return
    try:
        stockdata = yf.download(symbol, start=start_date, end=end_date)
        for index,row in stockdata.iterrows():
            session.add(portfolio(
                symbol=symbol,
                open_price=row['Open'],
                high_price=row['High'],
                close_price=row['Close'],
                #adj_price=row['Adj Close'],
                low_price=row['Low'],
                volume=row['Volume'],
                created_date=datetime.now()
            ))
        session.commit()
        print(f"portfolio is added now")
    except Exception as e:
        print (e)

def removingstock(symbol):
    try:
        result = session.execute(text(f"DELETE FROM portfolio WHERE symbol = '{symbol}'"))
        if result.rowcount>0:
            session.commit()
            print(f"removed {symbol} from portfolio")
        else:
            print(f"{symbol} not found in portfolio")
    except Exception as e:
        print ("error removing stock")

def displayingpf():
    portfolios = session.execute(text("SELECT * FROM portfolio"))
    for stock in portfolios:
        print(f"Symbol:{stock.symbol}, Date: {stock.date},Created Date: {stock.created_date}, Open: {stock.open_price}, High: {stock.high_price}, Close: {stock.close_price},Low: {stock.low_price}, Volume: {stock.volume}")

while True:
    print("Your Options Are:")
    print("1. Add stock to portfolio")
    print("2. Remove stock from portfolio")
    print("3. Display all portfolios")
    print("4. exit")
    userschoice = input("select one of the above")

    if userschoice=="1":
        symbol = input("enter stock symbol")
        start_date = input("enter start date in yyyy-mm-dd format")
        end_date = input("enter end date in yyyy-mm-dd format")
        addingstock(symbol, start_date, end_date)
    elif userschoice =="2":
        symbol = input("type the stock symbol to remove")
        removingstock(symbol)
    elif userschoice=="3":
        displayingpf()
    elif userschoice=="4":
        print("goodbye")
        break




