import sys
import requests
import datetime
import yfinance as yf
import sqlalchemy as sql
import concurrent.futures

from database import SessionLocal, StockData, TickerIndex, UserAsset
from yfinance_retrieve import insert_db, removingstock, stock_retrieve, workflow


def portfolio_ticker_list():
    try:
        session = SessionLocal()
        tickers = session.query(TickerIndex.ticker).distinct().all()
        if not tickers:
            print("No tickers found in the database.")
            return []
        else:
            ticker_list = [tkr[0] for tkr in tickers]
            return ticker_list
    
    except Exception as e:
        print(f"Error retrieving portfolio tickers: {e}")


def fetch_single_ticker(ticker_symbol):
    try:
        session_obj = getattr(fetch_single_ticker, "session", None)
        if session_obj is None:
            session_obj = requests.Session()
            setattr(fetch_single_ticker, "session", session_obj)
        
        yf_ticker = yf.Ticker(ticker_symbol, session=session_obj)
        info = yf_ticker.info
        
        current_info = {
            "ticker": ticker_symbol,
            "price": info.get("currentPrice"),
            "bid": info.get("bid"),
            "ask": info.get("ask"),
            "volume": info.get("volume"),
        }
        
        return current_info
    
    except Exception as e:
        print(f"Failed to fetch {ticker_symbol}: {e}")
        return None


#* Define the function for retrieving real-time price data of stocks that are in the database.
def parallel_fetch_tickers():
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_stock = {
            executor.submit(fetch_single_ticker, tkr): tkr
            for tkr in ticker_list
        }
        
        for future in concurrent.futures.as_completed(future_to_stock):
            ticker = future_to_stock[future]
            try:
                data = future.result()
                if data:
                    results[ticker] = data
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
    
    return [results.get(stock) for stock in ticker_list if stock in results]


#* Define the class for stock trading transctions.
class Transaction:
    def __init__(self, session=None):
        if session is None:
            self.session = SessionLocal()
        else:
            self.session = session
    
    def buy(self, symbol, quantity):
        try:
            ticker_info = fetch_single_ticker(symbol)
            
            if ticker_info is None or ticker_info["price"] is None:
                print(f"Failed to fetch {symbol} from Yahoo Finance.")
                return None
            
            current_price = ticker_info["price"]
            
            new_transaction = UserAsset(
                ticker=symbol,
                datetime=datetime.datetime.now(),
                quantity=quantity,
                price=current_price
            )
            
            self.session.add(new_transaction)
            self.session.commit()
            print(f"Transaction successful ({new_transaction.datetime}): purchased {quantity} shares of {symbol} at {current_price} per share")
            
            return new_transaction
        
        except Exception as e:
            print(f"Transaction failed: {e}")
            self.session.rollback()


    def sell(self, symbol, quantity):
        try:
            if not symbol or not quantity or quantity <= 0:
                print("Invalid input: symbol and quantity must be provided and quantity must be greater than 0.")
                return None
            
            #? Check if the user has enough shares to sell
            shares_held = self.session.query(
                sql.func.sum(UserAsset.quantity)
            ).filter(
                UserAsset.ticker == symbol
            ).scalar()
            
            if shares_held is None or shares_held < quantity:
                print(f"Insufficient shares of {symbol} to sell.")
                return None
            
            #? Fetch the current price of the ticker
            ticker_info = fetch_single_ticker(symbol)
            
            if ticker_info is None or ticker_info["price"] is None:
                print(f"Filed to fetch the current price of {symbol}.")
                return None
            
            current_price = ticker_info["price"]
            
            #? Record the selling transactions as a negative value to distinguish them from the buying transactions
            sell_transaction = UserAsset(
                ticker=symbol,
                datetime=datetime.datetime.now(),
                quantity=quantity,
                price=current_price
            )
            
            self.session.add(sell_transaction)
            self.session.commit()
            print(f"Transaction successful ({sell_transaction.datetime}): sold {quantity} shares of {symbol} at {current_price} per share")
            
            return sell_transaction
        
        except Exception as e:
            print(f"Transaction failed: {e}")
            self.session.rollback()


#* Define the function for calculating real-time total portfolio value and profit/loss.
def calculate_portfolio_value():
    try:
        session = getattr(calculate_portfolio_value, "session", None)
        if session is None:
            session = SessionLocal()
            setattr(calculate_portfolio_value, "session", session)
    
        if not tickers_info:
            print("No valid data retrieved from Yahoo Finance.")
            return None
        
        total_cost = 0
        total_current_value = 0
        portfolio_details = []
        
        #? Query user's total shares held and total cost for each ticker in the portfolio
        for info in tickers_info:
            ticker = info["ticker"]
            current_price = info["price"]
            
            #? Query user's total shares held and total cost for each ticker in the portfolio
            ticker_cost = UserAsset.ticker_total_cost(session, ticker)
            total_quantity = session.query(
                sql.func.sum(UserAsset.quantity)
            ).filter(UserAsset.ticker == ticker).scalar()
            
            #? Get current value of the ticker based on the current price
            current_value = current_price * total_quantity
            profit_loss = current_value - ticker_cost
            
            portfolio_details.append({
                "ticker": ticker,
                "current_price": current_price,
                "total_quantity": total_quantity,
                "current_value": current_value,
                "ticker_cost": ticker_cost,
                "profit_loss": profit_loss
            })
            
            #? Update totals
            total_cost += ticker_cost
            total_current_value += current_value
            
        #? Calculate total portfolio value and profit/loss
        total_profit_loss = total_current_value - total_cost
        
        
        portfolio_summary = {
            "portfolio_details": portfolio_details,
            "total_cost": total_cost,
            "total_current_value": total_current_value,
            "total_profit_loss": total_profit_loss
        }
        
        return portfolio_summary
    
    except Exception as e:
        print(f"Failed to calculate portfolio value: {e}")


# TODO: Define a function to update the historical data of the stock in the database.

# TODO: Define the function for Terminal-User Interface.


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
            start_date = input("Enter start date (YYYY-MM-DD): ") or None
            end_date = input("Enter end date (YYYY-MM-DD): ") or None
            workflow([symbol], start_date, end_date)
        elif userschoice == "2":
            symbol = input("Enter the stock symbol to remove: ").upper()
            removingstock(symbol)
        elif userschoice == "3":
#            displayingpf()
            continue
        elif userschoice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    ticker_list = portfolio_ticker_list()
    tickers_info = parallel_fetch_tickers()
    main()
