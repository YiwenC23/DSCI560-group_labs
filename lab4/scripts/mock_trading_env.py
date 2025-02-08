import requests
import datetime
import yfinance as yf
import sqlalchemy as sql
import concurrent.futures

from database import SessionLocal, StockData, TickerIndex, UserAsset
from yfinance_retrieve import stock_retrieve, removingstock, insert_workflow


def portfolio_ticker_list():
    try:
        session = SessionLocal()
        tickers = session.query(TickerIndex.ticker).distinct().all()
        if not tickers:
            print("\n[INFO] No tickers found in the database.")
            return []
        else:
            ticker_list = [tkr[0] for tkr in tickers]
            return ticker_list
    
    except Exception as e:
        raise ValueError(f"\n[ERROR] Error retrieving portfolio tickers: {e}")
    finally:
        session.close()


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
        raise ValueError(f"\n[ERROR] Failed to fetch {ticker_symbol}: {e}")
    finally:
        session_obj.close()


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
                print(f"\n[ERROR] Failed to fetch {ticker}: {e}")
    
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
                print(f"\n[ERROR] Failed to fetch {symbol} from Yahoo Finance.")
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
            print(f"\n[INFO] Transaction successful ({new_transaction.datetime}): purchased {quantity} shares of {symbol} at {current_price} per share")
            
            return new_transaction
        
        except Exception as e:
            print(f"\n[ERROR] Transaction failed: {e}")
            self.session.rollback()
        finally:
            self.session.close()
    
    
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
            
            #? If the user does not have enough shares to sell, return None
            if shares_held is None or shares_held < quantity:
                print(f"\n[ERROR] Insufficient shares of {symbol} to sell.")
                return None
            
            #? Fetch the current price of the ticker
            ticker_info = fetch_single_ticker(symbol)
            
            #? If the current price of the ticker is not available, return None
            if ticker_info is None or ticker_info["price"] is None:
                print(f"\n[ERROR] Failed to fetch the current price of {symbol}.")
                return None
            
            current_price = ticker_info["price"]
            
            #? Record the selling transactions as a negative value to distinguish them from the buying transactions
            sell_transaction = UserAsset(
                ticker=symbol,
                datetime=datetime.datetime.now(),
                quantity=-quantity,
                price=current_price
            )
            
            self.session.add(sell_transaction)
            self.session.commit()
            print(f"\n[INFO] Transaction successful ({sell_transaction.datetime}): sold {quantity} shares of {symbol} at {current_price} per share")
            
            return sell_transaction
        
        except Exception as e:
            print(f"\n[ERROR] Transaction failed: {e}")
            self.session.rollback()
        finally:
            self.session.close()


#* Define the function for calculating real-time total portfolio value and profit/loss.
def calculate_portfolio_value():
    try:
        session = SessionLocal()
        
        if not tickers_info:
            print("\n[INFO] No valid data retrieved from Yahoo Finance.")
            return None
        
        total_cost = 0
        total_current_value = 0
        portfolio_details = []
        
        #? Query user's total shares held and total cost for each ticker in the portfolio
        for info in tickers_info:
            ticker = info["ticker"]
            current_price = info["price"]
            
            #! When the stock market is closed, calculate the price based on the previous day's closing price
            if current_price is None:
                fallback_record = session.query(StockData).filter(StockData.ticker == ticker).order_by(StockData.date.desc()).first()
                if fallback_record is not None:
                    current_price = fallback_record.close
                    print(f"Using previous day's closing price for {ticker} as current price is unavailable.")
                else:
                    current_price = 0
                    print(f"No price data available for {ticker}, using 0 as the current price.")
            
            #? Query user's total shares held and total cost for each ticker in the portfolio
            ticker_cost = UserAsset.ticker_total_cost(session, ticker) or 0
            total_quantity = session.query(
                sql.func.sum(UserAsset.quantity)
            ).filter(UserAsset.ticker == ticker).scalar()
            total_quantity = float(total_quantity) if total_quantity is not None else 0
            
            #? Get current value of the ticker based on the current price
            current_value = current_price * total_quantity
            profit_loss = current_value - ticker_cost
            
            portfolio_details.append({
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "total_quantity": total_quantity,
                "current_value": round(current_value, 2),
                "ticker_cost": round(ticker_cost, 2),
                "profit_loss": round(profit_loss, 2)
            })
            
            #? Update totals
            total_cost += ticker_cost
            total_current_value += current_value
            
        #? Calculate total portfolio value and profit/loss
        total_profit_loss = total_current_value - total_cost
        
        
        portfolio_summary = {
            "portfolio_details": portfolio_details,
            "total_cost": round(total_cost, 2),
            "total_current_value": round(total_current_value, 2),
            "total_profit_loss": round(total_profit_loss, 2)
        }
        
        return portfolio_summary
    
    except Exception as e:
        raise ValueError(f"\n[ERROR] Failed to calculate portfolio value: {e}")
    finally:
        session.close()


#* Define the function for updating the current daily stock data into the database.
def update_daily_data():
    try:
        session = SessionLocal()
        
        #? Get end date for each ticker in the TickerIndex table
        ticker_indices = session.query(TickerIndex).all()
        
        today = datetime.datetime.now().date()
        
        
        for idx in ticker_indices:
            ticker = idx.ticker
            last_date = idx.end_date
            
            if last_date < today:
                next_date = last_date + datetime.timedelta(days=1)
                fresh_data = stock_retrieve([ticker], str(next_date), str(today))
                
                if fresh_data.empty:
                    continue
                else:
                    insert_workflow([ticker], str(next_date), str(today))
                    print(f"\n[INFO] {ticker} data has been successfully updated to {today}.")
    
    except Exception as e:
        print(f"\n[ERROR] Failed to update daily data: {e}")
    finally:
        session.close()


#* Define the function to display the portfolio information.
def display_portfolio_info():
    try:
        summary = calculate_portfolio_value()
        print("\n       Portfolio Summary")
        
        if summary and summary.get("portfolio_details"):
            print("--------------------------------")
            
            print("[Overall]")
            print(f"Cost Basis: {summary['total_cost']} | Total Holdings: {summary['total_current_value']} | Total Gain/Loss: {summary['total_profit_loss']}")
            print("--------------------------------")
            
            print("[Individual Stocks]")
            for detail in summary["portfolio_details"]:
                print(f"Ticker: {detail['ticker']} | Latest Price: {detail['current_price']} | Shares: {detail['total_quantity']} | Current Value: {detail['current_value']} | Profit/Loss: {detail['profit_loss']}")
        
        else:
            print("\n[INFO] No portfolio data available.")
    
    except Exception as e:
        raise ValueError(f"\n[ERROR] Failed to display portfolio: {e}")


#* Define the function for the transaction interface.
def transaction_interface():
    ticker_list = portfolio_ticker_list()
    
    if not ticker_list:
        print("\n[INFO] You do not have any stocks in your portfolio yet.")
        return
    
    #? Calculate the portfolio information
    try:
        summary = calculate_portfolio_value()
    except Exception as e:
        print(f"\n[ERROR] Failed to calculate portfolio value: {e}")
        return
    
    #? Display the detailed portfolio information
    print("\n------------------------------")
    print("  Current Portfolio Status")
    print("------------------------------")
    if summary and summary.get("portfolio_details"):
        for detail in summary["portfolio_details"]:
            print(f"Ticker: {detail['ticker']}")
            print(f"  Latest Price: {detail['current_price']}")
            print(f"  Shares: {detail['total_quantity']}")
            print(f"  Cost: {detail['ticker_cost']}")
            print(f"  Market Value: {detail['current_value']}")
            print(f"  Gain/Loss: {detail['profit_loss']}")
            print("-----------------------------")
        print(f"COST BASIS: {summary['total_cost']}")
        print(f"TOTAL HOLDINGS: {summary['total_current_value']}")
        print(f"TOTAL GAIN/LOSS: {summary['total_profit_loss']}")
    else:
        print("\n[INFO] No portfolio data available.")

    #? Provide the transaction operation submenu
    print("\nTransaction Options:")
    print("1. Buy Stock")
    print("2. Sell Stock")
    print("3. Back to Main Menu")
    
    choice = input("Select an option: ").strip()
    txn = Transaction()  # Apply the Transaction class
    
    if choice == "1":
        symbol = input("Enter stock symbol to buy: ").upper()
        qty_input = input("Enter quantity to buy: ").strip()
        
        try:
            quantity = int(qty_input)
        except ValueError as e:
            print(f"Invalid quantity: {e}")
            return
        
        txn.buy(symbol, quantity)
        
    elif choice == "2":
        symbol = input("Enter stock symbol to sell: ").upper()
        qty_input = input("Enter quantity to sell: ").strip()
        
        try:
            quantity = int(qty_input)
        except ValueError as e:
            print(f"Invalid quantity: {e}")
            return
        
        txn.sell(symbol, quantity)
        
    elif choice == "3":
        #? Return to the main menu
        return
    else:
        print("\n[INFO] Invalid option, returning to main menu.")


def main():
    while True:
        print("\nYour Options Are:")
        print("1. Add stock to portfolio")
        print("2. Remove stock from portfolio")
        print("3. Display portfolio information")
        print("4. Enter the transaction interface")
        print("5. Exit")
        
        userschoice = input("Select one of the above: ")
        
        if userschoice == "1":
            symbol = input("Enter stock symbol: ").upper()
            start_date = input("Enter start date (YYYY-MM-DD): ") or None
            end_date = input("Enter end date (YYYY-MM-DD): ") or None
            insert_workflow([symbol], start_date, end_date)
            print(f"\n[INFO] Stock {symbol} has been added to your portfolio.")
        
        elif userschoice == "2":
            symbol = input("Enter the stock symbol to remove: ").upper()
            removingstock(symbol)
        
        elif userschoice == "3":
            display_portfolio_info()
        elif userschoice == "4":
            transaction_interface()
        elif userschoice == "5":
            print("\n[INFO] Goodbye!")
            break
        else:
            print("\n[ERROR] Invalid option. Please try again.")


if __name__ == "__main__":
    ticker_list = portfolio_ticker_list()
    tickers_info = parallel_fetch_tickers()
    update_daily_data()
    main()
