import requests
import pandas as pd
import datetime
import yfinance as yf
import sqlalchemy as sql
import concurrent.futures

from algorithm import get_DBdata, algorithm
from database import SessionLocal, StockData, TickerIndex, UserAsset, UserAcc
from yfinance_retrieve import removingstock
from real_time_data import historical_stock_retrieve


def portfolio_ticker_list():
    global ticker_list
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
        
        #yf_ticker = yf.Ticker(ticker_symbol, session=session_obj)
        #info = yf_ticker.info
        info = historical_stock_retrieve([ticker_symbol])
        
        current_info = {
            "ticker": ticker_symbol,
            "date": info.get("date"),
            "open": info.get("open"),
            "high": info.get("high"),
            "low": info.get("low"),
            "close": info.get("close"),
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
        global tickers_info
        try:
            # if symbol in tickers_info:
            #     buy_price = tickers_info[symbol]["price"]
            #     date = tickers_info[symbol]["date"]
            # else:
            #     print(f"[ERROR] Failed to fetch {symbol} from Yahoo Finance.")
            #     return None
            
            tkr_info = next(info for info in tickers_info if info["ticker"] == symbol)
            buy_price = tkr_info.get("open")
            date = tkr_info.get("date")
            
            new_transaction = UserAsset(
                ticker=symbol,
                datetime=date,
                quantity=quantity,
                price=buy_price
            )
            
            self.session.add(new_transaction)
            self.session.commit()
            print(f"[INFO] Transaction successful ({new_transaction.datetime}): purchased {quantity} shares of {symbol} at {buy_price} per share!")
            
            #? Update the user's balance
            cost = buy_price * quantity
            update_user_balance(-cost)
            
            return new_transaction
        
        except Exception as e:
            print(f"[ERROR] Transaction failed: {e}")
            self.session.rollback()
        finally:
            self.session.close()
    
    
    def sell(self, symbol, quantity):
        global tickers_info
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
                print(f"[ERROR] Insufficient shares of {symbol} to sell.")
                return None
            
            # #? Fetch the current price of the ticker
            # ticker_info = fetch_single_ticker(symbol)
            
            # #? If the current price of the ticker is not available, return None
            # if ticker_info is None or ticker_info["price"] is None:
            #     print(f"[ERROR] Failed to fetch the current price of {symbol}.")
            #     return None
            
            # current_price = ticker_info["price"]
            # date = ticker_info["date"]
            # #? Record the selling transactions as a negative value to distinguish them from the buying transactions
            # sell_transaction = UserAsset(
            #     ticker=symbol,
            #     datetime=date,
            #     quantity=-quantity,
            #     price=current_price
            # )
            
            # if symbol in tickers_info:
            #     sell_price = tickers_info[symbol]["price"]
            #     date = tickers_info[symbol]["date"]
            # else:
            #     print(f"[ERROR] Failed to fetch {symbol} from Yahoo Finance.")
            #     return None
            
            tkr_info = next(info for info in tickers_info if info["ticker"] == symbol)
            sell_price = tkr_info.get("open")
            date = tkr_info.get("date")
            
            sell_transaction = UserAsset(
                ticker=symbol,
                datetime=date,
                quantity=-quantity,
                price=sell_price
            )
            
            self.session.add(sell_transaction)
            self.session.commit()
            print(f"[INFO] Transaction successful ({sell_transaction.datetime}): sold {quantity} shares of {symbol} at {sell_price} per share!")
            
            #? Update the user's balance
            income = sell_price * quantity
            update_user_balance(income)
            
            return sell_transaction
        
        except Exception as e:
            print(f"[ERROR] Transaction failed: {e}")
            self.session.rollback()
        finally:
            self.session.close()


#* Define the function to output a transaction signal for each ticker in the portfolio
def transaction_signal():
    global ticker_list
    try:
        session = SessionLocal()
        signal_dict = {}
        
        for tkr in ticker_list:
            tkr_info = next(info for info in tickers_info if info["ticker"] == tkr)
            current_price = tkr_info.get("open")
            data_train = get_DBdata(tkr)
            print("\ndata_train: \n", data_train.tail())

            prediction = algorithm(data_train)
            print(f"\nPrediction: {round(prediction, 3)} | Current Price: {round(current_price, 3)}\n")
            
            if prediction > current_price:
                signal_dict[tkr] = "buy"
            elif prediction < current_price:
                signal_dict[tkr] = "sell"
            else:
                signal_dict[tkr] = "no move"
        
        print(f"\nsignal_dict: {signal_dict}\n")
        return signal_dict
    
    except Exception as e:
        print(f"\n[ERROR] Failed to get the transaction signal: {e}")
    finally:
        session.close()

def init_user_account():
    session = SessionLocal()
    try:
        account = session.query(UserAcc).first()
        if account is None:
            while True:
                init_found = input("Enter your initial funds (skip for default, 1000): ").strip()
                try:
                    if init_found:
                        init_money = float(init_found)
                        break
                    else:
                        init_money = 1000.0
                        break
                
                except ValueError:
                    print("\n[ERROR] Invalid amount. Please enter a valid number.")
            
            account = UserAcc(balance=init_money)
            session.add(account)
            session.commit()
            print(f"\n[INFO] User account has been initialized with {init_money}.")
        else:
            print(f"\n[INFO] Welcome back, your current balance is {account.balance}.")
        return account.balance
    
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize user account: {e}")
        session.rollback()
    finally:
        session.close()


def update_user_balance(amount_change):
    session = SessionLocal()
    try:
        account = session.query(UserAcc).first()
        if account is None:
            print("\n[ERROR] User account not found.")
            return
        
        account.balance += amount_change
        session.commit()
    
    except Exception as e:
        print(f"\n[ERROR] Failed to update user balance: {e}")
        session.rollback()
    finally:
        session.close()


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
        
        balance = session.query(UserAcc).first().balance if session.query(UserAcc).first().balance is not None else 0.0
        
        #? Query user's total shares held and total cost for each ticker in the portfolio
        for info in tickers_info:
            ticker = info.get("ticker")
            current_price = info.get("open")
            
            #! When the stock market is closed, calculate the price based on the previous day's closing price
            if current_price is None:
                fallback_record = session.query(StockData) \
                    .filter(StockData.ticker == ticker) \
                    .order_by(StockData.date.desc()) \
                    .first()
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
        
        #? Calculate the annualized returns, and the Sharpe ratio
        num_stocks = len(portfolio_details) if portfolio_details else 1
        annualized_return = ((total_current_value / total_cost) ** (252 / num_stocks)) - 1 if total_cost else 0
        sharpe_ratio = annualized_return / ((0.03 / 252) ** 0.5) if annualized_return else 0
        
        portfolio_summary = {
            "balance": round(balance, 2),
            "portfolio_details": portfolio_details,
            "total_cost": round(total_cost, 2),
            "total_current_value": round(total_current_value, 2),
            "total_profit_loss": round(total_profit_loss, 2),
            "annualized_return": round(annualized_return, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }
        
        return portfolio_summary
    
    except Exception as e:
        raise ValueError(f"\n[ERROR] Failed to calculate portfolio value: {e}")
    finally:
        session.close()


#* Define the function for updating the current daily stock data into the database.
# def update_daily_data():
#     try:
#         session = SessionLocal()
        
#         #? Get end date for each ticker in the TickerIndex table
#         ticker_indices = session.query(TickerIndex).all()
        
#         today = datetime.datetime.now().date()
        
        
#         for idx in ticker_indices:
#             ticker = idx.ticker
#             last_date = idx.end_date
            
#             if last_date < today:
#                 next_date = last_date + datetime.timedelta(days=1)
#                 fresh_data = stock_retrieve([ticker], str(next_date), str(today))
                
#                 if fresh_data.empty:
#                     continue
#                 else:
#                     insert_workflow([ticker], str(next_date), str(today))
#                     print(f"\n[INFO] {ticker} data has been successfully updated to {today}.")
    
#     except Exception as e:
#         print(f"\n[ERROR] Failed to update daily data: {e}")
#     finally:
#         session.close()

def new_insert_workflow(ticker, start_date, end_date):
    try:
        session = SessionLocal()
        
        if start_date is not None and end_date is not None:
            latest_record = session.query(StockData).filter(StockData.ticker == ticker).order_by(StockData.date.desc()).first()
            
            if latest_record.date < end_date:
                new_info = next(info for info in tickers_info if info["ticker"] == ticker)
                new_record = StockData(**new_info)
                session.add(new_record)
            
                first_record = session.query(StockData).filter(StockData.ticker == ticker).order_by(StockData.date.asc()).first()
                first_date = first_record.date
                
                new_index = TickerIndex(
                    ticker=ticker,
                    start_date=first_date,
                    end_date=end_date
                )
                session.merge(new_index)
                
                
        session.commit()
    except Exception as e:
        print(f"\n[ERROR] Data insertion failed: {e}")
        session.rollback()
    finally:
        session.close()


def update_daily_data():
    try:
        session = SessionLocal()
        
        ticker_indices = session.query(TickerIndex).all()
        today = tickers_info[0].get("date")
        
        for idx in ticker_indices:
            ticker = idx.ticker
            last_date = idx.end_date
            
            if last_date < today:
                next_date = last_date + datetime.timedelta(days=1)
                fresh_data = historical_stock_retrieve([ticker])
                
                if not fresh_data:
                    continue
                else:
                    new_insert_workflow(ticker, next_date, today)
                    print(f"\n[INFO] {ticker} data has been successfully updated to {today}.")
    
    except Exception as e:
        print(f"\n[ERROR] Failed to update daily data: {e}")
    finally:
        session.close()


#* Define the function of buying and selling algorithm after the transaction signal is generated.
def transaction_workflow():
    global current_balance, tickers_info
    
    session = SessionLocal()
    txn = Transaction()  # Apply the Transaction class
    signal_dict = transaction_signal()
    
    try:
        sell_tickers = list(map(lambda r: r[0], filter(lambda r: r[1] == "sell", signal_dict.items())))
        if sell_tickers:
            print(f"\n[INFO] Transaction: Selling the following stocks: {sell_tickers}...")
            for tkr in sell_tickers:
                share_held = UserAsset.ticker_total_quantity(session, tkr)
                if share_held and share_held > 0:
                    tkr_info = next((info for info in tickers_info if info["ticker"] == tkr), None)
                    current_price = tkr_info.get("open")
                    quantity = int(share_held)
                    txn.sell(tkr, quantity)
                            
        current_balance = session.query(UserAcc).first().balance
        print(f"[INFO] Current balance: {current_balance}.")
        
        buy_tickers = list(map(lambda r: r[0], filter(lambda r: r[1] == "buy", signal_dict.items())))
        if buy_tickers:
            if len(buy_tickers) == 1:
                tkr = buy_tickers[0]
                tkr_info = next((info for info in tickers_info if info["ticker"] == tkr), None)
                current_price = tkr_info.get("open")
                if current_price and current_price > 0:
                    quantity = int(current_balance / current_price)
                    
                    if quantity > 0:
                        txn.buy(tkr, quantity)
            
            else:
                allocation = current_balance / len(buy_tickers)
                for tkr in buy_tickers:
                    tkr_info = next((info for info in tickers_info if info["ticker"] == tkr), None)
                    current_price = tkr_info.get("open")
                    
                    if current_price and current_price > 0:
                        quantity = int(allocation / current_price)
                        if quantity > 0:
                            txn.buy(tkr, quantity)
        current_balance = session.query(UserAcc).first().balance
        print(f"[INFO] Current balance: {current_balance}.")
    
    except Exception as e:
        print(f"[ERROR] Failed to making transactions: {e}")
    
    finally:
        session.close()


#* Define the function to display the portfolio information.
def display_portfolio_info():
    #? Calculate the portfolio information
    try:
        portfolio_summary = calculate_portfolio_value()
    except Exception as e:
        print(f"\n[ERROR] Failed to calculate portfolio value: {e}")
        return
    
    print("\n------------------------------")
    print("  Portfolio Details")
    print("------------------------------")
    if portfolio_summary and portfolio_summary.get("portfolio_details"):
        for info in portfolio_summary["portfolio_details"]:
            print(f"Ticker: {info['ticker']}")
            print(f"  Latest Price: {info['current_price']}")
            print(f"  Shares: {info['total_quantity']}")
            print(f"  Cost: {info['ticker_cost']}")
            print(f"  Market Value: {info['current_value']}")
            print(f"  Gain/Loss: {info['profit_loss']}")
            print("-----------------------------")
    else:
        print("\n[INFO] No portfolio data available.")
    
    main_menu = input("Option: Enter '1' to return to the main menu: ")
    if main_menu == "1":
        return
    else:
        print("\n[INFO] Invalid option, returning to main menu.")


#* Define the function for the portfolio summary.
def display_portfolio_summary(): 
    global ticker_list, tickers_info
    
    #? Calculate the portfolio information
    try:
        portfolio_summary = calculate_portfolio_value()
    except Exception as e:
        print(f"\n[ERROR] Failed to calculate portfolio value: {e}")
        return
    
    #? Display the detailed portfolio information
    print("\n------------------------------")
    print("  Current Portfolio Status")
    print("------------------------------")
    if portfolio_summary:
        print("User: root")
        print(f"  Balance: {portfolio_summary.get('balance')}")
        print(f"  Total Cost: {portfolio_summary.get('total_cost')}")
        print(f"  Total Market Value: {portfolio_summary.get('total_current_value')}")
        print(f"  Total Gain/Loss: {portfolio_summary.get('total_profit_loss')}")
        print(f"  Annualized Return: {portfolio_summary.get('annualized_return')}")
        print(f"  Sharpe Ratio: {portfolio_summary.get('sharpe_ratio')}")
        print("-----------------------------")
    else:
        print("\n[INFO] No portfolio data available.")
    

    main_menu = input("Option: Enter '1' to return to the main menu: ")
    if main_menu == "1":
        return
    else:
        print("\n[INFO] Invalid option, returning to main menu.")


def transaction_interface():
    global ticker_list, tickers_info, current_balance
    stop_date = datetime.datetime.strptime("2025-01-01", "%Y-%m-%d").date()
    
    
    while True:
        session = SessionLocal()
        # update the tickers_info
        tickers_info = historical_stock_retrieve(ticker_list)
        # get the latest date from the tickers_info
        current_date = tickers_info[0].get("date")
        # get the latest date from the database
        latest_record = session.query(StockData).order_by(StockData.date.desc()).first()
        latest_date = latest_record.date if latest_record else current_date
        
        print("\n[INFO] Transaction Interface")
        print(f"Current Date: {current_date} | Stock Date: {latest_date}")
        
        # if the market data is updated than the database, then execute the transaction and data update
        if current_date > latest_date:
            transaction_workflow()
            update_daily_data()
        
        # after the transaction, refresh the tickers_info and current_date
        tickers_info = historical_stock_retrieve(ticker_list)
        current_date = tickers_info[0].get("date")
        
        # if the current date is greater than or equal to the stop date, then exit the loop
        if current_date >= stop_date:
            print("\n[INFO] Transaction to the target data has been completed")
            session.close()
            break
        
        session.close()
    
    main_menu = input("\nOption: Enter '1' to return to the main menu: ")
    if main_menu == "1":
        return
    else:
        print("\n[INFO] Invalid option, returning to main menu.")


def main():
    global ticker_list, tickers_info, current_balance
    
    while True:
        print("\nYour Options Are:")
        print("1. Add stock to portfolio")
        print("2. Remove stock from portfolio")
        print("3. Enter the portfolio information interface")
        print("4. Enter the portfolio summary interface")
        print("5. Enter the transaction interface")
        print("6. Exit")
        
        userschoice = input("Select one of the above: ").strip()
        
        if userschoice == "1":
            symbol = input("Enter stock symbol: ").upper()
            start_date = input("Enter start date (YYYY-MM-DD): ") or None
            end_date = input("Enter end date (YYYY-MM-DD): ") or None
            insert_workflow(symbol, start_date, end_date)
            print(f"\n[INFO] Stock {symbol} has been added to your portfolio.")
        
            current_balance = init_user_account()
        
        elif userschoice == "2":
            symbol = input("Enter the stock symbol to remove: ").upper()
            removingstock(symbol)
        
        elif userschoice == "3":
            display_portfolio_info()
        
        elif userschoice == "4":
            display_portfolio_summary()
        
        elif userschoice == "5":
            transaction_interface()
            current_balance = init_user_account()
            
        
        elif userschoice == "6":
            print("\n[INFO] Goodbye!")
            break
        else:
            print("\n[ERROR] Invalid option. Please try again.")

if __name__ == "__main__":
    current_balance = init_user_account()
    ticker_list = portfolio_ticker_list()
    tickers_info = historical_stock_retrieve(ticker_list)
    
    # update_daily_data()
    main()
