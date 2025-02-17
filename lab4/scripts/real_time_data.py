import time
import random
import pandas as pd
import os
import yfinance as yf
from database import SessionLocal, StockData
import datetime
import pprint as pp

# Function to update Apple's stock data
def update_stock(stock_data):
    price_change = random.uniform(-20, 20)  # Price fluctuates within ±10-20
    bid_change = random.uniform(-5, 5)
    ask_change = random.uniform(-5, 5)
    volume_change = random.randint(-100, 100)

    # Update stock values
    stock_data["price"] = max(1, stock_data["price"] + price_change)
    stock_data["bid"] = max(1, stock_data["price"] - bid_change)
    stock_data["ask"] = max(1, stock_data["price"] + ask_change)
    stock_data["volume"] = max(1, stock_data["volume"] + volume_change)

def virtual_real_time_stock():
    apple_stock = {"price": 227.63, "bid": 229.44, "ask": 229.46, "volume": 38983016}

    while True:
        update_stock(apple_stock)  # Pass the stock data dictionary
        
        os.system("cls" if os.name == "nt" else "clear")  # Clears screen for a clean update

        df = pd.DataFrame([apple_stock])  # Convert to DataFrame for better display
        print(df.to_string(index=False))  # Print stock data in table format
    
        time.sleep(1)  # Wait for the next update

#* Define the function to retrieve historical stock data
# print(stock_retrieve(["AAPL", "MSFT"], "2024-02-14"))
def historical_stock_retrieve(ticker_list):
    session = SessionLocal()
    try:
        #? Get the latest date for the ticker
        latest_record = session.query(StockData).filter(StockData.ticker == ticker_list[0]).order_by(StockData.date.desc()).first()
        if latest_record is None or latest_record.date is None:
            print(f"[ERROR] No historical record found for ticker: {ticker_list}")
            return {}
            
        latest_date = latest_record.date
        
        target_date = latest_date + datetime.timedelta(days=1)
        
        while True:
            try: 
                next_date = target_date + datetime.timedelta(days=1)
                target_date_str = target_date.strftime("%Y-%m-%d")
                next_date_str = next_date.strftime("%Y-%m-%d")
            
                # Download historical stock data
                raw_data = yf.download(
                    ticker_list,
                    group_by="Ticker",
                    start=target_date_str,
                    end=next_date_str,
                    interval="1d",
                    progress=False
                )
                
                # Ensure data is available
                if raw_data.empty:
                    target_date = target_date + datetime.timedelta(days=1)
                    next_date = next_date + datetime.timedelta(days=1)
                    
                    target_date_str = target_date.strftime("%Y-%m-%d")
                    next_date_str = next_date.strftime("%Y-%m-%d")
                    
                    raw_data = yf.download(
                        ticker_list,
                        start=target_date_str,
                        end=next_date_str,
                        interval="1d",
                        progress=False
                    )
                elif not raw_data.empty:
                    break
                    
            except Exception:
                return None
        
        # Process and format data
        stock_data = (
            raw_data.stack(level=0, future_stack=True)
            .rename_axis(["Date", "Ticker"])
            .reset_index(level=1)
            .reset_index(drop=False)
            .rename(columns={
                "Date": "date",
                "Ticker": "ticker",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            .rename_axis(columns=None)
        )
        #? Covert the date to datetime date object
        stock_data["date"] = stock_data["date"].apply(lambda r: datetime.datetime.strptime(r.strftime("%Y-%m-%d"), "%Y-%m-%d").date())
        
        # Convert dataframe to list of dictionaries
        # result = stock_data[["date", "ticker", "open", "close", "bid", "ask", "volume"]].to_dict(orient="records")
        return [stock_data.set_index("date")]
    
    except Exception as e:
        print(f"\n[ERROR] Failed to retrieve stock data: {e}")
