import time
import random
import pandas as pd
import os
import yfinance

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
def stock_retrieve(ticker_list, target_date):
    try:
        # Download historical stock data
        raw_data = yf.download(
            ticker_list,
            start=target_date,
            end=target_date,
            interval="1d",
            progress=False
        )
        
        # Ensure data is available
        if raw_data.empty:
            print(f"No data found for the specified date: {target_date}")
            return []
        
        # Process and format data
        stock_data = (
            raw_data.stack(level=0, future_stack=True)
            .rename_axis(["Date", "Ticker"])
            .reset_index(level=1)
            .reset_index(drop=False)
            .rename(columns={
                "date": "date",
                "ticker": "ticker",
                "open_price": "open",
                "close_price": "close"
            })
            .rename_axis(columns=None)
        )
        
        # Convert dataframe to list of dictionaries
        result = stock_data[["date", "ticker", "open", "close"]].to_dict(orient="records")
        
        return result
    
    except Exception as e:
        print(f"\n[ERROR] Failed to retrieve stock data: {e}")
        return []

if __name__ == "__main__":
    virtual_real_time_stock()
