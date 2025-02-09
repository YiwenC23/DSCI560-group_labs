import time
import random
import pandas as pd
import os

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

if __name__ == "__main__":
    virtual_real_time_stock()
