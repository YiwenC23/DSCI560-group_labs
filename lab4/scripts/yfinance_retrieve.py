import pandas as pd
import yfinance as yf
import sqlalchemy as sql
from sqlalchemy import text
from datetime import datetime, timedelta

from database import SessionLocal, StockData, TickerIndex


#* Define the function for timestamp processing algorithm
def compare_timestamp(tickers_ranges):
    
    results = {}
    session = SessionLocal()
    
    #? Ensure tickers_ranges is a list
    if not isinstance(tickers_ranges, list):
        tickers_ranges = [tickers_ranges]
    
    #? Retrieve existing ticker index records for all tickers at once.
    tickers = [tr[0] for tr in tickers_ranges]
    indexes = session.query(TickerIndex).filter(TickerIndex.ticker.in_(tickers)).all()
    index_dict = {idx.ticker: idx for idx in indexes}
    
    for ticker, new_start, new_end in tickers_ranges:
        idx = index_dict.get(ticker)
        
        #! Case 1: If no stored data exists, the complete timeline is the new data range
        if not idx:
            results[ticker] = {
                "complete_range": (new_start, new_end),
                "missing_segments": [(new_start, new_end)]
            }
        else:
            #? Get the stored data interval and the new data interval
            results[ticker] = {}
            idx_interval = (idx.start_date, idx.end_date)
            new_interval = (new_start, new_end)
            
            #? The overall data range is the minimum start date and the maximum end date
            overall_start = min(new_interval[0], idx_interval[0])
            overall_end = max(new_interval[1], idx_interval[1])
            results[ticker]["complete_range"] = (overall_start, overall_end)
            
            #? Initialize the missing segments
            missing_segments = []
            
            #? Handle the gap relationships
            #! Case 2: New data interval left-apart from the stored data interval
            if new_interval[1] < idx_interval[0]:
                gap_start = new_interval[0]
                gap_end = idx_interval[0] - timedelta(days=1)
                if gap_start <= gap_end:
                    missing_segments.append((gap_start, gap_end))
            
            #! Case 3: New data interval right-apart from the stored data interval
            elif new_interval[0] > idx_interval[1]:
                gap_start = idx_interval[1] + timedelta(days=1)
                gap_end = new_interval[1]
                if gap_start <= gap_end:
                    missing_segments.append((gap_start, gap_end))
            
            #? Handle the containment relationships
            else:
                #! Case 4: New data interval left-overlaps with the stored data interval
                if new_interval[0] < idx_interval[0]:
                    missing_segments.append((new_interval[0], idx_interval[0] - timedelta(days=1)))
                #! Case 5: New data interval right-overlaps with the stored data interval
                if new_interval[1] > idx_interval[1]:
                    missing_segments.append((idx_interval[1] + timedelta(days=1), new_interval[1]))
            
            results[ticker]["missing_segments"] = missing_segments
    
    return results


#* Defined function for data insertion
def insert_db(data):
    try:
        session = SessionLocal()
        records = data.reset_index()[["date", "ticker", "open", "high", "low", "close", "volume"]]
        
        session.bulk_insert_mappings(StockData, records.to_dict(orient="records"))
        session.commit()
        
        tickers = data["ticker"].unique()
        for tkr in tickers:
            update_ticker_index(tkr)
    
    except Exception as e:
        print(f"Filed to insert data: {e}")
        session.rollback()


#* Define a function for ticker index update
def update_ticker_index(ticker):
    try:
        session = SessionLocal()
        #? Get the first record and latest records of the ticker
        first_record = session.query(StockData).filter(StockData.ticker == ticker).order_by(StockData.date.asc()).first()
        latest_record = session.query(StockData).filter(StockData.ticker == ticker).order_by(StockData.date.desc()).first()
        
        #? If there is no data for the ticker, that means the ticker data has been removed from the database, so delete the ticker index record
        if not first_record and not latest_record:
            session.query(TickerIndex).filter(TickerIndex.ticker == ticker).delete()
            session.commit()
        
        #? If there is data for the ticker, update the ticker index with the new complete range
        else:
            index = TickerIndex(ticker=ticker, start_date=first_record.date, end_date=latest_record.date)
            session.merge(index)
            session.commit()
    
    except Exception as e:
        print(f"Failed to update ticker index: {e}")
        session.rollback()

#* Defined function for data removal
def removingstock(symbol):
    try:
        session = SessionLocal()
        result = session.query(StockData).filter(StockData.ticker == symbol).delete()
        session.commit()
        
        update_ticker_index(symbol)
        
        if result > 0:
            print(f"Successfully removed {symbol} from database")
        else:
            print(f"{symbol} not found in database")
    
    except Exception as e:
        print(f"Error removing stock: {e}")
        session.rollback()


#* Define the function to retrieve historical stock data
def stock_retrieve(ticker_list, start_date=None, end_date=None):
    try:
        if start_date is None and end_date is None:
            raw_data = yf.download(
                ticker_list,
                group_by="Ticker",
                interval="1d",
                progress=False
            )
        else:
            raw_data = yf.download(
                ticker_list,
                group_by="Ticker",
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False
            )
        
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
        
        return stock_data.set_index("date")
    
    except Exception as e:
        print(e)


#* Define Workflow Function
def insert_workflow(ticker_list, start_date=None, end_date=None):
    try:
        session = SessionLocal()
        
        #? If no start or end date is provided, retrieve all data
        if start_date is None and end_date is None:
            stock_data = stock_retrieve(ticker_list)
            insert_db(stock_data)
            
            for tkr in ticker_list:
                update_ticker_index(tkr)
            
            session.commit()
                    
        #? If start and end date is provided, retrieve data for the specified range
        else:
            start_dateObj = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dateObj = datetime.strptime(end_date, "%Y-%m-%d").date()
            
            ticker_ranges = [(tk, start_dateObj, end_dateObj) for tk in ticker_list]
            needed_ranges = compare_timestamp(ticker_ranges)
            
            all_data = []
            for tk in ticker_list:
                needed_tr = needed_ranges.get(tk)
                
                overall_range = needed_tr["complete_range"]
                missing_segments = needed_tr["missing_segments"]
                
                #! If there are no missing segments (and an index exists), the complete range is already stored.
                if needed_tr and not missing_segments:
                    print(f"{tk}: Data from {overall_range[0]} to {overall_range[1]} already exists in the database.")
                    continue
                
                #? Fetch the missing segments data and insert into the database
                for ms in missing_segments:
                    all_data.append(stock_retrieve([tk], ms[0], ms[1]))
                
                update_ticker_index(tk)
            
            #? If any new data was fetched, combine and insert it into the database
            if all_data:
                combined_data = pd.concat(all_data)
                insert_db(combined_data)
            
            session.commit()
    
    except Exception as e:
        raise ValueError(f"Process failed: {e}")
        session.rollback()
