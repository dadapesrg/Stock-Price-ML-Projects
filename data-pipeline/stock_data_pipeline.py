import yfinance as yf
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

# Set up the database connection
DATABASE_URL = "sqlite:///data/stock_price_data.db"  # For SQLite

# Fetch stock data from the web using yfinance
def get_stock_data(ticker, start_date, end_date):
    # Fetch historical data using yfinance
    
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')

    # Reset the index to include the Date column in the DataFrame
    stock_data.reset_index(inplace=True)

    # Select and rename columns to match the specified table headings   
    stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    print("Before: ", stock_data)    
    return stock_data

# Clean and transform the data
def clean_and_transform_data(data):
    # Convert to a DataFrame
    df = pd.DataFrame(data)

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Change the column names
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    # Remove rows with missing data
    df = df.dropna()

    # Standard deviation statistical summary
    print(np.std(df))

    # Save the data to a CSV file
    df.to_csv('data/stock_data.csv', index=False)
       
    # Convert columns to appropriate types
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(',', ''), errors='coerce')
    print("After:::", df)
    return df

# Function to load data into the database
def load_data_to_db(df, database_url):
    # SQLAlchemy Engine Setup            
    engine = create_engine(database_url)
    
    # Insert data into the table
    df.to_sql('stock_prices', con=engine, if_exists="replace", index=False)

# Scheduler function to run the data pipeline
def run_pipeline():    
    # Define the stock ticker and period
    ticker = 'AAPL'  # Replace with the desired stock symbol
    start_date = '2018-12-01' # specify the start time
    end_date = '2024-12-31'   # specify the end time
    data = get_stock_data(ticker=ticker, start_date=start_date, end_date=end_date)
    cleaned_data = clean_and_transform_data(data)
    load_data_to_db(cleaned_data, DATABASE_URL)

# Set up the scheduler to run every day at 6 PM
scheduler = BlockingScheduler()
#scheduler = BackgroundScheduler()  # For non-blocking scheduler
run_time = datetime.now() + timedelta(seconds=10)
scheduler.add_job(run_pipeline, 'date', run_date=run_time)

# Set up for scheduler to run at specific time
#scheduler.add_job(run_pipeline, 'interval', days=1, start_date='2023-12-02 19:31:00')

# Start the scheduler
scheduler.start()