from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from stock_data_transformed import clean_and_transform_data
from web_extract_stock_data import get_stock_data
from stock_data_storage import load_data_to_db
from datetime import datetime, timedelta

DATABASE_URL = "sqlite:///data/stock_price_data.db"  # For SQLite

# Scheduler function to run the pipeline
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
