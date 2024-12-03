import yfinance as yf

def get_stock_data(ticker, start_date, end_date):
    # Fetch historical data using yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')

    # Reset the index to include the Date column in the DataFrame
    stock_data.reset_index(inplace=True)

    # Select and rename columns to match the specified table headings   
    stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    print("Before: ", stock_data)    
    return stock_data

# Display the data

