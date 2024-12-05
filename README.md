The stock price prediction project aims to develop different machine learning 
models for prediction of stock prices. The data used are obtained from the Yahoo Finance API. The project is organised into the following folders:
1. Data-pipeline: Used to scrap, clean and process the stock data from the Yahoo Finance website. The processed data is then stored in SQLite database using SQLALchemy. Models used SQLALchemy to retrieves the stored stock data data from the database to build the models.

2. Models: Consist of the followings:
    a. LSTM model using tensorFlow
    b. LSTM model using PyTorch
    c. Gradient Boost model with hyperparameters tuning

3. Data: Location of the SQLite database
4. Results: Location of stored trained models