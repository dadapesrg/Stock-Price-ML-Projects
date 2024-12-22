The stock price prediction project aims to develop different machine learning models for prediction of stock prices. The data used are obtained using the Yahoo Finance API. The project is organised into the following folders:
1. Data-pipeline: Modules used to scrap, clean and process the stock data from the Yahoo Finance website. The processed data is then stored in SQLite database using SQLALchemy. Models used SQLALchemy to retrieves the stored stock data data from the database to build the models.

2. Models: Modules for buiding the models. Contains the data preprocessing module and the model building modules. The models consist of the followings:\
    a. Long Short-Term Memory (LSTM) model using tensorFlow\
    b. Long Short-Term Memory (LSTM) model using PyTorch \
    c. Gradient Boost model with hyperparameters tuning \
    d. Convolution Neural Network (CNN) model \
    e. Random Forest model

3. Data: Location of the SQLite database where the stock data is stored.
4. Results: Location of stored trained models for prediction.
5. Tests: Location for the test cases for the project.