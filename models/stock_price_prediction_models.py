#import the libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
import keras.optimizers


from tensorflow.keras.layers import LSTM, Dropout

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,  mean_squared_error
from data_preprocessor import read_database_data, create_sequences

# Get data from the database using the database connection
DATABASE_URL = "sqlite:///data/stock_price_data.db"  # Replace with actual database URL

df_data = read_database_data(DATABASE_URL, "stock_prices")

# Convert the data to a time series format
df = pd.DataFrame(df_data).set_index("Date")

# Visualize the close price target data
plt.plot(df.index, df["Close"], label="Time Series Data for Stock Price Close Selected as Target")
plt.legend()
plt.show()

# Select the relevant features for prediction (e.g., 'Open', 'High', 'Low', 'Close', 'Volume')
features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = df[features].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define the sequence length
seq_length = 60

# Create sequences
table_column_index = 3  # Index of the 'close' price column as the target
X, y = create_sequences(scaled_data, seq_length, table_column_index)

#Prepare data for random forest and gradient boost models
# Drop features not used for prediction
df = df.drop('Adj Close', axis=1)
scaled_data = scaler.fit_transform(df)
# Convert to DataFrame for easier manipulation
df_scaled_data = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
# Seperate the dependant and independant variables
X_ml = df_scaled_data.drop('Close', axis=1)
y_ml = df_scaled_data['Close']
# Split the data into training and test sets
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_train_cnn_model(X_train, y_train, epochs=100, batch=24, lr=0.0003):
    # Define the CNN model    
    adam = keras.optimizers.Adam(lr)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=adam)
    #model.compile(optimizer='adam', loss='mean_squared_error')

    # Model summary
    model.summary()   

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch, verbose=2)

    return model

# Build and train the lstm model
def build_train_lstm_model(X_train, y_train, epochs=100, batch=24):
    # Initialize the model
    model = Sequential()

    # Add LSTM layers with Dropout regularization
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    # Add Dense layer
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # Output layer, predicting the 'close' price

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=batch, epochs=epochs)

    return model

def build_train_lstm_pytorch_model(X_train, y_train, epochs=100):    
    import torch.nn as nn
    import torch.optim as optim

    # Convert data to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)      
    
    # Define LSTM model
    class LSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            out = self.fc(hn[-1])
            return out

    # Model, loss function, and optimizer
    input_dim = X_train.shape[2]
    hidden_dim = 64
    output_dim = 1

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    train_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train).squeeze()
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model

# Build and train the random forest model
def build_train_random_forest_model(X_train, y_train, n_estimators=100, min_samples_split=2, max_depth=15):
    from sklearn.ensemble import RandomForestRegressor    
    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth)
    rf.fit(X_train, y_train)
    return rf

def build_train_decision_tree_model(X_train_ml, y_train_ml, max_depth=15):
    from sklearn.tree import DecisionTreeRegressor
    dt = DecisionTreeRegressor(max_depth=max_depth)
    dt.fit(X_train_ml, y_train_ml)
    return dt

# Build and train the xgboost model
def build_train_xgboost_model(X_train, y_train, n_estimators=100, max_depth=7, learning_rate=0.1):
    import xgboost as xgb
    params = {'objective': 'reg:squarederror', 'max_depth': max_depth, 'learning_rate': learning_rate, 'n_estimators': n_estimators}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model
def make_predictions(model, X_test):
    return model.predict(X_test)    

def invert_transform(data, shape, column_index, scaler):
    dummy_array = np.zeros((len(data), shape))    
    dummy_array[:, column_index] = data    
    return scaler.inverse_transform(dummy_array)[:, column_index]

#y_test_torch = torch.FloatTensor(y_test)
y_test = scaler.inverse_transform(np.hstack((np.zeros((y_test.shape[0], table_column_index)), y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 1)))))[:, table_column_index]
#y_test_torch = scaler.inverse_transform(np.hstack([np.zeros((y_test_torch.shape[0], scaled_data.shape[1] - 1)), y_test_torch.numpy().reshape(-1, 1)]))[:, -1]

y_test_ml = invert_transform(y_test_ml, len(df.columns), table_column_index, scaler)

models = {
    'LSTM': build_train_lstm_model(X_train, y_train),
    'CNN': build_train_cnn_model(X_train, y_train),
   # 'PTLSTM': build_train_lstm_pytorch_model(X_train, y_train),
    'RF': build_train_random_forest_model(X_train_ml, y_train_ml),    
    'XGB': build_train_xgboost_model(X_train_ml, y_train_ml), 
    'DT': build_train_decision_tree_model(X_train_ml, y_train_ml)
}

import torch
rmse_scores = dict()
predictions = dict()
for name, model in models.items():
    if name == 'PTLSTM':        
        model.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(X_test)
            y_pred = model(X_test).squeeze()  
            #y_pred = scaler.inverse_transform(np.hstack([np.zeros((y_pred.shape[0], scaled_data.shape[1] - 1)), y_pred.numpy().reshape(-1, 1)]))[:, -1]
            predictions[name] = y_pred
            rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
            rmse = float("{:.4f}".format(rmse))
            rmse_scores[name] = rmse
            print(f"{name} RMSE: {rmse}")            
    elif name == 'RF' or name == 'XGB' or name == 'DT':
        y_pred = model.predict(X_test_ml)
        y_pred = invert_transform(y_pred, len(df.columns), table_column_index, scaler)
        predictions[name] = y_pred
        rmse = np.sqrt(mean_squared_error(y_test_ml, y_pred))
        rmse = float("{:.4f}".format(rmse))
        rmse_scores[name] = rmse
        print(f"{name} RMSE: {rmse}")
    else:        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(np.hstack((np.zeros((y_pred.shape[0], 3)), y_pred, np.zeros((y_pred.shape[0], 1)))))[:, 3]
        predictions[name] = y_pred
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse = float("{:.4f}".format(rmse))
        rmse_scores[name] = rmse
        print(f"{name} RMSE: {rmse}")

def add_plot_labels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')

plt.bar(list(rmse_scores.keys()), list(rmse_scores.values()), color ='red')

add_plot_labels(list(rmse_scores.keys()), list(rmse_scores.values()))
plt.xlabel('') 
plt.ylabel('RMSE') 
plt.title('Models') 
plt.show()

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True,figsize=(22,12))
ax5, ax6, ax7 = axes[0]
ax8, ax9, ax10 = axes[1]

def add_plot(x,y):
    for i in range(len(x)):
        if x[i] == 'RF':
            ax5.plot(y[i], label=x[i])
        elif x[i] == 'XGB':
            ax6.plot(y[i], label=x[i])
        elif x[i] == 'LSTM':
            ax7.plot(y[i], label=x[i])
        elif x[i] == 'CNN':
            ax8.plot(y[i], label=x[i])
        elif x[i] == 'PTLSTM':
            ax9.plot(y[i], label=x[i])
        else:
            ax10.plot(y[i], label=x[i])

    ax5.plot(y_test_ml, label='Actual Stock Price')
    ax6.plot(y_test_ml, label='Actual Stock Price')
    ax7.plot(y_test, label='Actual Stock Price')   
    ax8.plot(y_test, label='Actual Stock Price')      
    ax9.plot(y_test, label='Actual Stock Price')
    ax10.plot(y_test_ml, label='Actual Stock Price')
    
    ax5.legend(loc='best')
    ax6.legend(loc='best')
    ax7.legend(loc='best')
    ax8.legend(loc='best')
    ax9.legend(loc='best')
    ax10.legend(loc='best')

    ax5.set_title('Random Forest Predictions')
    ax6.set_title('XGBoost Predictions')
    ax7.set_title('LSTM Predictions')
    ax8.set_title('CNN Predictions')
    ax9.set_title('PTLSTM Predictions')
    ax10.set_title('Decision Tree Predictions')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Stock Price')

add_plot(list(predictions.keys()), list(predictions.values()))  

plt.tight_layout()
plt.show()
