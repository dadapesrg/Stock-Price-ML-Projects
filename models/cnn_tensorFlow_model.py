#import the libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
import keras.optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from .data_preprocessor import read_database_data

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

# Define a function to create sequences of data
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 3])  # 'close' price is the target
    return np.array(X), np.array(y)

# Define the sequence length
seq_length = 60

# Create sequences
X, y = create_sequences(scaled_data, seq_length)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = Sequential()

lr = 0.0003
adam = keras.optimizers.Adam(lr)

#Develope convolusion CNN model for Time Series Forecasting

# Define the CNN model
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(1))
model_cnn.compile(loss='mse', optimizer=adam)

#CNN model summary
model_cnn.summary()

#Number of epoch and batch size
epochs = 150
batch = 24

# Train the model
cnn_history = model_cnn.fit(X_train, y_train, epochs=epochs, batch_size=batch, verbose=2)

# Make predictions on the test set
cnn_y_pred = model_cnn.predict(X_test)

# Make predictions on the test set
cnn_y_pred = model_cnn.predict(X_test)

# Since we scaled all features but predicting only one, we'll
# need to inverse transform the predictions using the appropriate feature column.
def invert_transform(data, shape, column_index, scaler):
    dummy_array = np.zeros((len(data), shape))
    dummy_array[:, column_index] = data.flatten()
    return scaler.inverse_transform(dummy_array)[:, column_index]

# Create a DataFrame to hold predictions and actual values
test_pred_df = pd.DataFrame({
    'Actual': invert_transform(y_test, X_train.shape[2], 0, scaler), 
    'Predicted': invert_transform(cnn_y_pred, X_train.shape[2], 0, scaler)
})

# Plot the results
plt.figure(figsize=(14,5))
plt.plot(test_pred_df['Actual'], color='blue', label='Actual Stock Price')
plt.plot(test_pred_df['Predicted'], color='red', label='Predicted Stock Price') #plt.plot(y_pred, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Save the model
#Import library for saving the model
import pickle
#Save the model to a file
with open('results/cnn_stock_model.pkl', 'wb') as f:
    pickle.dump(model, f)

#calculate the r2_score
#calculate the r2_score
R2_Score_dtr = round(r2_score(cnn_y_pred, y_test) * 100, 2)
print("R2 Score for CNN Model : ", R2_Score_dtr,"%")
