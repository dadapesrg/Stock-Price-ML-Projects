#import the libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from data_preprocessor import read_database_data, create_sequences, check_missing_data, visualize_correlation_matrix

# Get data from the database using the database connection
DATABASE_URL = "sqlite:///data/stock_price_data.db"  # Replace with actual database URL
df_data = read_database_data(DATABASE_URL, "stock_prices")

# Check for missing data
check_missing_data(df_data)

# Display the data statistics
df_data.describe()

# Convert the data to a time series format
df = pd.DataFrame(df_data).set_index("Date")

# Visualize the correlation matrix using a heatmap
visualize_correlation_matrix(df)

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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
model.fit(X_train, y_train, batch_size=8, epochs=100) #32

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.hstack((np.zeros((predictions.shape[0], 3)), predictions, np.zeros((predictions.shape[0], 1)))))[:, 3]

# Inverse transform the test data for comparison
y_test_scaled = scaler.inverse_transform(np.hstack((np.zeros((y_test.shape[0], 3)), y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 1)))))[:, 3]

# Plot the results
plt.figure(figsize=(14,5))
plt.plot(y_test_scaled, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Save the model
#Import library for saving the model
import pickle
#Save the model to a file
with open('results/lstm_stock_model.pkl', 'wb') as f:
    pickle.dump(model, f)

#calculate the r2_score
R2_Score_dtr = round(r2_score(predictions, y_test_scaled) * 100, 2)
print("R2 Score for LSTM : ", R2_Score_dtr,"%")