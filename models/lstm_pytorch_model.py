import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from .data_preprocessor import read_database_data, create_sequences

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

seq_length = 60
target_col_index = 3  # Index of the 'close' price column as the target

# Create sequences
X, y = create_sequences(scaled_data, seq_length, target_col_index)  # 'close' price is the target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

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
epochs = 50
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

# Testing
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test).squeeze()

# Rescale predictions and true values back to original scale
y_test = scaler.inverse_transform(np.hstack([np.zeros((y_test.shape[0], scaled_data.shape[1] - 1)), y_test.numpy().reshape(-1, 1)]))[:, -1]
y_test_pred = scaler.inverse_transform(np.hstack([np.zeros((y_test_pred.shape[0], scaled_data.shape[1] - 1)), y_test_pred.numpy().reshape(-1, 1)]))[:, -1]

# Plot results
plt.plot(y_test, label="Actual Stock Price")
plt.plot(y_test_pred, label="Predicted Stock Price")
plt.legend()
plt.title("Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()

#calculate the r2_score
R2_Score_dtr = round(r2_score(y_test_pred, y_test) * 100, 2)
print("R2 Score for LSTM : ", R2_Score_dtr,"%")