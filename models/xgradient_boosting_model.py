import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_preprocessor import read_database_data

# Get data from the database using the database connection
DATABASE_URL = "sqlite:///data/stock_price_data.db"  # Replace with actual database URL
df_data = read_database_data(DATABASE_URL, "stock_prices")

# Convert the data to a time series format
df = pd.DataFrame(df_data).set_index("Date")
print(df.info())

# Visualize the close price target data
plt.plot(df.index, df["Close"], label="Time Series Data for Stock Price Close Selected as Target")
plt.legend()
plt.show()

# Drop features not used for prediction
df = df.drop('Adj Close', axis=1)
print(df.head()) 

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Convert back to DataFrame for easier manipulation
df_scaled_data = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

# Seperate the dependant and independant variables
X = df_scaled_data.drop('Close', axis=1)
y = df_scaled_data['Close']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters
params = {'objective': 'reg:squarederror', 'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 200}

# Train model
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)
print("My Prediction", y_pred)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f'The Error MSE: {mse:.2f}')

# Evaluate model
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse:.2f}, R2: {r2:.2f}')

# Carry out hyperparameter tuning
#Hyperparameters tuning 
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.5, 1],
    'n_estimators': [50, 100, 200]
}

# Perform grid search
grid_search = GridSearchCV(xgb.XGBRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best score: {grid_search.best_score_}')


# Get the best hyperparameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)
print('Final Prediction:  ', y_pred)

#Function for inverse transform the predicted data
def invert_transform(data, shape, column_index, scaler):
    dummy_array = np.zeros((len(data), shape))    
    dummy_array[:, column_index] = data    
    return scaler.inverse_transform(dummy_array)[:, column_index]

# Create a DataFrame to hold predictions and actual values
test_pred_df = pd.DataFrame({
    'Actual': invert_transform(y_test, scaled_data.shape[1], 0, scaler), # scaler.inverse_transform(y_test),  # Inverse scale the nat_demand
    'Predicted': invert_transform(y_pred, scaled_data.shape[1], 0, scaler) #inversed_predictions #scaler.inverse_transform(np.concatenate([y_pred, np.zeros_like(y_pred)], axis=1))[:, 0]
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

#calculate the r2_score
R2_Score_dtr = round(r2_score(y_pred, y_test) * 100, 2)
print("R2 Score for XGBoost Model : ", R2_Score_dtr,"%")

# Evaluate the model on the testing data
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
