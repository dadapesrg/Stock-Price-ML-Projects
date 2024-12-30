#import the libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,  mean_squared_error, mean_absolute_error
from data_preprocessor import read_database_data, invert_transform

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

# Drop features not used for prediction
df = df.drop('Adj Close', axis=1)
scaled_data = scaler.fit_transform(df)

# Convert back to DataFrame for easier manipulation
df_scaled_data = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

# Seperate the dependant and independant variables
X = df_scaled_data.drop('Close', axis=1)
y = df_scaled_data['Close']


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

def build_train_gradient_boosting_model(X_train_ml, y_train_ml, rando_state=42):
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(random_state=rando_state)
    gb.fit(X_train_ml, y_train_ml)
    return gb

# Build and train the xgboost model
def build_train_xgboost_model(X_train, y_train, n_estimators=100, max_depth=7, learning_rate=0.1):
    import xgboost as xgb
    params = {'objective': 'reg:squarederror', 'max_depth': max_depth, 'learning_rate': learning_rate, 'n_estimators': n_estimators}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

table_column_index = 3  # Index of the 'close' price column as the target

models = {
    'RF': build_train_random_forest_model(X_train, y_train), 
    "GB": build_train_gradient_boosting_model(X_train, y_train), 
    'XGB': build_train_xgboost_model(X_train, y_train), 
    'DT': build_train_decision_tree_model(X_train, y_train)
}

rmse_scores = dict()
predictions = dict()
results = {}
for name, model in models.items():
        y_pred = model.predict(X_test)                
        predictions[name] = y_pred

        # calculating evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse = float("{:.4f}".format(rmse))
        rmse_scores[name] = rmse           
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # store the metrics
        results[name] = {"MAE": mae, "R²": r2,  "RMSE": rmse}     

results_df = pd.DataFrame(results).T  # convert the results to a DataFrame for better readability
print(results_df)

# Visualize the RMSE scores
def add_plot_labels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')

plt.bar(list(rmse_scores.keys()), list(rmse_scores.values()), color ='red')

add_plot_labels(list(rmse_scores.keys()), list(rmse_scores.values()))
plt.xlabel('') 
plt.ylabel('RMSE') 
plt.title('Models') 
plt.show()

# Visualize the predictions
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,figsize=(22,12))
fig.suptitle('Stock Price Predictions')
fig.supxlabel('Time')
fig.supylabel('Stock Price')
ax1, ax2 = axes[0]
ax3, ax4 = axes[1]

# Invert the transformation 
y_test = invert_transform(y_test, len(df.columns), table_column_index, scaler)
def add_plot(x,y):
    for i in range(len(x)):
        y[i] = invert_transform(y[i], len(df.columns), table_column_index, scaler)
        if x[i] == 'RF':
            ax1.plot(y[i], label=x[i])
        elif x[i] == 'XGB':
            ax2.plot(y[i], label=x[i])               
        elif x[i] == 'GB':
            ax3.plot(y[i], label=x[i])
        else:
            ax4.plot(y[i], label=x[i])

    ax1.plot(y_test, label='Actual Stock Price')    
    ax2.plot(y_test, label='Actual Stock Price')   
    ax3.plot(y_test, label='Actual Stock Price')      
    ax4.plot(y_test, label='Actual Stock Price')
    
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    ax4.legend(loc='best')  

    ax1.set_title('Random Forest Predictions')
    ax2.set_title('Extreme Gradient Boosting Predictions')
    ax3.set_title('Gradient Boosting Predictions')
    ax4.set_title('Decision Tree Predictions')    
   
add_plot(list(predictions.keys()), list(predictions.values()))  

plt.tight_layout()
plt.show()
