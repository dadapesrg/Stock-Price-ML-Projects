import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.metrics import r2_score
from models.cnn_tensorFlow_model import create_sequences, read_database_data

# FILE: Stock-Price-ML-Projects/models/test_cnn_tensorFlow_model.py


DATABASE_URL = "sqlite:///data/stock_price_data.db"

@pytest.fixture
def data():
    df_data = read_database_data(DATABASE_URL, "stock_prices")
    df = pd.DataFrame(df_data).set_index("Date")
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def test_create_sequences(data):
    scaled_data, _ = data
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == seq_length
    assert X.shape[2] == scaled_data.shape[1]

def test_model_training(data):
    scaled_data, scaler = data
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr=0.0003
    model_cnn = Sequential()
    model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(50, activation='relu'))
    model_cnn.add(Dense(1))
    adam = Adam(lr)
    model_cnn.compile(loss='mse', optimizer=adam)

    epochs = 1  # Use fewer epochs for testing
    batch = 24
    cnn_history = model_cnn.fit(X_train, y_train, epochs=epochs, batch_size=batch, verbose=2)
    
    assert cnn_history.history['loss'][-1] < 1  # Check if the loss is reasonable

def test_model_prediction(data):
    scaled_data, scaler = data
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr=0.0003
    model_cnn = Sequential()
    model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(50, activation='relu'))
    model_cnn.add(Dense(1))
    adam = Adam(lr)
    model_cnn.compile(loss='mse', optimizer=adam)

    epochs = 1  # Use fewer epochs for testing
    batch = 24
    model_cnn.fit(X_train, y_train, epochs=epochs, batch_size=batch, verbose=2)

    cnn_y_pred = model_cnn.predict(X_test)
    
    def invert_transform(data, shape, column_index, scaler):
        dummy_array = np.zeros((len(data), shape))
        dummy_array[:, column_index] = data.flatten()
        return scaler.inverse_transform(dummy_array)[:, column_index]

    actual = invert_transform(y_test, X_train.shape[2], 0, scaler)
    predicted = invert_transform(cnn_y_pred, X_train.shape[2], 0, scaler)
    
    assert len(actual) == len(predicted)
    assert r2_score(actual, predicted) > 0  # Check if the R2 score is positive