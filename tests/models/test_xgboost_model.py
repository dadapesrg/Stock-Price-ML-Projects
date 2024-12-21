import pytest
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.xgboost_model import invert_transform

@pytest.fixture
def sample_data():
    # Create a sample dataset
    data = np.random.rand(100)
    shape = 4
    column_index = 2
    scaler = MinMaxScaler()
    dummy_data = np.random.rand(100, shape)
    scaler.fit(dummy_data)
    return data, shape, column_index, scaler

def test_invert_transform(sample_data):
    data, shape, column_index, scaler = sample_data
    inverted_data = invert_transform(data, shape, column_index, scaler)
    
    # Check the shape of the output
    assert inverted_data.shape == data.shape
    
    # Check if the inversion is correct
    dummy_array = np.zeros((len(data), shape))
    dummy_array[:, column_index] = data
    expected_data = scaler.inverse_transform(dummy_array)[:, column_index]
    assert np.allclose(inverted_data, expected_data)