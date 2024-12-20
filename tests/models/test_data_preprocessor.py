import pytest
import numpy as np
from  models.data_preprocessor import prepare_dataset

@pytest.fixture
def sample_data():
    data = np.array([i for i in range(100)])
    steps = 5
    output_steps = 1
    return data, steps, output_steps

def test_prepare_dataset(sample_data):
    data, steps, output_steps = sample_data
    X, y = prepare_dataset(data, steps, output_steps)
    
    assert len(X) == len(y)
    assert X.shape[1] == steps + output_steps
    assert y.shape[0] == len(data) - steps - output_steps
    assert np.array_equal(X[0], data[:steps + output_steps])
    assert y[0] == data[steps + output_steps]

def test_prepare_dataset_empty():
    data = np.array([])
    steps = 5
    output_steps = 1
    X, y = prepare_dataset(data, steps, output_steps)
    
    assert X.size == 0
    assert y.size == 0

def test_prepare_dataset_single_step():
    data = np.array([i for i in range(10)])
    steps = 1
    output_steps = 1
    X, y = prepare_dataset(data, steps, output_steps)
    
    assert len(X) == len(y)
    assert X.shape[1] == steps + output_steps
    assert y.shape[0] == len(data) - steps - output_steps
    assert np.array_equal(X[0], data[:steps + output_steps])
    assert y[0] == data[steps + output_steps]