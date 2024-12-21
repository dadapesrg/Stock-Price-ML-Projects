import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from models.random_forest_model import train_model, make_predictions

# FILE: Stock-Price-ML-Projects/tests/test_random_forest_model.py

@pytest.fixture
def data():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def test_train_model(data):
    X_train, X_test, y_train, y_test = data
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestRegressor)


def test_make_predictions(data):
    X_train, X_test, y_train, y_test = data
    model = train_model(X_train, y_train)
    predictions = make_predictions(model, X_test)
    assert len(predictions) == len(X_test)