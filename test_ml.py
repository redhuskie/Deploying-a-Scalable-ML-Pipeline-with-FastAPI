import pytest
import pandas as pd
import numpy as np
from ml.model import train_model, inference
from ml.data import process_data
from ml.model import performance_on_categorical_slice  # if you put the slice code there
from sklearn.ensemble import RandomForestClassifier

# Test 1
def test_one():
    """
    Test that train_model returns a RandomForestClassifier with a predict method.
    """
    # Your code here
    # Create dummy data
    X = np.array([
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [1, 1],
    [0, 1]
])
    y = np.array([0, 1, 0, 1, 0, 1])

    # Train a model
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"
    assert hasattr(model, "predict"), "Model does not have a predict method"

# Test Two
def test_two():
    """
    # Test to ensure inference returns predictions matching input length.
    """
    # Your code here
    # have to have at least three samples so I'm padding. 
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]])
    y = np.array([0, 1, 0, 1, 0, 1])
    
    # Train a model
    model = train_model(X, y)
    # Perform inference
    preds = inference(model, X)
    
    assert len(preds) == len(X), "Predictions length does not match input data length"
# Test 3
def test_three():
    """
    Test that performance_on_categorical_slice returns valid precision, recall, and fbeta values.
    """
    
    df = pd.DataFrame({
        "education": ["Bachelors", "Masters", "Bachelors", "PhD", "Masters", "Bachelors"],
        "age": [25, 35, 45, 55, 30, 40],
        "income": [">50K", "<=50K", ">50K", "<=50K", ">50K", "<=50K"]
    })

    categorical_features = ["education"]
    label = "income"
    
    # Unpack all 5 return values from process_data
    X, y, encoder, lb, scaler = process_data(
        df,
        categorical_features=categorical_features,
        label=label,
        training=True,
    )

    model = train_model(X, y)

    # Run slice-based evaluation
    precision, recall, fbeta = performance_on_categorical_slice(
        df,
        column_name="education",
        slice_value="Bachelors",
        categorical_features=categorical_features,
        label=label,
        encoder=encoder,
        lb=lb,
        scaler=scaler,
        model=model
    )

    # Check that all metrics are returned
    assert precision is not None, "Precision is None"
    assert recall is not None, "Recall is None"
    assert fbeta is not None, "F-beta is None"