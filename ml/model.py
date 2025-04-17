import pickle
import pandas as pd
import joblib
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# TODO: Import census.csv
df = pd.read_csv('data/census.csv')

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # TODO: implement the function
    # Added Hyperparameter tuning using GridSearchCV
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50,100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    
    grid_search = GridSearchCV(
                               estimator=rf, 
                               param_grid=param_grid, 
                               cv=3, 
                               n_jobs=-1, 
                               verbose=2)
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.R
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # TODO: implement the function
    # Add preds = model.predict(X)
    preds = model.predict(X)
    return preds

def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    # TODO: implement the function
    # Saves the model. 
    joblib.dump(model, path)

def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    # TODO: implement the function
    # Returns the model.
    return joblib.load(path)

def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, scaler, model
):
    """
    Computes precision, recall, and fbeta for a slice of the data.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset.
    column_name : str
        Column to slice on.
    slice_value : str, int, float
        Value in the column to filter by.
    categorical_features : list
        Categorical columns used in processing.
    label : str
        Name of the label column.
    encoder : OneHotEncoder
        Fitted encoder from training.
    lb : LabelBinarizer
        Fitted label binarizer from training.
    scaler : StandardScaler
        Fitted scaler from training.
    model : sklearn model
        Trained model.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    # Filter the data to just the slice
    data_slice = data[data[column_name] == slice_value]

    # Return None if the slice has no data
    if data_slice.empty:
        return None, None, None

    # Process the sliced data — must unpack all 5 return values
    X_slice, y_slice, _, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
        scaler=scaler
    )

    # Run inference and compute metrics
    preds = inference(model, X_slice)
    precision = precision_score(y_slice, preds)
    recall = recall_score(y_slice, preds)
    fbeta = fbeta_score(y_slice, preds, beta=1)

    return precision, recall, fbeta
