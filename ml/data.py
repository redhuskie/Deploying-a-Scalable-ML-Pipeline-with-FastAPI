import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None, scaler=None
):
    # Add skilearn.Standardscaler for feature Scaling for Continuous Variables. 
    # These can be used for models that are sensitive to the scale of the features, such as SVMs or logistic regression. 
    # Probably won't use in this case but good to have in the pipeline.
    
    """
    Process the data used in the machine learning pipeline.

    Applies one-hot encoding to categorical features,
    scales continuous features, and binarizes the target label.

    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame containing features and the label.
    categorical_features : list[str]
        Column names for categorical features.
    label : str
        Column name of the target variable.
    training : bool
        Whether we are in training mode.
    encoder : OneHotEncoder
        Fitted encoder (used when training=False).
    lb : LabelBinarizer
        Fitted label binarizer (used when training=False).
    scaler : StandardScaler
        Fitted scaler (used when training=False).

    Returns
    -------
    X : np.array
        Processed feature array.
    y : np.array
        Processed label array (or empty array if label=None).
    encoder : OneHotEncoder
        Trained encoder.
    lb : LabelBinarizer
        Trained label binarizer.
    scaler : StandardScaler
        Trained scaler.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        scaler = StandardScaler()

        X_categorical = encoder.fit_transform(X_categorical)
        X_continuous = scaler.fit_transform(X_continuous)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        X_continuous = scaler.transform(X_continuous)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb, scaler


def apply_label(inference):
    """Convert the binary label in a single inference sample into string output."""
    if inference[0] == 1:
        return ">50K"
    elif inference[0] == 0:
        return "<=50K"
