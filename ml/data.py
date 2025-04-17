import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

def process_data(
        X, categorical_features=[], label=None, training=True, encoder=None, lb=None, scaler=None
    ):
        # Add skilearn.Standardscaler for feature Scaling for Continuous Variables. 
        # These can be used for models that are sensitive to the scale of the features, such as SVMs or logistic regression. 
        # Probably won't use in this case but good to have in the pipeline.
        
       ## preprocess the data

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

        # ✅ Only transform continuous features if scaler is provided
        if scaler:
            X_continuous = scaler.transform(X_continuous)
        else:
            X_continuous = X_continuous.values  # convert to NumPy if needed

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
