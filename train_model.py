import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Load Census Data
project_path = os.getcwd()  # or hardcode your path if needed
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Split the data into train and test sets
# Note: The test size is set to 0.2 (20% of the data) for testing purposes.
train, test = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["salary"]
)

# Categorical feature columns
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

#  Process the training data
X_train, y_train, encoder, lb, scaler = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data
X_test, y_test, _, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
    scaler=scaler
)

# Train the model
model = train_model(X_train, y_train)

# Save the model, encoder, and scaler
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)

save_model(model, os.path.join(model_dir, "model.pkl"))
save_model(encoder, os.path.join(model_dir, "encoder.pkl"))
save_model(scaler, os.path.join(model_dir, "scaler.pkl"))

# Inference and metrics
preds = inference(model, X_test)
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Evaluate model performance on slices
with open("slice_output.txt", "w") as f:
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
                data=test,
                column_name=col,
                slice_value=slicevalue,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                scaler=scaler,
                model=model
            )
            f.write(f"{col}: {slicevalue}, Count: {count:,}\n")
            f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n\n")
