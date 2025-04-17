import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model
import os

os.makedirs("model", exist_ok=True)
# DO NOT MODIFY
class Data(BaseModel):
    
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# Path to model and encoder, and scaler. 
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to ensure models are loaded safely and exist. 
def safe_load_model(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at {path}. Please ensure it is saved or trained first.")
    print(f"Loaded {label} from {path}")
    return load_model(path)

# Use function to load models. 
encoder = safe_load_model(os.path.join(MODEL_DIR, "encoder.joblib"), "Encoder")
scaler = safe_load_model(os.path.join(MODEL_DIR, "scaler.joblib"), "Scaler")
model = safe_load_model(os.path.join(MODEL_DIR, "model.joblib"), "Model")

# Create the FastAPI app
app = FastAPI()

# TODO: create a GET on the root giving a welcome message
@app.get("/")
 # Welcome message    
async def get_root():
    return {"message": "Welcome to the census Income API!"}


# TODO: create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

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
    data_processed, _, _, _, _= process_data(
        data,
        categorical_features=cat_features,
        encoder=encoder,
        lb=encoder,
        scaler=scaler,
        training=False,
    )
    # Run inference and return the result. 
    _inference = inference(model, data_processed)
    return {"result": apply_label(_inference)}
