from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

from deployment.schema import ChurnRequest

app = FastAPI()

# load model

BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "churn_pipeline.pkl"

model= joblib.load(model_path)

# home route

@app.get("/")
def home():
    return {"message":"Churn prediction API is running"}

# prediction route

@app.post("/predict")
def predict( data: ChurnRequest):
    # convert input to dataframe
    df = pd.DataFrame([data.dict()])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction" : int(prediction),
        "churn probability" : float(probability)
    }
