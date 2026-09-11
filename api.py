from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILE = "model/model.pkl"
PREDICTIONS_FILE = "data/predictions.csv"


@app.get("/")
def home():
    return {
        "message": "PL Predictor API is running"
    }


@app.get("/api/predictions")
def get_predictions():
    df = pd.read_csv(PREDICTIONS_FILE)

    predictions = df[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "Probability_H",
            "Probability_D",
            "Probability_A",
            "Prediction",
        ]
    ]

    return predictions.to_dict(orient="records")