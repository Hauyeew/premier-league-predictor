from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model.predict import generate_predictions


app = FastAPI(
    title="PL Predictor API",
    description="Premier League match prediction API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "PL Predictor API is running"
    }


@app.get("/api/predictions")
def get_predictions():
    predictions = generate_predictions()

    return predictions