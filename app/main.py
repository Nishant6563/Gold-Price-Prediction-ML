from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import FEATURES, MODEL_PATH
from src.train import train

model = None


class MarketFeatures(BaseModel):
    sp500: float = Field(gt=0)
    oil_usd: float = Field(gt=0)
    silver_usd: float = Field(gt=0)
    usd_inr: float = Field(gt=0)
    interest_rate: float = Field(ge=0, le=30)
    inflation: float = Field(ge=-10, le=50)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    if not MODEL_PATH.exists():
        train()
    model = joblib.load(MODEL_PATH)
    yield
    model = None


app = FastAPI(
    title="Gold Price Prediction API",
    version="1.0.0",
    description="Educational regression API; predictions are not financial advice.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "ready" if model is not None else "loading"}


@app.post("/predict")
def predict(payload: MarketFeatures) -> dict[str, float | str]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    frame = pd.DataFrame([[getattr(payload, feature) for feature in FEATURES]], columns=FEATURES)
    value = max(0.0, float(model.predict(frame)[0]))
    return {
        "predicted_gold_usd": round(value, 2),
        "unit": "USD per troy ounce (educational estimate)",
    }

