"""
deploy.py - FastAPI deployment for Breast Cancer Classification Model
Run with: uvicorn deploy:app --reload
"""

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ── Rebuild model architecture & load weights ──────────────────────────────────
def build_model():
    m = Sequential([
        Dense(30, activation='relu'),
        Dropout(0.5),
        Dense(15, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.build(input_shape=(None, 30))
    return m

model = build_model()

with open("model_weights.pkl", "rb") as f:
    weights = pickle.load(f)

model.set_weights([np.array(w) for w in weights])

with open("scaler_weights.pkl", "rb") as f:
    scaler = pickle.load(f)

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Breast Cancer Classification API",
    description="Predicts whether a tumor is Benign (0) or Malignant (1).",
    version="1.0.0"
)

class TumorFeatures(BaseModel):
    features: List[float]

class PredictionResult(BaseModel):
    prediction: int
    label: str
    probability: float

@app.get("/")
def root():
    return {"message": "Breast Cancer Classification API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict", response_model=PredictionResult)
def predict(data: TumorFeatures):
    if len(data.features) != 30:
        raise HTTPException(status_code=422, detail=f"Expected 30 features, got {len(data.features)}")

    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prob = float(model.predict(X_scaled, verbose=0)[0][0])
    pred = int(prob > 0.5)
    label = "Malignant" if pred == 1 else "Benign"

    return PredictionResult(prediction=pred, label=label, probability=round(prob, 4))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
