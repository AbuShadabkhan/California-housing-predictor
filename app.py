import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# ── Load model & pipeline once at startup ────────────────
MODEL_FILE    = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

try:
    model    = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
except FileNotFoundError:
    raise RuntimeError("model.pkl or pipeline.pkl not found. Run main.py first to train.")

app = FastAPI(
    title       = "California Housing Price Predictor",
    description = "Predicts median house value using a tuned RandomForest model.",
    version     = "1.0.0"
)

# ── Input schema ─────────────────────────────────────────
class HouseFeatures(BaseModel):
    longitude           : float = Field(..., example=-122.23)
    latitude            : float = Field(..., example=37.88)
    housing_median_age  : float = Field(..., example=41.0)
    total_rooms         : float = Field(..., example=880.0)
    total_bedrooms      : float = Field(..., example=129.0)
    population          : float = Field(..., example=322.0)
    households          : float = Field(..., example=126.0)
    median_income       : float = Field(..., example=8.3252)
    ocean_proximity     : str   = Field(..., example="NEAR BAY")

# ── Output schema ────────────────────────────────────────
class PredictionResult(BaseModel):
    predicted_house_value : float
    model_version         : str = "RandomForest-tuned-v1"

# ── Routes ───────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message" : "Housing Price Predictor API is running!",
        "docs"    : "Visit /docs for interactive Swagger UI",
        "predict" : "POST /predict with house features"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResult)
def predict(features: HouseFeatures):
    try:
        # Convert input to DataFrame (pipeline expects this)
        input_df = pd.DataFrame([features.model_dump()])

        # Validate ocean_proximity value
        valid_categories = ["NEAR BAY", "INLAND", "NEAR OCEAN", "<1H OCEAN", "ISLAND"]
        if features.ocean_proximity not in valid_categories:
            raise HTTPException(
                status_code = 422,
                detail      = f"ocean_proximity must be one of: {valid_categories}"
            )

        # Transform + predict
        transformed  = pipeline.transform(input_df)
        prediction   = model.predict(transformed)[0]

        # Cap at dataset range
        prediction = float(np.clip(prediction, 14999, 500001))

        return PredictionResult(predicted_house_value=round(prediction, 2))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ── Run directly ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)