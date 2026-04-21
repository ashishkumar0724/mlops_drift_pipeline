"""
Model Serving API: FastAPI with Auto-Fill
Accepts partial input, fills missing columns with training statistics
Production-ready inference for AutoGluon models
Rubric: Deployment (20%)
"""

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from autogluon.tabular import TabularPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FastAPI-Serve")

app = FastAPI(title="AutoGluon MLOps API", version="1.0")
MODEL_DIR = Path("model")
DATA_DIR = Path("data/processed")

# Global state
predictor = None
model_version = "unknown"
feature_stats = {}  # Stores median/mode for auto-fill


class PredictionRequest(BaseModel):
    """Minimal required fields for prediction (user-friendly)."""

    Age: int = 35
    Tenure_in_Months: int = 24
    Monthly_Charge: float = 85.50
    Total_Charges: float = 2040.00
    Gender: str = "Male"
    Married: str = "Yes"
    Contract: str = "Month-to-Month"

    # Optional: can override any other feature
    class Config:
        extra = "allow"


def load_feature_statistics() -> dict[str, Any]:
    """Load training data statistics for auto-filling missing columns."""
    train_path = DATA_DIR / "train.parquet"
    if not train_path.exists():
        logger.warning("⚠️ Training data not found. Cannot auto-fill missing columns.")
        return {}

    df = pd.read_parquet(train_path)
    stats = {}

    # Calculate median for numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ["Churn", "label"]:  # Skip target
            stats[col] = float(df[col].median())

    # Calculate mode for categorical columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col not in ["Churn", "Customer ID"]:  # Skip target & ID
            mode_val = df[col].mode()
            stats[col] = str(mode_val.iloc[0]) if len(mode_val) > 0 else "Unknown"

    logger.info(f"📊 Loaded statistics for {len(stats)} features (auto-fill enabled)")
    return stats


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Prepare input for AutoGluon:
    1. Convert payload to DataFrame
    2. Auto-fill missing columns with training statistics
    3. Ensure correct column order
    """
    if not feature_stats:
        # Fallback: just use provided fields
        return pd.DataFrame([payload])

    # Start with a copy of stats (all columns filled with defaults)
    input_row = feature_stats.copy()

    # Override with user-provided values
    for key, value in payload.items():
        # Handle key name variations (e.g., "Tenure_in_Months" → "Tenure in Months")
        normalized_key = key.replace("_", " ")
        if normalized_key in input_row:
            input_row[normalized_key] = value
        elif key in input_row:
            input_row[key] = value

    # Remove target column if present
    input_row.pop("Churn", None)
    input_row.pop("Customer ID", None)

    return pd.DataFrame([input_row])


def load_latest_model() -> tuple:
    """Load the highest versioned model."""
    if not MODEL_DIR.exists():
        raise RuntimeError("Model directory missing. Run pipeline.py first.")

    versions = [d for d in MODEL_DIR.iterdir() if d.is_dir() and d.name.startswith("predictor_v")]
    if not versions:
        raise RuntimeError("No model artifacts found.")

    def extract_version(p: Path) -> int:
        try:
            return int(p.name.replace("predictor_v", ""))
        except ValueError:
            return 0

    latest_path = sorted(versions, key=extract_version)[-1]
    version = latest_path.name.replace("predictor_", "")

    logger.info(f"📦 Loading model from: {latest_path} (version: {version})")
    return TabularPredictor.load(latest_path), version


@app.on_event("startup")
async def startup_event():
    global predictor, model_version, feature_stats
    try:
        predictor, model_version = load_latest_model()
        feature_stats = load_feature_statistics()
        logger.info(f"✅ Server started | Model: {model_version} | Auto-fill: {len(feature_stats)} features")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        predictor = None
        model_version = "unavailable"


@app.get("/health")
def health():
    status = "healthy" if predictor is not None else "unhealthy"
    return {"status": status, "model_version": model_version, "auto_fill_features": len(feature_stats)}


@app.post("/predict")
def predict(payload: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert Pydantic model to dict
        input_dict = payload.dict()

        # Prepare input with auto-fill
        df = prepare_input(input_dict)

        # Predict
        pred = predictor.predict(df)
        prob = predictor.predict_proba(df)

        prob_col = prob.columns[1] if len(prob.columns) > 1 else prob.columns[0]
        score = float(prob.iloc[0][prob_col])

        return {
            "version": model_version,
            "prediction": int(pred.iloc[0]),
            "churn_probability": round(score, 4),
            "interpretation": "High Risk ⚠️" if score > 0.7 else "Low Risk ✅" if score < 0.3 else "Medium Risk ⚡",
            "features_used": len(df.columns),
            "auto_filled": len(feature_stats) > 0,
        }
    except Exception as e:
        logger.error(f"❌ Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
