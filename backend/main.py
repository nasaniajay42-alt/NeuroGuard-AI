
import os
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# =====================================================
# PATH CONFIGURATION
# =====================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_DIR = os.path.join(
    ROOT, "backend", "saved_models"
)

OUTPUT_DIR = os.path.join(
    ROOT, "research", "evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, "latest_dashboard_predictions.csv"
)

# =====================================================
# AURA CONFIGURATION
# =====================================================

XGB_THRESHOLD = 0.30
RF_THRESHOLD = 0.018

# =====================================================
# LOAD TRAINED MODELS
# =====================================================

print("Loading NeuroGuard AI models...")

try:
    xgb_model = joblib.load(
        os.path.join(MODEL_DIR, "xgboost_classifier.pkl")
    )

    rf_model = joblib.load(
        os.path.join(MODEL_DIR, "adaptive_random_forest.pkl")
    )

    xgb_features = joblib.load(
        os.path.join(MODEL_DIR, "xgboost_features.pkl")
    )

    rf_features = joblib.load(
        os.path.join(MODEL_DIR, "adaptive_features.pkl")
    )

except Exception as error:
    raise RuntimeError(
        f"Unable to load trained models: {error}"
    )

print("XGBoost loaded successfully.")
print("Random Forest loaded successfully.")
print("AURA prediction engine ready.")

# =====================================================
# FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="NeuroGuard AI",
    description="AURA Adaptive Network Threat Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
def home():
    return {
        "project": "NeuroGuard AI",
        "architecture": "AURA",
        "status": "Online",
        "models": [
            "XGBoost",
            "Random Forest"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "aura_engine": "ready"
    }

# =====================================================
# CSV PREDICTION API
# =====================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        df = pd.read_csv(
            __import__("io").BytesIO(contents),
            low_memory=False
        )

        if len(df) == 0:
            raise HTTPException(
                status_code=400,
                detail="CSV contains no records."
            )

        # Remove spaces from column names
        df.columns = df.columns.astype(str).str.strip()

        # Validate required features
        missing_xgb = [
            f for f in xgb_features
            if f not in df.columns
        ]

        missing_rf = [
            f for f in rf_features
            if f not in df.columns
        ]

        if missing_xgb or missing_rf:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Required model features are missing.",
                    "missing_xgboost_features": missing_xgb,
                    "missing_random_forest_features": missing_rf
                }
            )

        # Prepare model inputs
        X_xgb = df[xgb_features].apply(
            pd.to_numeric, errors="coerce"
        ).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

        X_rf = df[rf_features].apply(
            pd.to_numeric, errors="coerce"
        ).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

        # Generate model probabilities
        xgb_prob = xgb_model.predict_proba(
            X_xgb
        )[:, 1]

        rf_prob = rf_model.predict_proba(
            X_rf
        )[:, 1]

        # Individual predictions
        xgb_pred = (
            xgb_prob >= XGB_THRESHOLD
        ).astype(int)

        rf_pred = (
            rf_prob >= RF_THRESHOLD
        ).astype(int)

        # AURA OR fusion
        aura_pred = (
            (xgb_pred == 1) |
            (rf_pred == 1)
        ).astype(int)

        # Risk scores for display
        aura_score = np.maximum(
            xgb_prob / XGB_THRESHOLD,
            rf_prob / RF_THRESHOLD
        )

        # Prepare output
        results = pd.DataFrame({
            "XGB_Probability": xgb_prob,
            "RF_Probability": rf_prob,
            "XGB_Prediction": xgb_pred,
            "RF_Prediction": rf_pred,
            "AURA_Prediction": aura_pred,
            "AURA_Score": aura_score,
            "Classification": np.where(
                aura_pred == 1,
                "SUSPICIOUS",
                "BENIGN"
            ),
            "Risk_Level": np.where(
                aura_pred == 1,
                np.where(
                    aura_score >= 5,
                    "HIGH",
                    "MEDIUM"
                ),
                "LOW"
            )
        })

        # Preserve original labels if supplied
        if "Label" in df.columns:
            results.insert(
                0,
                "Actual_Label",
                df["Label"].astype(str).values
            )

        results.to_csv(
            OUTPUT_PATH,
            index=False
        )

        total = len(results)

        attack_count = int(aura_pred.sum())
        benign_count = total - attack_count

        return {
            "project": "NeuroGuard AI",
            "engine": "AURA",
            "status": "success",
            "total_records": total,
            "suspicious_records": attack_count,
            "benign_records": benign_count,
            "suspicious_percentage": round(
                attack_count / total * 100, 2
            ),
            "xgboost_threshold": XGB_THRESHOLD,
            "random_forest_threshold": RF_THRESHOLD,
            "models": {
                "xgboost": "active",
                "random_forest": "active"
            },
            "download_endpoint": "/download-results",
            "sample_results": results.head(10).to_dict(
                orient="records"
            )
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )

# =====================================================
# DOWNLOAD PREDICTION RESULTS
# =====================================================

@app.get("/download-results")
def download_results():

    if not os.path.exists(OUTPUT_PATH):
        raise HTTPException(
            status_code=404,
            detail="No prediction results available."
        )

    return FileResponse(
        OUTPUT_PATH,
        media_type="text/csv",
        filename="neuroguard_aura_predictions.csv"
    )
