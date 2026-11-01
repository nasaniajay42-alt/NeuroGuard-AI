import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "cleaned_tuesday_dataset.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# Load Tuesday dataset
df = pd.read_csv(DATASET_PATH)

# Load Isolation Forest
model = joblib.load(
    os.path.join(MODEL_DIR, "isolation_forest.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "isolation_scaler.pkl")
)

features = joblib.load(
    os.path.join(MODEL_DIR, "isolation_features.pkl")
)

# Prepare features
X = df[features]

X_scaled = scaler.transform(X)

# Isolation Forest returns:
# -1 = anomaly
#  1 = normal

predictions = (
    model.predict(X_scaled) == -1
).astype(int)

# Higher anomaly score = more anomalous
scores = -model.score_samples(X_scaled)

# Ground-truth labels
y = (
    df["Label"].str.strip() != "BENIGN"
).astype(int).to_numpy()

print("\nISOLATION FOREST EVALUATION")

print("\nConfusion Matrix:")
print(confusion_matrix(y, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y,
        predictions,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)

print(
    "\nROC-AUC:",
    roc_auc_score(y, scores)
)

print(
    "PR-AUC:",
    average_precision_score(y, scores)
)

print("\nATTACK FAMILY DETECTION")

for label in ["FTP-Patator", "SSH-Patator"]:

    mask = (
        df["Label"].str.strip() == label
    ).to_numpy()

    detected = predictions[mask].sum()

    print(
        f"{label}: {detected}/{mask.sum()} "
        f"({detected / mask.sum() * 100:.2f}%)"
    )

print("\nIsolation Forest evaluation completed.")
