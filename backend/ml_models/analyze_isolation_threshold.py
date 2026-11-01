import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "datasets",
        "cleaned_tuesday_dataset.csv"
    )
)

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

model = joblib.load(
    os.path.join(MODEL_DIR, "isolation_forest.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "isolation_scaler.pkl")
)

features = joblib.load(
    os.path.join(MODEL_DIR, "isolation_features.pkl")
)

X = df[features]
X_scaled = scaler.transform(X)

# Continuous anomaly scores
scores = -model.score_samples(X_scaled)

y = (
    df["Label"].str.strip() != "BENIGN"
).astype(int).to_numpy()

print("\nISOLATION FOREST THRESHOLD ANALYSIS")

print(
    f"{'Percentile':<12}"
    f"{'Threshold':<14}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FPR':<12}"
)

for percentile in [90, 95, 97, 98, 99]:

    threshold = np.percentile(
        scores[y == 0],
        percentile
    )

    predictions = (
        scores > threshold
    ).astype(int)

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    f1 = f1_score(
        y, predictions, zero_division=0
    )

    benign_mask = y == 0

    fpr = predictions[benign_mask].mean()

    print(
        f"{percentile:<12}"
        f"{threshold:<14.5f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fpr:<12.4f}"
    )

print("\nATTACK FAMILY ANALYSIS")

threshold = np.percentile(scores[y == 0], 95)

predictions = (scores > threshold).astype(int)

for label in ["FTP-Patator", "SSH-Patator"]:

    mask = (
        df["Label"].str.strip() == label
    ).to_numpy()

    detected = predictions[mask].sum()

    print(
        f"{label}: {detected}/{mask.sum()} "
        f"({detected / mask.sum() * 100:.2f}%)"
    )

print("\nAnalysis completed.")