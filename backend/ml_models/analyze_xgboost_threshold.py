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

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "saved_models",
        "xgboost_classifier.pkl"
    )
)

features = joblib.load(
    os.path.join(
        BASE_DIR,
        "saved_models",
        "xgboost_features.pkl"
    )
)

X = df[features]

y = (
    df["Label"].str.strip() != "BENIGN"
).astype(int).to_numpy()

probabilities = model.predict_proba(X)[:, 1]

print("\nXGBOOST PROBABILITY DISTRIBUTION")

for label in ["BENIGN", "FTP-Patator", "SSH-Patator"]:

    mask = (
        df["Label"].str.strip() == label
    ).to_numpy()

    values = probabilities[mask]

    print(f"\n{label}")
    print("Mean:", np.mean(values))
    print("Median:", np.median(values))
    print("95th percentile:", np.percentile(values, 95))
    print("99th percentile:", np.percentile(values, 99))
    print("Maximum:", np.max(values))

print("\nTHRESHOLD ANALYSIS")

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

for threshold in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print(
        f"{threshold:<12.2f}"
        f"{precision_score(y, predictions, zero_division=0):<12.4f}"
        f"{recall_score(y, predictions, zero_division=0):<12.4f}"
        f"{f1_score(y, predictions, zero_division=0):<12.4f}"
    )

print("\nAttack family detection at different thresholds")

for label in ["FTP-Patator", "SSH-Patator"]:

    mask = (
        df["Label"].str.strip() == label
    ).to_numpy()

    print(f"\n{label}")

    for threshold in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:

        detection = (
            probabilities[mask] >= threshold
        ).mean()

        print(
            f"Threshold {threshold:.2f}: "
            f"{detection * 100:.2f}%"
        )

print("\nAnalysis completed.")