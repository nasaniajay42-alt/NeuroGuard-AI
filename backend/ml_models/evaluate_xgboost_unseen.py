import os
import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# Load Tuesday dataset
df = pd.read_csv(
    os.path.join(DATASET_DIR, "cleaned_tuesday_dataset.csv")
)

# Load existing XGBoost model
model = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_classifier.pkl")
)

features = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_features.pkl")
)

# Prepare features
X = df[features]

# Ground-truth labels
y = (
    df["Label"].str.strip() != "BENIGN"
).astype(int)

# Predict attack probabilities
probabilities = model.predict_proba(X)[:, 1]

# Use the default 0.5 threshold
predictions = (probabilities >= 0.5).astype(int)

print("\nXGBOOST UNSEEN ATTACK EVALUATION")

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
    "ROC-AUC:",
    roc_auc_score(y, probabilities)
)

print(
    "PR-AUC:",
    average_precision_score(y, probabilities)
)

print("\nATTACK FAMILY DETECTION")

for label in ["FTP-Patator", "SSH-Patator"]:

    mask = (
        df["Label"].str.strip() == label
    )

    detected = predictions[mask].sum()

    print(
        f"{label}: {detected}/{mask.sum()} "
        f"({detected / mask.sum() * 100:.2f}%)"
    )

print("\nXGBoost evaluation completed.")