import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

friday = pd.read_csv(
    os.path.join(DATASET_DIR, "cleaned_ddos_dataset.csv"),
    nrows=5
)

tuesday = pd.read_csv(
    os.path.join(DATASET_DIR, "cleaned_tuesday_dataset.csv"),
    nrows=5
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "independent_scaler.pkl")
)

features = joblib.load(
    os.path.join(MODEL_DIR, "features.pkl")
)

friday_features = [
    c for c in friday.columns if c != "Label"
]

tuesday_features = [
    c for c in tuesday.columns if c != "Label"
]

print("Friday features:", len(friday_features))
print("Tuesday features:", len(tuesday_features))
print("Model features:", len(features))

print(
    "Friday matches model:",
    friday_features == features
)

print(
    "Tuesday matches model:",
    tuesday_features == features
)

print(
    "Scaler feature names match model:",
    list(scaler.feature_names_in_) == features
)

print("\nFeature compatibility check completed.")