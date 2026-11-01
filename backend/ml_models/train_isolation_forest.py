import os
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "cleaned_ddos_dataset.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading Friday dataset...")

df = pd.read_csv(DATASET_PATH)

features = [
    column for column in df.columns
    if column != "Label"
]

# Train only on BENIGN traffic
benign = df[
    df["Label"].str.strip() == "BENIGN"
]

X = benign[features]

# Fit scaler using BENIGN training data
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print("Training records:", len(X_scaled))
print("Features:", len(features))

# Train Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)

# Save model and scaler
joblib.dump(
    model,
    os.path.join(MODEL_DIR, "isolation_forest.pkl")
)

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "isolation_scaler.pkl")
)

joblib.dump(
    features,
    os.path.join(MODEL_DIR, "isolation_features.pkl")
)

print("\nIsolation Forest training completed!")
print("Training data: BENIGN traffic only")
print("Saved model: isolation_forest.pkl")