import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

friday_path = os.path.join(
    DATASET_DIR, "cleaned_ddos_dataset.csv"
)

tuesday_path = os.path.join(
    DATASET_DIR, "cleaned_tuesday_dataset.csv"
)

# Load a sample from each dataset
friday = pd.read_csv(friday_path)
tuesday = pd.read_csv(tuesday_path)

features = joblib.load(
    os.path.join(MODEL_DIR, "features.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "independent_scaler.pkl")
)

# Use benign traffic only for this comparison
friday_benign = friday[
    friday["Label"].str.strip() == "BENIGN"
][features]

tuesday_benign = tuesday[
    tuesday["Label"].str.strip() == "BENIGN"
][features]

# Sample for faster analysis
friday_benign = friday_benign.sample(
    n=min(10000, len(friday_benign)),
    random_state=42
)

tuesday_benign = tuesday_benign.sample(
    n=min(10000, len(tuesday_benign)),
    random_state=42
)

# Transform while retaining feature names
friday_scaled = pd.DataFrame(
    scaler.transform(friday_benign),
    columns=features
)

tuesday_scaled = pd.DataFrame(
    scaler.transform(tuesday_benign),
    columns=features
)

# Compare the 99th percentile of absolute scaled values
friday_q99 = friday_scaled.abs().quantile(0.99)
tuesday_q99 = tuesday_scaled.abs().quantile(0.99)

comparison = pd.DataFrame({
    "Friday_BENIGN_Q99": friday_q99,
    "Tuesday_BENIGN_Q99": tuesday_q99
})

comparison["Shift_Ratio"] = (
    comparison["Tuesday_BENIGN_Q99"] /
    (comparison["Friday_BENIGN_Q99"] + 1e-9)
)

comparison = comparison.replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

comparison = comparison.sort_values(
    "Shift_Ratio", ascending=False
)

print("\nTOP 20 FEATURES WITH DISTRIBUTION SHIFT")
print(comparison.head(20).to_string())

print("\nMAXIMUM ABSOLUTE SCALED VALUE")
print("Friday:", friday_scaled.abs().max().max())
print("Tuesday:", tuesday_scaled.abs().max().max())

print("\nAnalysis completed.")