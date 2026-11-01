
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    ROOT, "backend", "datasets", "cleaned_tuesday_dataset.csv"
)

MODEL_DIR = os.path.join(ROOT, "backend", "saved_models")

OUTPUT_DIR = os.path.join(ROOT, "research", "experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["Label"] = df["Label"].astype(str).str.strip()

y = (df["Label"] != "BENIGN").astype(int).values

print("DATASET SHAPE:", df.shape)
print("\nLABEL DISTRIBUTION:")
print(df["Label"].value_counts())

# --------------------------------------------------
# 3. LOAD AUTOENCODER
# --------------------------------------------------

features = joblib.load(
    os.path.join(MODEL_DIR, "robust_features.pkl")
)

transformer = joblib.load(
    os.path.join(MODEL_DIR, "robust_transformer.pkl")
)

X = df[features].replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

X_scaled = transformer.transform(X)
X_scaled = np.clip(X_scaled, -5, 5)


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.network(x)


ae_model = Autoencoder(len(features)).to(device)

ae_model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "robust_autoencoder.pth"),
        map_location=device,
        weights_only=True
    )
)

ae_model.eval()

X_tensor = torch.tensor(
    X_scaled, dtype=torch.float32
).to(device)

with torch.no_grad():
    reconstructed = ae_model(X_tensor)

    ae_errors = torch.mean(
        (X_tensor - reconstructed) ** 2,
        dim=1
    ).cpu().numpy()

ae_threshold = np.percentile(ae_errors[y == 0], 95)
ae_predictions = (ae_errors > ae_threshold).astype(int)

# --------------------------------------------------
# 4. LOAD XGBOOST
# --------------------------------------------------

xgb_model = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_classifier.pkl")
)

xgb_features = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_features.pkl")
)

X_xgb = df[xgb_features].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

xgb_probabilities = xgb_model.predict_proba(X_xgb)[:, 1]

xgb_predictions = (
    xgb_probabilities >= 0.05
).astype(int)

# --------------------------------------------------
# 5. AURA OR FUSION
# --------------------------------------------------

aura_predictions = (
    (ae_predictions == 1) |
    (xgb_predictions == 1)
).astype(int)

# --------------------------------------------------
# 6. CLASSIFY EACH RECORD
# --------------------------------------------------

conditions = [
    (ae_predictions == 0) & (xgb_predictions == 0),
    (ae_predictions == 1) & (xgb_predictions == 0),
    (ae_predictions == 0) & (xgb_predictions == 1),
    (ae_predictions == 1) & (xgb_predictions == 1)
]

categories = [
    "Neither detected",
    "Autoencoder only",
    "XGBoost only",
    "Both detected"
]

df["AE_Error"] = ae_errors
df["XGB_Probability"] = xgb_probabilities
df["AE_Prediction"] = ae_predictions
df["XGB_Prediction"] = xgb_predictions
df["AURA_Prediction"] = aura_predictions

df["Detection_Category"] = np.select(
    conditions,
    categories,
    default="Unknown"
)

# --------------------------------------------------
# 7. ATTACK FAMILY ERROR ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 60)
print("ATTACK FAMILY ANALYSIS")
print("=" * 60)

for attack in df["Label"].unique():
    if attack == "BENIGN":
        continue

    subset = df[df["Label"] == attack]

    print(f"\nATTACK: {attack}")
    print("Total records:", len(subset))
    print("\nDetection categories:")
    print(subset["Detection_Category"].value_counts())

    print(
        "AURA detected:",
        int(subset["AURA_Prediction"].sum())
    )

    print(
        "AURA recall:",
        round(subset["AURA_Prediction"].mean() * 100, 2),
        "%"
    )

# --------------------------------------------------
# 8. FALSE POSITIVE ANALYSIS
# --------------------------------------------------

benign = df[df["Label"] == "BENIGN"]

print("\n" + "=" * 60)
print("BENIGN FALSE POSITIVE ANALYSIS")
print("=" * 60)

print("\nBenign detection categories:")
print(benign["Detection_Category"].value_counts())

print(
    "\nAutoencoder benign false positives:",
    int(benign["AE_Prediction"].sum())
)

print(
    "XGBoost benign false positives:",
    int(benign["XGB_Prediction"].sum())
)

print(
    "AURA benign false positives:",
    int(benign["AURA_Prediction"].sum())
)

# --------------------------------------------------
# 9. TOP FALSE POSITIVE FEATURES
# --------------------------------------------------

false_positives = benign[
    benign["AURA_Prediction"] == 1
]

selected_features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Flow IAT Mean",
    "Destination Port"
]

available_features = [
    feature for feature in selected_features
    if feature in false_positives.columns
]

print("\nFALSE POSITIVE FEATURE MEDIANS:")
print(false_positives[available_features].median())

print("\nBENIGN FEATURE MEDIANS:")
print(benign[available_features].median())

# --------------------------------------------------
# 10. SAVE DETAILED RESULTS
# --------------------------------------------------

output_path = os.path.join(
    OUTPUT_DIR,
    "aura_error_analysis.csv"
)

df.to_csv(output_path, index=False)

print("\nDetailed error analysis saved to:")
print(output_path)

print("\nAURA ERROR ANALYSIS COMPLETED")
