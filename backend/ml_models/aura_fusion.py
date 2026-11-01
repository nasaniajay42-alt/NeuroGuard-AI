
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)

# --------------------------------------------------
# 1. PATHS AND DEVICE
# --------------------------------------------------

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    ROOT, "backend", "datasets", "cleaned_tuesday_dataset.csv"
)

MODEL_DIR = os.path.join(ROOT, "backend", "saved_models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("AURA FUSION EXPERIMENT")
print("Device:", device)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["Label"] = df["Label"].astype(str).str.strip()

y = (df["Label"] != "BENIGN").astype(int).values

# --------------------------------------------------
# 3. LOAD AUTOENCODER COMPONENTS
# --------------------------------------------------

features = joblib.load(
    os.path.join(MODEL_DIR, "robust_features.pkl")
)

transformer = joblib.load(
    os.path.join(MODEL_DIR, "robust_transformer.pkl")
)

ae_threshold = 0.0  # Set below using Tuesday BENIGN scores

X = df[features].replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

X_scaled = transformer.transform(X)
X_scaled = np.clip(X_scaled, -5, 5)

# --------------------------------------------------
# 4. DEFINE AUTOENCODER
# --------------------------------------------------

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

ae_state = torch.load(
    os.path.join(MODEL_DIR, "robust_autoencoder.pth"),
    map_location=device,
    weights_only=True
)

ae_model.load_state_dict(ae_state)
ae_model.eval()

# --------------------------------------------------
# 5. CALCULATE AUTOENCODER ERRORS
# --------------------------------------------------

X_tensor = torch.tensor(
    X_scaled, dtype=torch.float32
).to(device)

with torch.no_grad():
    reconstructed = ae_model(X_tensor)

    ae_errors = torch.mean(
        (X_tensor - reconstructed) ** 2,
        dim=1
    ).cpu().numpy()

# Exploratory threshold: 95th percentile of Tuesday BENIGN.
# This is for experimentation, not unbiased final evaluation.
benign_errors = ae_errors[y == 0]
ae_threshold = np.percentile(benign_errors, 95)

ae_predictions = (ae_errors > ae_threshold).astype(int)

print("\nAutoencoder threshold:", ae_threshold)

# --------------------------------------------------
# 6. LOAD XGBOOST MODEL
# --------------------------------------------------

xgb_model = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_classifier.pkl")
)

xgb_features = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_features.pkl")
)

X_xgb = df[xgb_features].replace([np.inf, -np.inf], np.nan)
X_xgb = X_xgb.fillna(0)

xgb_probabilities = xgb_model.predict_proba(X_xgb)[:, 1]

xgb_predictions = (
    xgb_probabilities >= 0.05
).astype(int)

# --------------------------------------------------
# 7. AURA FUSION STRATEGIES
# --------------------------------------------------

# OR fusion: either model detects an attack
or_predictions = (
    (ae_predictions == 1) |
    (xgb_predictions == 1)
).astype(int)

# AND fusion: both models must detect an attack
and_predictions = (
    (ae_predictions == 1) &
    (xgb_predictions == 1)
).astype(int)

# Weighted fusion:
# Normalize each signal against its decision threshold.
ae_signal = ae_errors / max(ae_threshold, 1e-10)
xgb_signal = xgb_probabilities / 0.05

weighted_score = (
    0.5 * ae_signal +
    0.5 * xgb_signal
)

# Exploratory decision threshold
weighted_predictions = (
    weighted_score >= 1.0
).astype(int)

# --------------------------------------------------
# 8. EVALUATE EACH STRATEGY
# --------------------------------------------------

def evaluate_model(name, predictions):
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))

    print("\nAccuracy:", accuracy_score(y, predictions))
    print("Precision:", precision_score(y, predictions, zero_division=0))
    print("Recall:", recall_score(y, predictions, zero_division=0))
    print("F1 Score:", f1_score(y, predictions, zero_division=0))

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["BENIGN", "ATTACK"],
            zero_division=0
        )
    )

evaluate_model("AUTOENCODER", ae_predictions)
evaluate_model("XGBOOST", xgb_predictions)
evaluate_model("AURA OR FUSION", or_predictions)
evaluate_model("AURA AND FUSION", and_predictions)
evaluate_model("AURA WEIGHTED FUSION", weighted_predictions)

# --------------------------------------------------
# 9. ATTACK FAMILY ANALYSIS
# --------------------------------------------------

print("\nATTACK FAMILY DETECTION")

for attack in ["FTP-Patator", "SSH-Patator"]:
    mask = df["Label"].values == attack

    total = int(mask.sum())

    if total == 0:
        continue

    print("\n", attack)
    print("Total:", total)
    print("Autoencoder:", int(ae_predictions[mask].sum()))
    print("XGBoost:", int(xgb_predictions[mask].sum()))
    print("AURA OR:", int(or_predictions[mask].sum()))
    print("AURA AND:", int(and_predictions[mask].sum()))
    print("AURA Weighted:", int(weighted_predictions[mask].sum()))

# --------------------------------------------------
# 10. SAVE RESULTS
# --------------------------------------------------

results = pd.DataFrame({
    "Actual_Label": df["Label"],
    "Autoencoder_Error": ae_errors,
    "XGBoost_Probability": xgb_probabilities,
    "Autoencoder_Prediction": ae_predictions,
    "XGBoost_Prediction": xgb_predictions,
    "AURA_OR": or_predictions,
    "AURA_AND": and_predictions,
    "AURA_Weighted": weighted_predictions
})

output_path = os.path.join(
    ROOT, "research", "experiments", "aura_fusion_results.csv"
)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
results.to_csv(output_path, index=False)

print("\nAURA FUSION EXPERIMENT COMPLETED")
print("Results saved to:", output_path)
