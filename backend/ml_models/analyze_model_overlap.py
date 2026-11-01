import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

df = pd.read_csv(
    os.path.join(DATASET_DIR, "cleaned_tuesday_dataset.csv")
)

features = joblib.load(
    os.path.join(MODEL_DIR, "robust_features.pkl")
)

transformer = joblib.load(
    os.path.join(MODEL_DIR, "robust_transformer.pkl")
)

threshold = joblib.load(
    os.path.join(MODEL_DIR, "robust_threshold.pkl")
)

X = df[features].to_numpy(dtype=np.float64)

X_scaled = transformer.transform(X)
X_scaled = np.clip(X_scaled, -5, 5).astype(np.float32)


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


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = Autoencoder(len(features)).to(device)

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "robust_autoencoder.pth"),
        map_location=device,
        weights_only=True
    )
)

model.eval()

ae_scores = []

with torch.no_grad():

    for start in range(0, len(X_scaled), 4096):

        batch = torch.tensor(
            X_scaled[start:start + 4096]
        ).to(device)

        output = model(batch)

        errors = torch.mean(
            (batch - output) ** 2,
            dim=1
        )

        ae_scores.extend(errors.cpu().numpy())

ae_scores = np.array(ae_scores)

# XGBoost probabilities
xgb = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_classifier.pkl")
)

xgb_features = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_features.pkl")
)

xgb_scores = xgb.predict_proba(
    df[xgb_features]
)[:, 1]

# Diagnostic thresholds from previous experiments
ae_predictions = (
    ae_scores > np.percentile(ae_scores[df["Label"].str.strip() == "BENIGN"], 95)
)

xgb_predictions = xgb_scores >= 0.05

print("\nAURA MODEL OVERLAP ANALYSIS")

for label in ["FTP-Patator", "SSH-Patator"]:

    mask = (
        df["Label"].str.strip() == label
    ).to_numpy()

    ae_detected = ae_predictions[mask]
    xgb_detected = xgb_predictions[mask]

    both = ae_detected & xgb_detected
    either = ae_detected | xgb_detected

    print(f"\n{label}")
    print("Total attacks:", mask.sum())
    print("Autoencoder detected:", ae_detected.sum())
    print("XGBoost detected:", xgb_detected.sum())
    print("Both detected:", both.sum())
    print("Either detected:", either.sum())

print("\nAnalysis completed.")