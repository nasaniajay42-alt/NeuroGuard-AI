
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = ROOT / "backend/datasets/cleaned_ddos_dataset.csv"
MODEL_PATH = ROOT / "backend/saved_models/autoencoder.pth"
SCALER_PATH = ROOT / "backend/saved_models/scaler.pkl"
FEATURE_PATH = ROOT / "backend/saved_models/features.pkl"

print("Loading dataset and trained model...")

df = pd.read_csv(DATA_PATH)

features = joblib.load(FEATURE_PATH)
scaler = joblib.load(SCALER_PATH)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


model = Autoencoder(len(features)).to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
model.eval()

X = df[features].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median()).fillna(0)

labels = (
    df["Label"].astype(str).str.strip().str.upper() != "BENIGN"
).astype(int).to_numpy()

X_scaled = np.clip(scaler.transform(X), -10, 10)
X_tensor = torch.tensor(
    X_scaled, dtype=torch.float32
).to(DEVICE)

print("Calculating reconstruction errors...")

errors = []

with torch.no_grad():
    for batch in X_tensor.split(4096):
        reconstructed = model(batch)
        batch_errors = torch.mean(
            (batch - reconstructed) ** 2, dim=1
        )
        errors.extend(batch_errors.cpu().numpy())

errors = np.asarray(errors)

print("\nThreshold comparison")
print("-" * 75)
print(
    f"{'Threshold':>12} {'Precision':>12} "
    f"{'Recall':>12} {'F1':>12} {'FPR':>12}"
)

# Compare thresholds derived from the normal validation
# distribution, not arbitrary score choices.
normal_errors = errors[labels == 0]

for percentile in [95, 97, 98, 99, 99.5, 99.9]:
    threshold = np.percentile(normal_errors, percentile)
    predictions = (errors > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        labels, predictions, zero_division=0
    )
    recall = recall_score(
        labels, predictions, zero_division=0
    )
    f1 = f1_score(labels, predictions, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0

    print(
        f"{percentile:>11}% "
        f"{precision:>11.4f} "
        f"{recall:>11.4f} "
        f"{f1:>11.4f} "
        f"{fpr:>11.4f}"
    )

print("\nThreshold analysis completed.")
