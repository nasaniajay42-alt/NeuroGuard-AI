import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")


# Autoencoder architecture must match the trained model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
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


# Load Tuesday dataset
path = os.path.join(
    DATASET_DIR, "cleaned_tuesday_dataset.csv"
)

df = pd.read_csv(path)

features = joblib.load(
    os.path.join(MODEL_DIR, "features.pkl")
)

X = df[features].to_numpy(dtype=np.float32)

# Labels: BENIGN = 0, attacks = 1
y = (df["Label"].str.strip() != "BENIGN").astype(int).to_numpy()

# Load the independently trained scaler and threshold
scaler = joblib.load(
    os.path.join(MODEL_DIR, "independent_scaler.pkl")
)

threshold = joblib.load(
    os.path.join(MODEL_DIR, "independent_threshold.pkl")
)

X_scaled = scaler.transform(X).astype(np.float32)

# Load model
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = Autoencoder(len(features)).to(device)

state_dict = torch.load(
    os.path.join(MODEL_DIR, "autoencoder_independent.pth"),
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

# Calculate reconstruction errors
with torch.no_grad():
    tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    reconstructed = model(tensor)
    errors = torch.mean((tensor - reconstructed) ** 2, dim=1)
    scores = errors.cpu().numpy()

# Predict anomalies
predictions = (scores > threshold).astype(int)


benign_mask = df["Label"].str.strip() == "BENIGN"

extreme_mask = (
    (df["Fwd IAT Mean"] > 10000000) |
    (df["Bwd IAT Mean"] > 10000000) |
    (df["Flow IAT Mean"] > 10000000)
)

normal_benign = benign_mask & (~extreme_mask)
extreme_benign = benign_mask & extreme_mask

print("\nFALSE POSITIVE INVESTIGATION")

for name, mask in [
    ("Regular BENIGN records", normal_benign),
    ("Extreme BENIGN records", extreme_benign)
]:
    total = mask.sum()
    false_positives = predictions[mask.to_numpy()].sum()

    print(f"\n{name}")
    print("Total:", total)
    print("Flagged as attack:", false_positives)

    if total > 0:
        print(
            "False positive rate:",
            round(false_positives / total * 100, 2),
            "%"
        )
# Display results
print("\nUNSEEN ATTACK FAMILY EVALUATION")
print("Threshold:", threshold)

print("\nConfusion Matrix:")
print(confusion_matrix(y, predictions))

print("\nClassification Report:")
print(classification_report(
    y,
    predictions,
    target_names=["BENIGN", "ATTACK"],
    zero_division=0
))

print("ROC-AUC:", roc_auc_score(y, scores))
print("PR-AUC:", average_precision_score(y, scores))

print("\nAttack-family detection rates:")

for label in ["FTP-Patator", "SSH-Patator"]:
    mask = df["Label"].str.strip() == label

    detection_rate = predictions[mask.to_numpy()].mean()

    print(
        f"{label}: {detection_rate * 100:.2f}% "
        f"({predictions[mask.to_numpy()].sum()}/{mask.sum()})"
    )

print("\nEvaluation completed.")
print("\nANOMALY SCORE DISTRIBUTION")

for label in ["BENIGN", "FTP-Patator", "SSH-Patator"]:
    mask = df["Label"].str.strip() == label
    group_scores = scores[mask.to_numpy()]

    print(f"\n{label}")
    print("Mean:", np.mean(group_scores))
    print("Median:", np.median(group_scores))
    print("95th percentile:", np.percentile(group_scores, 95))
    print("99th percentile:", np.percentile(group_scores, 99))