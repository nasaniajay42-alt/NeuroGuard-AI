import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

features = joblib.load(
    os.path.join(MODEL_DIR, "robust_features.pkl")
)

transformer = joblib.load(
    os.path.join(MODEL_DIR, "robust_transformer.pkl")
)

threshold = joblib.load(
    os.path.join(MODEL_DIR, "robust_threshold.pkl")
)

X = df[features]

# Apply the training transformer
X_scaled = transformer.transform(X)

X_scaled = np.clip(
    X_scaled, -5, 5
).astype(np.float32)

y = (
    df["Label"].str.strip() != "BENIGN"
).astype(int).to_numpy()

# Autoencoder architecture
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

# Calculate reconstruction errors in batches
scores = []

with torch.no_grad():

    for start in range(0, len(X_scaled), 4096):

        batch = torch.tensor(
            X_scaled[start:start + 4096]
        ).to(device)

        reconstructed = model(batch)

        errors = torch.mean(
            (batch - reconstructed) ** 2,
            dim=1
        )

        scores.extend(errors.cpu().numpy())

scores = np.array(scores)

predictions = (
    scores > threshold
).astype(int)

# Evaluation
print("\nROBUST AUTOENCODER EVALUATION")

print("Threshold:", threshold)

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
    "\nROC-AUC:",
    roc_auc_score(y, scores)
)

print(
    "PR-AUC:",
    average_precision_score(y, scores)
)

# Individual attack detection
print("\nATTACK FAMILY DETECTION")

for label in ["FTP-Patator", "SSH-Patator"]:

    mask = (
        df["Label"].str.strip() == label
    ).to_numpy()

    detected = predictions[mask].sum()

    print(
        f"{label}: "
        f"{detected}/{mask.sum()} "
        f"({detected / mask.sum() * 100:.2f}%)"
    )

# False-positive investigation
print("\nBENIGN FALSE POSITIVE RATE")

benign_mask = (
    df["Label"].str.strip() == "BENIGN"
).to_numpy()

false_positives = predictions[benign_mask].sum()

print(
    "False positives:",
    false_positives,
    "/",
    benign_mask.sum()
)

print(
    "False positive rate:",
    round(
        false_positives / benign_mask.sum() * 100,
        2
    ),
    "%"
)

print("\nEvaluation completed.")
