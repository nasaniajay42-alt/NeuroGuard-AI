import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

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

X = df[features].to_numpy(dtype=np.float64)

X_scaled = transformer.transform(X)
X_scaled = np.clip(X_scaled, -5, 5).astype(np.float32)

y = (
    df["Label"].str.strip() != "BENIGN"
).astype(int).to_numpy()


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

scores = []

with torch.no_grad():

    for start in range(0, len(X_scaled), 4096):

        batch = torch.tensor(
            X_scaled[start:start + 4096]
        ).to(device)

        output = model(batch)

        error = torch.mean(
            (batch - output) ** 2,
            dim=1
        )

        scores.extend(error.cpu().numpy())

scores = np.array(scores)

print("\nTHRESHOLD COMPARISON")
print("-" * 75)

print(
    f"{'Percentile':<12}"
    f"{'Threshold':<14}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FPR':<12}"
)

for percentile in [90, 95, 97, 98, 99]:

    threshold = np.percentile(scores[y == 0], percentile)

    predictions = (scores > threshold).astype(int)

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    f1 = f1_score(
        y, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y, predictions
    ).ravel()

    fpr = fp / (fp + tn)

    print(
        f"{percentile:<12}"
        f"{threshold:<14.5f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fpr:<12.4f}"
    )

print("\nATTACK FAMILY ANALYSIS AT 95TH PERCENTILE")

threshold_95 = np.percentile(scores[y == 0], 95)

predictions_95 = (scores > threshold_95).astype(int)

for label in ["BENIGN", "FTP-Patator", "SSH-Patator"]:

    mask = (
        df["Label"].str.strip() == label
    ).to_numpy()

    total = mask.sum()
    detected = predictions_95[mask].sum()

    print(f"\n{label}")
    print("Total records:", total)
    print("Detected:", detected)
    print("Detection rate:", round(detected / total * 100, 2), "%")

print("\nAnalysis completed.")