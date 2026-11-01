
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# ==========================================
# NEUROGUARD AI - AURA
# INDEPENDENT MODEL EVALUATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "backend/datasets/cleaned_ddos_dataset.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_STATE = 42
EPOCHS = 20
BATCH_SIZE = 1024

print("=" * 55)
print("     NEUROGUARD AI - INDEPENDENT EVALUATION")
print("=" * 55)

print("\nComputing device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 1. LOAD DATA
# ==========================================

print("\n[1] Loading dataset...")

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.astype(str).str.strip()
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Label"])

labels = df["Label"].astype(str).str.strip().str.upper()

X = df.drop(columns=["Label"])
X = X.select_dtypes(include=np.number)

X = X.fillna(X.median()).fillna(0)

normal_mask = labels == "BENIGN"

X_normal = X.loc[normal_mask].copy()
X_attack = X.loc[~normal_mask].copy()

print("Normal records:", len(X_normal))
print("Attack records:", len(X_attack))


# ==========================================
# 2. SPLIT NORMAL TRAFFIC
# ==========================================

print("\n[2] Creating independent data splits...")

X_train, X_remaining = train_test_split(
    X_normal,
    test_size=0.40,
    random_state=RANDOM_STATE
)

X_val, X_test_normal = train_test_split(
    X_remaining,
    test_size=0.50,
    random_state=RANDOM_STATE
)

# Keep attack traffic out of model training
_, X_test_attack = train_test_split(
    X_attack,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("Training normal:", len(X_train))
print("Validation normal:", len(X_val))
print("Testing normal:", len(X_test_normal))
print("Testing attacks:", len(X_test_attack))


# ==========================================
# 3. SCALE FEATURES
# ==========================================

print("\n[3] Scaling features...")

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_normal_scaled = scaler.transform(X_test_normal)
X_test_attack_scaled = scaler.transform(X_test_attack)

X_train_scaled = np.clip(X_train_scaled, -10, 10)
X_val_scaled = np.clip(X_val_scaled, -10, 10)
X_test_normal_scaled = np.clip(X_test_normal_scaled, -10, 10)
X_test_attack_scaled = np.clip(X_test_attack_scaled, -10, 10)

X_train_scaled = X_train_scaled.astype(np.float32)
X_val_scaled = X_val_scaled.astype(np.float32)
X_test_normal_scaled = X_test_normal_scaled.astype(np.float32)
X_test_attack_scaled = X_test_attack_scaled.astype(np.float32)


# ==========================================
# 4. CREATE DATA LOADER
# ==========================================

train_tensor = torch.tensor(X_train_scaled)

train_loader = DataLoader(
    TensorDataset(train_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================================
# 5. DEFINE AUTOENCODER
# ==========================================

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


model = Autoencoder(X_train_scaled.shape[1]).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 6. TRAIN ON NORMAL TRAFFIC ONLY
# ==========================================

print("\n[4] Training independent Autoencoder...")

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for (batch,) in train_loader:

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch)

    average_loss = total_loss / len(train_tensor)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.6f}"
    )


# ==========================================
# 7. CALCULATE THRESHOLD ON VALIDATION
# ==========================================

print("\n[5] Calculating validation threshold...")

model.eval()

def reconstruction_errors(data):

    tensor = torch.tensor(
        data,
        dtype=torch.float32
    )

    all_errors = []

    with torch.no_grad():

        for batch in tensor.split(4096):

            batch = batch.to(DEVICE)

            reconstructed = model(batch)

            errors = torch.mean(
                (batch - reconstructed) ** 2,
                dim=1
            )

            all_errors.extend(errors.cpu().numpy())

    return np.asarray(all_errors)


validation_errors = reconstruction_errors(X_val_scaled)

threshold = float(np.percentile(validation_errors, 99))

print("Validation threshold:", threshold)


# ==========================================
# 8. INDEPENDENT TESTING
# ==========================================

print("\n[6] Evaluating independent test data...")

X_test = np.vstack([
    X_test_normal_scaled,
    X_test_attack_scaled
])

y_test = np.concatenate([
    np.zeros(len(X_test_normal_scaled), dtype=int),
    np.ones(len(X_test_attack_scaled), dtype=int)
])

test_errors = reconstruction_errors(X_test)

predictions = (test_errors > threshold).astype(int)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["BENIGN", "DDoS"],
        zero_division=0
    )
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, test_errors), 4)
)

print(
    "PR-AUC:",
    round(average_precision_score(y_test, test_errors), 4)
)


# ==========================================
# 9. SAVE INDEPENDENT MODEL
# ==========================================

MODEL_DIR = ROOT / "backend/saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

torch.save(
    model.state_dict(),
    MODEL_DIR / "autoencoder_independent.pth"
)

joblib.dump(
    scaler,
    MODEL_DIR / "independent_scaler.pkl"
)

joblib.dump(
    threshold,
    MODEL_DIR / "independent_threshold.pkl"
)

print("\nIndependent evaluation completed!")
print("Independent model and scaler saved.")
