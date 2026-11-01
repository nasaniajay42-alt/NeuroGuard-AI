
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# ==========================================
# NEUROGUARD AI - AURA
# AUTOENCODER TRAINING MODULE
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "backend" / "datasets" / "cleaned_ddos_dataset.csv"
MODEL_DIR = ROOT / "backend" / "saved_models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "autoencoder.pth"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_PATH = MODEL_DIR / "features.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print("       NEUROGUARD AI - AURA")
print("       AUTOENCODER TRAINING")
print("=" * 50)

print("\nComputing device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n[1] Loading cleaned dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH, low_memory=False)

df.columns = df.columns.astype(str).str.strip()
df = df.replace([np.inf, -np.inf], np.nan)

if "Label" not in df.columns:
    raise ValueError("Label column is missing.")

df = df.dropna(subset=["Label"])

labels = df["Label"].astype(str).str.strip().str.upper()

# Keep only numeric input features
X = df.drop(columns=["Label"])
X = X.select_dtypes(include=np.number)

# Replace missing values
X = X.fillna(X.median()).fillna(0)

feature_columns = X.columns.tolist()

print("Dataset shape:", df.shape)
print("Input features:", len(feature_columns))


# ==========================================
# 2. SEPARATE NORMAL AND ATTACK TRAFFIC
# ==========================================

print("\n[2] Separating traffic classes...")

normal_mask = labels == "BENIGN"
attack_mask = ~normal_mask

X_normal = X.loc[normal_mask].copy()

X_attack = X.loc[attack_mask].copy()
y_attack = labels.loc[attack_mask].copy()

if len(X_normal) < 100:
    raise ValueError("Not enough normal traffic for training.")

if len(X_attack) == 0:
    raise ValueError("No attack records found for evaluation.")

print("Normal records:", len(X_normal))
print("Attack records:", len(X_attack))


# ==========================================
# 3. SPLIT NORMAL TRAFFIC
# ==========================================

print("\n[3] Splitting normal traffic...")

X_train, X_val = train_test_split(
    X_normal,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))


# ==========================================
# 4. FEATURE SCALING
# ==========================================

print("\n[4] Scaling network features...")

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_attack_scaled = scaler.transform(X_attack)

# Limit extreme scaled values for stable training
X_train_scaled = np.clip(X_train_scaled, -10, 10)
X_val_scaled = np.clip(X_val_scaled, -10, 10)
X_attack_scaled = np.clip(X_attack_scaled, -10, 10)

X_train_scaled = X_train_scaled.astype(np.float32)
X_val_scaled = X_val_scaled.astype(np.float32)
X_attack_scaled = X_attack_scaled.astype(np.float32)

joblib.dump(scaler, SCALER_PATH)
joblib.dump(feature_columns, FEATURE_PATH)


# ==========================================
# 5. CONVERT DATA TO PYTORCH
# ==========================================

print("\n[5] Preparing GPU tensors...")

train_tensor = torch.tensor(X_train_scaled)
val_tensor = torch.tensor(X_val_scaled)

train_loader = DataLoader(
    TensorDataset(train_tensor),
    batch_size=1024,
    shuffle=True
)


# ==========================================
# 6. DEFINE AUTOENCODER
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

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


model = Autoencoder(
    input_dim=len(feature_columns)
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 7. TRAIN AUTOENCODER
# ==========================================

print("\n[6] Training Autoencoder...")

EPOCHS = 20

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
# 8. CALCULATE NORMAL RECONSTRUCTION ERROR
# ==========================================

print("\n[7] Calculating anomaly threshold...")

model.eval()

with torch.no_grad():

    val_tensor = val_tensor.to(DEVICE)

    reconstructed = model(val_tensor)

    normal_errors = torch.mean(
        (val_tensor - reconstructed) ** 2,
        dim=1
    )

    normal_errors = normal_errors.cpu().numpy()

# 99th percentile of normal validation errors
threshold = float(np.percentile(normal_errors, 99))

print("Anomaly threshold:", threshold)


# ==========================================
# 9. EVALUATE ON HELD-OUT ATTACK TRAFFIC
# ==========================================

print("\n[8] Evaluating attack detection...")

test_features = np.vstack([
    X_val_scaled,
    X_attack_scaled
])

y_true = np.concatenate([
    np.zeros(len(X_val_scaled), dtype=int),
    np.ones(len(X_attack_scaled), dtype=int)
])

test_tensor = torch.tensor(
    test_features,
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    reconstructed = model(test_tensor)

    test_errors = torch.mean(
        (test_tensor - reconstructed) ** 2,
        dim=1
    ).cpu().numpy()

y_pred = (test_errors > threshold).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["BENIGN", "DDoS"],
        zero_division=0
    )
)


# ==========================================
# 10. SAVE TRAINED MODEL
# ==========================================

torch.save(model.state_dict(), MODEL_PATH)

joblib.dump(threshold, THRESHOLD_PATH)

print("\n" + "=" * 50)
print("AUTOENCODER TRAINING COMPLETED")
print("=" * 50)

print("Model saved:", MODEL_PATH)
print("Scaler saved:", SCALER_PATH)
print("Threshold saved:", THRESHOLD_PATH)

print("\nNeuroGuard AI AURA module is ready!")
