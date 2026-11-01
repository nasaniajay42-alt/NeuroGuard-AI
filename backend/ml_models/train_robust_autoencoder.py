import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_PATH = os.path.join(
    BASE_DIR, "datasets", "cleaned_ddos_dataset.csv"
)
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Training device:", device)

# Load Friday dataset
df = pd.read_csv(DATASET_PATH)

features = [
    column for column in df.columns if column != "Label"
]

# Train only on BENIGN traffic
benign = df[df["Label"].str.strip() == "BENIGN"]

X = benign[features].to_numpy(dtype=np.float64)

# Remove invalid numerical values
X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# Split normal traffic into train and validation
X_train, X_val = train_test_split(
    X,
    test_size=0.25,
    random_state=42
)

# Quantile transformation
transformer = QuantileTransformer(
    n_quantiles=min(1000, len(X_train)),
    output_distribution="normal",
    random_state=42
)

X_train = transformer.fit_transform(X_train)
X_val = transformer.transform(X_val)

X_train = np.clip(X_train, -5, 5).astype(np.float32)
X_val = np.clip(X_val, -5, 5).astype(np.float32)

# Autoencoder
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


model = Autoencoder(len(features)).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.MSELoss()

train_tensor = torch.tensor(X_train)

loader = DataLoader(
    TensorDataset(train_tensor),
    batch_size=512,
    shuffle=True
)

# Training
epochs = 20

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for (batch,) in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        output = model(batch)
        loss = criterion(output, batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"Loss: {total_loss / len(loader):.6f}"
    )

# Calibrate threshold using validation BENIGN traffic
model.eval()

with torch.no_grad():

    val_tensor = torch.tensor(X_val).to(device)

    reconstructed = model(val_tensor)

    errors = torch.mean(
        (val_tensor - reconstructed) ** 2,
        dim=1
    ).cpu().numpy()

threshold = float(np.percentile(errors, 99))

# Save independent experimental model
torch.save(
    model.state_dict(),
    os.path.join(MODEL_DIR, "robust_autoencoder.pth")
)

joblib.dump(
    transformer,
    os.path.join(MODEL_DIR, "robust_transformer.pkl")
)

joblib.dump(
    threshold,
    os.path.join(MODEL_DIR, "robust_threshold.pkl")
)

joblib.dump(
    features,
    os.path.join(MODEL_DIR, "robust_features.pkl")
)

print("\nRobust Autoencoder training completed!")
print("Threshold:", threshold)
print("Features:", len(features))
print("Device:", device)