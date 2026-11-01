
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# ==================================================
# 1. PATHS
# ==================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    ROOT, "backend", "datasets", "cleaned_tuesday_dataset.csv"
)

MODEL_DIR = os.path.join(ROOT, "backend", "saved_models")
OUTPUT_DIR = os.path.join(ROOT, "research", "experiments")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==================================================
# 2. LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH)
df["Label"] = df["Label"].astype(str).str.strip()

y = (df["Label"] != "BENIGN").astype(int).values

# ==================================================
# 3. LOAD AUTOENCODER
# ==================================================

features = joblib.load(
    os.path.join(MODEL_DIR, "robust_features.pkl")
)

transformer = joblib.load(
    os.path.join(MODEL_DIR, "robust_transformer.pkl")
)

X = df[features].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

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

# Exploratory Tuesday benign threshold
ae_threshold = np.percentile(
    ae_errors[y == 0], 95
)

ae_predictions = (
    ae_errors >= ae_threshold
).astype(int)

# ==================================================
# 4. LOAD XGBOOST
# ==================================================

xgb_model = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_classifier.pkl")
)

xgb_features = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_features.pkl")
)

X_xgb = df[xgb_features].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

xgb_probabilities = xgb_model.predict_proba(
    X_xgb
)[:, 1]

xgb_predictions = (
    xgb_probabilities >= 0.05
).astype(int)

# ==================================================
# 5. LOAD RANDOM FOREST
# ==================================================

rf_model = joblib.load(
    os.path.join(MODEL_DIR, "adaptive_random_forest.pkl")
)

rf_features = joblib.load(
    os.path.join(MODEL_DIR, "adaptive_features.pkl")
)

X_rf = df[rf_features].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

rf_probabilities = rf_model.predict_proba(
    X_rf
)[:, 1]

# Exploratory threshold from previous analysis
rf_predictions = (
    rf_probabilities >= 0.02
).astype(int)

# ==================================================
# 6. THREE-MODEL FUSION
# ==================================================

fusion_models = {
    "Autoencoder + XGBoost": (
        (ae_predictions == 1) |
        (xgb_predictions == 1)
    ),

    "Autoencoder + Random Forest": (
        (ae_predictions == 1) |
        (rf_predictions == 1)
    ),

    "XGBoost + Random Forest": (
        (xgb_predictions == 1) |
        (rf_predictions == 1)
    ),

    "AURA Three-Model OR": (
        (ae_predictions == 1) |
        (xgb_predictions == 1) |
        (rf_predictions == 1)
    )
}

# ==================================================
# 7. EVALUATION
# ==================================================

def evaluate(name, predictions):

    predictions = predictions.astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    f1 = f1_score(
        y, predictions, zero_division=0
    )

    fpr = fp / (fp + tn)

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("Confusion Matrix:")
    print([[tn, fp], [fn, tp]])

    print("Accuracy:", accuracy_score(y, predictions))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("False Positive Rate:", fpr)

    return {
        "Model": name,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr
    }


# ==================================================
# 8. RUN ALL EXPERIMENTS
# ==================================================

results = []

results.append(
    evaluate("Autoencoder + XGBoost",
             fusion_models["Autoencoder + XGBoost"])
)

results.append(
    evaluate("Autoencoder + Random Forest",
             fusion_models["Autoencoder + Random Forest"])
)

results.append(
    evaluate("XGBoost + Random Forest",
             fusion_models["XGBoost + Random Forest"])
)

results.append(
    evaluate("AURA Three-Model OR",
             fusion_models["AURA Three-Model OR"])
)

# ==================================================
# 9. ATTACK FAMILY DETECTION
# ==================================================

print("\nATTACK FAMILY DETECTION")

for attack in ["FTP-Patator", "SSH-Patator"]:

    mask = df["Label"].values == attack

    print("\n", attack)
    print("Total:", int(mask.sum()))

    for name, predictions in fusion_models.items():

        detected = int(predictions[mask].sum())

        print(
            name,
            ":",
            detected,
            "/",
            int(mask.sum())
        )

# ==================================================
# 10. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_path = os.path.join(
    OUTPUT_DIR,
    "aura_three_model_comparison.csv"
)

results_df.to_csv(results_path, index=False)

prediction_data = pd.DataFrame({
    "Actual_Label": df["Label"],
    "AE_Prediction": ae_predictions,
    "XGB_Prediction": xgb_predictions,
    "RF_Prediction": rf_predictions,
    "AURA_Three_Model": (
        fusion_models["AURA Three-Model OR"].astype(int)
    )
})

prediction_data.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "aura_three_model_predictions.csv"
    ),
    index=False
)

print("\nAURA THREE-MODEL FUSION COMPLETED")
print("Results saved:", results_path)
