
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# ==========================================
# NEUROGUARD AI - AURA
# XGBOOST ATTACK CLASSIFIER
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "backend/datasets/cleaned_ddos_dataset.csv"
MODEL_DIR = ROOT / "backend/saved_models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 55)
print("       NEUROGUARD AI - XGBOOST")
print("=" * 55)


# ==========================================
# 1. LOAD DATASET
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

y = (labels != "BENIGN").astype(int)

print("Dataset shape:", X.shape)
print("Normal records:", int((y == 0).sum()))
print("Attack records:", int((y == 1).sum()))


# ==========================================
# 2. CREATE STRATIFIED SPLITS
# ==========================================

print("\n[2] Splitting dataset...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))


# ==========================================
# 3. CALCULATE CLASS WEIGHT
# ==========================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_weight = negative_count / positive_count

print("\nClass weight:", scale_weight)


# ==========================================
# 4. TRAIN XGBOOST
# ==========================================

print("\n[3] Training XGBoost classifier...")

model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_weight,
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    early_stopping_rounds=30
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)


# ==========================================
# 5. VALIDATION PERFORMANCE
# ==========================================

print("\n[4] Validation performance...")

val_probabilities = model.predict_proba(X_val)[:, 1]

print(
    "Validation ROC-AUC:",
    round(roc_auc_score(y_val, val_probabilities), 4)
)

print(
    "Validation PR-AUC:",
    round(average_precision_score(y_val, val_probabilities), 4)
)


# ==========================================
# 6. TEST PERFORMANCE
# ==========================================

print("\n[5] Independent test evaluation...")

test_probabilities = model.predict_proba(X_test)[:, 1]

# Default probability threshold
test_predictions = (test_probabilities >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["BENIGN", "DDoS"],
        zero_division=0
    )
)

print(
    "Test ROC-AUC:",
    round(roc_auc_score(y_test, test_probabilities), 4)
)

print(
    "Test PR-AUC:",
    round(average_precision_score(y_test, test_probabilities), 4)
)


# ==========================================
# 7. SAVE MODEL
# ==========================================

print("\n[6] Saving model...")

joblib.dump(
    model,
    MODEL_DIR / "xgboost_classifier.pkl"
)

joblib.dump(
    list(X.columns),
    MODEL_DIR / "xgboost_features.pkl"
)

print("Model saved successfully!")

print("\n" + "=" * 55)
print("XGBOOST TRAINING COMPLETED")
print("=" * 55)
