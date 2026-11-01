
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(ROOT, "backend", "datasets")
MODEL_DIR = os.path.join(ROOT, "backend", "saved_models")
OUTPUT_DIR = os.path.join(ROOT, "research", "experiments")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRIDAY_PATH = os.path.join(
    DATA_DIR,
    "cleaned_ddos_dataset.csv"
)

TUESDAY_PATH = os.path.join(
    DATA_DIR,
    "cleaned_tuesday_dataset.csv"
)

# ==================================================
# 2. LOAD TRAINING DATA
# ==================================================

print("\nAURA ADAPTIVE CLASSIFIER")
print("=" * 60)

print("\nLoading Friday training dataset...")

friday = pd.read_csv(FRIDAY_PATH)

friday["Label"] = friday["Label"].astype(str).str.strip()

features = [
    col for col in friday.columns
    if col != "Label"
]

X_train_all = friday[features].replace(
    [np.inf, -np.inf], np.nan
)

X_train_all = X_train_all.fillna(0)

y_train_all = (
    friday["Label"] != "BENIGN"
).astype(int)

print("Friday dataset:", friday.shape)
print("\nFriday labels:")
print(friday["Label"].value_counts())

# ==================================================
# 3. TRAIN-VALIDATION SPLIT
# ==================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all,
    y_train_all,
    test_size=0.20,
    random_state=42,
    stratify=y_train_all
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))

# ==================================================
# 4. RANDOM FOREST MODEL
# ==================================================

print("\nTraining Adaptive Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed.")

# ==================================================
# 5. FRIDAY VALIDATION
# ==================================================

val_probabilities = model.predict_proba(X_val)[:, 1]

val_predictions = (
    val_probabilities >= 0.50
).astype(int)

print("\n" + "=" * 60)
print("FRIDAY VALIDATION RESULTS")
print("=" * 60)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_predictions))

print("\nAccuracy:", accuracy_score(y_val, val_predictions))
print(
    "Precision:",
    precision_score(y_val, val_predictions, zero_division=0)
)
print(
    "Recall:",
    recall_score(y_val, val_predictions, zero_division=0)
)
print(
    "F1 Score:",
    f1_score(y_val, val_predictions, zero_division=0)
)

# ==================================================
# 6. LOAD TUESDAY UNSEEN DATA
# ==================================================

print("\nLoading Tuesday independent evaluation dataset...")

tuesday = pd.read_csv(TUESDAY_PATH)

tuesday["Label"] = tuesday["Label"].astype(str).str.strip()

missing_features = [
    col for col in features
    if col not in tuesday.columns
]

if missing_features:
    raise ValueError(
        f"Missing Tuesday features: {missing_features}"
    )

X_test = tuesday[features].replace(
    [np.inf, -np.inf], np.nan
)

X_test = X_test.fillna(0)

# Labels are used only for evaluation.
y_test = (
    tuesday["Label"] != "BENIGN"
).astype(int)

print("Tuesday dataset:", tuesday.shape)

# ==================================================
# 7. TUESDAY PREDICTIONS
# ==================================================

print("\nEvaluating on Tuesday traffic...")

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= 0.50
).astype(int)

print("\n" + "=" * 60)
print("TUESDAY EVALUATION RESULTS")
print("=" * 60)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)

print("Accuracy:", accuracy_score(y_test, test_predictions))
print(
    "Attack Precision:",
    precision_score(y_test, test_predictions, zero_division=0)
)
print(
    "Attack Recall:",
    recall_score(y_test, test_predictions, zero_division=0)
)
print(
    "Attack F1:",
    f1_score(y_test, test_predictions, zero_division=0)
)

print(
    "ROC-AUC:",
    roc_auc_score(y_test, test_probabilities)
)

print(
    "PR-AUC:",
    average_precision_score(y_test, test_probabilities)
)

# ==================================================
# 8. ATTACK FAMILY ANALYSIS
# ==================================================

print("\n" + "=" * 60)
print("ATTACK FAMILY DETECTION")
print("=" * 60)

for attack in ["FTP-Patator", "SSH-Patator"]:

    mask = tuesday["Label"] == attack

    total = int(mask.sum())
    detected = int(test_predictions[mask].sum())

    recall = detected / total if total else 0

    print("\nAttack:", attack)
    print("Total:", total)
    print("Detected:", detected)
    print("Missed:", total - detected)
    print("Recall:", round(recall * 100, 2), "%")

# ==================================================
# 9. FEATURE IMPORTANCE
# ==================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 60)

print(importance.head(20).to_string(index=False))

importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "adaptive_feature_importance.csv"
    ),
    index=False
)

# ==================================================
# 10. SAVE MODEL AND RESULTS
# ==================================================

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "adaptive_random_forest.pkl"
    )
)

joblib.dump(
    features,
    os.path.join(
        MODEL_DIR,
        "adaptive_features.pkl"
    )
)

results = pd.DataFrame({
    "Actual_Label": tuesday["Label"],
    "Attack_Probability": test_probabilities,
    "Prediction": test_predictions
})

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "adaptive_classifier_results.csv"
    ),
    index=False
)

print("\n" + "=" * 60)
print("ADAPTIVE CLASSIFIER COMPLETED")
print("=" * 60)

print("Model saved: adaptive_random_forest.pkl")
print("Features saved: adaptive_features.pkl")
print("Results saved: adaptive_classifier_results.csv")
