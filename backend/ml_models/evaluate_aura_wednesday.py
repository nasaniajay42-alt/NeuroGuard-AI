
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    ROOT, "backend", "datasets",
    "cleaned_wednesday_dataset.csv"
)

MODEL_DIR = os.path.join(ROOT, "backend", "saved_models")
OUTPUT_DIR = os.path.join(ROOT, "research", "evaluation")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed thresholds selected using Tuesday experiments
XGB_THRESHOLD = 0.30
RF_THRESHOLD = 0.02

print("=" * 65)
print("AURA INDEPENDENT WEDNESDAY EVALUATION")
print("=" * 65)

# 1. Load dataset
print("\nLoading Wednesday dataset...")

df = pd.read_csv(DATA_PATH)

df["Label"] = df["Label"].astype(str).str.strip()

y = (df["Label"] != "BENIGN").astype(int).values

print("Dataset shape:", df.shape)
print("Total records:", len(df))

# 2. Load trained models
print("\nLoading trained models...")

xgb_model = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_classifier.pkl")
)

rf_model = joblib.load(
    os.path.join(MODEL_DIR, "adaptive_random_forest.pkl")
)

xgb_features = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_features.pkl")
)

rf_features = joblib.load(
    os.path.join(MODEL_DIR, "adaptive_features.pkl")
)

# 3. Prepare features
X_xgb = df[xgb_features].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

X_rf = df[rf_features].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

# 4. Generate probabilities
print("\nGenerating predictions...")

xgb_prob = xgb_model.predict_proba(X_xgb)[:, 1]
rf_prob = rf_model.predict_proba(X_rf)[:, 1]

# 5. Individual model predictions
xgb_pred = (xgb_prob >= XGB_THRESHOLD).astype(int)
rf_pred = (rf_prob >= RF_THRESHOLD).astype(int)

# 6. AURA OR fusion
aura_pred = (
    (xgb_pred == 1) | (rf_pred == 1)
).astype(int)

# Combined score for ranking / ROC-AUC
aura_score = np.maximum(
    xgb_prob / XGB_THRESHOLD,
    rf_prob / RF_THRESHOLD
)

# 7. Evaluation function
def evaluate_model(name, predictions, scores):

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

    fpr = fp / (fp + tn) if (fp + tn) else 0

    roc = roc_auc_score(y, scores)
    pr = average_precision_score(y, scores)

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("\nConfusion Matrix:")
    print([[tn, fp], [fn, tp]])

    print("\nPrecision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("False Positive Rate:", round(fpr, 4))
    print("ROC-AUC:", round(roc, 4))
    print("PR-AUC:", round(pr, 4))

    return {
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }

# 8. Evaluate all models
results = []

results.append(
    evaluate_model(
        "XGBoost",
        xgb_pred,
        xgb_prob
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        rf_pred,
        rf_prob
    )
)

results.append(
    evaluate_model(
        "AURA XGBoost + Random Forest",
        aura_pred,
        aura_score
    )
)

# 9. Attack family detection
print("\n" + "=" * 65)
print("ATTACK FAMILY DETECTION")
print("=" * 65)

family_results = []

for attack in sorted(df["Label"].unique()):

    if attack == "BENIGN":
        continue

    mask = df["Label"].values == attack

    total = int(mask.sum())

    xgb_detected = int(xgb_pred[mask].sum())
    rf_detected = int(rf_pred[mask].sum())
    aura_detected = int(aura_pred[mask].sum())

    family_results.append({
        "Attack": attack,
        "Total": total,
        "XGB_Detected": xgb_detected,
        "RF_Detected": rf_detected,
        "AURA_Detected": aura_detected,
        "AURA_Recall": aura_detected / total if total else 0
    })

    print("\nAttack:", attack)
    print("Total:", total)
    print("XGBoost detected:", xgb_detected)
    print("Random Forest detected:", rf_detected)
    print("AURA detected:", aura_detected)
    print(
        "AURA detection rate:",
        round(aura_detected / total * 100, 2),
        "%"
    )

# 10. Detailed classification report
print("\n" + "=" * 65)
print("AURA CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        y,
        aura_pred,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)

# 11. Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "aura_wednesday_model_comparison.csv"
    ),
    index=False
)

pd.DataFrame(family_results).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "aura_wednesday_attack_detection.csv"
    ),
    index=False
)

pd.DataFrame({
    "Actual_Label": df["Label"],
    "XGB_Probability": xgb_prob,
    "RF_Probability": rf_prob,
    "XGB_Prediction": xgb_pred,
    "RF_Prediction": rf_pred,
    "AURA_Prediction": aura_pred
}).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "aura_wednesday_predictions.csv"
    ),
    index=False
)

print("\n" + "=" * 65)
print("WEDNESDAY EVALUATION COMPLETED")
print("=" * 65)

print("Results saved in:", OUTPUT_DIR)

print("\nThresholds used:")
print("XGBoost:", XGB_THRESHOLD)
print("Random Forest:", RF_THRESHOLD)

print("\nThresholds were not tuned on Wednesday.")
