
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
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

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 2. LOAD DATA
# ==================================================

print("\nAURA FUSION OPTIMIZATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df["Label"] = df["Label"].astype(str).str.strip()

y = (df["Label"] != "BENIGN").astype(int).values

# ==================================================
# 3. LOAD MODELS
# ==================================================

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

X_xgb = df[xgb_features].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

X_rf = df[rf_features].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

# ==================================================
# 4. PREDICT PROBABILITIES
# ==================================================

print("\nGenerating model probabilities...")

xgb_prob = xgb_model.predict_proba(X_xgb)[:, 1]
rf_prob = rf_model.predict_proba(X_rf)[:, 1]

print("XGBoost predictions completed.")
print("Random Forest predictions completed.")

# ==================================================
# 5. DEFINE THRESHOLD RANGES
# ==================================================

xgb_thresholds = [
    0.001, 0.005, 0.01, 0.02,
    0.05, 0.10, 0.20, 0.30, 0.50
]

rf_thresholds = [
    0.001, 0.002, 0.005, 0.01,
    0.02, 0.03, 0.05, 0.075,
    0.10, 0.15, 0.20
]

results = []

# ==================================================
# 6. TEST ALL THRESHOLD COMBINATIONS
# ==================================================

print("\nTesting fusion combinations...")

for xgb_threshold in xgb_thresholds:

    xgb_pred = (
        xgb_prob >= xgb_threshold
    ).astype(int)

    for rf_threshold in rf_thresholds:

        rf_pred = (
            rf_prob >= rf_threshold
        ).astype(int)

        # OR fusion
        aura_pred = (
            (xgb_pred == 1) |
            (rf_pred == 1)
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y,
            aura_pred,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y, aura_pred, zero_division=0
        )

        recall = recall_score(
            y, aura_pred, zero_division=0
        )

        f1 = f1_score(
            y, aura_pred, zero_division=0
        )

        fpr = fp / (fp + tn) if (fp + tn) else 0

        results.append({
            "XGB_Threshold": xgb_threshold,
            "RF_Threshold": rf_threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "FPR": fpr,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        })

# ==================================================
# 7. SORT RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "F1",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 15 FUSION COMBINATIONS BY F1")
print("=" * 60)

print(
    results_df.head(15).round(4).to_string(index=False)
)

# ==================================================
# 8. BEST RESULT WITH FPR <= 2%
# ==================================================

safe_results = results_df[
    results_df["FPR"] <= 0.02
]

print("\n" + "=" * 60)
print("BEST RESULTS WITH FALSE POSITIVE RATE <= 2%")
print("=" * 60)

if len(safe_results) > 0:

    best_safe = safe_results.iloc[0]

    print(best_safe.to_string())

else:
    print("No combination satisfies this condition.")

# ==================================================
# 9. ATTACK FAMILY ANALYSIS
# ==================================================

if len(safe_results) > 0:

    selected = best_safe

else:

    selected = results_df.iloc[0]

xgb_threshold = selected["XGB_Threshold"]
rf_threshold = selected["RF_Threshold"]

final_xgb = (xgb_prob >= xgb_threshold).astype(int)
final_rf = (rf_prob >= rf_threshold).astype(int)

final_aura = (
    (final_xgb == 1) |
    (final_rf == 1)
).astype(int)

print("\n" + "=" * 60)
print("SELECTED EXPLORATORY FUSION")
print("=" * 60)

print("XGBoost threshold:", xgb_threshold)
print("Random Forest threshold:", rf_threshold)

for attack in ["FTP-Patator", "SSH-Patator"]:

    mask = df["Label"].values == attack

    total = int(mask.sum())
    detected = int(final_aura[mask].sum())

    recall = detected / total if total else 0

    print("\nAttack:", attack)
    print("Total:", total)
    print("Detected:", detected)
    print("Recall:", round(recall * 100, 2), "%")

# ==================================================
# 10. SAVE RESULTS
# ==================================================

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "aura_fusion_optimization.csv"
    ),
    index=False
)

pd.DataFrame({
    "Actual_Label": df["Label"],
    "XGB_Probability": xgb_prob,
    "RF_Probability": rf_prob,
    "AURA_Prediction": final_aura
}).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "aura_optimized_predictions.csv"
    ),
    index=False
)

print("\n" + "=" * 60)
print("AURA OPTIMIZATION COMPLETED")
print("=" * 60)

print("Results saved in research/experiments.")

print("\nIMPORTANT:")
print(
    "All threshold combinations were evaluated on Tuesday. "
    "These are exploratory results, not independent final-test metrics."
)
