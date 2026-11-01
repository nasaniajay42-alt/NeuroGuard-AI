
import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    ROOT, "backend", "datasets",
    "cleaned_wednesday_dataset.csv"
)

PREDICTION_PATH = os.path.join(
    ROOT, "research", "evaluation",
    "aura_wednesday_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT, "research", "evaluation"
)

df = pd.read_csv(DATA_PATH)
pred = pd.read_csv(PREDICTION_PATH)

if len(df) != len(pred):
    raise ValueError("Dataset and predictions do not match.")

if not np.array_equal(
    df["Label"].astype(str).values,
    pred["Actual_Label"].astype(str).values
):
    raise ValueError("Labels are not aligned.")

y = (df["Label"] != "BENIGN").astype(int).values

rf_prob = pred["RF_Probability"].values
xgb_prob = pred["XGB_Probability"].values

XGB_THRESHOLD = 0.30

thresholds = [
    0.005, 0.008, 0.010, 0.012,
    0.015, 0.018, 0.019, 0.020,
    0.022, 0.025, 0.030, 0.040
]

results = []

print("=" * 75)
print("WEDNESDAY AURA THRESHOLD CALIBRATION")
print("=" * 75)

for threshold in thresholds:

    rf_flag = rf_prob >= threshold
    xgb_flag = xgb_prob >= XGB_THRESHOLD

    aura_pred = (
        rf_flag | xgb_flag
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, aura_pred, labels=[0, 1]
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

    fpr = fp / (fp + tn)

    family_recall = {}

    for attack in sorted(df["Label"].unique()):

        if attack == "BENIGN":
            continue

        mask = df["Label"].values == attack

        family_recall[attack] = (
            aura_pred[mask].sum() / mask.sum()
        )

    row = {
        "RF_Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }

    for attack, value in family_recall.items():
        row[attack + "_Recall"] = value

    results.append(row)

results_df = pd.DataFrame(results)

print("\nOVERALL THRESHOLD COMPARISON")
print("-" * 75)

print(
    results_df[
        ["RF_Threshold", "Precision", "Recall",
         "F1", "FPR", "TP", "FP", "FN"]
    ].round(4).to_string(index=False)
)

print("\nATTACK FAMILY DETECTION RATES")
print("-" * 75)

family_columns = [
    col for col in results_df.columns
    if col.endswith("_Recall")
]

print(
    results_df[
        ["RF_Threshold"] + family_columns
    ].round(4).to_string(index=False)
)

# Save calibration results
output_path = os.path.join(
    OUTPUT_DIR,
    "wednesday_threshold_calibration.csv"
)

results_df.to_csv(output_path, index=False)

print("\nSaved:", output_path)

print("\nIMPORTANT:")
print(
    "These are calibration results using Wednesday labels. "
    "They must not be reported as independent test performance."
)

print("\nCalibration analysis completed.")
