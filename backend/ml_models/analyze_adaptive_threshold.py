
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# ==================================================
# 1. PATHS
# ==================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    ROOT,
    "backend",
    "datasets",
    "cleaned_tuesday_dataset.csv"
)

MODEL_PATH = os.path.join(
    ROOT,
    "backend",
    "saved_models",
    "adaptive_random_forest.pkl"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "research",
    "experiments"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 2. LOAD DATA AND MODEL
# ==================================================

print("\nAURA ADAPTIVE THRESHOLD ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df["Label"] = df["Label"].astype(str).str.strip()

features = joblib.load(
    os.path.join(
        ROOT,
        "backend",
        "saved_models",
        "adaptive_features.pkl"
    )
)

model = joblib.load(MODEL_PATH)

X = df[features].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

y = (df["Label"] != "BENIGN").astype(int).values

probabilities = model.predict_proba(X)[:, 1]

print("Dataset:", df.shape)
print("Attack records:", int(y.sum()))
print("Benign records:", int((y == 0).sum()))

# ==================================================
# 3. OVERALL MODEL RANKING
# ==================================================

print("\nROC-AUC:", roc_auc_score(y, probabilities))
print("PR-AUC:", average_precision_score(y, probabilities))

# ==================================================
# 4. THRESHOLD ANALYSIS
# ==================================================

thresholds = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50
]

results = []

print("\n" + "=" * 60)
print("THRESHOLD PERFORMANCE")
print("=" * 60)

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
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

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "True_Positives": tp,
        "False_Positives": fp,
        "False_Negatives": fn
    })

    print(
        f"\nThreshold: {threshold:.3f}"
        f"\nPrecision: {precision:.4f}"
        f"\nRecall: {recall:.4f}"
        f"\nF1: {f1:.4f}"
        f"\nFPR: {fpr:.4f}"
        f"\nDetected attacks: {tp}"
        f"\nFalse alarms: {fp}"
    )

# ==================================================
# 5. ATTACK FAMILY ANALYSIS
# ==================================================

print("\n" + "=" * 60)
print("ATTACK FAMILY DETECTION")
print("=" * 60)

family_thresholds = [0.001, 0.01, 0.05, 0.10, 0.20, 0.50]

family_results = []

for threshold in family_thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print(f"\nThreshold: {threshold}")

    for attack in ["FTP-Patator", "SSH-Patator"]:

        mask = (
            df["Label"].values == attack
        )

        total = int(mask.sum())
        detected = int(predictions[mask].sum())

        recall = (
            detected / total if total else 0
        )

        print(
            f"{attack}: {detected}/{total} "
            f"({recall * 100:.2f}%)"
        )

        family_results.append({
            "Threshold": threshold,
            "Attack": attack,
            "Total": total,
            "Detected": detected,
            "Recall": recall
        })

# ==================================================
# 6. SAVE RESULTS
# ==================================================

threshold_df = pd.DataFrame(results)

family_df = pd.DataFrame(family_results)

threshold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "adaptive_threshold_results.csv"
    ),
    index=False
)

family_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "adaptive_family_threshold_results.csv"
    ),
    index=False
)

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 60)

print("Saved adaptive_threshold_results.csv")
print("Saved adaptive_family_threshold_results.csv")

print("\nIMPORTANT:")
print(
    "These Tuesday threshold results are exploratory. "
    "Do not use them as independent final test metrics."
)
