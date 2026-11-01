
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(ROOT, "backend", "datasets")
MODEL_DIR = os.path.join(ROOT, "backend", "saved_models")
OUTPUT_DIR = os.path.join(ROOT, "research", "evaluation")

os.makedirs(OUTPUT_DIR, exist_ok=True)

XGB_THRESHOLD = 0.30
RF_THRESHOLD = 0.018

DATASETS = {
    "WebAttacks": "cleaned_thursday_webattacks.csv",
    "Infiltration": "cleaned_thursday_infiltration.csv"
}

print("=" * 70)
print("AURA THURSDAY INDEPENDENT EVALUATION")
print("=" * 70)

# Load models
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


def evaluate_dataset(name, filename):

    print("\n" + "=" * 70)
    print("DATASET:", name)
    print("=" * 70)

    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)

    df["Label"] = df["Label"].astype(str).str.strip()

    y = (df["Label"] != "BENIGN").astype(int).values

    X_xgb = df[xgb_features].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    X_rf = df[rf_features].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    print("Total records:", len(df))
    print("Benign:", int((y == 0).sum()))
    print("Attacks:", int((y == 1).sum()))

    # Predictions
    print("\nGenerating predictions...")

    xgb_prob = xgb_model.predict_proba(X_xgb)[:, 1]
    rf_prob = rf_model.predict_proba(X_rf)[:, 1]

    xgb_pred = (xgb_prob >= XGB_THRESHOLD).astype(int)
    rf_pred = (rf_prob >= RF_THRESHOLD).astype(int)

    aura_pred = (
        (xgb_pred == 1) | (rf_pred == 1)
    ).astype(int)

    aura_score = np.maximum(
        xgb_prob / XGB_THRESHOLD,
        rf_prob / RF_THRESHOLD
    )

    models = {
        "XGBoost": (xgb_pred, xgb_prob),
        "Random Forest": (rf_pred, rf_prob),
        "AURA": (aura_pred, aura_score)
    }

    results = []

    for model_name, (prediction, score) in models.items():

        tn, fp, fn, tp = confusion_matrix(
            y, prediction, labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y, prediction, zero_division=0
        )

        recall = recall_score(
            y, prediction, zero_division=0
        )

        f1 = f1_score(
            y, prediction, zero_division=0
        )

        fpr = fp / (fp + tn) if (fp + tn) else 0

        # AUC is undefined if only one class exists
        if len(np.unique(y)) == 2:
            roc = roc_auc_score(y, score)
            pr = average_precision_score(y, score)
        else:
            roc = np.nan
            pr = np.nan

        results.append({
            "Dataset": name,
            "Model": model_name,
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
        })

        print("\nMODEL:", model_name)
        print("-" * 45)
        print("Confusion Matrix:", [[tn, fp], [fn, tp]])
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1:", round(f1, 4))
        print("FPR:", round(fpr, 4))
        print("ROC-AUC:", round(roc, 4) if not np.isnan(roc) else "N/A")
        print("PR-AUC:", round(pr, 4) if not np.isnan(pr) else "N/A")

    # Attack-family detection
    print("\nATTACK FAMILY DETECTION")
    print("-" * 55)

    family_results = []

    for attack in sorted(df["Label"].unique()):

        if attack == "BENIGN":
            continue

        mask = df["Label"].values == attack

        total = int(mask.sum())
        detected = int(aura_pred[mask].sum())

        rate = detected / total if total else 0

        family_results.append({
            "Dataset": name,
            "Attack": attack,
            "Total": total,
            "Detected": detected,
            "Missed": total - detected,
            "Detection_Rate": rate
        })

        print(
            f"{attack}: {detected}/{total} "
            f"({rate * 100:.2f}%)"
        )

    # Save predictions
    prediction_df = pd.DataFrame({
        "Actual_Label": df["Label"],
        "XGB_Probability": xgb_prob,
        "RF_Probability": rf_prob,
        "XGB_Prediction": xgb_pred,
        "RF_Prediction": rf_pred,
        "AURA_Prediction": aura_pred
    })

    prediction_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"aura_thursday_{name.lower()}_predictions.csv"
        ),
        index=False
    )

    return results, family_results


all_results = []
all_families = []

for name, filename in DATASETS.items():

    results, families = evaluate_dataset(name, filename)

    all_results.extend(results)
    all_families.extend(families)


# Save combined reports
pd.DataFrame(all_results).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "aura_thursday_model_comparison.csv"
    ),
    index=False
)

pd.DataFrame(all_families).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "aura_thursday_attack_detection.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("THURSDAY EVALUATION COMPLETED")
print("=" * 70)

print("XGBoost threshold:", XGB_THRESHOLD)
print("Random Forest threshold:", RF_THRESHOLD)

print("\nReports saved in research/evaluation/")
print("Thresholds were not tuned on Thursday.")
