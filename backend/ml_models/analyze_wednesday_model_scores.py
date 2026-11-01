
import os
import pandas as pd
import numpy as np

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

PREDICTION_PATH = os.path.join(
    ROOT, "research", "evaluation",
    "aura_wednesday_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT, "research", "evaluation"
)

df = pd.read_csv(PREDICTION_PATH)

print("=" * 70)
print("AURA WEDNESDAY MODEL CONFIDENCE ANALYSIS")
print("=" * 70)

# Fixed thresholds used in the independent evaluation
XGB_THRESHOLD = 0.30
RF_THRESHOLD = 0.02

# 1. Analyze model probabilities by traffic category
print("\n1. MODEL PROBABILITY DISTRIBUTION")
print("-" * 70)

summary = []

for attack in sorted(df["Actual_Label"].unique()):

    subset = df[df["Actual_Label"] == attack]

    xgb = subset["XGB_Probability"]
    rf = subset["RF_Probability"]

    xgb_detected = int(
        (xgb >= XGB_THRESHOLD).sum()
    )

    rf_detected = int(
        (rf >= RF_THRESHOLD).sum()
    )

    total = len(subset)

    row = {
        "Attack": attack,
        "Total": total,
        "XGB_Mean": xgb.mean(),
        "XGB_Median": xgb.median(),
        "XGB_P95": xgb.quantile(0.95),
        "XGB_P99": xgb.quantile(0.99),
        "XGB_Detected": xgb_detected,
        "RF_Mean": rf.mean(),
        "RF_Median": rf.median(),
        "RF_P95": rf.quantile(0.95),
        "RF_P99": rf.quantile(0.99),
        "RF_Detected": rf_detected
    }

    summary.append(row)

summary_df = pd.DataFrame(summary)

print(
    summary_df.round(5).to_string(index=False)
)

# 2. Analyze RF scores for missed attacks
print("\n2. RANDOM FOREST SCORES FOR MISSED ATTACKS")
print("-" * 70)

missed = df[
    (df["Actual_Label"] != "BENIGN") &
    (df["AURA_Prediction"] == 0)
]

for attack in sorted(missed["Actual_Label"].unique()):

    subset = missed[
        missed["Actual_Label"] == attack
    ]

    print("\nAttack:", attack)
    print("Missed records:", len(subset))

    print(
        "RF score percentiles:",
        subset["RF_Probability"].quantile(
            [0.50, 0.75, 0.90, 0.95, 0.99]
        ).round(5).to_dict()
    )

    print(
        "XGB score percentiles:",
        subset["XGB_Probability"].quantile(
            [0.50, 0.75, 0.90, 0.95, 0.99]
        ).round(5).to_dict()
    )

# 3. Compare model contribution
print("\n3. MODEL CONTRIBUTION ANALYSIS")
print("-" * 70)

xgb_flag = (
    df["XGB_Probability"] >= XGB_THRESHOLD
)

rf_flag = (
    df["RF_Probability"] >= RF_THRESHOLD
)

categories = {
    "Both models detected": xgb_flag & rf_flag,
    "Only XGBoost detected": xgb_flag & ~rf_flag,
    "Only Random Forest detected": ~xgb_flag & rf_flag,
    "Neither model detected": ~xgb_flag & ~rf_flag
}

for name, mask in categories.items():

    print(
        name + ":",
        int(mask.sum())
    )

# 4. Save results
summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "wednesday_model_score_summary.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("MODEL CONFIDENCE ANALYSIS COMPLETED")
print("=" * 70)

print(
    "Saved: wednesday_model_score_summary.csv"
)

print(
    "\nNote: Probability distributions describe model confidence, "
    "not necessarily calibrated probabilities."
)
