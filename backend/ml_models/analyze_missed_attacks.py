

import os
import numpy as np
import pandas as pd

# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_PATH = os.path.join(
    ROOT,
    "research",
    "experiments",
    "aura_error_analysis.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "research",
    "experiments"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 2. LOAD DATA
# ==================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("AURA MISSED ATTACK ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)

# ==================================================
# 3. IDENTIFY ERROR CATEGORIES
# ==================================================

missed_attacks = df[
    (df["Label"] != "BENIGN") &
    (df["AURA_Prediction"] == 0)
].copy()

detected_attacks = df[
    (df["Label"] != "BENIGN") &
    (df["AURA_Prediction"] == 1)
].copy()

false_positives = df[
    (df["Label"] == "BENIGN") &
    (df["AURA_Prediction"] == 1)
].copy()

true_negatives = df[
    (df["Label"] == "BENIGN") &
    (df["AURA_Prediction"] == 0)
].copy()

print("\nMissed attack distribution:")
print(missed_attacks["Label"].value_counts())

print("\nTotal missed attacks:", len(missed_attacks))
print("Total detected attacks:", len(detected_attacks))
print("Total false positives:", len(false_positives))
print("Total true negatives:", len(true_negatives))

# ==================================================
# 4. ATTACK FAMILY ANALYSIS
# ==================================================

for attack in ["FTP-Patator", "SSH-Patator"]:

    missed = missed_attacks[
        missed_attacks["Label"] == attack
    ]

    detected = detected_attacks[
        detected_attacks["Label"] == attack
    ]

    print("\n" + "=" * 60)
    print("ATTACK:", attack)
    print("=" * 60)

    print("Missed attacks:", len(missed))
    print("Detected attacks:", len(detected))

    if len(missed) > 0:
        print("\nMissed attack feature medians:")
        print(
            missed.select_dtypes(include=np.number)
            .median()
            .round(3)
            .to_string()
        )

    if len(detected) > 0:
        print("\nDetected attack feature medians:")
        print(
            detected.select_dtypes(include=np.number)
            .median()
            .round(3)
            .to_string()
        )

# ==================================================
# 5. BENIGN TRAFFIC COMPARISON
# ==================================================

print("\n" + "=" * 60)
print("BENIGN TRAFFIC COMPARISON")
print("=" * 60)

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

excluded_columns = [
    "AE_Prediction",
    "XGB_Prediction",
    "AURA_Prediction"
]

feature_columns = [
    col for col in numeric_columns
    if col not in excluded_columns
]

comparison = pd.DataFrame({
    "False_Positive_Median": false_positives[
        feature_columns
    ].median(),

    "True_Negative_Median": true_negatives[
        feature_columns
    ].median()
})

comparison["Absolute_Median_Difference"] = (
    comparison["False_Positive_Median"] -
    comparison["True_Negative_Median"]
).abs()

comparison = comparison.sort_values(
    "Absolute_Median_Difference",
    ascending=False
)

print("\nTop 15 features with different medians:")
print(comparison.head(15).round(4).to_string())

# ==================================================
# 6. MODEL SCORE COMPARISON
# ==================================================

print("\n" + "=" * 60)
print("MODEL SCORE COMPARISON")
print("=" * 60)

groups = {
    "Missed FTP": missed_attacks[
        missed_attacks["Label"] == "FTP-Patator"
    ],

    "Missed SSH": missed_attacks[
        missed_attacks["Label"] == "SSH-Patator"
    ],

    "Detected attacks": detected_attacks,

    "False positives": false_positives,

    "True negatives": true_negatives
}

for name, group in groups.items():

    print("\n" + "-" * 50)
    print(name, "(n =", len(group), ")")

    if len(group) == 0:
        print("No records available.")
        continue

    score_summary = group[
        ["AE_Error", "XGB_Probability"]
    ].describe(
        percentiles=[0.50, 0.90, 0.95, 0.99]
    )

    print(
        score_summary.loc[
            ["mean", "50%", "90%", "95%", "99%"]
        ].round(6).to_string()
    )

# ==================================================
# 7. SAVE ANALYSIS RESULTS
# ==================================================

missed_path = os.path.join(
    OUTPUT_DIR,
    "aura_missed_attacks.csv"
)

false_positive_path = os.path.join(
    OUTPUT_DIR,
    "aura_false_positives.csv"
)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "aura_feature_comparison.csv"
)

missed_attacks.to_csv(
    missed_path,
    index=False
)

false_positives.to_csv(
    false_positive_path,
    index=False
)

comparison.to_csv(
    comparison_path
)

# ==================================================
# 8. COMPLETION
# ==================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nSaved files:")
print(missed_path)
print(false_positive_path)
print(comparison_path)
