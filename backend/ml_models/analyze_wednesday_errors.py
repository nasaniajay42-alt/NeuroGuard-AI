
import os
import numpy as np
import pandas as pd

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

print("=" * 65)
print("AURA WEDNESDAY ERROR ANALYSIS")
print("=" * 65)

# 1. Load original data and predictions
df = pd.read_csv(DATA_PATH)
pred = pd.read_csv(PREDICTION_PATH)

if len(df) != len(pred):
    raise ValueError("Dataset and prediction row counts do not match.")

if not np.array_equal(
    df["Label"].astype(str).values,
    pred["Actual_Label"].astype(str).values
):
    raise ValueError("Dataset and prediction labels are not aligned.")

df["AURA_Prediction"] = pred["AURA_Prediction"].values

# 2. Identify traffic groups
df["Actual_Class"] = np.where(
    df["Label"] == "BENIGN", "BENIGN", "ATTACK"
)

df["Detection_Status"] = "Correct"

df.loc[
    (df["Actual_Class"] == "ATTACK") &
    (df["AURA_Prediction"] == 0),
    "Detection_Status"
] = "Missed Attack"

df.loc[
    (df["Actual_Class"] == "BENIGN") &
    (df["AURA_Prediction"] == 1),
    "Detection_Status"
] = "False Positive"

df.loc[
    (df["Actual_Class"] == "ATTACK") &
    (df["AURA_Prediction"] == 1),
    "Detection_Status"
] = "Detected Attack"

# 3. Overall error summary
print("\n1. OVERALL DETECTION SUMMARY")
print("-" * 65)

print(df["Detection_Status"].value_counts().to_string())

# 4. Attack-family error analysis
print("\n2. ATTACK FAMILY ANALYSIS")
print("-" * 65)

attack_results = []

for attack in sorted(df["Label"].unique()):

    if attack == "BENIGN":
        continue

    subset = df[df["Label"] == attack]

    total = len(subset)

    detected = int(
        (subset["AURA_Prediction"] == 1).sum()
    )

    missed = total - detected

    detection_rate = detected / total * 100 if total else 0

    attack_results.append({
        "Attack": attack,
        "Total": total,
        "Detected": detected,
        "Missed": missed,
        "Detection_Rate_Percent": detection_rate
    })

    print(f"\nAttack: {attack}")
    print("Total:", total)
    print("Detected:", detected)
    print("Missed:", missed)
    print("Detection rate:", round(detection_rate, 2), "%")

pd.DataFrame(attack_results).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "wednesday_attack_error_summary.csv"
    ),
    index=False
)

# 5. Select numeric features
feature_columns = [
    col for col in df.select_dtypes(include=[np.number]).columns
    if col not in ["AURA_Prediction"]
]

# 6. Compare detected attacks and missed attacks
print("\n3. DETECTED VS MISSED ATTACKS")
print("-" * 65)

detected_attacks = df[
    df["Detection_Status"] == "Detected Attack"
]

missed_attacks = df[
    df["Detection_Status"] == "Missed Attack"
]

comparison = []

for feature in feature_columns:

    detected_values = detected_attacks[feature].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    missed_values = missed_attacks[feature].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(detected_values) == 0 or len(missed_values) == 0:
        continue

    detected_median = detected_values.median()
    missed_median = missed_values.median()

    pooled_std = np.sqrt(
        (
            detected_values.var() +
            missed_values.var()
        ) / 2
    )

    if pd.isna(pooled_std) or pooled_std == 0:
        effect_size = 0
    else:
        effect_size = (
            detected_values.mean() -
            missed_values.mean()
        ) / pooled_std

    comparison.append({
        "Feature": feature,
        "Detected_Median": detected_median,
        "Missed_Median": missed_median,
        "Detected_Mean": detected_values.mean(),
        "Missed_Mean": missed_values.mean(),
        "Absolute_Standardized_Difference": abs(effect_size)
    })

comparison_df = pd.DataFrame(comparison)

comparison_df = comparison_df.sort_values(
    "Absolute_Standardized_Difference",
    ascending=False
)

print("\nTop 15 features separating detected and missed attacks:")
print(
    comparison_df.head(15).round(4).to_string(index=False)
)

comparison_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "wednesday_detected_vs_missed_features.csv"
    ),
    index=False
)

# 7. Compare benign false positives and true negatives
print("\n4. BENIGN FALSE POSITIVES VS TRUE NEGATIVES")
print("-" * 65)

false_positives = df[
    df["Detection_Status"] == "False Positive"
]

true_negatives = df[
    (df["Actual_Class"] == "BENIGN") &
    (df["AURA_Prediction"] == 0)
]

benign_comparison = []

for feature in feature_columns:

    fp_values = false_positives[feature].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    tn_values = true_negatives[feature].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(fp_values) == 0 or len(tn_values) == 0:
        continue

    pooled_std = np.sqrt(
        (fp_values.var() + tn_values.var()) / 2
    )

    if pd.isna(pooled_std) or pooled_std == 0:
        effect_size = 0
    else:
        effect_size = (
            fp_values.mean() - tn_values.mean()
        ) / pooled_std

    benign_comparison.append({
        "Feature": feature,
        "False_Positive_Median": fp_values.median(),
        "True_Negative_Median": tn_values.median(),
        "False_Positive_Mean": fp_values.mean(),
        "True_Negative_Mean": tn_values.mean(),
        "Absolute_Standardized_Difference": abs(effect_size)
    })

benign_df = pd.DataFrame(benign_comparison)

benign_df = benign_df.sort_values(
    "Absolute_Standardized_Difference",
    ascending=False
)

print("\nTop 15 features separating benign false positives and true negatives:")
print(
    benign_df.head(15).round(4).to_string(index=False)
)

benign_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "wednesday_benign_error_features.csv"
    ),
    index=False
)

# 8. Save individual error records
df[df["Detection_Status"] == "Missed Attack"].to_csv(
    os.path.join(OUTPUT_DIR, "wednesday_missed_attacks.csv"),
    index=False
)

df[df["Detection_Status"] == "False Positive"].to_csv(
    os.path.join(OUTPUT_DIR, "wednesday_false_positives.csv"),
    index=False
)

print("\n" + "=" * 65)
print("WEDNESDAY ERROR ANALYSIS COMPLETED")
print("=" * 65)

print("\nGenerated files:")
print("- wednesday_attack_error_summary.csv")
print("- wednesday_detected_vs_missed_features.csv")
print("- wednesday_benign_error_features.csv")
print("- wednesday_missed_attacks.csv")
print("- wednesday_false_positives.csv")

print("\nNote: Feature differences show associations, not causation.")
