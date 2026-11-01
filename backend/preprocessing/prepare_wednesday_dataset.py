
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_PATH = os.path.join(
    ROOT,
    "backend",
    "datasets",
    "Wednesday-workingHours.pcap_ISCX.csv"
)

OUTPUT_PATH = os.path.join(
    ROOT,
    "backend",
    "datasets",
    "cleaned_wednesday_dataset.csv"
)

print("=" * 60)
print("WEDNESDAY DATASET PREPROCESSING")
print("=" * 60)

# 1. Load dataset
print("\nLoading dataset...")

df = pd.read_csv(INPUT_PATH, low_memory=False)

print("Original shape:", df.shape)

# 2. Clean column names
df.columns = df.columns.str.strip()

# 3. Clean labels
if "Label" not in df.columns:
    raise ValueError("Label column not found!")

df["Label"] = df["Label"].astype(str).str.strip()

# 4. Replace infinite values
df = df.replace([np.inf, -np.inf], np.nan)

# 5. Remove completely empty columns
df = df.dropna(axis=1, how="all")

# 6. Remove rows with missing labels
df = df.dropna(subset=["Label"])

# 7. Convert feature columns to numeric
feature_columns = [
    col for col in df.columns if col != "Label"
]

for col in feature_columns:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

# 8. Fill missing numerical values with medians
for col in feature_columns:
    median_value = df[col].median()

    if pd.notna(median_value):
        df[col] = df[col].fillna(median_value)
    else:
        df[col] = df[col].fillna(0)

# 9. Remove duplicate rows
before = len(df)

df = df.drop_duplicates().reset_index(drop=True)

duplicates_removed = before - len(df)

# 10. Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

# 11. Display results
print("\nPREPROCESSING COMPLETED")
print("-" * 60)

print("Final shape:", df.shape)
print("Duplicates removed:", duplicates_removed)
print("Missing values:", df.isnull().sum().sum())
print("Infinite values:", np.isinf(
    df.select_dtypes(include=[np.number]).to_numpy()
).sum())

print("\nAttack distribution:")
print(df["Label"].value_counts().to_string())

print("\nSaved to:")
print(OUTPUT_PATH)

print("=" * 60)
