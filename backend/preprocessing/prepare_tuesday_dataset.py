import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_PATH = os.path.join(
    BASE_DIR, "datasets",
    "Tuesday-WorkingHours.pcap_ISCX.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "datasets",
    "cleaned_tuesday_dataset.csv"
)

print("Loading Tuesday dataset...")
df = pd.read_csv(INPUT_PATH, low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()

# Remove completely empty columns
df.dropna(axis=1, how="all", inplace=True)

# Clean labels
df["Label"] = df["Label"].astype(str).str.strip()

# Convert feature columns to numeric
feature_columns = [c for c in df.columns if c != "Label"]

for column in feature_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Replace infinite values with missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows without labels
df.dropna(subset=["Label"], inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Fill missing numeric values using column medians
for column in feature_columns:
    df[column] = df[column].fillna(df[column].median())

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaning completed!")
print("Dataset shape:", df.shape)
print("\nLabel distribution:")
print(df["Label"].value_counts())
print("\nMissing values:", df.isna().sum().sum())
print("\nSaved to:", OUTPUT_PATH)