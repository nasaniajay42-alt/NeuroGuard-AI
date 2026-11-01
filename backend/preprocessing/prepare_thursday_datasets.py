
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(ROOT, "backend", "datasets")

FILES = {
    "WebAttacks": (
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    ),
    "Infiltration": (
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    )
}

def clean_dataset(name, filename):

    input_path = os.path.join(DATA_DIR, filename)

    output_path = os.path.join(
        DATA_DIR,
        f"cleaned_thursday_{name.lower()}.csv"
    )

    print("\n" + "=" * 65)
    print("PROCESSING:", name)
    print("=" * 65)

    if not os.path.exists(input_path):
        print("ERROR: File not found:", input_path)
        return

    print("Loading dataset...")

    df = pd.read_csv(input_path, low_memory=False)

    print("Original shape:", df.shape)

    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError("Label column not found!")

    df["Label"] = df["Label"].astype(str).str.strip()

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(axis=1, how="all")

    df = df.dropna(subset=["Label"])

    feature_columns = [
        col for col in df.columns if col != "Label"
    ]

    for col in feature_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

        median_value = df[col].median()

        if pd.notna(median_value):
            df[col] = df[col].fillna(median_value)
        else:
            df[col] = df[col].fillna(0)

    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    duplicates_removed = before - len(df)

    df.to_csv(output_path, index=False)

    print("\nPREPROCESSING COMPLETED")
    print("Final shape:", df.shape)
    print("Duplicates removed:", duplicates_removed)
    print("Missing values:", df.isnull().sum().sum())

    print("\nAttack distribution:")
    print(df["Label"].value_counts().to_string())

    print("\nSaved:", output_path)


for name, filename in FILES.items():
    clean_dataset(name, filename)

print("\n" + "=" * 65)
print("THURSDAY DATASET PREPARATION FINISHED")
print("=" * 65)
