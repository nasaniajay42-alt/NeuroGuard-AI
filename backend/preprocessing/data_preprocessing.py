
import pandas as pd
import numpy as np
from pathlib import Path


def clean_column_names(df):
    """Remove unwanted spaces from column names."""
    df.columns = df.columns.astype(str).str.strip()
    return df


def clean_network_data(df, label_column="Label"):
    """Clean network traffic data without changing its labels."""

    print("\n[NEUROGUARD AI] Starting data preprocessing...")

    # Clean column names
    df = clean_column_names(df.copy())

    # Replace infinite values with missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove completely empty columns
    df.dropna(axis=1, how="all", inplace=True)

    # Remove rows without a target label
    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df.dropna(subset=[label_column], inplace=True)

    # Remove extra spaces from labels
    df[label_column] = df[label_column].astype(str).str.strip()

    print("[SUCCESS] Data cleaning completed.")

    return df


def load_network_csv(file_path, label_column="Label"):
    """Load and clean a network traffic CSV dataset."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    print(f"\nLoading dataset: {file_path.name}")

    df = pd.read_csv(file_path, low_memory=False)

    print(f"Original dataset shape: {df.shape}")

    df = clean_network_data(df, label_column)
    # Remove duplicate records
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed duplicates: {before - len(df)}")

   
    numeric_columns = df.select_dtypes(include=np.number).columns
    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].median()
    )

    print("Missing values after cleaning:",
          df.isnull().sum().sum())
    print(f"Cleaned dataset shape: {df.shape}")
    print("\nTraffic label distribution:")
    print(df[label_column].value_counts().head(15))

    print("\nMissing values:", int(df.isna().sum().sum()))

    return df



if __name__ == "__main__":

    print("======================================")
    print("       NEUROGUARD AI - AURA")
    print("       Dataset Loading Test")
    print("======================================")

    dataset_path = (
        "backend/datasets/"
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    )

    try:
        df = load_network_csv(dataset_path)
        output_path = "backend/datasets/cleaned_ddos_dataset.csv"

        df.to_csv(output_path, index=False)

        print("\nCleaned dataset saved successfully!")
        print("\nFirst 5 records:")
        print(df.head())

        print("\nTotal records:", len(df))
        print("Total features:", len(df.columns))

        print("\nDataset information:")
        df.info()

        print("\nDataset shape:")
        print(df.shape)

        print("\nAttack categories:")
        print(df["Label"].value_counts())
        print("\nMissing values:")
        print(df.isnull().sum().sum())

        print("\nDuplicate records:")
        print(df.duplicated().sum())

        print("\nInfinite values:")
        print(np.isinf(df.select_dtypes(include=np.number)).sum().sum())
        print("\nNormal traffic count:")
        print((df["Label"] == "BENIGN").sum())

        print("\nAttack traffic count:")
        print((df["Label"] != "BENIGN").sum())

    except Exception as error:
        print("\nERROR:", error)
