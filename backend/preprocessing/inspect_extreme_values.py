import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

files = {
    "Friday": "cleaned_ddos_dataset.csv",
    "Tuesday": "cleaned_tuesday_dataset.csv"
}

features_to_check = [
    "Fwd IAT Min",
    "Bwd IAT Min",
    "Bwd IAT Mean",
    "Fwd IAT Mean",
    "Flow IAT Mean",
    "Flow Packets/s",
    "FIN Flag Count"
]

for dataset_name, filename in files.items():

    path = os.path.join(DATASET_DIR, filename)
    df = pd.read_csv(path)

    print(f"\n========== {dataset_name} ==========")

    for feature in features_to_check:

        if feature not in df.columns:
            continue

        values = pd.to_numeric(
            df[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        print(f"\n{feature}")
        print("Minimum:", values.min())
        print("Median:", values.median())
        print("99th percentile:", values.quantile(0.99))
        print("Maximum:", values.max())
        print("Values above 1 billion:", (values > 1e9).sum())