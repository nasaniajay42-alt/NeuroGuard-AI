import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

path = os.path.join(
    BASE_DIR,
    "datasets",
    "cleaned_tuesday_dataset.csv"
)

df = pd.read_csv(path)

features = [
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Flow IAT Mean",
    "Fwd IAT Min",
    "Bwd IAT Min",
    "Flow Packets/s"
]

benign = df[df["Label"].str.strip() == "BENIGN"].copy()

print("BENIGN records:", len(benign))

for feature in features:

    values = pd.to_numeric(
        benign[feature], errors="coerce"
    )

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 3 * iqr

    outliers = values > upper_limit

    print(f"\nFeature: {feature}")
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR upper limit:", upper_limit)
    print("Extreme records:", outliers.sum())
    print("Maximum:", values.max())

print("\nInspection completed.")